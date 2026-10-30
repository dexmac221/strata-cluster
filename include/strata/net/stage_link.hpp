#pragma once
/// LAYER SPLIT ACROSS TWO PCs (experimental, strati-in-rete): one TCP connection between the stage that runs layers
/// [0, K) and the stage that runs [K, n_layers) plus the head, on another machine.  A verify window crosses it once
/// each way: the first stage sends its hand-off (Verifier::handoff_floats per token, 51,216 B on qwen4exp) with the
/// tokens and positions; the last stage answers with its picks.  Measured on the gigabit LAN this was written for
/// (two Ryzen desktops, 28 Oct 2026): 2 KB round trip 158 us, 64 KB 1.36 ms, 116 MB/s at most.
///
/// Blocking sockets with TCP_NODELAY (a window is one request and one reply: Nagle's delay would sit on every one),
/// 4 MiB socket buffers, and a 16-byte frame header so either side can check what it reads.  No threads, no copies
/// beyond the kernel's: a message is written from its parts with writev and read straight into the caller's buffers.
/// Nothing here knows about CUDA; the caller passes host pointers (mapped hand-off memory is host-visible).

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>

namespace strata::net {

/// What a frame carries.  The payload layouts are the caller's (see the structs below).
enum class Msg : uint32_t {
    Hello = 1,    ///< both ways at connect: StageHello, so a mismatched pair stops before the first window
    Window = 2,   ///< first stage -> last: WindowHead, int32 tokens[T], float hand-off[T * handoff_floats]
    Result = 3,   ///< last stage -> first: ResultHead, int32 picks[T]
    Bye = 4,      ///< either way: the sender is closing
    Line = 5,     ///< first stage -> last: one request line, as the engine read it from its stdin
    Drafts = 6,   ///< last stage -> first: DraftsHead, int32 drafts[n], float probabilities[n]
    Chunk = 7,    ///< first stage -> last, a prompt chunk: ChunkHead, int64 tokens[T], float rows[T * hc * n_embd]
    ChunkAck = 8, ///< last stage -> first: AckHead (+ an error string), once its first stage has read the chunk
    PromptEnd = 9,///< first stage -> last: the prompt read is over (every chunk was acknowledged): EndHead (+ an error)
};

/// One prompt chunk the first PC's last stage hands on (Prefill's chunk hand-off, the rows without bo/inject).
struct ChunkHead {
    int64_t c0 = 0;            ///< the chunk's first token, counted in the prompt part this run reads
    int64_t T = 0;             ///< its tokens
    int64_t p0 = 0;            ///< its first position
    int32_t single_chunk = 0;  ///< the flags a later stage takes from the stage before it
    int32_t pipe_chunked = 0;
};

struct AckHead {
    int32_t ok = 0;
    uint32_t err_bytes = 0;
};

/// How the first PC's prompt read ended: the last PC's ends the same way (a STOP mid-prompt is "cancelled" on both).
struct EndHead {
    int32_t ok = 0;            ///< the first PC's Prefill::run result; 0: an error string of `err_bytes` follows
    int32_t stop = 0;          ///< the first PC's STOP, latched here (see WindowHead::flags)
    uint32_t err_bytes = 0;
    uint32_t pad = 0;
};

/// The drafter's answer after a window (it runs on the last stage, next to the final rows it reads).
struct DraftsHead {
    int32_t ok = 0;   ///< MtpDrafter::draft's return
    int32_t n = 0;    ///< the entries of drafts and probabilities that follow
};

/// What both stages must agree on before a window crosses.
struct StageHello {
    uint32_t version = 2;       ///< this file's protocol (2: the STOP travels with the windows and the prompt's end)
    uint32_t layer_split = 0;   ///< K: the first layer of the last stage
    uint32_t n_layers = 0;
    uint32_t n_embd = 0;
    uint32_t hc = 0;
    uint32_t max_t = 0;         ///< the most tokens a window holds
    uint64_t handoff_floats = 0;   ///< per token
    uint64_t prefill_chunk = 0;    ///< the prompt path's chunk (the last PC reads the first PC's chunks)
};

/// WindowHead::flags: the first PC had a STOP when it sent this window.  A STOP takes effect on both PCs at the same
/// point: the first PC's engine latches the STOP its stdin read into its request state as it sends a window (or the
/// prompt's end), and the last PC's takes that state from the message, so both end the request after the same window.
constexpr int32_t kWindowStop = 1;

struct WindowHead {
    int32_t T = 0;
    int32_t flags = 0;          ///< kWindowStop
    int64_t pos0 = 0;
};

struct ResultHead {
    int32_t T = 0;
    int32_t ok = 0;             ///< 1: picks follow; 0: an error string of `err_bytes` follows instead
    int32_t nf_row = -1;        ///< Verifier::nonfinite_row() of the last stage
    uint32_t nf_mask = 0;       ///< rows with a non-finite logit (bit t)
    int64_t nf_pos = -1;
    uint32_t err_bytes = 0;
    uint32_t pad = 0;
};

class StageLink {
public:
    StageLink() = default;
    StageLink(const StageLink&) = delete;
    StageLink& operator=(const StageLink&) = delete;
    ~StageLink() { close(); }

    /// The last stage's side: listen on `port` (all interfaces) and wait for the first stage to connect.
    bool listen_accept(int port, std::string& err);
    /// The first stage's side: connect to `host:port`, retrying for up to `timeout_s` while the other side starts.
    bool connect_to(const std::string& host, int port, int timeout_s, std::string& err);
    void close();
    bool connected() const { return fd_ >= 0; }

    /// One frame from `parts` (pointer, bytes), written with writev: the header first, then every part in order.
    bool send(Msg kind, std::initializer_list<std::pair<const void*, size_t>> parts, std::string& err);
    /// The next frame's header; `kind` and `bytes` are what it says.  A bad magic is an error (a desynced stream).
    bool recv_header(Msg& kind, uint64_t& bytes, std::string& err);
    /// Exactly `n` payload bytes into `p`.
    bool recv_bytes(void* p, size_t n, std::string& err);
    /// A whole frame of the expected kind into `parts`, whose sizes must add up to the frame's payload.
    bool recv(Msg kind, std::initializer_list<std::pair<void*, size_t>> parts, std::string& err);

    /// Counters for the engine log: frames and bytes each way, and the time spent waiting inside recv_header.
    uint64_t frames_sent = 0, frames_received = 0, bytes_sent = 0, bytes_received = 0;
    double wait_us = 0.0;

private:
    bool tune(std::string& err);   // TCP_NODELAY and the socket buffers on fd_
    int fd_ = -1;
};

}   // namespace strata::net
