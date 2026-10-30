// strati-in-rete: StageLink over loopback (or two PCs).  The "last stage" thread receives windows shaped like the real
// ones (WindowHead, T tokens, T * 12804 floats of hand-off), checks every float against the pattern the sender wrote,
// and answers with picks derived from the tokens.  Prints the round trip per T.
//   stage_link_test                      both sides in this process over 127.0.0.1
//   stage_link_test serve PORT           the last stage only (run it on the other PC)
//   stage_link_test connect HOST PORT    the first stage only
#include "strata/net/stage_link.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using strata::net::Msg;
using strata::net::ResultHead;
using strata::net::StageHello;
using strata::net::StageLink;
using strata::net::WindowHead;

namespace {
constexpr uint64_t HB = 4 * 2560 + 2560 + 4;   // Verifier::handoff_floats on qwen4exp (hc 4, n_embd 2560)
constexpr int kMaxT = 8;

float pattern(int64_t pos0, uint64_t i) { return (float) ((pos0 * 131 + (int64_t) i * 7) % 100003) * 0.001f; }

StageHello hello() {
    StageHello h;
    h.layer_split = 24; h.n_layers = 48; h.n_embd = 2560; h.hc = 4; h.max_t = kMaxT; h.handoff_floats = HB;
    return h;
}

bool same(const StageHello& a, const StageHello& b) { return std::memcmp(&a, &b, sizeof a) == 0; }

int serve(int port) {
    StageLink link;
    std::string err;
    if (!link.listen_accept(port, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    StageHello mine = hello(), theirs;
    if (!link.recv(Msg::Hello, {{&theirs, sizeof theirs}}, err) || !link.send(Msg::Hello, {{&mine, sizeof mine}}, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    if (!same(mine, theirs)) { std::fprintf(stderr, "hello mismatch\n"); return 1; }
    std::vector<int32_t> tok(kMaxT), picks(kMaxT);
    std::vector<float> hand((size_t) kMaxT * HB);
    uint64_t windows = 0, bad = 0;
    for (;;) {
        Msg k;
        uint64_t bytes = 0;
        if (!link.recv_header(k, bytes, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        if (k == Msg::Bye) break;
        WindowHead w;
        if (k != Msg::Window || bytes < sizeof w || !link.recv_bytes(&w, sizeof w, err) || w.T < 1 || w.T > kMaxT ||
            bytes != sizeof w + (uint64_t) w.T * (sizeof(int32_t) + HB * sizeof(float)) ||
            !link.recv_bytes(tok.data(), (size_t) w.T * sizeof(int32_t), err) ||
            !link.recv_bytes(hand.data(), (size_t) w.T * HB * sizeof(float), err)) {
            std::fprintf(stderr, "bad window: %s\n", err.c_str());
            return 1;
        }
        for (uint64_t i = 0; i < (uint64_t) w.T * HB; ++i)
            if (hand[i] != pattern(w.pos0, i)) { ++bad; break; }
        for (int t = 0; t < w.T; ++t) picks[t] = tok[t] * 3 + t;
        ResultHead r;
        r.T = w.T;
        r.ok = 1;
        if (!link.send(Msg::Result, {{&r, sizeof r}, {picks.data(), (size_t) w.T * sizeof(int32_t)}}, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        ++windows;
    }
    std::printf("last stage: %llu windows, %llu with a wrong hand-off, %.1f MB received\n", (unsigned long long) windows,
                (unsigned long long) bad, link.bytes_received / 1e6);
    return bad == 0 ? 0 : 2;
}

int first_stage(const std::string& host, int port) {
    StageLink link;
    std::string err;
    if (!link.connect_to(host, port, 30, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    StageHello mine = hello(), theirs;
    if (!link.send(Msg::Hello, {{&mine, sizeof mine}}, err) || !link.recv(Msg::Hello, {{&theirs, sizeof theirs}}, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    std::vector<int32_t> tok(kMaxT), picks(kMaxT);
    std::vector<float> hand((size_t) kMaxT * HB);
    int wrong = 0;
    std::printf("%3s %10s %8s %8s %8s\n", "T", "byte", "p50 us", "p90 us", "p99 us");
    int64_t pos = 0;
    for (int T : {1, 2, 3, 4, 6, 8}) {
        std::vector<double> v;
        for (int i = -20; i < 400; ++i, pos += T) {
            WindowHead w;
            w.T = T;
            w.pos0 = pos;
            for (int t = 0; t < T; ++t) tok[t] = (int32_t) (pos + t);
            for (uint64_t j = 0; j < (uint64_t) T * HB; ++j) hand[j] = pattern(pos, j);
            const auto t0 = std::chrono::steady_clock::now();
            ResultHead r;
            if (!link.send(Msg::Window, {{&w, sizeof w}, {tok.data(), (size_t) T * sizeof(int32_t)},
                                         {hand.data(), (size_t) T * HB * sizeof(float)}}, err) ||
                !link.recv(Msg::Result, {{&r, sizeof r}, {picks.data(), (size_t) T * sizeof(int32_t)}}, err)) {
                std::fprintf(stderr, "%s\n", err.c_str());
                return 1;
            }
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            for (int t = 0; t < T; ++t) wrong += picks[t] != tok[t] * 3 + t;
            if (i >= 0) v.push_back(us);
        }
        std::sort(v.begin(), v.end());
        std::printf("%3d %10llu %8.0f %8.0f %8.0f\n", T,
                    (unsigned long long) (sizeof(WindowHead) + T * (4 + HB * 4)), v[v.size() / 2], v[v.size() * 9 / 10],
                    v[v.size() * 99 / 100]);
    }
    if (!link.send(Msg::Bye, {}, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::printf("first stage: %d wrong picks\n", wrong);
    return wrong == 0 ? 0 : 2;
}
}   // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "serve") return serve(std::atoi(argv[2]));
    if (argc >= 4 && std::string(argv[1]) == "connect") return first_stage(argv[2], std::atoi(argv[3]));
    int rc_serve = 0;
    std::thread t([&] { rc_serve = serve(47811); });
    const int rc = first_stage("127.0.0.1", 47811);
    t.join();
    return rc != 0 ? rc : rc_serve;
}
