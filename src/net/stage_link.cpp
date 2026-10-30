#include "strata/net/stage_link.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace strata::net {
namespace {

constexpr uint32_t kMagic = 0x4C525453u;   // "STRL" in memory order
constexpr int kSockBuf = 4 << 20;

struct Frame {
    uint32_t magic = kMagic;
    uint32_t kind = 0;
    uint64_t bytes = 0;
};
static_assert(sizeof(Frame) == 16, "the frame header is 16 bytes on both sides");

std::string sys_err(const char* what) { return std::string("stage link: ") + what + ": " + std::strerror(errno); }

double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}   // namespace

bool StageLink::tune(std::string& err) {
    const int one = 1;
    if (setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one) != 0) { err = sys_err("TCP_NODELAY"); return false; }
    // larger than one T=8 window's hand-off (410 KB) so a send never waits on the receiver's reads
    setsockopt(fd_, SOL_SOCKET, SO_SNDBUF, &kSockBuf, sizeof kSockBuf);
    setsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &kSockBuf, sizeof kSockBuf);
    // A silent failure (a pulled cable, a frozen PC) sends no reset: without this a recv() waits for ever.  The kernel
    // probes an idle link after 5 s, every 2 s, and gives up after 3 unanswered probes; data left unacknowledged for
    // STRATA_NET_TIMEOUT_S (default 10 s) also ends it.  A peer that is busy computing still answers the probes (its
    // kernel does), so a long prompt chunk or an idle gap between requests is not a failure.  0 turns both off.
    const char* t = std::getenv("STRATA_NET_TIMEOUT_S");
    const int timeout_s = t != nullptr ? std::atoi(t) : 10;
    if (timeout_s > 0) {
        const int idle = 5, intvl = 2, cnt = 3;
        const unsigned user_ms = (unsigned) timeout_s * 1000u;
        if (setsockopt(fd_, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof one) != 0 ||
            setsockopt(fd_, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof idle) != 0 ||
            setsockopt(fd_, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof intvl) != 0 ||
            setsockopt(fd_, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof cnt) != 0 ||
            setsockopt(fd_, IPPROTO_TCP, TCP_USER_TIMEOUT, &user_ms, sizeof user_ms) != 0) {
            err = sys_err("keepalive");
            return false;
        }
    }
    return true;
}

bool StageLink::listen_accept(int port, std::string& err) {
    close();
    const int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) { err = sys_err("socket"); return false; }
    const int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t) port);
    a.sin_addr.s_addr = INADDR_ANY;
    if (bind(s, (sockaddr*) &a, sizeof a) != 0 || listen(s, 1) != 0) {
        err = sys_err("bind/listen");
        ::close(s);
        return false;
    }
    fd_ = accept(s, nullptr, nullptr);
    ::close(s);   // one first stage per process: no second connection is taken
    if (fd_ < 0) { err = sys_err("accept"); return false; }
    return tune(err);
}

bool StageLink::connect_to(const std::string& host, int port, int timeout_s, std::string& err) {
    close();
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* ai = nullptr;
    const std::string ps = std::to_string(port);
    if (getaddrinfo(host.c_str(), ps.c_str(), &hints, &ai) != 0 || ai == nullptr) {
        err = "stage link: cannot resolve " + host;
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ >= 0 && connect(fd_, ai->ai_addr, ai->ai_addrlen) == 0) break;
        const int e = errno;
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
        // the last stage loads its weights first and listens only then: a refused connect is retried
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(timeout_s)) {
            errno = e;
            err = sys_err(("connect to " + host + ":" + ps).c_str());
            freeaddrinfo(ai);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    freeaddrinfo(ai);
    return tune(err);
}

void StageLink::close() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
}

bool StageLink::send(Msg kind, std::initializer_list<std::pair<const void*, size_t>> parts, std::string& err) {
    Frame f;
    f.kind = (uint32_t) kind;
    std::vector<iovec> iov;
    iov.reserve(parts.size() + 1);
    iov.push_back({&f, sizeof f});
    for (const auto& [p, n] : parts) {
        if (n == 0) continue;
        iov.push_back({const_cast<void*>(p), n});
        f.bytes += n;
    }
    const uint64_t total = sizeof f + f.bytes;
    uint64_t done = 0;
    size_t first = 0;
    while (done < total) {
        const ssize_t w = writev(fd_, iov.data() + first, (int) (iov.size() - first));
        if (w < 0) {
            if (errno == EINTR) continue;
            err = sys_err("send");
            return false;
        }
        done += (uint64_t) w;
        // drop the parts written whole, trim the one written in part
        size_t left = (size_t) w;
        while (first < iov.size() && left >= iov[first].iov_len) left -= iov[first++].iov_len;
        if (first < iov.size()) {
            iov[first].iov_base = (char*) iov[first].iov_base + left;
            iov[first].iov_len -= left;
        }
    }
    ++frames_sent;
    bytes_sent += total;
    return true;
}

bool StageLink::recv_bytes(void* p, size_t n, std::string& err) {
    char* c = (char*) p;
    while (n > 0) {
        const ssize_t r = read(fd_, c, n);
        if (r == 0) { err = "stage link: the other stage closed the connection"; return false; }
        if (r < 0) {
            if (errno == EINTR) continue;
            err = sys_err("recv");
            return false;
        }
        c += r;
        n -= (size_t) r;
        bytes_received += (uint64_t) r;
    }
    return true;
}

bool StageLink::recv_header(Msg& kind, uint64_t& bytes, std::string& err) {
    Frame f;
    const double t0 = now_us();
    if (!recv_bytes(&f, sizeof f, err)) return false;
    wait_us += now_us() - t0;
    if (f.magic != kMagic) { err = "stage link: a frame without its magic (the stream is out of step)"; return false; }
    kind = (Msg) f.kind;
    bytes = f.bytes;
    ++frames_received;
    return true;
}

bool StageLink::recv(Msg kind, std::initializer_list<std::pair<void*, size_t>> parts, std::string& err) {
    Msg k;
    uint64_t bytes = 0;
    if (!recv_header(k, bytes, err)) return false;
    if (k != kind) {
        err = "stage link: expected frame " + std::to_string((uint32_t) kind) + ", got " + std::to_string((uint32_t) k);
        return false;
    }
    uint64_t want = 0;
    for (const auto& pr : parts) want += pr.second;
    if (want != bytes) {
        err = "stage link: frame " + std::to_string((uint32_t) kind) + " holds " + std::to_string(bytes) +
              " bytes, the reader expected " + std::to_string(want);
        return false;
    }
    for (const auto& [p, n] : parts)
        if (n > 0 && !recv_bytes(p, n, err)) return false;
    return true;
}

}   // namespace strata::net
