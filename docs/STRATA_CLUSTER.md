# strata-cluster: one model, two PCs (unofficial, experimental)

This fork adds one feature to Strata 0.2.7 (engine 0.1.52): **a layer split across two PCs over TCP.** It is not
part of Strata. The authors decided not to build cross-machine splitting
([#1256](https://github.com/Niko1221/Strata/issues/1256#issuecomment-6038109482)), so do not report problems with this
fork to them. Everything else in this repository is Strata 0.2.7 as released.

## What it does

Strata's multi-GPU mode (`--layer-split`, [MULTI_GPU.md](MULTI_GPU.md)) gives each card a block of consecutive layers
and passes only the hidden state between blocks. This fork carries that hand-off over a network connection instead:

- **The first PC** (`--net-role first`) runs layers `0..K-1`, the HTTP server, the tokenizer and the n-gram table.
- **The last PC** (`--net-role last`) runs layers `K..47`, the output head and the MTP draft layer.
- **Each PC holds only its own layers' experts**, in its RAM and on its GPUs, with its own expert cache and adaptive
  tier. Each PC can still split its layers over its own cards with `--layer-split`.
- **One round trip per verify window.** The first PC sends the hand-off of the 1-4 tokens being verified (about 51 KB
  per token). The last PC accepts drafts, drafts the next tokens and replies with a few hundred bytes.
- **Prompts travel in chunks** of `--prefill` tokens (40 KB per token). The first PC computes the next chunk while the
  last PC works on the previous one, but a chunk is sent only after the previous one has been acknowledged.
- **A cancelled request** (STOP) travels inside the messages the PCs already exchange, so both engines stop after the
  same window.

The new code runs only when `--net-role` is given. Without it, the engine behaves exactly like Strata 0.2.7.

## When it helps

It helps when the model does not fit in the RAM of the best PC you have. Measured with Qwen3.8-Flash-Next (125B),
median of six 400-token answers:

| Setup | Tokens/s |
|---|---|
| 4-bit UD-Q4_K_XL, one PC (Ryzen 7 3700X, 64 GB, 2× RTX 5060 Ti), same quality as the pair | 16.2 / 16.4 |
| 4-bit UD-Q4_K_XL, one PC, Strata's default settings (route tail skip on) | 20.6 / 19.7 |
| 4-bit UD-Q4_K_XL, that PC + a Ryzen 5 5600X with 32 GB and an RTX 4070, split at layer 32 | 28.6 / 30.5 / 30.7 |
| 3.5-bit IQ3_S, one 32 GB PC with two cards, experts read from the file | 37.9 |
| 3.5-bit IQ3_S, two 32 GB PCs, split at layer 24, one card each | 48.8 |
| 3.5-bit IQ3_S, the 64 GB PC alone (the model fits) | 57.8 |
| 3.5-bit IQ3_S, the 64 GB PC + the 5600X | 56.9 |

It does not help when one PC already holds the model (the last two rows), and long prompts are slower than on one
PC: one PC reads a long prompt layer by layer, while the pair reads it chunk by chunk. The speed of the pair is set by
the RAM of the smaller PC, which must hold its half of the experts.

## Requirements

- **Linux on both PCs, NVIDIA cards.** The link uses POSIX sockets: **this fork does not build on Windows.** HIP and
  Metal builds were not tried.
- **The same model files and the same engine arguments on both PCs**, except `--net-role` and each PC's own
  `--layer-split`. At start the two engines compare the protocol version, the split, the model's shape, the largest
  window and the prompt chunk, and refuse to run if any of them differ. They do not check that the files are the same.
- **A fixed `--prefill N`** (not `auto`): both engines must cut the prompt into the same chunks.
- `--suffix-draft 0 --prompt-cache 0 --conversation-cache-mib 0`. No `--batch` and no pipelined windows.
- **A trusted network.** The link is plain, unauthenticated TCP. The last PC listens on the given port on every
  interface and takes the first connection. Firewall the port to the first PC's address.

## Running it

Pick the split `K` (the first layer of the last PC) and a port. On the **last PC**, start the engine on its own, with
the same arguments as the first PC plus the role. It loads its weights, then listens:

```sh
strata --serve --pack PACK --native MODEL.gguf [the same arguments as the first PC] \
       --net-split 32 --net-role last --net-addr FIRST_PC_HOSTNAME_OR_IP:47811
```

(The host part of `--net-addr` is not used by the last PC; the port is.) Run it under a supervisor that restarts it
on failure, for example a systemd unit with `Restart=on-failure`. When the link is lost it exits with code 1, and a
new instance waits for the first PC's next connection.

On the **first PC**, add the role to the engine arguments of the normal server config (`serve/server.py --config`):

```json
"args": ["--pack", "PACK", "--native", "MODEL.gguf", "--prefill", "2048",
         "--suffix-draft", "0", "--prompt-cache", "0", "--conversation-cache-mib", "0",
         "--layer-split", "16",
         "--net-split", "32", "--net-role", "first", "--net-addr", "LAST_PC_HOSTNAME_OR_IP:47811"]
```

The first PC connects to the last one (it retries for up to 30 minutes while the last PC is still loading) and then
forwards every request line, so both engines run the same requests in step. Clients talk only to the first PC's
server.

## When the link fails

- A closed connection is noticed at once: the request in flight ends with an error. The last PC's engine exits and
  its supervisor restarts it. The first PC's engine ends too, at the latest at the next request (which may get "busy",
  503), and its server starts it again.
- A silent failure (a pulled cable, a frozen PC) is caught by TCP keepalive (5 s idle, then 3 probes 2 s apart) and by
  `TCP_USER_TIMEOUT` on unacknowledged data. `STRATA_NET_TIMEOUT_S` sets that limit (default 10 s; 0 turns it off).
- Recovery means reloading the model on both PCs: about 3 minutes in my tests. There is no reconnection without
  a reload yet.

## Known limits

- Token probabilities (`logprobs`) are not returned through the API.
- Images (`--vision`) were not tried across PCs.
- The prompt hand-off is stop-and-wait: a chunk's transfer is not overlapped with the previous chunk's computation.
- Errors from the link reach the client as HTTP 400, like any other engine error, not as a 5xx.
- The internal name in the code and in the logs is `strati-in-rete` ("layers on the network").

## Validation

| Check | Result |
|---|---|
| CUDA 12.9 build, Linux, sm_75/86/89/120 | Passed |
| CUDA 12.9 build, Linux, sm_61 (Pascal, `STRATA_EXPERIMENTAL_SM60`) | Passed |
| `tests/net/stage_link_test` over loopback: 2,520 windows shaped like the real ones | Passed: 0 wrong hand-offs, 0 wrong picks |
| One process with `--layer-split 24` against two processes joined over TCP, same cards for the same layers, one expert cached per card, adaptive features off, temperature 0, 8 prompts | Passed: 8 of 8 answers identical (RTX 5060 Ti ×2: 880 tokens; Titan Xp ×2: 889 tokens) |
| Cancel mid-answer, mid-prompt and after the first chunk, then the same long prompt again | Passed: 4 of 4 |
| 256K context, 60 requests in a row | Passed: 0 errors |
| Code word found in prompts of 60K, 121K and 227K tokens | Passed: 6 of 6 |
| Connection reset mid-answer (`ss -K`); all packets dropped for 45 s mid-answer (`iptables DROP`) | Passed: error returned, both engines restarted, the pair answered again |
| Windows build | Not run (expected to fail: POSIX sockets) |
| HIP and Metal builds | Not run |

The modification was written with the help of Claude (Anthropic). Measured on the author's own machines, October 2026.
