# Gufo in Rust

Gufo's host code is Rust; its GPU kernels stay HIP. This page explains why,
how the workspace is built and checked, where it stands against upstream
Gufo (the C++ engine, [gufo-org/gufo](https://github.com/gufo-org/gufo)) and
how it follows upstream releases.

The reference is upstream release **v0.28.1**. Every model of that release
except MiniMax H3 runs in Rust, with the same command line, HTTP API and
numerics (see [Differences from upstream](#differences-from-upstream)).

## Why Rust

Gufo is developed with coding agents; the repository carries instructions
([AGENTS.md](../AGENTS.md)) and skills for them. In C++, a lifetime, aliasing
or thread-safety mistake compiles and shows up later as memory corruption,
a wrong token or a hang under load. The Rust compiler rejects those mistakes
before anything runs, with the line at fault, which an agent can act on; a
corruption at runtime names nothing. The port keeps what C++ did well and
moves only the host code:

| Stays HIP/C++ | Rust |
| --- | --- |
| `__global__` kernels | Engines, weight loading, sessions, KV and snapshot management |
| Launchers: grid and block choice, template dispatch, geometry checks | Server, scheduler, continuation caches, CLI, `eval` |
| Adapted llama.cpp MMQ sources | GGUF, sampling, tokenizers, quantization, diagnostics |
| | hipBLAS/hipBLASLt plan selection, HIP graphs, streams, allocation |

What it buys:

- **Hand-written `unsafe` lives in one crate.** It is confined to
  `gufo-hip`, apart from the kernel build script's jobserver handoff;
  kernel crates allow `unsafe` only in their generated launcher module,
  every other crate forbids it, and `tools/ci/check-unsafe.py` enforces
  that.
- **Kernel launches are checked before they are enqueued.** Each launcher's
  contract (`launch.toml`) declares the extent of every buffer, the stream it
  must belong to and the bounds of every index buffer; a buffer one element
  short, from another stream or aliased as input and output is refused (the
  last at compile time).
- **Graph replays cannot touch freed or replaced memory.** A captured graph
  holds every allocation it uses, and every replay is validated against the
  launches that captured it.
- **One build system.** Cargo builds the one `gufo` binary; CMake and the
  Nix package are gone. Production dependencies are small and mostly pure
  Rust (PNG, JPEG and WebP decoding, Unicode normalization, rustls).

Kernels, launch geometry and arithmetic order are upstream's, so outputs are
bit-identical to the C++ engine wherever the C++ path is deterministic, and
speed, load time and memory are checked against it
([Performance against upstream](#performance-against-upstream)). Built with
the same ROCm, every device function the Rust binary shares with upstream's
C++ build has the same instructions and resources
(`tools/bench/kernel-isa.py`; llama.cpp's abort stubs aside, which print
their source path). The HIP sources differ from upstream's only in host
code: the C++ launchers that Rust replaces are gone, adapters give the
launcher contract its flat signatures, and the two vendored quantized-GEMM
tiers take the caller's scratch (Flash-Next) and the engine's own stream
(DeepSeek), see
[Implemented differently](#implemented-differently).

## Workspace

| Crate | Path | Role |
| --- | --- | --- |
| `gufo` | `src/cli` | The binary: every command and the model bindings of `serve` |
| `gufo-serve` | `src/cli/serve` | HTTP server, APIs, scheduler, runner pool, continuation caches |
| `gufo-core` | `src/core` | GGUF, tokenizers, sampling, grammars, the text runner interface |
| `gufo-eval` | `src/eval` | `gufo eval` against an OpenAI-compatible server |
| `gufo-formats` | `src/formats` | GGML storage formats and block geometry |
| `gufo-hip` | `src/hip` | Streams, memory, graphs, weights, BLAS: the hand-written `unsafe` of the GPU |
| `gufo-ffmpeg` | `src/ffmpeg` | Video decoding through FFmpeg, loaded on the first video: the `unsafe` of the FFmpeg calls |
| `gufo-kernel-build` | `src/kernel_build` | `launch.toml` to C++ shim and Rust wrappers, `hipcc` |
| `gufo-cli-kernels` | `src/cli/kernels` | The binary's own kernels (the allocation benchmark of `gufo diagnose`) |
| `gufo-hip-smoke` | `src/hip/smoke` | Starts the HIP runtime as a `gufo` executable does; the template for executables' build scripts |
| `gufo-<model>`, `gufo-<model>-kernels` | `src/models/<model>` | A model's engine (`#![forbid(unsafe_code)]`) and its kernels with their contract |

The models are `qwen35` (the `qwen35` architecture: Qwen3.8 27B and the
dense Qwen3.5 family), `qwen38_flash_next`, `deepseek_v4_flash`,
`qwen3_asr`, `qwen3_tts`, `qwen_image_21` and `embedding_gemma2`. `qwen`
holds what the Qwen language models share: the chat template, the vision
encoder and the kernel crate over upstream's `src/models/qwen` HIP sources,
which stay at upstream's paths. Model code, kernels, tests and numerical
contracts stay together, as upstream keeps them.

## Safety model

### `unsafe` policy

- Workspace lints: `unsafe_op_in_unsafe_fn = "deny"`,
  `clippy::undocumented_unsafe_blocks = "deny"` and `clippy::unwrap_used =
  "deny"` (tests may opt out).
- `#![forbid(unsafe_code)]` in every crate except `gufo-hip`,
  `gufo-ffmpeg`, `gufo-kernel-build` and the kernel crates. Kernel crates
  deny it with one `#[allow(unsafe_code)]` on their generated module;
  `gufo-kernel-build` denies it except for the build-time jobserver handoff.
- `gufo-ffmpeg` holds the FFmpeg boundary of video embeddings: `dlopen` of
  libavformat, libavcodec, libavutil and libswscale on the first video, the
  calls, and FFmpeg's IO and interrupt callbacks into the decoder. Struct
  fields go through a C shim compiled against the build's headers
  (`src/ffmpeg/shim/fields.c`), so no Rust code assumes FFmpeg's layouts, and
  libraries of another major version than the headers are refused.
- Changes to `src/hip/` and `src/ffmpeg/` need human review: that is where a
  mistake becomes memory corruption.

A safe launch wrapper is only as sound as its contract, so the **trusted
base** is `gufo-hip`, `gufo-kernel-build`, every `launch.toml` (a wrong
extent compiles and lets a kernel overrun a buffer that passed the check),
the C++ launchers and kernels behind them, the BLAS, device-property and
pointer-attribute shims in `src/hip/shim/`, and `gufo-ffmpeg` with its field
shim. Changes there get the same
review as `unsafe` code. Every launcher declares `probes`: the generated
test runs each with buffers of exactly the declared extent and a 4,096-element
guard region, and fails when a kernel writes into the guard. Probes catch
extents too small for what a kernel writes, not for what it reads; those
remain a review item.

### `gufo-hip`

- `Stream`: `Send`, not `Sync`, so launch order is program order. After a
  sticky device fault the stream is poisoned and every further operation
  fails fast; the owning engine is recreated.
- `DeviceBuffer<T>`: the single owner of mutable memory of one stream, freed
  in stream order. `DeviceTensor<T>`: immutable memory readable on every
  stream (weights). `DeviceSlice`/`DeviceSliceMut`: borrowed views with
  length and allocation; read-only arguments accept tensors and buffers of
  the launch stream, written arguments only buffers of the launch stream.
- `Stream::alloc_persistent` sizes long-lived memory exactly; the stream
  pool behind `Stream::alloc` rounds up by up to 128 MiB.
- `DeviceIndices`: `u32` values within a bound fixed at allocation, written
  only by a checked host upload or by kernels whose contract guarantees the
  bound, so a kernel that indexes a table with them cannot be handed an
  out-of-range index.
- `PinnedBuffer<T>`: host memory for asynchronous copies whose host view
  waits for the last copy, so host code can neither read a half-transferred
  buffer nor overwrite one the DMA engine still reads.
- Frees that synchronise the device, and frees on a capturing stream, go
  through a queue that drains only while no stream captures; otherwise a
  free on one thread would invalidate a capture on another.
- `GraphCache`: the first call runs eagerly, the second captures, later calls
  replay. A graph retains every allocation it captured, so it never writes
  freed memory; a replaced buffer is detected before replay and recaptured;
  allocation or synchronisation during capture keeps that key eager. Every
  replay re-runs the body record-only and compares buffers, lengths, scalars
  and launch order with the capture (about 160 ns per launch, while the
  device executes the graph); a mismatch fails the step and demotes the key
  to eager execution. Per-step values therefore live in device buffers.
- Weights: `MappedFile` (read-only mmap) and `WeightLoader` with the
  placements `DirectCopy` (O_DIRECT reads through pinned staging),
  `CachedCopy` (device copies from the registered mapping into exactly sized
  arenas), `Map` (the registered mapping itself) and `HostCopy` (a
  huge-page copy, registered read-only). Tensors exist only after `finish()`,
  so no stream reads a weight still being copied.
- BLAS: `Blas` (hipBLAS) and `BlasLt` (hipBLASLt with per-shape plans and
  workspace) behind `src/hip/shim/blas.cpp`, whose `static_assert`s check
  every call and struct against the installed ROCm headers. Operands are
  checked like launch arguments; the model's `AlgoPolicy` chooses the
  algorithm, since that decides the accumulation order.

### `launch.toml`

One file per kernel crate declares its launchers; the build generates a C++
shim and a Rust wrapper for each. A launcher is a non-template C++ function
`LaunchX(pointers..., scalars..., hipStream_t) -> void | bool`, where `false`
means "geometry unsupported, nothing enqueued".

```toml
[[launcher]]
name = "text_swiglu"
cpp = "LaunchTextSwiGLU"
returns = "void"                       # or "supported" (bool)
params = [
  { name = "gate", kind = "in", elem = "f32", len = "count" },
  { name = "up", kind = "in", elem = "f32", len = "count" },
  { name = "output", kind = "out", elem = "bf16", len = "count" },
  { name = "count", kind = "scalar", type = "usize" },
]
probes = [{ count = 257 }]
```

- Parameter kinds: `in`, `inout`, `out` (device views), `indices` and
  `indices_out` (bounded `u32` buffers), `scalar`, `quant` (GGML-quantized
  weights checked against their format and block size), `format`, `const`,
  `alias` and `stream`. `len` expressions use integers, scalar parameters,
  `+ - * / %` and parentheses, all overflow-checked; `requires` adds
  relations no extent expresses; `contract_only` scalars size extents
  without being C++ arguments; `host_visible` marks a buffer the C++
  launcher writes from the host, which must be memory the host addresses at
  its device address.
- The shim `static_assert`s the launcher's exact function-pointer type (any
  drift in constness, width or order fails the build), converts exceptions
  to a status so nothing unwinds into Rust, and reads `hipGetLastError()`.
- The Rust wrapper checks streams, extents, index bounds and requirements,
  records the launch for graph capture and validation, then calls the shim.
  The checked path costs the same host time per launch as a direct C++
  `<<<>>>` launch (about 1.9 µs).
- Template dispatch stays inside C++; adapters (`adapters.hip` in a kernel
  crate) flatten what the contract cannot describe directly.

## Build

Linux x86-64 on Strix Halo (`gfx1151`) only; see the
[README](../README.md#build-from-source) for the prerequisites.

- `cargo build --release -p gufo` builds `target/release/gufo`; each kernel
  crate's `build.rs` is `gufo_kernel_build::compile_kernels("launch.toml")`.
- One ROCm prefix supplies compiler, headers and runtime: `$ROCM_PATH`, else
  `/opt/rocm/core`, else `/usr`. The binary's RUNPATH points to its `lib`, and
  `gufo-hip` refuses to run when another HIP runtime was loaded (the
  distribution and AMD runtimes share the soname `libamdhip64.so.7`). Every
  crate producing an executable calls `gufo_kernel_build::link_hip_runtime()`.
- Kernels compile with `--offload-arch=gfx1151 -O2 -g -DNDEBUG -fPIC -Wall
  -Wextra` (upstream's release flags) plus each source's `std`, `flags` and
  `defines`; `hipcc` gets the prefix's `--rocm-path` and include directory and
  `--gcc-install-dir` of the linker driver, so ROCm's clang uses the GCC whose
  `libstdc++` the link takes.
- Kernel objects are keyed by compiler identity and command line and rebuilt
  when a file of their depfile changes; `hipcc` runs in parallel within
  Cargo's jobserver.
- `GUFO_RELEASE_VERSION` and `GUFO_REVISION` set what `gufo --version`
  prints ([releasing](RELEASING.md)).
- Nix provides the development shell (profiling, reference and client tools);
  the binary is built and measured outside it, against the ROCm prefix.

## Tests and the gate

`tools/ci/check-rust.sh` is the gate before every commit: rustfmt, clippy
with GPU features and warnings as errors, the GPU and model tests of the
packages a change reaches, the DeepSeek operator checks when their kernels
change, clang-format of the hand-written C++, the unsafe policy, the tool
tests, the CLI parser, kernel scratch against
`tools/ci/kernel-resources.json`, and the documentation and dependency
checks. `tools/ci/gate_scope.py` maps each changed file to the packages it
reaches (through Cargo's and `hipcc`'s dependency records); `--full` runs
every test and is required after shared or environment changes. Hosted CI
runs only what needs neither ROCm nor a gfx1151, including clippy and the
tests of the crates that link no ROCm library and `cargo deny`. Details,
test features and the local artifacts some tests read are in
[testing](TESTING.md).

On ROCm 10.1 two of upstream's DeepSeek operator checks fail, for its C++
build as for this one: they are standalone HIP programs, built with
`-ffast-math`, that expect two kernel variants to agree bit for bit, and
ROCm 10.1's clang 24 rounds them apart in the last bit (`IQ2 tile8 row32`
against the scalar kernel in `ds4.projections`; the indexer at one token
and 17 rows in `ds4.attention`). The Rust build's outputs still equal the
C++ build's, which compiles the same kernels; the gate reports both checks
until the kernels or the compiler change.

Beyond the gate: model QUALITY checks (`#[ignore]` tests), the text API
functional suites ([tests/functional](../tests/functional/README.md)) and
A/B runs against another binary with `tools/bench/ab.py`.

## Differences from upstream

Against v0.28.1. Everything not listed behaves as upstream does.

### Not in Rust

| Upstream feature | Status |
| --- | --- |
| MiniMax H3 video generation (`gufo video`) | Not ported: its 134 GiB of BF16 safetensors fit neither the disk nor the 128 GiB of the development host, so a port could not be compared with the C++ engine. A Rust implementation on GGUFs is the planned replacement. |
| `serve llm --trace` (request and reply content to a JSONL file) | Not ported, by decision: it writes client content to disk. |
| `prompt`/`chat --cpu` (CPU fallback) | Not ported: without a device, loading fails. |
| `bench --enable-yarn` | Not ported: `gufo bench` measures the native context; `serve llm --enable-yarn` exists. |
| Dense Qwen with F32 or Q8_K projections, or with other head and state shapes (upstream's hipBLAS/hipBLASLt projection routes, unfused QK-norm/RoPE and plain recurrence) | Not ported: no published checkpoint needs them, and loading refuses such files. Both sides refuse F16 projections. |
| The DeepSeek V4 Flash profiling and trace hooks (`GUFO_DEEPSEEK_ROCM_*_PROFILE`, `GUFO_DEEPSEEK_DSPARK_TRACE`, `_WIDTH_TRACE`, `_TIMING`) | Not ported: diagnostics only; `tools/prof/prof.py` profiles the Rust binary. |
| `tune_hipblaslt` and its plan database (`GUFO_HIPBLASLT_PLAN_CACHE`), `gufo-kernel-bench` | Not ported: tools outside `gufo`. Plans are chosen per shape at run time, as upstream does without a database. |
| Windows support (upstream's `src/core/platform` layer, Windows builds and their workflows, #775, #805, #872) | Not ported: Rust uses the standard library and `gufo-hip`, not upstream's platform layer. |
| The Nix package (`nix build`), its ROCr queue-memory patch and the Devlab CI workflows | Not ported: Cargo builds the binary against the installed ROCm; hosted CI runs the ROCm-free checks. A ROCm built with upstream's [queue-memory patch](https://github.com/gufo-org/gufo/blob/v0.27.0/.devops/nix/patches/rocr-fixed-cwsr-rocm10.patch) keeps the same protection against queue evictions during host memory compaction. |

### Implemented differently

| Area | Rust |
| --- | --- |
| Conversation cache | Rust keeps its own continuation cache (RAM and disk tiers, same-slot in-place restore) with upstream's policies: the deepest checkpoint across RAM and disk, metadata-only startup, oversized checkpoints on disk only and the wait for checkpoints being published. It does not use upstream's `src/cache` component. Disk checkpoints are written in 1 MiB pieces but serialised and restored as whole payloads, so a disk transfer stages the whole checkpoint in host memory. Two consequences show in the functional suites on Flash-Next: an exactly repeated request restores the state before the assistant's opening and prefills its 7 tokens again, where `src/cache` keeps the whole prompt (`sampling-defaults`); and after earlier suites have filled the cache, `cache-edits` misses an intermediate checkpoint (alone, it passes). Restoring a Flash-Next snapshot into another session copies all of its rows; upstream tracks snapshot lineage and keeps rows both sessions already share. A Flash-Next capture freezes the mutable state on the device and leaves the rows in the session; once a request has generated 10 tokens, the frozen state of the snapshots captured so far moves into pageable host memory on a thread of its own and the device blocks return for the next captures. Upstream copies captures into pinned host memory, 2 GiB of which it commits at load and grows before each prefill; restoring the state from pageable memory takes about 1.5 ms longer. |
| Flash-Next n-gram rows across cache checkpoints | Read through the page cache; the prefetch across checkpoint boundaries is not ported. |
| Image sessions and HIP graphs | Graph keys carry whether a session has an image RoPE table, so text graphs survive an image's arrival and image graphs are captured once per shape. |
| JPEG decoding | `zune-jpeg` instead of libjpeg-turbo: pixels may differ by a few levels (another IDCT and upsampling), so outputs on JPEG images may differ. `tests/functional/embedding_images.py` therefore accepts a cosine of 0.999 between a JPEG's embedding and that of Pillow's decoded copy, where upstream asks for 0.99999 (measured: 0.99983 against C++'s 0.999999). PNG and WebP decode as upstream's do. |
| Flash-Next quantized-GEMM scratch | Each routed call of the vendored tier allocates from scratch the executor passes (`qfn_mmq_set_scratch`), so a captured graph keeps memory that cannot be freed under it; upstream's tier grows one arena per stream in blocks (#740). |
| DeepSeek V4 Flash quantized-GEMM scratch | The vendored tier reuses its persistent scratch on the engine's stream; upstream's C++ engine runs on the legacy stream, the only one where the tier reuses it. |
| `--cache-ram-bytes off` with `--cache-disk` | Keeps no RAM checkpoint and pins no cache memory; the Rust pool still restores the disk checkpoints. Upstream's component cache instead keeps up to 512 MiB of backing for Qwen and Flash-Next captures (#783). |
| Dense Qwen weights on the device (#708, #794) | A whole-file device image, filled through eight pinned staging buffers that drop the mapping's pages as they go; when the image would leave the device heap less than its dispatch reserve, or its allocation fails, the file falls back to a registered host copy. Upstream falls back to its per-tensor weight upload. |
| Qwen-Image weight upload (#830) | `gufo_hip::FileUpload` owns each destination buffer from `copy` until `finish` hands it back, so no safe caller can reach memory a reader thread is still filling. The reader count, staging size, queue bound and the overlap of staging and copying are upstream's `WeightUpload`. |
| Video embedding input (#869) | The same FFmpeg libraries, loaded the same way on the first video; `gufo-ffmpeg` reaches struct fields through a C shim built against the build's headers and refuses libraries of another major version, where a mismatch would be undefined behaviour in C++. |
| Audio embedding input (#819, #828) | WAV, FLAC, Ogg Vorbis and MP3 decode in Rust (`hound`, `claxon`, `lewton`, `nanomp3`) and other sample rates resample with `rubato`; upstream uses libsndfile and libsoxr. PCM WAV and FLAC decode exactly, MP3 and Ogg within 3e-7; resampled inputs differ (a few percent of the signal energy near the new Nyquist frequency for 24 to 48 kHz stereo). From the 16 kHz PCM on, features, encoder and embeddings equal upstream's bit for bit. |
| Generated-token logprobs (#814) | Streaming buffer limits count the logprob records at Rust struct sizes, a few bytes apart from C++. Requests that select the unmodified argmax verify in the batched path, which upstream also uses unless batched verification is off. |
| DFlash2 prompt features | The 27B target captures prompt features into one pinned buffer per model, which grows to the longest prompt and serves every session; upstream pins about 400 MiB of host memory per session at startup. |
| Message text that spells a control token | Tokenized as text by whole message span; upstream decides per tokenized slice, so a prompt slice that cuts such a span can tokenize differently (and reuse less cache). |
| Responses reasoning | Trimmed like the chat completion's reasoning, as llama.cpp does; upstream keeps the newline before `</think>`. |
| A stream cancelled before any output | Still commits its empty response (logged as `disconnected` once the client is gone); upstream writes nothing and logs `completed`. |
| Range errors | Bounds print as integers, without `std::to_string`'s decimals. |
| Unicode property escapes in schema patterns | General categories and the binary properties listed under [structured output](SERVER.md#structured-output) are accepted; scripts, blocks and other properties are rejected, where upstream resolves them through ICU. |
| A panic on the scheduler thread | Fails resident and queued requests with 503 `scheduler_failed` and exits with status 70 ([lifecycle](SERVER.md#lifecycle-and-limits)); upstream terminates. |
| Structured-output vocabulary at startup | A build failure does not stop the server; the first constrained request builds it again and reports the error. Upstream fails startup. |
| DeepSeek V4 Flash startup lines | Written to stderr directly, so `--log-level` does not filter them. |
| Load and image-cache diagnostics | No `phase=artifact_identity` load line and no `event=image_cache` debug line. |

### Only in Rust

| Option | Meaning |
| --- | --- |
| `--api-compat gufo\|llama.cpp` | Request semantics where Gufo and llama.cpp differ; `gufo` (the default) is upstream's ([server](SERVER.md)). |
| `--responses-reasoning-target summary\|reasoning-text\|reasoning` | Where `/v1/responses` puts the raw reasoning; `summary` (the default) is upstream's. |
| `--sse-ping-interval N` | llama-server's SSE keep-alive interval; upstream keeps streams alive at a fixed interval. |
| `gufo arbiter`, `--arbiter HOST:PORT` | GPU sharing between servers: text generation yields while a streamed TTS falls behind real time ([sharing the GPU](ARBITER.md)). |

## Performance against upstream

A/B runs of `target/release/gufo` against a C++ build of v0.28.1 with the
same ROCm 10.1 runtime (`tools/bench/ab.py`; `tools/bench/ab-upstream.sh`
runs every workload below, with the models named in its header). A
workload runs as whole runs of one binary, the C++ build twice and then
Rust twice, with a third run of a side whose two runs disagree; the C++
runs stay the baseline for later candidates while the reference and the
toolchain stay the same.
Rust must be no slower than C++ in prompt processing and generation within
0.5%, load no more than 5% slower and peak memory no more than 1% higher;
each verdict uses the upper bound of the bootstrapped 95% interval of the
ratio of a Rust run's rounds to the pooled C++ rounds. Peak memory is
everything a run takes from the machine: host memory the kernel cannot
reclaim, which on the APU includes the GTT, plus VRAM. Device memory (VRAM
plus GTT) is reported but not judged, as both sides keep some buffers in
different places: C++ copies serving snapshots into pinned host memory it
commits at load, Rust keeps them on the device until a decode is under way.

### Results

Measured on 2026-11-04 and 05 on a Radeon 8060S (gfx1151) with 128 GiB and
ROCm 10.1, 4 to 8 rounds per run, with the build of this tree. Each cell
gives, per Rust run, the median ratio Rust/C++ as a change in percent (bold
where that run fails the gate), and in brackets the highest upper bound of
the runs' 95% intervals.

| Workload | Request | Gated stages | Load | Memory | Device memory | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| q27-q4-ar | −0.08 / −0.04 (≤ +0.04) | pp **+0.68** / **+0.56** (≤ +1.19) | +0.58 / +1.30 (≤ +2.23) | +0.19 / +0.11 (≤ +0.81) | −0.37 | FAIL (0 of 2) |
| q27-q4-dflash2 | +0.01 / +0.05 (≤ +0.11) | pp **+1.09** / **+1.20** (≤ +1.38) | +0.92 / +1.71 (≤ +2.85) | −0.18 / −0.30 (≤ +0.39) | −0.34 | FAIL (0 of 2) |
| q27-q8-ar | ±0.00 / ±0.00 (≤ +0.14) | pp −0.97 / −0.51 (≤ −0.18) | +0.57 / +0.55 (≤ +1.40) | +0.08 / +0.17 (≤ +0.56) | −0.22 | PASS (2 of 2) |
| q27-q8-dflash2 | −0.02 / ±0.00 (≤ +0.04) | pp −0.26 / −0.14 (≤ +0.38) | +1.36 / +1.52 (≤ +2.58) | −0.13 / −0.07 (≤ +0.21) | −0.20 | PASS (2 of 2) |
| q27-q4-ar-d32k | −0.05 / ±0.00 (≤ +0.09) | pp −0.06 / ±0.00 (≤ +0.11) | +2.34 / +1.65 (≤ +3.46) | −0.38 / −0.32 (≤ −0.11) | −0.30 | PASS (2 of 2) |
| fn-ar | −0.39 / −0.42 / −0.39 (≤ −0.20) | pp +0.04 / −0.04 / −0.04 (≤ +0.15) | **+9.49** / −0.25 / **+3.10** (≤ +20.66) | −0.36 / −0.29 / −0.38 (≤ −0.27) | −0.52 | FAIL (1 of 3) |
| fn-mtp | −3.62 / −3.62 (≤ −1.06) | pp +0.03 / +0.02 (≤ +0.16) | −3.27 / −3.64 (≤ +4.92) | −0.49 / −0.51 (≤ −0.42) | −0.53 | PASS (2 of 2) |
| fn-ar-d32k | −0.19 / −0.24 (≤ −0.17) | pp −0.03 / +0.06 (≤ +0.15) | **+2.82** / **+0.79** (≤ +22.65) | −0.19 / −0.16 (≤ −0.12) | −0.38 | FAIL (0 of 2) |
| ds-ar | −0.09 / −0.06 (≤ −0.03) | pp −1.26 / −1.68 (≤ −0.26) | **−2.23** / **−1.16** (≤ +18.94) | −0.04 / −0.04 (≤ +0.03) | −0.09 | FAIL (0 of 2) |
| ds-dspark | −1.44 / −1.44 / −1.44 (≤ −1.39) | pp ±0.00 / −0.08 / +0.10 (≤ +0.26) | −3.87 / **−2.45** / −3.99 (≤ +5.18) | −0.04 / −0.05 / −0.04 (≤ −0.02) | −0.08 | PASS (2 of 3) |
| serve-q27-dflash2 | −0.16 / ±0.00 (≤ +0.19) | c1 −0.16 / ±0.00 (≤ +0.19); c4 +0.04 / +0.24 (≤ +0.37); TTFT +0.12 / +0.27 (≤ +0.44); decode −0.25 / −0.02 (≤ +0.12) | −26.18 / −25.57 (≤ −24.29) | −27.91 / −27.91 (≤ −27.56) | −9.37 | PASS (2 of 2) |
| serve-fn-mtp | −0.38 / −0.36 / −0.36 (≤ −0.23) | c1 −0.38 / −0.36 / −0.36 (≤ −0.23); c4 **−0.18** / **+0.45** / **−0.03** (≤ +2.53); TTFT −0.03 / +0.03 / −0.03 (≤ +0.20); decode −0.39 / −0.36 / −0.37 (≤ −0.24) | **−0.39** / **−0.21** / **+3.93** (≤ +13.75) | −1.04 / −1.04 / −1.01 (≤ −0.91) | +0.98 | FAIL (0 of 3) |
| serve-ds-dspark | **−0.91** / −0.75 / −0.80 (≤ +3.15) | c1 **−0.91** / −0.75 / −0.80 (≤ +3.15); c4 **+3.13** / **−0.03** / −1.18 (≤ +7.48); TTFT **−0.77** / −0.45 / −0.45 (≤ +3.34); decode **−0.83** / −0.70 / −0.78 (≤ +3.24) | **+10.79** / **+5.63** / **−0.27** (≤ +17.36) | +0.20 / +0.08 / +0.14 (≤ +0.35) | +0.03 | FAIL (0 of 3) |
| serve-q27-dflash2-t08 | +0.14 / +0.30 (≤ +0.46) | c1 +0.14 / +0.30 (≤ +0.46); c4 **+0.49** / **+0.84** (≤ +1.17); TTFT **+0.44** / **+0.58** (≤ +0.76); decode +0.07 / +0.26 (≤ +0.38) | −27.40 / −27.88 (≤ −26.98) | −27.43 / −27.44 (≤ −27.21) | −9.33 | FAIL (0 of 2) |
| serve-fn-mtp-t08 | −0.70 / −0.73 (≤ −0.53) | c1 −0.70 / −0.73 (≤ −0.53); c4 **+0.22** / **−0.16** (≤ +2.34); TTFT +0.11 / +0.20 (≤ +0.32); decode −0.81 / −0.85 (≤ −0.62) | **+2.37** / **+6.94** (≤ +18.97) | −0.98 / −0.98 (≤ −0.91) | +0.98 | FAIL (0 of 2) |
| serve-ds-dspark-t08 | −1.13 / −1.10 / −1.11 (≤ −0.89) | c1 −1.13 / −1.10 / −1.11 (≤ −0.89); c4 **+1.23** / −1.14 / −1.12 (≤ +3.49); TTFT −0.69 / −0.73 / −0.78 (≤ −0.13); decode −1.31 / −1.26 / −1.29 (≤ −1.16) | **−0.19** / −0.15 / **−0.24** (≤ +14.33) | +0.16 / +0.31 / +0.06 (≤ +0.35) | +0.12 | FAIL (1 of 3) |
| asr | −4.31 / −4.37 / −4.40 (≤ −4.28) | – | **+1.28** / −9.95 / −6.14 (≤ +6.45) | −4.59 / −4.75 / −4.34 (≤ −1.19) | +0.04 | PASS (2 of 3) |
| tts-customvoice | −4.78 / −3.53 / −3.34 (≤ −2.17) | first audio −2.82 / −2.48 / −2.50 (≤ −1.81); streamed −3.60 / −3.50 / −3.52 (≤ −3.40) | **+4.82** / +3.20 / **+3.14** (≤ +8.01) | **−0.09** / **−0.51** / **−0.50** (≤ +1.52) | ±0.00 | FAIL (0 of 3) |
| tts-voicedesign | −3.07 / −2.95 / −4.14 (≤ −2.64) | first audio −2.31 / −2.10 / −2.21 (≤ −1.60); streamed −3.48 / −3.54 / −3.45 (≤ −2.77) | **+15.39** / −1.57 / **+19.97** (≤ +29.14) | **+0.51** / **+0.16** / **+0.34** (≤ +2.54) | ±0.00 | FAIL (0 of 3) |
| image | −0.56 / −0.20 / −0.53 (≤ +0.06) | denoise −0.35 / +0.03 / −0.32 (≤ +0.27); prompt −4.94 / −5.17 / −5.30 (≤ −4.59); VAE −0.92 / −0.86 / −0.94 (≤ −0.34) | −16.32 / **−8.17** / **−0.33** (≤ +10.01) | −0.08 / −0.02 / +0.08 (≤ +0.53) | ±0.00 | FAIL (1 of 3) |
| embedding | −1.63 / −1.26 / −1.58 (≤ −0.31) | single **−1.39** / **−0.22** / **+0.93** (≤ +5.23) | −0.13 / −1.47 / −0.13 (≤ +0.15) | **+0.62** / **+0.20** / **+0.72** (≤ +2.20) | ±0.00 | FAIL (0 of 3) |

The text workloads (`gufo bench`) prefill 512 prompt tokens and generate
128 (64 after 32,768 tokens of context for `d32k`) on Qwen3.8 27B
(UD-Q4_K_XL, UD-Q8_K_XL; autoregressive or with the DFlash2 draft),
Flash-Next (UD-Q4_K_XL; autoregressive or with MTP) and DeepSeek V4 Flash
(IQ2XXS; autoregressive or with DSpark); `pp` is prefill time per token. The serving workloads run the HTTP server at concurrency 1 and 4
(`c1`, `c4`: time per output token); those named `t08` sample at temperature
0.8 with a fixed seed and without top-k. Speech, image and embedding requests
go through their servers: an English recording, the CustomVoice and
VoiceDesign checkpoints, eight steps at 1024×1024, and EmbeddingGemma 2 with
a batch of 32 inputs of about 512 tokens (the request) and one short input
(`single`).

Every invocation of every workload gave the same outputs on both sides.
Image and video embeddings, which these workloads do not cover, take one of
two values per server process at some budgets on both sides: the BLAS plan of
a shape is chosen by timing at its first call, so C++ differs from C++ as
often as from Rust (up to 3e-4 per element, a cosine of at least 0.999997).

Seven workloads pass. Rust needs less memory in every run of 14 of 21
workloads, 28% less when serving the 27B with DFlash2 (see
[DFlash2 prompt features](#implemented-differently)). Most failures have
causes that are not Rust's. Of the alternating rounds cited here, q27-q4-ar,
the first 27B prefill number and the first sampled 27B numbers ran with this
build; the others ran with earlier builds of this sync that differ only in the
host sampler and in the waits after host copies, which measure the same:

- **Upstream's speech loaders.** Since v0.27.0, the TTS and ASR loads drop
  the checkpoint's pages after the upload
  ([#845](https://github.com/gufo-org/gufo/pull/845)), so every load reads
  the files again and its peak memory varies with what the kernel has
  reclaimed. The C++ builds of v0.23.0 and v0.27.0 measured in alternating
  rounds load the CustomVoice checkpoint in 615 and 1,644 ms, with a peak of
  6.47–6.49 and 6.48–6.75 GiB. In alternating rounds against v0.28.1, the peak
  spreads over 6.35–6.96 GiB on C++ and 6.49–6.81 GiB on Rust, so the median
  ratios of +1.0 and +1.9% have intervals that reach below 1; per process,
  Rust holds about 19 MiB (+0.3%) more. The load takes +3.3% on CustomVoice,
  one step of the harness's 50 ms poll, and +1.3% (≤ +3.2) on VoiceDesign.
- **Machine state.** Decoding the large models alternates between fast and
  slow rounds on both sides, and slow rounds follow slow loads: with
  80 to 100 GiB of weights on a 128 GiB machine, the kernel reclaims and swaps
  while a round runs. The Flash-Next and DeepSeek workloads fail only through
  this spread: load intervals reach +23%, `c4` intervals +7.5%, and in one
  serve-ds-dspark run the request and decode intervals +3.2%. Their request and
  decode medians lie below C++'s in every run.
- **Run order.** The APU runs at its power limit, so a hotter GPU clocks
  lower and a run that follows other runs is slower; the Rust runs of a
  workload follow its C++ runs. Measured in alternating rounds (ABBA, 8
  rounds per side), q27-q4-ar's prefill is +0.04% (≤ +0.56).
- **Load time.** Loads of the large models take 20 to 25 s on both sides,
  from round to round. The image workload's load median lies below C++'s in
  every run, its interval reaches +10%.

Rust is slower than C++ here, and the causes are open:

- **27B DFlash2 prefill:** +0.47% (≤ +0.56) in alternating rounds, +0.38 and
  +0.63% in two earlier series (+0.62% at v0.27.0). Over 21 repetitions per
  run the hot gap is +0.19% (419.24 against 420.02 tokens per second), and from
  a GPU cooled to 45 °C before each run, in swapped order, Rust leads by 0.15%
  (468.42 against 467.71). Prefill moves far more with temperature than
  between the binaries: 468 tokens per second at 45 to 57 °C, 420 at 80 °C.
  Ruled out: bench arguments, the waits after host copies (a build without
  them measures the same), CPU time per prefill, clocks and power. AB and BA
  rounds show the same gap, so it is not the later slot running hotter.
  Generation is 0.16% ahead in the same rounds.
- **27B DFlash2 serving, the first token:** about 4 ms. +0.32% (≤ +0.59;
  1,331.6 against 1,327.2 ms) sampled in alternating rounds, +0.40% greedy, and
  +0.32% from a cooled GPU. Greedy `c1` and `c4` take +0.2% hot and +0.07 to
  +0.09% cooled. Sampled, decoding is equal (+0.05%) and `c4` takes +0.28%
  (≤ +0.70), +0.51% (≤ +0.89) in the series before.
- **Embedding, one request:** +0.60% over 200 requests per server (5.871
  against 5.836 ms, 35 µs), and +1.63% (≤ +2.78) for three requests right
  after a batch, over 24 rounds, where latency varies by about 15% from
  request to request. The batch request is 0.82% faster and memory is +0.29%
  (≤ +1.45) in the same rounds.
- **Flash-Next serving, a repeated prompt** (against v0.27.0, below): +1.4
  to +1.7 ms to the first token, +0.5 to +1.4% to the tenth. Here the cause
  is known. The restore copies the snapshot's frozen state (about 118 MB) from
  pageable host memory ([conversation cache](#implemented-differently)), which
  takes 3.5 to 4.6 ms; C++ copies it from the pinned host memory it commits at
  load, in 2.2 to 2.8 ms. Matching that needs pinned memory held for the
  cache, and Rust reserves no host memory up front. Pinning a buffer for each
  capture instead costs 35 to 55 ms to the first token, and batching the
  restore's copies gains nothing, as the transfer from pageable memory
  dominates.

Sampling without top-k is new in this comparison: per verified token the host
normalises a target distribution over the whole vocabulary, and per rejected
draft it draws from the residual. Flash-Next with MTP took 20.34 ms per token
against C++'s 19.56 (+4%), with identical outputs, while top-k 40 was equal
(20.79 ms on both). `src/core/src/sampling.rs` now divides in pairs, hashes a
token ID with one multiplication and filters non-candidates before the lookup.
Against the same C++ source in a microbenchmark, the distribution takes 1.37
against 1.47 ms and the residual draw 0.77 against 0.75 ms, and
`serve-fn-mtp-t08` decodes 0.8% and `serve-ds-dspark-t08` 1.3% faster than
C++.

Against v0.27.0 on 2026-11-01, Flash-Next serving with MTP was measured from a
GPU cooled to 45 °C, alternating and in swapped order, with the snapshot
handling of the [conversation cache](#implemented-differently): time to the
first and the tenth streamed token at concurrency 1, four servers per side. A
new 443-token prompt takes 440.7 to 440.9 and 571.6 to 572.4 ms against C++'s
441.0 to 442.8 and 572.8 to 575.0 ms (one server per side, whose first requests
ran in the slow machine state, left out). The second turn of a conversation
(568 tokens cached, 29 prefilled) takes 132.6 to 133.1 and 326.3 to 330.9 ms
against 134.7 to 137.0 and 329.8 to 332.2 ms, the third 136.6 to 139.0 and
269.3 to 271.0 ms against 138.5 to 142.1 and 272.1 to 275.9 ms. An exactly
repeated prompt restores its snapshot from host memory and takes 6.7 to 6.9 and
132.7 to 133.5 ms against 5.2 to 5.3 and 131.6 to 132.0 ms.

### The C++ reference

Build the reference from its release tag in a directory of its own,
against the same ROCm prefix as `target/release/gufo`. With a
distribution ROCm and a newer GCC installed beside it, the release preset
alone yields a mixed binary, so the HIP compilations get the prefix's
include directory (`/usr/include/hip` shadows it otherwise) and the GCC
whose `libstdc++` headers the C++ files use, and the binary's RUNPATH is
set to the prefix:

```sh
R=/opt/rocm/core-10.1 G=/usr/lib/gcc/x86_64-linux-gnu/15
mkdir -p target/upstream-v0.28.1/src
git archive v0.28.1 | tar -x -C target/upstream-v0.28.1/src
cd target/upstream-v0.28.1/src
cmake --preset release -DCMAKE_PREFIX_PATH=$R \
  -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++ \
  "-DCMAKE_HIP_FLAGS=--rocm-path=$R -isystem $R/include --gcc-install-dir=$G"
cmake --build --preset release
cp build/release/gufo ../gufo-cpp
patchelf --set-rpath $R/lib:$G ../gufo-cpp
ldd ../gufo-cpp | grep -E 'amdhip64|hipblas|rocblas'   # every one from $R
```

`tools/bench/kernel-isa.py ../gufo-cpp target/release/gufo` compares the
device code of the two builds function by function: with the same ROCm,
every function both have must have the same instructions. Each side also
holds a few template variants the other does not instantiate (the C++
launchers instantiate some no path launches).

`ab.py` refuses to compare binaries that resolve different `libamdhip64`
libraries and records both binaries' hashes, the runtime, the GPU DPM level
and the CPU governor. `gufo diagnose` prints the toolchain the build pinned,
not the runtime it loads; check that with `ldd`.

## Open work

- **Conversation cache:** the parts of upstream's `src/cache` behind the two
  suite results under "Implemented differently"; a fallback to a retained RAM
  snapshot when a deeper disk restore fails while RAM holds only a live
  frontier (today the request prefills its whole prompt); and the functional
  runners that still read upstream's component-cache logs
  ([functional tests](../tests/functional/README.md)). The log check of
  `disk-spacing` reports two stored Flash-Next checkpoints 381 tokens apart
  (C++ passes it); Rust's events carry no checkpoint ids, and the cause is
  open.
- **27B DFlash2, short replies:** two requests that arrive together and end
  within a speculative cycle or two run one after the other
  (`serial-fallback`), where C++ verifies them in one batch, with the same
  texts and about the same total time. The width check of the `batch` suite
  therefore fails on the 27B with DFlash2; the Rust build of v0.27.0 does the
  same.
- **Performance gaps:** the open causes under [results](#results); also the
  17 MiB of additional anonymous memory in the speech servers.
- **Small dense Qwen:** decode stays 0.2 to 0.4% behind C++ below the 9B
  ([optimizations](OPTIMIZATIONS.md), Q8); Flash-Next's bit-exact
  follow-ups F10 and F11 are listed there as well.
- **References:** the 27B BF16 comparison as a Rust test (upstream's
  `qwen27b_target_test`), Flash-Next's independent MTP audit and deep
  execution check (`--mtp-audit`, `--execution-only`), Flash-Next's F32
  reference fixtures with their GPU probe and scorer (#758), Flash-Next's
  operator chains `hc_route_ops_test` and `routed_wmma_ops_test` and the
  m=17 split of `blaslt_ops_test` (the C++ session fixtures cover these routes
  end to end), the two disabled DeepSeek baseline checks and the `reference`
  suite of `tools/ds4/check.py`.
- **Freeing shared tensors:** the last drop of an immutable tensor waits for
  the whole device (not during a graph capture). Swapping models at run time
  would need it to wait for the tensor's actual users instead.
- **Dependencies:** `tools/ci/check-dependencies.py` checks the crate
  inventory against [third-party notices](../THIRD_PARTY_NOTICES.md) and
  hosted CI the policy of [`deny.toml`](../deny.toml); generated crate
  licences (`cargo-about`) are open.
- **Responses reasoning:** `--responses-reasoning-target` picks one target.
  Several at once, none (which needs `encrypted_content` for tool loops) and
  generated summaries for `reasoning.summary` are ideas.
- **Images on Qwen3.5 0.8B and 2B:** their `mmproj` carries another vision
  tower (12 layers of width 768), which upstream does not take either.

## Following upstream

1. Pick an upstream release as the reference, not a moving `main`.
2. List its changes since the last reference; port each into the crate that
   owns it, or record it above when it does not apply (platform layers,
   packaging, CI) or is deliberately left out. HIP and C++ sources keep
   upstream's paths; host code of upstream's `src/models/qwen` goes to
   `qwen` (chat template, vision) or `qwen35` (everything else). Comments
   that cite a C++ file (`src/cli/serve/openai_chat.cpp`,
   `tests/cli/http_server_test.cpp`) mean upstream's tree at the reference
   release, where `git show v0.28.1:<path>` finds them.
3. Keep numerics: kernels merge as source onto the Rust tree, host
   arithmetic follows the C++ order, and fixtures that upstream's changes
   alter are recorded again from a C++ build of the reference.
4. Run the gate with `--full`, the affected functional suites against both
   binaries, and the A/B runs above against a C++ build of the reference
   with the same ROCm ([the C++ reference](#the-c-reference)).
5. Update this page: the reference, the differences and the A/B results.
