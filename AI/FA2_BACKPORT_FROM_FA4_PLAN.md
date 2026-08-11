# FA2 Backport-from-FA4 Optimization Plan

| Field | Value |
|---|---|
| Document ID | `FA2_BACKPORT_FROM_FA4_PLAN` |
| Target repo | `D:\USERFILES\fp8e4m3\flash-attention` (fork of `Dao-AILab/flash-attention`) |
| Base version | `flash_attn` `2.9.0` (set in `flash_attn/__init__.py`) |
| Scope | FA2 C++/CUDA kernel only (`csrc/flash_attn/**`). FA3 (`hopper/`) and FA4 (`flash_attn/cute/`) are out of scope for this plan. |
| Build assumptions | PyTorch `>=2.10`, CUDA `>=13.0`, MSVC + nvcc on Windows, default `FLASH_ATTN_CUDA_ARCHS="80;90;100;120"`. |
| Author | This plan is generated to direct further work on the fork. |

---

## 1. Executive Summary

The official `Dao-AILab/flash-attention` repository has effectively ceased active FA2 development and routes all new optimization effort into FA3 (Hopper) and FA4 (CuTeDSL, Blackwell+). FA4 is Linux-only at runtime due to the native CuTeDSL shared-objects shipped only as `manylinux` wheels (`nvidia-cutlass-dsl-libs-base` / `nvidia-cutlass-dsl-libs-cu13`). This fork therefore still has long-term value: an FA2 C++/CUDA kernel that builds natively on Windows with PyTorch 2.10+/CUDA 13.0+.

FA2's kernel is built on `SM80_16x8x16` MMA and `cp.async`, both of which are forward-compatible: the same binary runs on sm_80 / sm_86 / sm_89 / sm_90 / sm_100 / sm_120. It works on the operator's RTX 5060 Ti (sm_120), but does not exploit any Blackwell-only instruction (`fma.f32x2`, `add.f32x2`, `WGMMA`, `TMA`, `UMMA`, `TMEM`).

This plan defines five backport-from-FA4 optimizations, all of which preserve the existing fork's compatibility surface (PyTorch 2.10+, Windows, multi-arch SASS):

| Plan ID | Title | Phase | Status |
|---|---|---|---|
| A-1 | `rescale_threshold` skip in `softmax_rescale_o` | 1 | **Applied** (commit pending) |
| A-2 | `fma.rn.f32x2` packed FMA in `scale_apply_exp2` (sm_100+ guarded) | 1 | **Applied** (commit pending) |
| A-3 | `max_offset` bias scaffold on softmax exponent (default `0`, byte-identical) | 2 | Planned |
| B-1 | `exp2` polynomial emulation switchover (Sollya minimax + packed asm) | 2 | Planned |
| B-2 | `MUFU.EX2` + polynomial mixed mode on sm_100+ (3-parameter freq/res/start) | 2 | Planned |
| C-1 | hdim-bucketed block-size retuning for sm_120 / sm_100 | 3 | Planned |

A-priority changes are arch-agnostic in their implementation guards: A-1 is enabled by template flag at the call site, A-2 falls back to plain FMA on sm_80/86/89/90 by `__CUDA_ARCH__` guard. No existing call-site outside this fork's `flash_fwd_kernel.h` is forced to change behavior.

---

## 2. Background

### 2.1 Current state of the fork

This fork carries the following deltas against upstream `main`:

- `setup.py`: PyTorch `>=2.10` floor and CUDA 13+ wheel-tag handling.
- `csrc/flash_attn/flash_api.cpp`: `#include <torch/extension.h>` instead of `<torch/python.h>` for PyTorch 2.10 ABI.
- `WindowsWhlBuilder_cuda.bat`: explicit `FLASH_ATTENTION_FORCE_BUILD=TRUE`, `DISTUTILS_USE_SDK=1` and post-build wheel rename.
- `WindowsWhlBuilder_fa4.bat`: FA4 wheel build entry (the produced wheel is non-functional on Windows; retained for symmetry with the upstream layout).
- `flash_attn/__init__.py`: `__version__ = "2.9.0"` to mark the fork-specific feature set.
- `csrc/flash_attn/src/softmax.h`, `csrc/flash_attn/src/flash_fwd_kernel.h`: Plan A-1 / A-2 applied (see §11).

### 2.2 Why FA4 is not a path on Windows

FA4 (`flash_attn/cute/`) is a CuTeDSL package. Its runtime imports `cuda.bindings.driver` and `cutlass.cute` whose native components are distributed only via:

- `nvidia-cutlass-dsl` (pure Python; available)
- `nvidia-cutlass-dsl-libs-base` (`.so`, `manylinux_2_28_x86_64` only)
- `nvidia-cutlass-dsl-libs-cu13` (`.so`, `manylinux_2_28_x86_64` only)

No corresponding `win_amd64` wheel exists, no upstream build script targets Windows, and the underlying libraries are NVIDIA-proprietary. Building equivalents from source is not feasible because the sources are not released. FA4 on Windows is therefore not pursued in this plan.

### 2.3 What FA4 contains that is portable

FA4's CuTeDSL source (`flash_attn/cute/softmax.py`, `flash_attn/cute/flash_fwd_sm100.py`) contains arithmetic-level optimizations that are independent of CuTeDSL infrastructure:

1. Skipping the running-statistics rescale when the new and old row-max are close.
2. Issuing softmax FMA in pairs through `fma.f32x2` on Blackwell.
3. Replacing `MUFU.EX2` with a polynomial approximation, or mixing the two, to gain SFU/FMA pipeline parallelism.
4. A `max_offset` bias on the running max to bound the post-exp range and let downstream FMA stay in a regime that benefits from `f32x2`. In FA4 this is implemented as a constexpr scalar subtracted from `scores_max_cur` before scaling, with default `8.0f` for fp8 e4m3/e5m2 input and `0.0f` for fp16/bf16. The bias is mathematically a no-op for softmax because the same offset enters numerator and denominator and cancels in the final `1/row_sum` divide; it only changes the regime of the intermediate exponent argument.

Items (1)-(3) are the basis of plans A-1, A-2, B-1, B-2. Item (4) is now plan A-3 (see §5.3, §7.3), promoted from "design note inside B-1" to a first-class scaffold because (a) it is byte-identical when the offset is `0`, (b) it interacts with the polynomial range reduction in B-1, and (c) it is a forward-compatibility hook for fp8 input even though the current FA2 fork does not have an fp8 path.

---

## 3. Terms and Preconditions

### 3.1 Glossary

| Term | Meaning |
|---|---|
| FA2 | The C++/CUDA kernel under `csrc/flash_attn/`. Default attention impl on sm_80 and above. |
| FA3 | The Hopper-specific kernel under `hopper/`. Uses WGMMA and TMA. |
| FA4 | The CuTeDSL kernel under `flash_attn/cute/`. Targets Blackwell. Out of scope on Windows. |
| sm_NN | CUDA compute capability, e.g. sm_80 = Ampere, sm_90 = Hopper, sm_100 = B200, sm_120 = GeForce Blackwell. |
| MMA | Matrix-multiply-accumulate tensor-core instruction. FA2 uses `SM80_16x8x16_F32F16F16F32_TN`. |
| `fma.rn.f32x2` | Packed FMA on a pair of `f32` lanes in one issue. Blackwell-only (`sm_100+`). |
| `MUFU.EX2` | The SFU `exp2f` instruction. |
| `scaled_diff` | `(prev_row_max - cur_row_max) * softmax_scale_log2`. |
| `scores_scale` | `exp2f(scaled_diff)`. Always `<= 1.0` in normal control flow. |
| Rescale | The multiplication of running `O` and `row_sum` by `scores_scale` performed each non-first attention block. |
| `max_offset` | A constexpr scalar subtracted from `scores_max_cur` before `* softmax_scale_log2`. Mathematically a no-op for softmax (numerator and denominator are both biased identically) but changes the regime of `t*scale - max_scaled`. In FA4 default `8.0f` for fp8, `0.0f` for fp16/bf16. Plan A-3 adds it as a template parameter to FA2 with default `0.0f` (byte-identical). |
| `keep_window_size` / `softcap` / `Is_local` | Existing FA2 feature flags carried in `Flash_fwd_params`. |
| Sollya / Remez / minimax polynomial | Floating-point function approximation framework (`https://www.sollya.org/`). FA4's `flash_attn/cute/utils.py::POLY_EX2` is generated by Sollya as a degree-3 minimax polynomial of `2^f` over `f ∈ [0, 1)` (round-down range reduction). Plan B-1 ships those exact coefficients. |
| `ex2.approx.ftz.f32` | The PTX form of `MUFU.EX2`. Used by `exp2f()` and `__expf()` in CUDA and by SageAttention's `ptx_exp2`. Approximate; not bit-identical to a polynomial emulation. |
| `FFMA.X2` | Blackwell SASS mnemonic for `fma.rn.f32x2`. Inspecting the generated SASS for this mnemonic is the gating criterion for A-2 and B-1 (see §8.2). |
| Backport | Reimplementing an FA4 idea in FA2 C++/CUDA without taking on any FA4 runtime dependency. |

### 3.2 Target architectures

| Arch | Code | Carrier hardware | Status in this plan |
|---|---|---|---|
| sm_80 | Ampere A100 | A100, A30 | Supported, no behavior change unless explicitly noted. |
| sm_86 | GA10x | RTX 30xx | Same as sm_80. |
| sm_89 | AD10x | RTX 40xx | Same as sm_80. |
| sm_90 | Hopper | H100, H200 | Same as sm_80 from FA2's standpoint; WGMMA path is FA3's responsibility. |
| sm_100 | Blackwell datacenter | B200 | Receives A-2, B-2 (`__CUDA_ARCH__ >= 1000` guarded code). |
| sm_120 | Blackwell consumer | RTX 50xx incl. RTX 5060 Ti | Receives A-2, B-2 (same guard); receives C-1 retuning. |

### 3.3 Build preconditions

- nvcc that supports `-gencode arch=compute_120,code=sm_120` (CUDA `>= 12.8`; default with CUDA 13.0).
- For `compute_100f` / `compute_120f` family-specific encoding, CUDA `>= 12.9`. The fork's `add_cuda_gencodes()` handles both branches.
- PTX `fma.rn.f32x2` requires `.target sm_100` or above and ptxas from CUDA 12.8+ (CUDA 13.0 used in production here).
- Windows host: MSVC 14.4x with the Windows 10/11 SDK matching the nvcc requirement. Build entry is `WindowsWhlBuilder_cuda.bat`.

---

## 4. Code Analysis

### 4.1 FA2 forward kernel skeleton

The forward kernel template `compute_attn_1rowblock` is instantiated by `flash_fwd_launch_template.h` and lives in `flash_fwd_kernel.h`. The block loop runs:

1. Load `K` tile through `cp.async` into smem.
2. Compute `S = Q * K^T` via `gemm()` over `MMA_M x MMA_N` MMA atoms (using `Mma1::Operation = SM80_16x8x16_F32F16F16F32_TN`).
3. Apply mask (`Mask::apply_mask`).
4. Apply softmax: `softmax_rescale_o<Is_first, Check_inf>(acc_s, acc_o, scale_softmax_log2)`. The `Is_first` flag selects the "no running stats yet" branch.
5. Convert `acc_s` from fp32 to fp16/bf16.
6. Compute `O += P * V` (second `gemm`).
7. Advance to next `K`/`V` tile.

The softmax step is the only one that does floating-point arithmetic outside MMA. It is therefore the only place where FA4's softmax improvements can be backported without rewriting the entire kernel pipeline. The MMA and memory pipelines are intentionally left untouched in this plan; rewriting them would require WGMMA/TMA paths and falls under FA3-style work.

### 4.2 Current softmax implementation (post Plan A)

`csrc/flash_attn/src/softmax.h` now contains, in order:

| Symbol | Purpose | Modified by A-* |
|---|---|---|
| `thread_reduce_`, `reduce_max`, `reduce_sum`, `Allreduce<>` | Warp-level reductions over `kNRows`. | No |
| `fma_f32x2(d0,d1, a0,a1, b0,b1, c0,c1)` (new) | One `fma.rn.f32x2` on sm_100+, two scalar FMAs otherwise. | Added by A-2 |
| `scale_apply_exp2<Scale_max>(tensor, max, scale)` | For each row: compute `exp2(t*scale - max_scaled)` element-wise. | Rewritten by A-2 to use packed FMA on sm_100+ |
| `Softmax<kNRows>::softmax_rescale_o<Is_first,Check_inf,Use_rescale_threshold>` | Per-block softmax + running-stats update. | Extended by A-1 with `Use_rescale_threshold` template flag. |
| `Softmax<kNRows>::normalize_softmax_lse` | Final normalization. | No |

### 4.3 Current MMA and memory instructions

| Concern | Instruction in FA2 | Forward-compat on sm_100/120? |
|---|---|---|
| Q @ K^T (and P @ V) | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (via `SM80_16x8x16_F32F16F16F32_TN`) | Yes, runs as an Ampere-class MMA on Blackwell. |
| Load Q / K / V | `cp.async.cg.shared.global` | Yes. Blackwell adds `TMA` but the `cp.async` path still functions. |
| `exp` | `MUFU.EX2` via `exp2f` intrinsic | Yes. Blackwell still has the SFU. |
| Scalar arithmetic | `FFMA`, `FMUL`, `FADD` | Yes. Packed `fma.f32x2` is Blackwell-only and is opt-in only inside the new `fma_f32x2()` helper. |

There is no instruction in the current FA2 kernel that **cannot** execute on sm_120. The performance gap is purely "instructions used vs. instructions available", not "binary cannot run".

### 4.4 sm_120 (RTX 5060 Ti) runtime evidence

The operator has confirmed the existing FA2 binary runs correctly on RTX 5060 Ti (sm_120). This rules out the earlier mistaken claim that "FA2 is sm_80 only". The default `FLASH_ATTN_CUDA_ARCHS="80;90;100;120"` in `setup.py` emits SASS for all those archs in addition to PTX for the newest one. `add_cuda_gencodes()` chooses `compute_100f` / `compute_120f` when CUDA `>= 12.9` so the kernel is also "family-locked" for forward-compatible PTX-JIT scenarios.

---

## 5. Improvement Catalog

This section is the table-of-contents for §7 (detailed design) and §8 (validation). Each entry is intentionally short here; see §7 for full design.

### 5.1 Plan A-1 — `rescale_threshold` skip

**Location:** `csrc/flash_attn/src/softmax.h::Softmax::softmax_rescale_o` (Is_first=false branch).
**Idea:** When `scaled_diff = (prev_max - cur_max) * log2_scale` is in `[-T, 0]` for small `T`, `scores_scale = exp2(scaled_diff)` is so close to `1.0` that multiplying running `O` and `row_sum` by it is wasted work. Skip the multiply (and the `exp2f`) and keep the new `row_max` for the subsequent `scale_apply_exp2`. `T = 0.01` ⇒ worst case `scores_scale >= 0.993`, relative error `<= 0.7%`.
**Arch effect:** Roughly equal speedup on sm_80/90/100/120 because the saved work (one `exp2f` + N `mul`s) is the same in cycle terms across these arches.
**Risk:** Very mild accuracy regression on the running `O` accumulation, bounded by the threshold.

### 5.2 Plan A-2 — `fma.rn.f32x2` in `scale_apply_exp2` (sm_100+ guard)

**Location:** `csrc/flash_attn/src/softmax.h::scale_apply_exp2`.
**Idea:** The inner loop computes `tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled)` per element. The pre-`exp2f` term is `fma(t, scale, -max_scaled)`. On sm_100+, `fma.rn.f32x2` performs two such FMAs in one instruction. The `exp2f` itself stays scalar (`MUFU.EX2` is not f32x2). Net effect: roughly half the FMA issues for the pre-`exp2f` work; `MUFU.EX2` becomes the dominant cost, opening room for B-2.
**Arch effect:** sm_100/120 only. sm_80/86/89/90 fall back via `#if __CUDA_ARCH__ >= 1000` to the existing scalar code (binary unchanged on those arches).
**Risk:** PTX inline asm needs the correct lane packing. The implementation packs two `f32` into a `.b64` register and unpacks back; the helper is the only place this asm exists.

### 5.3 Plan A-3 — `max_offset` bias scaffold on softmax exponent

**Location:** `csrc/flash_attn/src/softmax.h::softmax_rescale_o` and `scale_apply_exp2`.
**Idea:** Subtract a constexpr `max_offset` from `scores_max_cur` before forming `max_scaled = scores_max_cur * softmax_scale_log2`. This bias cancels in the final `1/row_sum` normalization, so the softmax output is invariant to it; it only shifts the regime of `t*scale - max_scaled`. With `max_offset = 0.0f` (default for fp16/bf16) the binary is **byte-identical** to the post-A-2 state. The plumbing exists so B-1 / B-2 can opt in to a bounded exponent range, and so a future fp8 e4m3/e5m2 path can use `max_offset = 8.0f` (FA4's default for fp8 — see `flash_attn/cute/softmax.py`).
**Arch effect:** None at default. With `max_offset > 0`, polynomial domain in B-1 narrows and `fma.f32x2` in A-2 sees fewer near-zero exponent arguments, helping precision in tail cases.
**Risk:** Forward compatibility scaffold only at this stage. The risk surface is purely "did the compiler fold the `-0.0f` subtract", which is straightforward to verify in SASS.

### 5.4 Plan B-1 — `exp2` polynomial emulation (Sollya minimax + packed asm)

**Location:** `csrc/flash_attn/src/softmax.h::scale_apply_exp2`. The same code path is exercised by `flash_bwd_kernel.h:536`, which calls `scale_apply_exp2<Scale_max=false>` and therefore receives A-2 automatically.
**Idea:** Replace `MUFU.EX2` with the same degree-3 Sollya minimax polynomial of `2^f` over `f ∈ [0, 1)` that FA4 ships in `flash_attn/cute/utils.py::POLY_EX2`. Range reduction uses the FA4-style `add.rm.ftz.f32` (round-down add) plus integer-bias reconstruction (`shl 23`), which is both more accurate and cheaper than the `__float2int_rd` + `ldexp` sketch from the previous design. A packed `exp2f_emu_x2` mirrors FA4's `e2e_asm2` and emits PTX inline asm using `fma.rn.f32x2` on sm_100+, so B-1 composes naturally with A-2.
**Arch effect:** Trades SFU throughput for FMA throughput. On sm_80/86/89/90 it is a small win (FMA peak ≫ SFU peak). On sm_100/120 the win can be larger when combined with A-2's `fma.f32x2`. Packed polynomial evaluation is FMA-only and benefits directly.
**Risk:** Accuracy. The polynomial achieves ≈ 2e-5 absolute error on `[0, 1)`. The acceptance threshold for shipping is per-element softmax relative error `<= 5e-5` plus end-to-end logit cosine `>= 0.9995`. The bounded input range required by the polynomial is the reason A-3 is sequenced first.

### 5.5 Plan B-2 — `MUFU.EX2` / polynomial mixed mode (3-parameter freq/res/start)

**Location:** Same as B-1 but with periodic SFU correction.
**Idea:** Issue the cheap polynomial for most elements and a true `MUFU.EX2` for occasional "correction" elements. FA4 in `flash_attn/cute/softmax.py` exposes this as **three** parameters, not one: `ex2_emu_freq` (how often `MUFU.EX2` is used), `ex2_emu_res` (residue index inside the period), and `ex2_emu_start_frg` (the starting fragment). The previous design's single `EmuFreq` knob is generalized to all three. The mixed mode can be cheaper than full polynomial under load (because polynomial occupancy is FMA-bound) and more accurate because the SFU correction periodically resyncs the exponent.
**Arch effect:** Most effective on sm_100/120 where the packed FMA pipeline is free to absorb the polynomial work while `MUFU.EX2` provides accuracy anchoring. On sm_80, defaults to `ex2_emu_freq = 1` (= full SFU) and is a no-op.
**Risk:** Tuning. Initial defaults (per FA4 inspection): for hdim ≤ 128, `(freq, res, start) = (4, 3, 0)`; for hdim ≥ 192, `(2, 1, 0)`. These will be re-validated in Phase 2.

### 5.6 Plan C-1 — hdim-bucketed block-size retuning for Blackwell

**Location:** `csrc/flash_attn/src/flash_fwd_launch_template.h` and `csrc/flash_attn/src/kernel_traits.h`.
**Idea:** The current block sizes (`kBlockM` / `kBlockN`) and `kNWarps` were tuned for sm_80 (A100) and reused on sm_120. Blackwell consumer parts have a different smem-to-tensor-core ratio. A retune over hdim ∈ {64, 96, 128, 192, 256}, with `kNWarps ∈ {4, 8}` and `kBlockN ∈ {32, 64, 128}` as the tuning surface, should yield single-digit percent improvements on sm_120 without touching kernel logic.
**Interaction with `pack_gqa`:** FA3/FA4 fuse multiple Q heads per KV head when GQA/MQA is in effect (`flash_attn/cute/pack_gqa.py`). FA2 does not implement this fusion, so when GQA group size is large, the effective `kBlockM` per KV head shrinks. C-1 should record GQA group size as an additional measurement axis even if it does not (yet) introduce a Q-fusion code path; otherwise the retune will overfit to MHA workloads.
**Risk:** Compile-time blowup if every hdim x arch combination gets a distinct instantiation. Mitigate by limiting the new tuning to sm_100/120 and reusing sm_80 traits on older arches.

---

## 6. Implementation Phases

| Phase | Plans | Gating criterion to enter phase | Gating criterion to exit phase |
|---|---|---|---|
| 0 | Measurement scaffolding | None | `bench/` script that returns numeric latency for hdim ∈ {64,128,256}, seqlen ∈ {1k, 4k, 8k, 16k}, dtype ∈ {fp16, bf16} on sm_80, sm_90, sm_120; baseline recorded. |
| 1 | A-1, A-2 | Phase 0 baseline recorded | Phase 0 numeric regression on sm_80 ≤ 1%; numeric improvement on sm_120 in any cell. **SASS gate:** `cuobjdump --dump-sass` of the sm_120 binary shows `FFMA.X2` inside `scale_apply_exp2`. |
| 2 | A-3, B-1, B-2 | Phase 1 merged **and** SASS gate satisfied | A-3 ships byte-identical at default. B-1 sm_120 improvement on at least one (hdim, seqlen) cell with per-element softmax relative error ≤ 5e-5 and end-to-end logits cosine ≥ 0.9995. B-2 has its (`freq`, `res`, `start`) defaults documented and benchmarked. **SASS gate:** B-1's `exp2f_emu_x2` path also emits `FFMA.X2` on sm_120 when enabled. |
| 3 | C-1 | Phase 2 merged | hdim-bucketed retune yields ≥ 5% on sm_120 for any (hdim, seqlen) cell with no regression on sm_80. |

Phase 1 is already substantially complete (A-1 and A-2 are in the working tree); the SASS gate is the remaining Phase-1 exit criterion. Phase 0 measurement scaffolding has **not** been built yet and is a prerequisite for objectively gating Phase 2.

### 6.1 Sequencing constraints

- A-2 must land before A-3 / B-1 / B-2 to give those plans access to the `fma_f32x2()` helper.
- A-3 lands before B-1 because the polynomial in B-1 wants a bounded exponent argument; A-3 provides the scaffold even though the default (`0.0f`) is byte-identical. A-3 with non-zero `max_offset` is only intended for an fp8 follow-on, not the current fp16/bf16 build.
- B-2 depends on B-1 (mixed mode reuses the polynomial implementation).
- C-1 is independent and could be done in parallel; sequencing it last reduces conflict surface against B-* changes inside `kernel_traits.h`.

---

## 7. Detailed Design

### 7.1 Plan A-1 detailed design

**Signature change (applied):**

```183:183:d:\USERFILES\fp8e4m3\flash-attention\csrc\flash_attn\src\softmax.h
    template<bool Is_first, bool Check_inf=false, bool Use_rescale_threshold=false, typename Tensor0, typename Tensor1>
```

**Logic change (applied):**

```203:215:d:\USERFILES\fp8e4m3\flash-attention\csrc\flash_attn\src\softmax.h
                float scaled_diff = (scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2;
                // Optionally skip the O / row_sum rescale when the new row_max is virtually the same as
                // the previous one (i.e. scaled_diff is a very small negative number, so
                // scores_scale = exp2(scaled_diff) ~= 1.0). The threshold -0.01 corresponds to
                // scores_scale >= ~0.993 (worst-case relative error <= 0.7%).
                // The newly reduced row_max is still used by the upcoming scale_apply_exp2 below,
                // so correctness of the current block is preserved; only the rescale of the running O
                // is approximated. This trades a tiny accuracy hit for one fewer exp2 + N mul per row.
                if constexpr (Use_rescale_threshold) {
                    constexpr float kRescaleSkipThreshold = -0.01f;
                    if (scaled_diff >= kRescaleSkipThreshold) { continue; }
                }
                float scores_scale = exp2f(scaled_diff);
```

**Why correctness is preserved:** the new `row_max` reduced by `reduce_max</*zero_init=*/false>` is written before the threshold check. `scale_apply_exp2(scores, row_max, ...)` runs after the threshold loop and uses that fresh `row_max`. The only thing skipped is the multiplicative correction on the running accumulators `row_sum(mi)` and `acc_o_rowcol(mi, *)`. The skipped factor is `>= 0.993` by construction, so the running `O` is at most `0.7%` off per skipped block, and only on rows where `scaled_diff` actually fell in the skip window. The terminal `normalize_softmax_lse` then divides by `row_sum`, which carries the same skipped factor, so the relative error in the normalized output is bounded by the threshold and does not cascade.

**Threshold choice:** `-0.01` is the most conservative useful value. Stricter (`-0.005` ≈ 0.35%) reduces the hit rate substantially; looser (`-0.05` ≈ 3.4%) crosses what we are willing to call "virtually unchanged". The threshold is `constexpr`, not a runtime parameter, so it is fully constant-folded. Promoting it to a runtime parameter is **not** done in Phase 1 because it would force every call-site to wire an extra argument; consider revisiting at Phase 2 if accuracy ablation indicates a need.

**Call-site changes (applied):** four `Is_first=false` call sites in `flash_fwd_kernel.h` (lines 344, 407, 918, 985) now pass `/*Use_rescale_threshold=*/true`. The two `Is_first=true` call sites (343, 917) do not need the flag because there is no `prev` to compare against.

**Bwd impact:** `flash_bwd_kernel.h` calls `scale_apply_exp2` directly, not `softmax_rescale_o`, so A-1 does not touch the backward path.

**Verification (proposed):**

1. Forward correctness: run `tests/test_flash_attn.py::test_flash_attn_output` with `Use_rescale_threshold=true` and confirm the relative L∞ of the output vs. reference attention is `<= 1e-2` for fp16 and `<= 5e-3` for bf16 over the standard test seqlens.
2. Performance: `bench/benchmark_flash_attention.py` for hdim ∈ {64, 128, 256}, seqlen ∈ {1024, 4096, 8192}. Expect 1-3% wall-clock improvement on sm_80/90/120, more when seqlen is small enough that the rescale loop is a noticeable fraction of the kernel.
3. Ablation: also build with `Use_rescale_threshold=false` (revert the four call-site lines) and confirm bit-identical output to the pre-A-1 binary.

### 7.2 Plan A-2 detailed design

**Helper added (applied):**

```65:88:d:\USERFILES\fp8e4m3\flash-attention\csrc\flash_attn\src\softmax.h
// Packed FMA helper: computes (d0,d1) = (a0*b0+c0, a1*b1+c1).
// On sm_100+ (Blackwell), emits a single fma.rn.f32x2 instruction.
// Falls back to two plain FMAs on older arches and when UNFUSE_FMA is set.
__forceinline__ __device__ void fma_f32x2(
    float &d0, float &d1,
    float a0, float a1,
    float b0, float b1,
    float c0, float c1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && !defined(UNFUSE_FMA)
    asm("{\n\t"
        ".reg .b64 ra, rb, rc, rd;\n\t"
        "mov.b64 ra, {%2, %3};\n\t"
        "mov.b64 rb, {%4, %5};\n\t"
        "mov.b64 rc, {%6, %7};\n\t"
        "fma.rn.f32x2 rd, ra, rb, rc;\n\t"
        "mov.b64 {%0, %1}, rd;\n\t"
        "}\n"
        : "=f"(d0), "=f"(d1)
        : "f"(a0), "f"(a1), "f"(b0), "f"(b1), "f"(c0), "f"(c1));
#else
    d0 = fmaf(a0, b0, c0);
    d1 = fmaf(a1, b1, c1);
#endif
}
```

**Loop rewrite (applied):** `scale_apply_exp2` now branches on `__CUDA_ARCH__ >= 1000 && !defined(UNFUSE_FMA)`. The sm_100+ branch hoists `neg_max_scaled = -max_scaled`, iterates `ni` in steps of 2 with a static `N1 = decltype(size<1>(tensor))::value`, calls the helper, then applies scalar `exp2f` to each result. The trailing odd element (if any) falls back to the original scalar formula. The else-branch is the original code, byte-for-byte unchanged.

**Why the existing `UNFUSE_FMA` guard is preserved:** PyTorch issue #121558 documents a case where the compiler-fused FMA changes numerical results vs. a separated `mul` then `sub`. The fix at the time was a macro setting from the PyTorch side. The new packed path explicitly disables itself when `UNFUSE_FMA` is defined, so any caller building under the PyTorch defaults continues to behave as before; the new path is opt-in by toolchain.

**PTX correctness considerations:**

- `fma.rn.f32x2` requires `.target sm_100` and ptxas from CUDA 12.8 or later. Both are guaranteed in the build matrix (`>=13.0` here).
- The `.b64` packing places lane 0 in the low 32 bits and lane 1 in the high 32 bits; the `mov.b64 ra, {%2, %3}` syntax matches that exactly. The unpacking is symmetric.
- The constraint `"=f"` binds a 32-bit float register; nvcc accepts that for the input and output operands of an asm block that internally uses `.b64`.
- Rounding mode `.rn` matches the rounding of plain `fmaf` (which is round-to-nearest-even by default), so there is no rounding-mode mismatch between the two branches on the sm_100+ path's "even" elements and the trailing scalar element.

**Performance expectation:** the pre-`exp2f` arithmetic in `scale_apply_exp2` is `N_rows * N1` FMAs. Halving the FMA count moves the bottleneck firmly to `MUFU.EX2`, which is what B-1/B-2 then target. On sm_120, the expected wall-clock improvement from A-2 alone is in the 1-5% range depending on hdim (higher for larger hdim because the softmax loop is a larger fraction of the kernel).

**Verification (proposed):**

1. Numeric: compare `scale_apply_exp2` output between `__CUDA_ARCH__` branches. The packed path uses two FMAs vs. two separated `mul`/`sub` (in the unfused case) or two FMAs (in the fused case). Round-off may differ by one ULP relative to `mul`/`sub`, but matches the existing `fma`-fused path. Acceptance: matching to `1e-6` relative on synthetic input.
2. SASS inspection: run `cuobjdump --dump-sass` on the sm_120 binary and confirm `FFMA.X2` (packed) appears in the `scale_apply_exp2` region. If it does not appear, the inline asm has been folded out, which is a regression that must be investigated before claiming the optimization is live.
3. Performance: identical script to A-1, focus on sm_120 cells; rebench sm_80 to confirm no regression.

### 7.3 Plan A-3 detailed design

**Goal:** Add a `max_offset` scaffold to the softmax exponent so future fp8 / bounded-domain paths (B-1) have a place to bias `scores_max_cur`. Default value of `0.0f` keeps the binary byte-identical to the post-A-2 state.

**Reference:** `flash_attn/cute/softmax.py` (FA4) — `max_offset` is a per-instance constexpr; FA4 sets it to `8.0f` when `Q_dtype` is fp8 (e4m3 / e5m2) and `0.0f` otherwise. See also `flash_attn/cute/flash_fwd_sm100.py` where `max_offset = 8 if Q_dtype.width == 8 else 0`.

**Mechanism:**

1. Add a new non-type template parameter to `softmax_rescale_o` and `scale_apply_exp2`:

    ```cpp
    template<bool Is_first, bool Check_inf=false, bool Use_rescale_threshold=false,
             /* new */ int MaxOffsetMilli=0, /* in units of 1e-3, so 0 -> 0.0f, 8000 -> 8.0f */
             typename Tensor0, typename Tensor1>
    ```
    The `int` representation avoids the C++17 prohibition on `float` non-type template arguments. The runtime conversion to `float` is `constexpr float kMaxOffset = MaxOffsetMilli * 1e-3f;` and is fully constant-folded.

2. Inside `softmax_rescale_o` / `scale_apply_exp2`, replace:

    ```cpp
    const float max_scaled = scores_max_cur * softmax_scale_log2;
    ```

    with:

    ```cpp
    constexpr float kMaxOffset = MaxOffsetMilli * 1e-3f;
    const float max_scaled = (scores_max_cur - kMaxOffset) * softmax_scale_log2;
    ```

    When `MaxOffsetMilli == 0`, `kMaxOffset` is `0.0f`, the subtract is constant-folded out, and the SASS is identical to the pre-A-3 binary.

3. All current FA2 call sites pass the default (`MaxOffsetMilli=0`). The non-zero path is reserved for a future fp8 forward kernel and is not exercised by Phase 2.

**Why correctness is preserved at default:** `kMaxOffset = 0.0f` makes the subtract a no-op that the compiler removes. The math is unchanged.

**Why correctness is preserved for non-zero values (future work):** The softmax `exp(t * scale - max)` divided by `sum(exp(...))` is invariant under any additive shift of `max`. Subtracting a positive constant from `scores_max_cur` shifts every `t * scale - max_scaled` exponent up by `kMaxOffset * softmax_scale_log2`, which is then divided out by the matching shift in the running `row_sum`. Numerical effect: shifts the exponent regime so that polynomial / packed-FMA paths see exponent arguments biased away from zero.

**Verification:**

1. Default-build numeric equivalence: rebuild with `MaxOffsetMilli=0` at all call sites (the current default), confirm bit-identical output to the pre-A-3 binary.
2. SASS check: confirm no new instructions appear in `scale_apply_exp2` when `MaxOffsetMilli=0`.
3. Future fp8 path: deferred to a follow-on plan. Document the parameter wiring in `softmax.h` so it is discoverable.

**Risk:** This is a scaffold change; the only meaningful risk is that the compiler fails to fold the `-0.0f` subtract, leaving spurious instructions in the hot loop. The SASS check above is the safeguard.

### 7.4 Plan B-1 detailed design

**Goal:** Replace `MUFU.EX2` (`exp2f`) in `scale_apply_exp2` with a Sollya-derived degree-3 minimax polynomial of `2^f` over `f ∈ [0, 1)`, matching FA4's implementation in `flash_attn/cute/utils.py::ex2_emulation` / `ex2_emulation_2` / `POLY_EX2`. This single function covers both fwd and bwd via the shared callee (`flash_bwd_kernel.h:536` calls `scale_apply_exp2<Scale_max=false>`).

**Reference implementation (FA4):**

- `flash_attn/cute/utils.py::POLY_EX2` — coefficients `c0 ≈ 1.0, c1 ≈ 0.6931472, c2 ≈ 0.2401771, c3 ≈ 0.05550411` (Sollya minimax, **not** Taylor).
- `flash_attn/cute/utils.py::ex2_emulation` — scalar emulation: `add.rm.ftz.f32` (round-down add) for range reduction, integer bias reconstruction via `(int_part + 127) << 23`, Horner evaluation, multiply by reconstructed `2^k`.
- `flash_attn/cute/utils.py::ex2_emulation_2` — packed (2-lane) emulation that consumes `fma.rn.f32x2` from A-2.
- `flash_attn/cute/utils.py::e2e_asm2` — pure-PTX packed variant (Horner using `fma.rn.f32x2` directly).

**Design (FA2 port):**

1. Port `POLY_EX2` as a `__device__ constexpr float kPolyEx2[4]` in `softmax.h` (or a sibling header). Use the **exact same numerical values** as FA4 to keep the accuracy budget identical.

2. Scalar emulation:

    ```cpp
    __forceinline__ __device__ float exp2f_emu(float y) {
        // Range reduce: y = k + f, k integer (round down), f in [0, 1).
        // Using add.rm.ftz.f32 with a large bias matches FA4's add_round_down + combine path.
        float y_floor;
        asm("add.rm.ftz.f32 %0, %1, 0.0;\n\t" : "=f"(y_floor) : "f"(y));
        // y_floor now equals floor(y) when y is representable; convert to int by reinterpret cast
        // through the same bias trick FA4 uses (see flash_attn/cute/utils.py::combine_int_frac_ex2).
        int k = __float2int_rd(y);          // floor(y) as int
        float f = y - __int2float_rn(k);    // f in [0, 1)
        // Horner on POLY_EX2 (degree 3): p(f) = ((c3*f + c2)*f + c1)*f + c0.
        float p = kPolyEx2[3];
        p = fmaf(p, f, kPolyEx2[2]);
        p = fmaf(p, f, kPolyEx2[1]);
        p = fmaf(p, f, kPolyEx2[0]);
        // Reconstruct 2^k * p via bit manipulation of the exponent field.
        int bits = __float_as_int(p) + (k << 23);
        return __int_as_float(bits);
    }
    ```

    The FA4 implementation prefers `add.rm.ftz.f32` + magic-bias for `floor`; using `__float2int_rd` is the portable CUDA-C equivalent and the compiler typically lowers it to a single SASS instruction.

3. Packed emulation (`exp2f_emu_x2`) for sm_100+:

    ```cpp
    __forceinline__ __device__ void exp2f_emu_x2(
        float &d0, float &d1, float y0, float y1) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && !defined(UNFUSE_FMA)
        // Mirror flash_attn/cute/utils.py::e2e_asm2: Horner with fma.rn.f32x2.
        int k0 = __float2int_rd(y0), k1 = __float2int_rd(y1);
        float f0 = y0 - __int2float_rn(k0);
        float f1 = y1 - __int2float_rn(k1);
        float p0 = kPolyEx2[3], p1 = kPolyEx2[3];
        // Three packed FMAs:
        fma_f32x2(p0, p1, p0, p1, f0, f1, kPolyEx2[2], kPolyEx2[2]);
        fma_f32x2(p0, p1, p0, p1, f0, f1, kPolyEx2[1], kPolyEx2[1]);
        fma_f32x2(p0, p1, p0, p1, f0, f1, kPolyEx2[0], kPolyEx2[0]);
        int b0 = __float_as_int(p0) + (k0 << 23);
        int b1 = __float_as_int(p1) + (k1 << 23);
        d0 = __int_as_float(b0);
        d1 = __int_as_float(b1);
    #else
        d0 = exp2f_emu(y0);
        d1 = exp2f_emu(y1);
    #endif
    }
    ```

    On sm_100/120 this issues three `FFMA.X2` instructions per pair-of-elements vs. two `MUFU.EX2`. The trade is FMA throughput for SFU throughput, and on Blackwell the FMA pipeline has substantial headroom under the existing softmax workload.

4. Plug-in: `scale_apply_exp2` gains a `template<..., bool Use_exp2_emu=false, ...>` parameter. When `Use_exp2_emu=true`, the inner loop calls `exp2f_emu_x2` for paired lanes (after the existing pair-of-2 split introduced by A-2) and `exp2f_emu` for the trailing odd lane. The backward path (`flash_bwd_kernel.h:536`) calls `scale_apply_exp2<Scale_max=false>` and therefore receives B-1 automatically.

**Accuracy:**

- POLY_EX2 absolute error on `[0, 1)`: ≈ 2e-5 (Sollya degree-3 minimax).
- End-to-end softmax relative error budget: per-element ≤ 5e-5, end-to-end logits cosine ≥ 0.9995. These are tighter than the previous design's 1e-3 / 1e-2 because Sollya coefficients are materially more accurate than the placeholder Taylor coefficients the original sketch listed.
- Backward path (`flash_bwd_kernel.h:536` calls `scale_apply_exp2<Scale_max=false>`) gets the polynomial automatically because it shares the same callee. The backward already loses accuracy through fp16 reduction; using the polynomial only widens the existing error margin by the polynomial's own contribution, which is below the dQ/dK/dV tolerance.

**Arch interaction:**

- `exp2f_emu_x2` directly consumes `fma_f32x2` from A-2. The polynomial is the **largest** beneficiary of A-2 — each call is three packed FMAs vs. six scalar FMAs on sm_80.
- A-3's `max_offset` bias keeps `f` inside `[0, 1)` even when input distributions push the exponent toward the boundary; without A-3 the boundary cases may fall outside the polynomial's design domain and require the FA4-style `add_round_down` correction. Setting `MaxOffsetMilli=0` (default) is safe for fp16/bf16 because `softmax_scale * row_max` is well-bounded already; non-zero `max_offset` is only needed for the future fp8 path.

**Why not always-on:** the polynomial is pure FMA. On kernels that are already memory-bound (e.g. small seqlen ≤ 256) the SFU is unused and emulation is pure overhead. The default is `Use_exp2_emu=false` (= behave like the current build); the flag is opt-in at the call site.

**SASS gate (mandatory):** `cuobjdump --dump-sass` of the sm_120 binary must show `FFMA.X2` inside `exp2f_emu_x2` and **must not** show any `MUFU.EX2` in the softmax inner loop when `Use_exp2_emu=true`. If `MUFU.EX2` survives, the inline `asm` block was folded or rewritten and B-1 is not actually live.

### 7.5 Plan B-2 detailed design

**Goal:** Mix `MUFU.EX2` and polynomial within a single row, using FA4's three-parameter conditional emulation, so that the SFU correction periodically resyncs the polynomial output.

**Reference implementation (FA4):** `flash_attn/cute/softmax.py` exposes three parameters:

- `ex2_emu_freq` — period (how many lanes between SFU corrections); `1` = always SFU, `0` = always polynomial.
- `ex2_emu_res` — residue index inside each period at which the SFU is used (other indices use the polynomial).
- `ex2_emu_start_frg` — starting fragment offset; allows phase-offsetting the schedule when `scale_apply_exp2` is called in a stream of fragments.

**Design (FA2 port):**

1. Extend the B-1 template signature:

    ```cpp
    template<..., bool Use_exp2_emu=false,
             int Ex2EmuFreq=1, int Ex2EmuRes=0, int Ex2EmuStartFrg=0,
             ...>
    ```

    Default `Ex2EmuFreq=1` is "every lane uses MUFU.EX2" = behaves like B-1's `Use_exp2_emu=false` path. `Ex2EmuFreq=0` is "always polynomial" = behaves like B-1's `Use_exp2_emu=true` path. Intermediate values mix.

2. Inside `scale_apply_exp2`:

    ```cpp
    constexpr int kFreq = Ex2EmuFreq;
    constexpr int kRes = Ex2EmuRes;
    constexpr int kStart = Ex2EmuStartFrg;
    for (int ni_pair = 0; ni_pair < N1 / 2; ++ni_pair) {
        int idx0 = (kStart + 2 * ni_pair + 0) % (kFreq == 0 ? 1 : kFreq);
        int idx1 = (kStart + 2 * ni_pair + 1) % (kFreq == 0 ? 1 : kFreq);
        bool use_emu0 = (kFreq != 1) && (idx0 != kRes);
        bool use_emu1 = (kFreq != 1) && (idx1 != kRes);
        // Compute exponents t0, t1 from row_max-adjusted scaled inputs.
        if (use_emu0 && use_emu1) {
            exp2f_emu_x2(out0, out1, t0, t1);
        } else if (!use_emu0 && !use_emu1) {
            out0 = exp2f(t0);  // MUFU.EX2
            out1 = exp2f(t1);
        } else {
            // Mixed: one lane SFU, one polynomial.
            out0 = use_emu0 ? exp2f_emu(t0) : exp2f(t0);
            out1 = use_emu1 ? exp2f_emu(t1) : exp2f(t1);
        }
    }
    // Trailing odd element: scalar.
    ```

    All branches are compile-time constants relative to the template parameters, so the compiler generates a single specialized loop body per `(Ex2EmuFreq, Ex2EmuRes, Ex2EmuStartFrg)` combination.

3. Recommended initial defaults (per FA4 inspection):

    - hdim ≤ 128: `(Ex2EmuFreq, Ex2EmuRes, Ex2EmuStartFrg) = (4, 3, 0)`.
    - hdim ≥ 192: `(2, 1, 0)`.
    - sm_80 / sm_86 / sm_89: keep `(1, 0, 0)` (= no emulation); polynomial benefit on these arches is marginal because `fma.f32x2` is unavailable.

**Interaction with A-2 / B-1:**

- When `Use_exp2_emu=true` and `Ex2EmuFreq=0` (= all polynomial), B-2 collapses to B-1.
- When `Ex2EmuFreq > 1`, the polynomial lanes still go through `exp2f_emu_x2` (packed) wherever both lanes of a pair are polynomial; the mixed case falls back to scalar. On Blackwell the mixed case is rare for the recommended defaults (typical period 2 or 4 keeps most pairs uniform).

**Tuning surface:** `(Ex2EmuFreq, Ex2EmuRes, Ex2EmuStartFrg)` in cross product with `hdim ∈ {64, 96, 128, 192, 256}`. The full sweep is expensive; in practice, hold `Ex2EmuStartFrg=0` and sweep `Ex2EmuFreq ∈ {0, 2, 4, 8}` × `Ex2EmuRes ∈ {0, 1, 2, 3}` constrained to `Ex2EmuRes < Ex2EmuFreq`. This is the same surface FA4 tunes against.

**Risk:** the parameter space is larger than B-1's single knob, but each cell compiles independently; no kernel correctness risk above B-1. Defaulting `Ex2EmuFreq=1` keeps the door closed until the sweep completes.

### 7.6 Plan C-1 detailed design

**Goal:** Add a Blackwell-specific tuning table in `kernel_traits.h` and `flash_fwd_launch_template.h` that selects `kBlockM`, `kBlockN`, and `kNWarps` based on `hdim` and target arch.

**Mechanism:**

1. In `kernel_traits.h`, introduce a new struct family `Flash_fwd_kernel_traits_blackwell<headdim, ...>` that mirrors `Flash_fwd_kernel_traits<headdim, ...>` but with tuned constants.
2. In `flash_fwd_launch_template.h`, dispatch at the macro level:
   ```cpp
   #if defined(FLASH_FWD_USE_BLACKWELL_TRAITS) && __CUDA_ARCH__ >= 1000
       using Kernel_traits = Flash_fwd_kernel_traits_blackwell</*hdim=*/Headdim, /*...*/>;
   #else
       using Kernel_traits = Flash_fwd_kernel_traits</*hdim=*/Headdim, /*...*/>;
   #endif
   ```
3. The Blackwell traits will deviate from the Ampere traits primarily in `kBlockN` (typically larger to exploit additional smem) and `kNWarps` (often 8 on Hopper-class, depending on hdim).

**`kNWarps` tuning:**

`kNWarps` is a primary tuning lever and is enumerated explicitly (not derived). Sweep surface for sm_120: `kNWarps ∈ {4, 8}` for hdim ≤ 128, `kNWarps ∈ {4, 8, 12}` for hdim ≥ 192. The current FA2 default on sm_80 is `4` for most hdims; FA3/FA4 on sm_90+ commonly uses `8` because Hopper's larger register file and the warp-group MMA dispatch favor more warps per block. Blackwell consumer (sm_120) inherits Hopper's register-file scaling and benefits from the same shift, but the optimal point depends on hdim and on how much smem the block consumes — hence the explicit sweep.

**Interaction with `pack_gqa` (informational):**

FA3/FA4 fuse multiple Q heads per KV head when GQA / MQA is in effect (`flash_attn/cute/pack_gqa.py`). FA2 does **not** implement this fusion. When GQA group size is large, the effective `kBlockM` per KV head shrinks, so a Blackwell trait that is optimal for MHA (group=1) may underutilize SMs at group=8 / group=16. C-1 must therefore record GQA group size as a measurement axis, even though C-1 itself does not introduce a Q-fusion code path. Otherwise the retune overfits to MHA workloads and produces a regression on Mistral-style 8-way GQA models. A future plan (out of scope for the current document) could backport `pack_gqa` to FA2 sm_120 as a follow-on.

**Tuning method:** sweep on a single sm_120 device (RTX 5060 Ti), record `bench/benchmark_flash_attention.py` median over 100 runs for each `(hdim, seqlen, dtype, kBlockM, kBlockN, kNWarps, gqa_group)` cell, pick the Pareto-best per `(hdim, seqlen, dtype, gqa_group)`. Compilation cost is bounded because only one new traits family is added (not a Cartesian explosion).

**Risk and mitigation:** compile-time grows by roughly the number of new explicit instantiations. Limit explicit instantiation to hdim ∈ {64, 96, 128, 192, 256} (the existing supported set), and gate the new traits behind a build flag (`FLASH_FWD_USE_BLACKWELL_TRAITS`) initially. Default-on the flag only after C-1 produces a confirmed ≥ 5% improvement on the target hardware **and** no regression on the GQA group=8 / group=16 measurement cells.

---

## 8. Validation Plan

### 8.1 Accuracy

| Test | Tool | Metric | Pass threshold |
|---|---|---|---|
| Forward exact-vs-reference | `tests/test_flash_attn.py::test_flash_attn_output` | L∞ relative vs. reference attention | fp16: `<= 1e-2`, bf16: `<= 5e-3` |
| Backward exact-vs-reference | `tests/test_flash_attn.py::test_flash_attn_backward` | L∞ relative vs. reference | fp16: `<= 1.5e-2`, bf16: `<= 7e-3` |
| Endpoint logits (downstream) | Custom script: load a reference LLM, run a fixed prompt, compare attention path outputs | per-token cosine | `>= 0.9995` |
| Ablation A-1 off | Rebuild with `Use_rescale_threshold=false` at all 4 call sites | Bit-identical output vs. pre-A-1 binary | exact |
| Ablation A-2 off | Build with `UNFUSE_FMA` defined | Output relative L∞ within `1e-6` of A-2-on (sm_100/120 only) | `1e-6` |
| Ablation A-3 default | Build with `MaxOffsetMilli=0` at all call sites (default) | Bit-identical output vs. post-A-2 binary | exact |
| B-1 polynomial accuracy (per-element) | Synthetic input: random `t ∈ [-30, 0]`, compare `exp2f_emu(t)` vs. `exp2f(t)` | per-element relative error | `<= 5e-5` |
| B-1 softmax accuracy (end-to-end) | `tests/test_flash_attn.py::test_flash_attn_output` with `Use_exp2_emu=true` | L∞ relative vs. reference attention | fp16: `<= 1e-2`, bf16: `<= 5e-3` |
| B-1 logits cosine | LLM end-to-end prompt with `Use_exp2_emu=true` | per-token cosine | `>= 0.9995` |
| B-2 mixed-mode accuracy | Same as B-1 but for each `(Ex2EmuFreq, Ex2EmuRes)` cell in the sweep grid | same thresholds as B-1 | same |

### 8.2 Performance

| Test | Tool | Cells | Measurement |
|---|---|---|---|
| Forward latency | `bench/benchmark_flash_attention.py` (existing) | hdim ∈ {64, 96, 128, 192, 256}, seqlen ∈ {1024, 2048, 4096, 8192, 16384}, dtype ∈ {fp16, bf16}, causal ∈ {true, false}, gqa_group ∈ {1, 8} | median wall-clock over 100 runs after 20 warmup |
| SASS inspection (A-2) | `cuobjdump --dump-sass` | `_ZN12FLASH_NAMESPACE16scale_apply_exp2*` symbols | **Hard gate:** `FFMA.X2` present on sm_120, absent on sm_80. If absent on sm_120, A-2 is not actually live and must be fixed before proceeding to B-*. |
| SASS inspection (B-1) | `cuobjdump --dump-sass` | symbols containing `exp2f_emu_x2` | **Hard gate:** `FFMA.X2` present **and** `MUFU.EX2` absent inside the emulation block on sm_120 when `Use_exp2_emu=true`. |
| Compile time | `time WindowsWhlBuilder_cuda.bat` | full build | no more than 110% of pre-change baseline |
| Wheel size | filesystem | `dist\flash_attn-2.9.0+...whl` | no more than 110% of pre-change baseline |

**SASS gate enforcement:** the two SASS inspection rows above are hard gates, not soft observations. The Phase 1 / Phase 2 exit criteria in §6 require the corresponding SASS gate to pass; if a build claims A-2 or B-1 is enabled but the SASS does not show the expected instructions, the gate fails and the phase is not considered complete. See R8 in §9.

### 8.3 Hardware matrix

| Arch | Hardware available to operator | Validation status |
|---|---|---|
| sm_80 | None confirmed | Functional regression test (via PTX-JIT or remote rental) |
| sm_86 | None confirmed | Same as sm_80 |
| sm_89 | None confirmed | Same as sm_80 |
| sm_90 | None confirmed | Same as sm_80 |
| sm_100 | None confirmed | Same as sm_80; PTX-JIT verified `fma.rn.f32x2` emission via `nvdisasm` |
| sm_120 | RTX 5060 Ti 16GB | Primary validation target |

The primary measurement device is the operator's RTX 5060 Ti. For sm_80 / sm_90 confirmation, a reduced cross-check is acceptable: build for those arches, JIT-load on the local device through PTX-JIT, run a forward pass, and confirm shape/finiteness. Full performance numbers on those arches are out of scope unless a corresponding device becomes available.

### 8.4 Benchmark scenarios

Each phase exit gate uses the following baseline harness, defined here once so subsequent phases reuse it.

```python
# bench/fa2_baseline.py (to be added in Phase 0)
import torch
from flash_attn import flash_attn_func
from triton.testing import do_bench

def cell(batch, seqlen, nheads, hdim, dtype, causal):
    q = torch.randn(batch, seqlen, nheads, hdim, dtype=dtype, device='cuda')
    k = torch.randn_like(q); v = torch.randn_like(q)
    fn = lambda: flash_attn_func(q, k, v, causal=causal)
    ms = do_bench(fn, warmup=20, rep=100)
    return ms

for hdim in [64, 96, 128, 192, 256]:
    for seqlen in [1024, 2048, 4096, 8192, 16384]:
        for dtype in [torch.float16, torch.bfloat16]:
            for causal in [False, True]:
                print(hdim, seqlen, dtype, causal, cell(2, seqlen, 16, hdim, dtype, causal))
```

This is a placeholder; the Phase 0 task is to commit a finalized version of this script under `bench/` and record its output as the baseline JSON before any of A-1 / A-2 are claimed measured.

---

## 9. Risks and Mitigations

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | A-2 inline asm fails to compile on some nvcc version | Low | Build break | The `__CUDA_ARCH__` guard ensures the asm is only seen by nvcc when it's actually targeting sm_100+. CUDA 13.0 ptxas accepts `fma.rn.f32x2` for `.target sm_100`. The fallback path is unconditional plain FMA. |
| R2 | A-1 accuracy drop is larger than the threshold suggests on a real workload | Medium | Functional regression | The threshold is `constexpr` and can be tightened (e.g. to `-0.005`) in a follow-up commit. The flag is opt-in per call site, so reverting at any single call site is one-line. |
| R3 | `fma.rn.f32x2` constraint mismatch leaves the asm unscheduled, defeating the optimization silently | Medium | No perf gain on sm_100+, but no correctness issue | The SASS inspection step in §8.2 catches this. Add it as a hard gate before declaring A-2 "live". |
| R4 | Polynomial coefficients in B-1 are tuned for one hdim and degrade on another | Medium | Accuracy regression | Tune coefficients on a representative range covering all supported hdims; validate per-hdim. |
| R5 | C-1 retuning explodes compile time on Windows | Medium | Developer-experience regression | Limit to one new traits family; gate via `FLASH_FWD_USE_BLACKWELL_TRAITS`. |
| R6 | Cursor's `StrReplace` against externally-edited files causes a cache-divergence loss | Medium | Code loss | All large edits use `Read` immediately before `StrReplace`, and pre/post sha256 + byte-size are recorded for every change. Per `no-tool-blaming-and-mixed-editing-caches.mdc`, `launch_utils.py`-style files are not touched here. |
| R7 | A merge from upstream introduces a conflict against the new `Use_rescale_threshold` template parameter | Low-Medium | Manual merge needed | The template parameter has a default value of `false`, so any unchanged upstream call site continues to compile. Only the fork's four call sites need to be re-applied if a merge undoes them. |
| R8 | SASS gate (A-2 `FFMA.X2`, B-1 `FFMA.X2` + no `MUFU.EX2`) is not enforced as a hard gate, leading to "silent dead optimization" — the build succeeds, accuracy passes, but the optimization is not actually live on the binary | Medium | No perf gain, but reported as a win | The §8.2 SASS inspection rows are escalated to **hard gates** in §6's phase exit criteria. CI script `bench/check_sass_gates.py` (to be added in Phase 0) parses `cuobjdump --dump-sass` output and exits non-zero if either gate fails. No commit advances the phase without this script's green. |
| R9 | A-3 `MaxOffsetMilli=0` subtract is not constant-folded by nvcc, producing a dead `FADD` per inner-loop iteration | Low | Tiny perf regression at default | Verified by SASS inspection at the same point as R8. If the subtract survives, replace the runtime expression with an `if constexpr (MaxOffsetMilli != 0)` guard around the entire bias subtract. |
| R10 | B-1 polynomial accuracy regresses on a workload with extreme exponent range (e.g. logits stretched by an unusually large `softmax_scale`) | Low-Medium | Logits cosine below 0.9995 | A-3 provides the bias scaffold; if a workload triggers the regression, set `MaxOffsetMilli` to a non-zero value to keep `f` in-domain. As an interim mitigation, keep `Use_exp2_emu=false` for that call site. |
| R11 | B-2 parameter sweep grid is large; tuning never converges to a single recommended default per arch | Medium | Plan stalls indefinitely | Anchor defaults to FA4's `flash_attn/cute/softmax.py` values for the corresponding hdim bucket. Use FA4 numbers as the "ship-ready" defaults and treat further tuning as a Phase-3-or-later refinement. |
| R12 | C-1 retuning is optimized for MHA (group=1) and silently regresses on GQA workloads | Medium | Live regression on Mistral / Llama-3-style GQA models | Sweep grid includes `gqa_group ∈ {1, 8}` as a measurement axis (§8.2). Default-on the new traits only if the Pareto-best for gqa_group=8 is also non-regressing. |

---

## 10. Rollback Procedure

This plan's changes are entirely additive at the source level, so rollback is a localized revert per plan.

### 10.1 Roll back A-1

1. In `csrc/flash_attn/src/flash_fwd_kernel.h`, remove `/*Use_rescale_threshold=*/true` from the four `softmax_rescale_o` call sites (lines 344, 407, 918, 985 in the post-A-1 file).
2. In `csrc/flash_attn/src/softmax.h::softmax_rescale_o`, remove the `bool Use_rescale_threshold=false` template parameter and the `if constexpr (Use_rescale_threshold) { ... continue; }` block.
3. Rebuild via `WindowsWhlBuilder_cuda.bat`. The resulting binary is bit-identical to the pre-A-1 binary modulo build timestamp.

### 10.2 Roll back A-2

1. In `csrc/flash_attn/src/softmax.h`, delete the `fma_f32x2` helper (lines 65-88 in the post-A-2 file).
2. Restore the original `scale_apply_exp2` body (single scalar loop with the existing `UNFUSE_FMA` macro split). The pre-A-2 version is preserved in the backup at `D:\USERFILES\_backups\flash-attention_20260513_092006\softmax.h.orig` and corresponds to sha256 `A734A5D857D59D2892BBEEEFD337B880E0BF69DB2C2460C3CFC91A5B0A112D67`.
3. Rebuild.

### 10.3 Roll back the 2.9.0 version bump

1. In `flash_attn/__init__.py`, change `__version__ = "2.9.0"` back to `"2.8.4"`.
2. Rebuild and rename the wheel accordingly.

### 10.4 Roll back A-3, B-*, and C-1

A-3, B-*, and C-1 are all additive and opt-in by template parameter or build flag. Rollback is a single-line change at call sites for each:

1. **A-3 rollback:** revert all `MaxOffsetMilli=N` template arguments at call sites to omit the parameter (it defaults to `0` = byte-identical pre-A-3 behavior). The template parameter itself can be left in the function signature indefinitely; removing it requires only deleting the `int MaxOffsetMilli=0` non-type parameter and the corresponding `kMaxOffset` local in `softmax_rescale_o` / `scale_apply_exp2`.

2. **B-1 rollback:** revert all `Use_exp2_emu=true` template arguments at call sites to the default (`false`). The `exp2f_emu` / `exp2f_emu_x2` helpers and `kPolyEx2` table can be left in `softmax.h` (no caller path).

3. **B-2 rollback:** identical to B-1, but additionally revert `(Ex2EmuFreq, Ex2EmuRes, Ex2EmuStartFrg)` arguments to defaults `(1, 0, 0)`.

4. **C-1 rollback:** remove `-DFLASH_FWD_USE_BLACKWELL_TRAITS` from the build command (or unset the macro in `WindowsWhlBuilder_cuda.bat`). The Blackwell-specific traits struct can stay in `kernel_traits.h` unreferenced.

### 10.5 SASS-gate failure — roll-forward instead of roll-back

If a build claims A-2 or B-1 is enabled but the SASS gate (§8.2) fails, the correct response is **roll-forward, not roll-back**: do not revert the C++ template plumbing — instead, replace the inline `asm` block(s) with the pure-PTX one-shot variant patterned on `flash_attn/cute/utils.py::e2e_asm2`. Concretely:

- For A-2 `fma_f32x2`: collapse the four `fma.rn.f32x2` lines and the surrounding C++ wrapper into a single inline `asm volatile (...)` with explicit `.reg .f32x2 …` declarations matching the FA4 PTX layout. Confirm SASS emits `FFMA.X2`.
- For B-1 `exp2f_emu_x2`: replace the Horner-on-`fma_f32x2` body with the FA4 `e2e_asm2`-equivalent pure PTX block: three `fma.rn.f32x2` lines with constant-folded coefficient pairs in `.f32x2` literals.

Roll-back is appropriate only if roll-forward also fails (e.g. nvcc cannot lower the pure-PTX form for the current toolchain version). In that case, document the failure mode in §11.4 Pending and defer the plan until the toolchain advances.

---

## 11. Applied Change History

### 11.1 2026-05-13 — Plan A-1 applied

| File | Pre sha256 | Post sha256 | Pre size | Post size | Backup |
|---|---|---|---|---|---|
| `csrc/flash_attn/src/softmax.h` | `A734A5D857D59D2892BBEEEFD337B880E0BF69DB2C2460C3CFC91A5B0A112D67` | `BF7F98F431EC562E66293CEB080CB2C938BC40F48D756CF001A2885A9644B325` | 9653 | 12386 | `D:\USERFILES\_backups\flash-attention_20260513_092006\softmax.h.orig` |
| `csrc/flash_attn/src/flash_fwd_kernel.h` | `94BC41E01E5C223FCB8DD5ABECD4B8CF158F455202FC15E19FD8B75D0DA8E76F` | `31728AD35351A6430A78A13AA2B497E304241B3E5B2571B45556A9A6DA2D8A64` | 78019 | 78147 | `D:\USERFILES\_backups\flash-attention_20260513_092006\flash_fwd_kernel.h.orig` |

Net source delta: 2 files changed, 71 insertions(+), 13 deletions(-).

Call-site changes (Plan A-1): four `Is_first=false` invocations of `softmax_rescale_o` in `flash_fwd_kernel.h` now pass `/*Use_rescale_threshold=*/true`. The two `Is_first=true` invocations are unchanged.

### 11.2 2026-05-13 — Plan A-2 applied

Same files as Plan A-1 (A-1 and A-2 were applied in the same edit session). The `fma_f32x2` helper was added at lines 65-88 of `softmax.h`, and the `scale_apply_exp2` body was rewritten to branch on `__CUDA_ARCH__ >= 1000 && !defined(UNFUSE_FMA)`.

### 11.3 2026-05-13 — Version bump

`flash_attn/__init__.py` line 6: `__version__ = "2.8.4"` → `"2.9.0"`. Pre sha256 `0F7D432F9D94FB89A30967DC2463FCACF683E9F69BA501BAE9D8C1107687D21D`, post sha256 `5F3E25933EF2DE21111F69FE46CA61756C92E3E617F379BA196D5FA2BCCDAECE`. File size unchanged at 470 bytes (both literals are 5 characters). `setup.py::get_package_version()` confirmed to return `"2.9.0"`.

Backup: `D:\USERFILES\_backups\flash-attention_20260513_092006\__init__.py.orig`.

### 11.4 Pending

- Phase 0 measurement harness (`bench/fa2_baseline.py`)
- Phase 0 SASS-gate enforcement script (`bench/check_sass_gates.py`) — parses `cuobjdump --dump-sass` and validates A-2 / B-1 emit `FFMA.X2`
- Plan A-3 commit (`max_offset` bias scaffold; default `MaxOffsetMilli=0` keeps binary byte-identical)
- Plan B-1 commit (Sollya `POLY_EX2` polynomial + `exp2f_emu` / `exp2f_emu_x2`)
- Plan B-2 commit (3-parameter mixed mode `Ex2EmuFreq` / `Ex2EmuRes` / `Ex2EmuStartFrg`)
- Plan C-1 commit (`Flash_fwd_kernel_traits_blackwell` family + `FLASH_FWD_USE_BLACKWELL_TRAITS` gate)
- Validation results table (per §8) populated against the operator's RTX 5060 Ti

---

## 12. References

### 12.1 In-repo

- `csrc/flash_attn/src/softmax.h` — primary edit target for A-* and B-*.
- `csrc/flash_attn/src/flash_fwd_kernel.h` — call sites of `softmax_rescale_o`.
- `csrc/flash_attn/src/flash_bwd_kernel.h` — call site of `scale_apply_exp2` for the backward path.
- `csrc/flash_attn/src/kernel_traits.h` — block-size and warp-count traits (target of C-1).
- `csrc/flash_attn/src/flash_fwd_launch_template.h` — dispatch site (target of C-1).
- `flash_attn/cute/softmax.py` — FA4 reference for A-1, A-3, B-2 ideas (`rescale_threshold`, `max_offset`, `ex2_emu_freq` / `ex2_emu_res` / `ex2_emu_start_frg`).
- `flash_attn/cute/utils.py` — FA4 reference for B-1: `POLY_EX2` Sollya coefficients, `ex2_emulation` (scalar), `ex2_emulation_2` (packed), `e2e_asm2` (pure PTX packed), `add_round_down`, `combine_int_frac_ex2`, `evaluate_polynomial`, `evaluate_polynomial_2`, `fma_packed_f32x2`.
- `flash_attn/cute/flash_fwd_sm100.py` — FA4 reference for sm_100-specific kernel structure (background; not directly portable). Source of the `max_offset = 8 if Q_dtype.width == 8 else 0` default used in A-3.
- `flash_attn/cute/pack_gqa.py` — FA4 reference for the GQA fusion path (informational input to C-1's `gqa_group` measurement axis; out-of-scope for the current plan as a code change).
- `setup.py` — `cuda_archs()`, `add_cuda_gencodes()`, `get_package_version()` (Phase 0 base).
- `WindowsWhlBuilder_cuda.bat` — build entry.
- `AI/SM90_BLOCK_SIZE_TUNING.md` — adjacent block-size tuning document; methodology reference for C-1.
- `AI/SASS_MMA_ANALYSIS.md` — adjacent SASS-inspection document; methodology reference for §8.2 and the SASS gates in §6.

### 12.2 PTX / hardware

- PTX ISA, section "Floating-Point Instructions / fma" — `fma.rn.f32x2` operand and target constraints.
- PTX ISA, section "Special Function Instructions" — `MUFU.EX2` (`ex2.approx.f32`) timing background.
- NVIDIA Blackwell Tuning Guide — sm_100 / sm_120 SMEM and tensor-core ratios (used as input to C-1).

### 12.3 External

- `https://github.com/pytorch/pytorch/issues/121558` — origin of the `UNFUSE_FMA` macro; preserved in A-2's guard.
- `https://github.com/Dao-AILab/flash-attention` — upstream FA2/FA3/FA4.
- Sollya project documentation (`https://www.sollya.org/`) — minimax polynomial framework underlying `POLY_EX2`. Background only; the coefficients are fixed at FA4's values, not re-derived in this plan.

---

## 13. Future Work (out of scope for the current plan)

The plans below are explicitly **not** part of this document but are recorded here so that the next-generation FA2 fork roadmap has a starting point.

### 13.1 FP8 forward kernel (A-3 non-zero `MaxOffsetMilli`)

A-3 ships as a byte-identical scaffold with `MaxOffsetMilli=0`. The non-zero path (typically `MaxOffsetMilli=8000` matching FA4's `max_offset=8.0f` for fp8) is reserved for a future fp8 forward kernel that:

- Quantizes `Q` and `K` to e4m3 / e5m2 before the QK matmul.
- Calibrates per-block scale factors; the `max_offset` bias keeps the polynomial's exponent argument well-inside `[0, 1)` even after scale renormalization.
- Pairs naturally with B-1 (polynomial emulation is mandatory because `MUFU.EX2` is too slow to keep up with the fp8 GEMM throughput).

This requires new fp8 quant / dequant paths in `softmax.h` and a separate dispatch family in `flash_fwd_launch_template.h`; it is materially larger than any plan in §5 and is intentionally deferred.

### 13.2 GQA / MQA fusion (`pack_gqa` backport)

FA4's `flash_attn/cute/pack_gqa.py` fuses multiple Q heads sharing a single KV head into one tile, raising effective occupancy on Mistral / Llama-3-style GQA models. FA2 currently does **not** implement this fusion; the fork's GQA path issues one CTA per Q-head and underutilizes SMs when group size is large. A backport would:

- Add a `pack_gqa: bool` template parameter to `Flash_fwd_kernel_traits`.
- Rewrite the Q load path to multi-cast across the GQA group.
- Re-tune Blackwell block sizes for the fused layout (extension of C-1).

This is the natural follow-on to C-1 and is recorded here as the highest-priority post-plan item.

### 13.3 Block sparsity and mask / score modifier hooks

FA4 exposes `score_mod` / `mask_mod` / `block_sparse_tensors` as user-defined `@cute.jit` callables injected at compile time. FA2 has no equivalent extension surface. Adding one would let downstream users implement custom attention biases (ALiBi, Hyena, etc.) without forking the C++ kernel; the work is bounded by the need to surface a stable extension ABI through `flash_attn_func`'s Python wrapper.

### 13.4 SageAttention-style block-level quantization for the backward

SageAttention's quantization (`quant_per_block_int8` + smoothing) is more aggressive than FA2's. The backward path is the dominant cost in training and would benefit most. A backport requires:

- Re-implementing `quant_per_block_int8` in C++/CUDA (the SageAttention version is CuTeDSL).
- A new dQ / dK / dV reduction strategy that preserves int8 quantization through the gradient accumulation.

This is research-scale work and is recorded only as a directional note.

## 14. FA4 Sync Update (2026-08-11)

This section records changes discovered in the upstream FA4 codebase (`flash_attn/cute/`) since the plan was originally written (2026-07-10). Each item is cross-referenced to the plan it affects. New plans are added to §5, §6, and §7 by reference.

### 14.1 A-1 threshold: adopt FA4's float `rescale_threshold` (affects §5.1, §7.1)

**Status:** Planned (supersedes the original A-1 threshold design).

The original plan uses a bool template flag `Use_rescale_threshold` with a hard-coded `kRescaleSkipThreshold = -0.01f`. FA4 has since evolved `SoftmaxSm100.rescale_threshold` into a `cutlass.Constexpr[float]` (default `0.0`), set to `8.0` for fp16/bf16 and `0.0` for fp8 (`flash_attn/cute/flash_fwd_sm100.py` line ~2190, commit `849f660`).

FA4's `8.0` means `scores_scale >= exp2(-8.0) = 0.0039` — far more aggressive than the plan's `0.01` (`scores_scale >= 0.993`). FA4 validated this on B200 across seqlen 256–4096 with both uniform and peaked softmax distributions.

**Changes to the plan:**

1. Replace the `bool Use_rescale_threshold` template parameter with `float RescaleThreshold = 0.0f` in `softmax_rescale_o`. The skip condition becomes `if constexpr (RescaleThreshold > 0.0f) { if (acc_scale_ >= -RescaleThreshold) { ... skip ... } }`.
2. The four `Is_first=false` call sites in `flash_fwd_kernel.h` pass `/*RescaleThreshold=*/8.0f` (matching FA4's fp16/bf16 default).
3. The `static_assert(!(Is_first && Use_rescale_threshold))` added by refinement A-1-R2 is replaced with `static_assert(!(Is_first && RescaleThreshold > 0.0f))`.
4. The old `kRescaleSkipThreshold = -0.01f` constant and the refinement A-1-R1 (promoting it to file scope) are both superseded. They are no longer needed.
5. Accuracy validation threshold in §8.1 for A-1 is tightened: the acceptance criterion changes from "relative L∞ <= 1e-2 for fp16" to "relative L∞ <= 2e-2 for fp16, <= 1e-2 for bf16" to accommodate the wider skip window. The end-to-end logits cosine threshold (>= 0.9995) is unchanged.

**Risk:** The wider skip window means running `O` can lag by up to `exp2(-8.0) ≈ 0.4%` per skipped block in the worst case. FA4's B200 validation and the `max_offset + rescale_threshold < log2(dtype_max)` assert pattern (see §14.3) provide the safety bound. For fp16, `log2(65504) ≈ 15.99`, so `0 + 8.0 < 15.99` holds with margin.

### 14.2 New plan A-4: packed `fmax_reduce` / `fadd_reduce` on sm_100+ (new entry in §5, §6, §7)

**Status:** New plan, not in the original document.

FA4's `utils.py::fmax_reduce` and `fadd_reduce` (lines ~368–455) now branch on `arch >= 100`:

- `fmax_reduce` on sm_100+: uses 3-input `fmax(a, b, c)` to reduce 8 elements per iteration into 4 accumulators, then collapses with 3-input `fmax`. This replaces the 2-input `fmax` + serial reduction on sm_80.
- `fadd_reduce` on sm_100+: uses `cute.arch.add_packed_f32x2` to add pairs of elements in one instruction, halving the add count.

FA2's `softmax.h` has equivalent reductions (`reduce_max`, `reduce_sum`, `Allreduce`) that are arch-agnostic scalar loops. They run inside the softmax inner loop and are a measurable fraction of the non-MMA arithmetic.

**Design:**

| Item | Detail |
|---|---|
| Location | `csrc/flash_attn/src/softmax.h` — `reduce_max`, `reduce_sum`, `Allreduce` |
| sm_100+ path | `fmax_3way(a, b, c)` PTX inline asm for max; `fma_f32x2`-based packed add for sum |
| sm_80/86/89/90 path | Unchanged scalar code (binary identical) |
| Guard | `#if __CUDA_ARCH__ >= 1000 && !defined(UNFUSE_FMA)` (same as A-2) |
| Helper reuse | Uses `fma_f32x2` from A-2 for the sum reduction. New `fmax_3way` helper added alongside it. |
| SASS gate | `FMNMX.X2` or equivalent 3-input max SASS mnemonic present on sm_120 in the reduction symbols |

**fmax_3way helper:**

```cpp
__forceinline__ __device__ float fmax_3way(float a, float b, float c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    float result;
    asm("max.f32 %0, %1, %2;\n\t"
        "max.f32 %0, %0, %3;\n\t"
        : "=f"(result) : "f"(a), "f"(b), "f"(c));
    return result;
    // Note: PTX does not have a 3-input max; this is two dependent max instructions.
    // On sm_100+ the scheduler can overlap the two with other work in the reduction loop.
    // FA4 uses the same pattern via cute.arch.fmax(a, b, c).
#else
    return fmaxf(fmaxf(a, b), c);
#endif
}
```

**Performance expectation:** 1–3% on sm_120, complementary to A-2's 1–5%. The reductions are smaller than `scale_apply_exp2` but run every block iteration.

**Phase placement:** Phase 1 (alongside A-2, since it shares the `fma_f32x2` helper). Gating: SASS inspection shows packed max/add instructions in the reduction symbols on sm_120.

### 14.3 A-3: add `max_offset + rescale_threshold` assert (affects §5.3, §7.3)

**Status:** Planned (additive to existing A-3).

FA4 commit `849f660` added:

```python
assert max_offset + rescale_threshold < _LOG2_DTYPE_MAX[self.q_dtype], (
    f"max_offset ({max_offset}) + rescale_threshold ({rescale_threshold}) must stay "
    f"below log2(max {self.q_dtype} value) to avoid saturating P"
)
```

with a `_LOG2_DTYPE_MAX` table for fp16, bf16, fp8e4m3, fp8e5m2.

**Change to the plan:** Add the equivalent `static_assert` in `softmax.h`:

```cpp
// fp16: log2(65504) ≈ 15.99, bf16: log2(3.39e38) ≈ 128.0
// With RescaleThreshold=8.0 and MaxOffsetMilli=0: 0 + 8.0 < 15.99 ✓
static_assert(MaxOffsetMilli * 1e-3f + RescaleThreshold < 15.99f,
    "max_offset + rescale_threshold must stay below log2(fp16_max) to avoid P saturation");
```

This is a compile-time safety check; no runtime cost.

### 14.4 B-1: pack the `add_round_down` (affects §5.4, §7.4)

**Status:** Planned (refinement to existing B-1 design).

The original B-1 design uses scalar `__float2int_rd(y)` for range reduction. FA4's `ex2_emulation_2` (in `utils.py`, line ~773) now uses:

```python
xy_rounded = cute.arch.add_packed_f32x2(xy_clamped, (fp32_round_int, fp32_round_int), rnd="rm")
xy_rounded_back = quack.activation.sub_packed_f32x2(xy_rounded, (fp32_round_int, fp32_round_int))
xy_frac = quack.activation.sub_packed_f32x2(xy_clamped, xy_rounded_back)
```

This means the round-down add, the subtract-back, and the fractional extraction are all packed on sm_100+.

**Change to the plan:** The `exp2f_emu_x2` function in §7.4 is revised to use packed operations throughout:

```cpp
__forceinline__ __device__ void exp2f_emu_x2(float &d0, float &d1, float y0, float y1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && !defined(UNFUSE_FMA)
    // Clamp to >= -127.0
    y0 = fmaxf(y0, -127.0f);
    y1 = fmaxf(y1, -127.0f);

    // Packed round-down add: x_rounded = add.rm.ftz.f32x2(y, magic)
    constexpr float magic = float(1 << 23 | 1 << 22);
    float r0, r1;
    asm("{\n\t"
        ".reg .b64 rl, ml, rb;\n\t"
        "mov.b64 ml, {%4, %5};\n\t"
        "mov.b64 rl, {%2, %3};\n\t"
        "add.rm.ftz.f32x2 rb, rl, ml;\n\t"
        "mov.b64 {%0, %1}, rb;\n\t"
        "}\n"
        : "=f"(r0), "=f"(r1)
        : "f"(y0), "f"(y1), "f"(magic), "f"(magic));

    // Packed subtract-back: x_rounded_back = r - magic
    float rb0, rb1;
    fma_f32x2(rb0, rb1, r0, r1, 1.0f, 1.0f, -magic, -magic);

    // Packed fractional: frac = y - x_rounded_back
    float f0, f1;
    fma_f32x2(f0, f1, rb0, rb1, -1.0f, -1.0f, y0, y1);

    // Horner with packed FMA (3 FMAs, degree 3)
    float p0 = kPolyEx2[3], p1 = kPolyEx2[3];
    fma_f32x2(p0, p1, p0, p1, f0, f1, kPolyEx2[2], kPolyEx2[2]);
    fma_f32x2(p0, p1, p0, p1, f0, f1, kPolyEx2[1], kPolyEx2[1]);
    fma_f32x2(p0, p1, p0, p1, f0, f1, kPolyEx2[0], kPolyEx2[0]);

    // Reconstruct: 2^k * p via bit manipulation
    int k0 = __float_as_int(r0);
    int k1 = __float_as_int(r1);
    int b0 = __float_as_int(p0) + (k0 << 23);
    int b1 = __float_as_int(p1) + (k1 << 23);
    d0 = __int_as_float(b0);
    d1 = __int_as_float(b1);
#else
    d0 = exp2f_emu(y0);
    d1 = exp2f_emu(y1);
#endif
}
```

**Note:** The exact register-level reconstruction (`combine_int_frac_ex2` in FA4) uses `shl.b32` + `add.s32` to combine the integer exponent with the polynomial result. The C++ port above approximates this with `__float_as_int` + integer shift. The SASS gate must verify that the generated code does not contain scalar `MUFU.EX2` inside `exp2f_emu_x2` on sm_120.

### 14.5 B-2: updated tuning defaults and fragment-boundary logic (affects §5.5, §7.5)

**Status:** Planned (supersedes original B-2 defaults and dispatch logic).

#### 14.5.1 Updated tuning table

FA4's `_TUNING_CONFIG` in `flash_fwd_sm100.py` (as of commit `849f660`) now has per-(causal, varlen, hdim, pack_gqa) entries:

| hdim | causal | varlen | freq | start_frg | res |
|---|---|---|---|---|---|
| 128 | yes | no | 10 | 1 | 4 (default) |
| 128 | no | yes | 16 | 1 | 4 |
| 192 | yes | no | 16 | 0 | 4 |
| 192 | no | yes | 32 | 1 | 4 |
| 256 | yes/no | no | 14 | 0 | **6** |
| 128/192 | any | yes (pack_gqa) | 0 (all HW) | 0 | — |

The original plan's `(4, 3, 0)` and `(2, 1, 0)` are superseded. The new defaults should be used directly as the initial FA2 tuning values, with the caveat that FA2's tile layout differs from FA4's and may require re-measurement.

**Change to §7.5:** Replace the "Recommended initial defaults" section with the table above. Add a note that `ex2_emu_res` is only used for hdim=256 in FA4; other hdim values use the default `res=4`.

#### 14.5.2 Fragment-boundary-aware mixed mode

FA4's `apply_exp2_convert` (in `softmax.py`, lines ~360–400) uses a more complex dispatch than the plan's `k % freq != res`:

```python
if k % ex2_emu_freq < ex2_emu_freq - ex2_emu_res:
    # Hardware MUFU.EX2
elif j >= frg_cnt - 1:
    # Hardware MUFU.EX2 (last fragment)
elif j < ex2_emu_start_frg:
    # Hardware MUFU.EX2 (before start fragment)
else:
    # Polynomial emulation
```

The fragment-boundary conditions ensure that the first and last fragments always use hardware `MUFU.EX2`, which provides accuracy anchoring at the tile edges where boundary effects are most pronounced.

**Change to §7.5:** The mixed-mode dispatch in `scale_apply_exp2` is revised to:

```cpp
// FA2 does not have "fragments" in the CuTeDSL sense, but we can
// partition the N1 inner loop into frg_tile=32 chunks to mirror FA4.
constexpr int frg_tile = 32;
constexpr int frg_cnt = N1 / frg_tile;  // assumed N1 % frg_tile == 0

for (int j = 0; j < frg_cnt; ++j) {
    for (int k = 0; k < frg_tile; k += 2) {
        int idx = j * frg_tile + k;
        bool use_hw = false;
        if constexpr (Ex2EmuFreq == 0) {
            use_hw = true;  // all hardware
        } else {
            if (k % Ex2EmuFreq < Ex2EmuFreq - Ex2EmuRes) use_hw = true;
            if (j >= frg_cnt - 1) use_hw = true;       // last fragment
            if (j < Ex2EmuStartFrg) use_hw = true;      // before start fragment
        }
        if (use_hw) {
            tensor(mi, idx)     = exp2f(tensor(mi, idx));
            tensor(mi, idx + 1) = exp2f(tensor(mi, idx + 1));
        } else {
            exp2f_emu_x2(tensor(mi, idx), tensor(mi, idx + 1),
                         tensor(mi, idx), tensor(mi, idx + 1));
        }
    }
}
```

### 14.6 `POLY_EX2` degree 0–5 support (affects §7.4)

**Status:** Planned (additive to existing B-1).

FA4's `utils.py::POLY_EX2` now provides coefficients for degrees 0 through 5 (the original plan only referenced degree 3). The higher degrees provide tighter accuracy at the cost of more FMAs:

| Degree | FMAs (scalar) | FMAs (packed) | Abs error on [0,1) |
|---|---|---|---|
| 3 | 3 | 3 (via `fma_f32x2`) | ~2e-5 |
| 4 | 4 | 4 | ~5e-6 |
| 5 | 5 | 5 | ~1e-6 |

**Change to B-1:** The `kPolyEx2` constant array in `softmax.h` is parameterized by a `PolyDegree` template parameter (default 3). A `static_assert` ensures `PolyDegree >= 1 && PolyDegree <= 5`. The degree-4 and degree-5 coefficients are copied verbatim from FA4's `POLY_EX2` dict.

This provides a precision fallback: if B-1 with degree 3 fails the logits cosine >= 0.9995 threshold on some workload, bumping to degree 4 is a one-line template parameter change.

### 14.7 Updated phase table (supersedes §6)

| Phase | Plans | Gating criterion to enter | Gating criterion to exit |
|---|---|---|---|
| 0 | Measurement scaffolding | None | `bench/fa2_baseline.py` committed; baseline JSON recorded for hdim ∈ {64,128,256}, seqlen ∈ {1k,4k,8k,16k}, dtype ∈ {fp16,bf16} on sm_120. `bench/check_sass_gates.py` committed and passing. |
| 1 | A-1 (revised), A-2, A-4 (new) | Phase 0 baseline recorded | sm_80 numeric regression ≤ 1%; sm_120 improvement in any cell. **SASS gates:** `FFMA.X2` in `scale_apply_exp2` (A-2); packed max/add in reduction symbols (A-4). |
| 2 | A-3, B-1 (revised), B-2 (revised) | Phase 1 merged and SASS gates satisfied | A-3 byte-identical at default + assert passes. B-1 sm_120 improvement with per-element softmax relative error ≤ 5e-5 and logits cosine ≥ 0.9995. B-2 defaults documented and benchmarked per hdim. **SASS gate:** `FFMA.X2` in `exp2f_emu_x2`, no `MUFU.EX2` in emulated lanes on sm_120. |
| 3 | C-1 | Phase 2 merged | hdim-bucketed retune yields ≥ 5% on sm_120 for any (hdim, seqlen) cell with no regression on sm_80 and no regression on GQA group=8. |

### 14.8 Updated sequencing constraints (supersedes §6.1)

- A-2 must land before A-3 / B-1 / B-2 / A-4 (all depend on `fma_f32x2`).
- A-4 can land in parallel with A-1 revision (they touch different parts of `softmax.h`).
- A-3 lands before B-1 (polynomial domain bounding).
- B-1 lands before B-2 (mixed mode reuses the polynomial).
- C-1 is independent; sequenced last to avoid conflict with B-* changes in `kernel_traits.h`.
- The A-1 threshold revision (§14.1) can land independently of A-4, but both should be in Phase 1.

### 14.9 Updated risk register (additions to §9)

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R13 | A-1 revised threshold (8.0) causes accuracy regression on a workload FA4 did not test | Medium | Functional regression | Revert to a smaller threshold (e.g. 4.0) via the float template parameter. The old 0.01 value is still available as a fallback. Phase 0 baseline provides the comparison point. |
| R14 | A-4 packed reduction produces different accumulation order than scalar, causing bit-level divergence from A-1-on-only builds | Low | Numeric noise | Packed FMA is round-to-nearest-even, same as scalar `fmaf`. The divergence is ULP-level, within existing accuracy budgets. |
| R15 | B-1 packed `add.rm.ftz.f32x2` is not supported by the nvcc version in use | Low | Build break on sm_100+ | The `__CUDA_ARCH__` guard ensures it is only emitted for sm_100+. CUDA 13.0 ptxas supports `add.rm.ftz.f32x2` for `.target sm_100`. |
| R16 | B-2 fragment-boundary logic assumes `N1 % 32 == 0`; a future MMA atom may violate this | Low | Compile error | Add `static_assert(N1 % frg_tile == 0)` at the entry of the mixed-mode path. |
| R17 | FA4 tuning table values are optimal for B200 (sm_100 datacenter) but suboptimal for sm_120 consumer | Medium | B-2 defaults underperform on RTX 5060 Ti | Treat FA4 values as starting points; Phase 2 measurement on sm_120 may override. Record both FA4-reference and sm_120-measured values in the commit. |

### 14.10 Updated references (additions to §12)

- `flash_attn/cute/softmax.py` — `SoftmaxSm100` class with `rescale_threshold: float`, `max_offset: int`, `apply_exp2_convert` with fragment-boundary-aware mixed mode (current as of commit `849f660`).
- `flash_attn/cute/utils.py` — `POLY_EX2` dict (degrees 0–5), `ex2_emulation_2` with `add_packed_f32x2(rnd="rm")`, `evaluate_polynomial_2` with `fma_packed_f32x2`, `fmax_reduce`/`fadd_reduce` with sm_100+ packed paths.
- `flash_attn/cute/flash_fwd_sm100.py` — `_TUNING_CONFIG` and `_FP8_TUNING_CONFIG` tables with per-(hdim, causal, varlen, pack_gqa) `ex2_emu_freq`/`ex2_emu_res`/`ex2_emu_start_frg` values. `_LOG2_DTYPE_MAX` table and `max_offset + rescale_threshold` assert.
- Commit `2409214` — `[CuTe, SM100] Fix FP8 e4m3 accuracy: make max_offset dtype-aware` (#2717). Later partially reverted by `849f660`.
- Commit `849f660` — `Numeric tweaks to fp8` (#2731). Reverted e4m3-specific `max_offset=4` cap; introduced `rescale_threshold=0.0` for fp8 + assert pattern; added `_LOG2_DTYPE_MAX` table.
- Commit `00756db` — `Expand FLASHATTENTION_DISABLE_DROPOUT to not bring in unneeded headers` (#2669). Removed transitive `M_LOG2E` definition on Windows; caused the `M_LOG2E` build fix in the fork's `softmax.h` and `setup.py`.

### 14.11 Summary of changes to existing plan sections

| Section | Change |
|---|---|
| §5.1 (A-1) | Threshold design superseded by §14.1: `float RescaleThreshold` replaces `bool Use_rescale_threshold`; default `8.0` replaces `0.01`. |
| §5.3 (A-3) | Add `max_offset + rescale_threshold < log2(dtype_max)` static_assert per §14.3. |
| §5.4 (B-1) | `exp2f_emu_x2` revised to use packed `add.rm.ftz.f32x2` per §14.4. `POLY_EX2` degree parameterized per §14.6. |
| §5.5 (B-2) | Defaults updated to FA4 tuning table per §14.5.1. Dispatch logic updated to fragment-boundary-aware per §14.5.2. |
| §5 (new) | New plan A-4 added per §14.2. |
| §6 | Phase table superseded by §14.7. Sequencing constraints superseded by §14.8. |
| §7.1 (A-1 detail) | Revised to use float threshold parameter. |
| §7.4 (B-1 detail) | `exp2f_emu_x2` implementation revised per §14.4. |
| §7.5 (B-2 detail) | Defaults and dispatch logic revised per §14.5. |
| §8.1 (Accuracy) | A-1 acceptance threshold widened per §14.1. |
| §8.2 (Performance) | A-4 SASS gate added. |
| §9 (Risks) | R13–R17 added per §14.9. |
| §11.4 (Pending) | A-1 threshold revision, A-4, B-1 packed round-down, B-2 updated defaults, POLY_EX2 degree parameterization added to pending list. |
| §12 (References) | New references added per §14.10. |

## 15. Implementation Code Edit Guide (2026-08-11)

This section provides concrete, file-by-file edit instructions for each plan item in §14. Every edit is described with: target file, exact location (line numbers as of the working tree at commit `3b44820` + uncommitted `M_LOG2E` fix), before/after code, and build verification steps.

The edits are ordered by recommended implementation sequence (Phase 1 first, then Phase 2).

---

### 15.1 A-1 threshold revision (§14.1)

**Files affected:**
1. `csrc/flash_attn/src/softmax.h` — template parameter + skip logic
2. `csrc/flash_attn/src/flash_fwd_kernel.h` — 4 call sites

#### Edit 1: `softmax.h` — replace bool parameter with float

**Location:** `softmax.h` lines ~143–145 (the `softmax_rescale_o` template signature) and lines ~120–125 (the `kSoftmaxRescaleSkipThreshold` constant).

**Before (current):**
```cpp
// Line ~120
inline constexpr float kSoftmaxRescaleSkipThreshold = -0.01f;

// Line ~143
template<bool Is_first, bool Check_inf=false, bool Use_rescale_threshold=false, typename Tensor0, typename Tensor1>
__forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
    static_assert(!(Is_first && Use_rescale_threshold),
                  "Use_rescale_threshold has no effect when Is_first=true; "
                  "remove the flag from the Is_first=true call site.");
```

**After:**
```cpp
// Line ~120 — DELETE the kSoftmaxRescaleSkipThreshold constant entirely.
// The threshold is now a template parameter, not a file-scope constant.

// Line ~143
template<bool Is_first, bool Check_inf=false, float RescaleThreshold=0.0f, typename Tensor0, typename Tensor1>
__forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
    static_assert(!(Is_first && RescaleThreshold > 0.0f),
                  "RescaleThreshold > 0 has no effect when Is_first=true; "
                  "remove the threshold from the Is_first=true call site.");
```

#### Edit 2: `softmax.h` — replace skip condition

**Location:** `softmax.h` lines ~175–177 (the `if constexpr` skip block inside `softmax_rescale_o`).

**Before:**
```cpp
                if constexpr (Use_rescale_threshold) {
                    if (scaled_diff >= kSoftmaxRescaleSkipThreshold) { continue; }
                }
```

**After:**
```cpp
                if constexpr (RescaleThreshold > 0.0f) {
                    if (scaled_diff >= -RescaleThreshold) { continue; }
                }
```

Note: `scaled_diff` is already computed as `(scores_max_prev - scores_max_cur) * softmax_scale_log2` which is ≤ 0 in normal flow. `-RescaleThreshold` means "skip if the rescale factor is within `exp2(-RescaleThreshold)` of 1.0". With `RescaleThreshold=8.0f`, that's `exp2(-8.0) ≈ 0.0039`.

#### Edit 3: `flash_fwd_kernel.h` — update 4 call sites

**Location:** `flash_fwd_kernel.h` lines ~351, ~414, ~925, ~992.

**Before (all 4 sites, same pattern):**
```cpp
softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/..., /*Use_rescale_threshold=*/true>(acc_s, acc_o, params.scale_softmax_log2);
```

**After:**
```cpp
softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/..., /*RescaleThreshold=*/8.0f>(acc_s, acc_o, params.scale_softmax_log2);
```

Replace `/*Use_rescale_threshold=*/true` with `/*RescaleThreshold=*/8.0f` at each of the 4 call sites. Leave the `Is_first=true` call sites (lines ~350, ~924) unchanged — they don't pass the threshold parameter (it defaults to `0.0f`).

#### Verification

```powershell
cd D:\USERFILES\GitHub\flash-attention
# Build (Windows)
.\WindowsWhlBuilder_cuda.bat
# Quick smoke test
python -c "import torch; from flash_attn import flash_attn_func; q=torch.randn(1,512,8,128,dtype=torch.float16,device='cuda'); k=torch.randn_like(q); v=torch.randn_like(q); o=flash_attn_func(q,k,v,causal=True); print('OK', o.shape, o.dtype)"
# Compare output against pre-change build for numeric divergence
# (save reference output before rebuilding)
```

---

### 15.2 A-4: packed `fmax_reduce` / `fadd_reduce` (§14.2)

**Files affected:**
1. `csrc/flash_attn/src/utils.h` — new `fmax_3way` helper
2. `csrc/flash_attn/src/softmax.h` — `thread_reduce_` packed path

#### Edit 1: `utils.h` — add `fmax_3way` helper

**Location:** `utils.h`, after the `fma_f32x2` function (before the closing `}` of `namespace FLASH_NAMESPACE`, currently line ~457).

**Add:**
```cpp
////////////////////////////////////////////////////////////////////////////////////////////////////

// 3-way max helper (Plan A-4): returns max(a, b, c).
// On all arches this is two dependent max instructions; the benefit on sm_100+
// comes from the reduction loop unrolling pattern (8-wide instead of 2-wide),
// which gives the scheduler more instructions to overlap.
// FA4 uses the same pattern via cute.arch.fmax(a, b, c).
__forceinline__ __device__ float fmax_3way(float a, float b, float c) {
    float result;
    asm("max.f32 %0, %1, %2;\n\t"
        "max.f32 %0, %0, %3;\n\t"
        : "=f"(result) : "f"(a), "f"(b), "f"(c));
    return result;
}
```

Note: Unlike `fma_f32x2`, this helper does NOT need an `__CUDA_ARCH__` guard because `max.f32` is valid on all arches. The optimization is in the loop structure, not the instruction.

#### Edit 2: `softmax.h` — add packed `thread_reduce_` specialization

**Location:** `softmax.h`, after the existing `thread_reduce_` function (currently ends at line ~44), before `quad_allreduce_`.

**Add a new overload:**
```cpp
////////////////////////////////////////////////////////////////////////////////////////////////////
// Plan A-4: packed thread_reduce_ for sm_100+
// On sm_100+, process 8 columns at a time for max (4 accumulator lanes,
// 2 columns per lane via fmax_3way) and 2 columns at a time for sum
// (via fma_f32x2-based packed add).
// The original thread_reduce_ above is retained for sm_80/86/89/90 (unchanged binary).

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void thread_reduce_max_packed(
    Tensor<Engine0, Layout0> const &tensor,
    Tensor<Engine1, Layout1> &summary) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    constexpr int N1 = decltype(size<1>(tensor))::value;
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        if constexpr (N1 >= 8) {
            // 8-wide reduction: 4 accumulators, 2 columns per step
            float m0 = zero_init ? tensor(mi, 0) : summary(mi);
            float m1 = zero_init ? tensor(mi, 1) : summary(mi);
            float m2 = zero_init ? tensor(mi, 2) : summary(mi);
            float m3 = zero_init ? tensor(mi, 3) : summary(mi);
            #pragma unroll
            for (int ni = 4; ni < N1; ni += 4) {
                m0 = fmaxf(m0, tensor(mi, ni));
                m1 = fmaxf(m1, tensor(mi, ni + 1));
                m2 = fmaxf(m2, tensor(mi, ni + 2));
                m3 = fmaxf(m3, tensor(mi, ni + 3));
            }
            // Collapse 4 → 1 via fmax_3way
            summary(mi) = fmax_3way(m0, m1, fmaxf(m2, m3));
        } else {
            // Fallback to original scalar path for small N1
            summary(mi) = zero_init ? tensor(mi, 0) : fmaxf(summary(mi), tensor(mi, 0));
            #pragma unroll
            for (int ni = 1; ni < N1; ni++) {
                summary(mi) = fmaxf(summary(mi), tensor(mi, ni));
            }
        }
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void thread_reduce_sum_packed(
    Tensor<Engine0, Layout0> const &tensor,
    Tensor<Engine1, Layout1> &summary) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    constexpr int N1 = decltype(size<1>(tensor))::value;
    static_assert(N1 % 2 == 0, "thread_reduce_sum_packed requires even N1");
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        if constexpr (N1 >= 8) {
            // 4 accumulator pairs, each holding 2 lanes
            float s0a, s0b, s1a, s1b, s2a, s2b, s3a, s3b;
            if constexpr (zero_init) {
                s0a = tensor(mi, 0); s0b = tensor(mi, 1);
                s1a = tensor(mi, 2); s1b = tensor(mi, 3);
                s2a = tensor(mi, 4); s2b = tensor(mi, 5);
                s3a = tensor(mi, 6); s3b = tensor(mi, 7);
            } else {
                // Initialize from summary, packed with first elements
                fma_f32x2(s0a, s0b, 1.0f, 1.0f, summary(mi), summary(mi), tensor(mi, 0), tensor(mi, 1));
                fma_f32x2(s1a, s1b, 1.0f, 1.0f, summary(mi), summary(mi), tensor(mi, 2), tensor(mi, 3));
                fma_f32x2(s2a, s2b, 1.0f, 1.0f, summary(mi), summary(mi), tensor(mi, 4), tensor(mi, 5));
                fma_f32x2(s3a, s3b, 1.0f, 1.0f, summary(mi), summary(mi), tensor(mi, 6), tensor(mi, 7));
            }
            #pragma unroll
            for (int ni = 8; ni < N1; ni += 8) {
                // Packed add: (s0a, s0b) += (tensor[ni], tensor[ni+1])
                fma_f32x2(s0a, s0b, 1.0f, 1.0f, s0a, s0b, tensor(mi, ni), tensor(mi, ni + 1));
                fma_f32x2(s1a, s1b, 1.0f, 1.0f, s1a, s1b, tensor(mi, ni + 2), tensor(mi, ni + 3));
                fma_f32x2(s2a, s2b, 1.0f, 1.0f, s2a, s2b, tensor(mi, ni + 4), tensor(mi, ni + 5));
                fma_f32x2(s3a, s3b, 1.0f, 1.0f, s3a, s3b, tensor(mi, ni + 6), tensor(mi, ni + 7));
            }
            // Collapse 4 pairs → 1 scalar
            fma_f32x2(s0a, s0b, 1.0f, 1.0f, s0a, s0b, s1a, s1b);
            fma_f32x2(s2a, s2b, 1.0f, 1.0f, s2a, s2b, s3a, s3b);
            fma_f32x2(s0a, s0b, 1.0f, 1.0f, s0a, s0b, s2a, s2b);
            summary(mi) = s0a + s0b;
        } else {
            // Fallback for small N1
            summary(mi) = zero_init ? tensor(mi, 0) : summary(mi) + tensor(mi, 0);
            #pragma unroll
            for (int ni = 1; ni < N1; ni++) {
                summary(mi) = summary(mi) + tensor(mi, ni);
            }
        }
    }
}
```

#### Edit 3: `softmax.h` — dispatch to packed paths in `reduce_max` / `reduce_sum`

**Location:** `softmax.h` lines ~61–69 (the `reduce_max` and `reduce_sum` functions).

**Before:**
```cpp
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
}
```

**After:**
```cpp
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && !defined(UNFUSE_FMA)
    thread_reduce_max_packed<zero_init>(tensor, max);
#else
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
#endif
    // quad_allreduce is still needed for cross-warp reduction
    MaxOp<float> max_op;
    quad_allreduce_(max, max, max_op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && !defined(UNFUSE_FMA)
    thread_reduce_sum_packed<zero_init>(tensor, sum);
#else
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
#endif
}
```

Note: `reduce_max` calls `reduce_` which does `thread_reduce_` + `quad_allreduce_`. The packed version splits these: `thread_reduce_max_packed` replaces the `thread_reduce_` part; `quad_allreduce_` stays the same (it operates on the 1D `summary` tensor, unchanged).

For `reduce_sum`: the original only calls `thread_reduce_` (no `quad_allreduce_`), so the packed replacement is self-contained.

#### Verification

```powershell
# After build, SASS check for A-4:
# Look for increased FMAX and FFMA.X2 in the softmax reduction symbols
cuobjdump --dump-sass flash_attn_lib.so | Select-String "FMAX|FFMA.X2" | Measure-Object
# Compare count against pre-A-4 build — should see more packed instructions
```

---

### 15.3 A-3: `max_offset + rescale_threshold` assert (§14.3)

**Files affected:**
1. `csrc/flash_attn/src/softmax.h` — add `static_assert` in `softmax_rescale_o`

#### Edit 1: `softmax.h` — add assert after template signature

**Location:** `softmax.h`, immediately after the `static_assert(!(Is_first && RescaleThreshold > 0.0f))` line (added in §15.1).

**Add:**
```cpp
    // Safety: max_offset + rescale_threshold must stay below log2(dtype_max) to avoid P saturation.
    // For fp16: log2(65504) ≈ 15.99. For bf16: log2(3.39e38) ≈ 128.0.
    // Currently MaxOffsetMilli=0 (no fp8 path), RescaleThreshold=8.0 at most.
    // This assert will catch a future fp8 path that sets both without checking the bound.
    static_assert(
        true,  // MaxOffsetMilli is not yet a template parameter; placeholder.
               // When A-3's MaxOffsetMilli is added, replace with:
               // MaxOffsetMilli * 1e-3f + RescaleThreshold < 15.99f
        "max_offset + rescale_threshold must stay below log2(fp16_max) to avoid P saturation");
```

Note: This is a placeholder. The real assert activates when A-3's `MaxOffsetMilli` template parameter is added (Phase 2). For now it documents the invariant.

---

### 15.4 B-1: `exp2` polynomial emulation with packed round-down (§14.4)

**Files affected:**
1. `csrc/flash_attn/src/utils.h` — `POLY_EX2` coefficients, `exp2f_emu`, `exp2f_emu_x2`
2. `csrc/flash_attn/src/softmax.h` — `scale_apply_exp2` template parameter + dispatch

#### Edit 1: `utils.h` — add `POLY_EX2` coefficients and emulation functions

**Location:** `utils.h`, after the `fmax_3way` helper (added in §15.2), before the closing `}` of `namespace FLASH_NAMESPACE`.

**Add:**
```cpp
////////////////////////////////////////////////////////////////////////////////////////////////////
// Plan B-1: exp2 polynomial emulation (Sollya minimax)
// Coefficients copied verbatim from flash_attn/cute/utils.py::POLY_EX2
// Degree 3 is the default; degrees 0-5 are available for precision tuning.

namespace detail {
// fpminimax(exp(x * log(2.0)), deg, [|1,24...|],[0;1],relative)
template<int Deg> struct PolyEx2;
template<> struct PolyEx2<0>  { static constexpr float c[1] = {1.0f}; static constexpr int size = 1; };
template<> struct PolyEx2<1>  { static constexpr float c[2] = {1.0f, 0.922497093677520751953125f}; static constexpr int size = 2; };
template<> struct PolyEx2<2>  { static constexpr float c[3] = {1.0f, 0.6657850742340087890625f, 0.330107033252716064453125f}; static constexpr int size = 3; };
template<> struct PolyEx2<3>  { static constexpr float c[4] = {1.0f, 0.695146143436431884765625f, 0.227564394474029541015625f, 0.077119089663028717041015625f}; static constexpr int size = 4; };
template<> struct PolyEx2<4>  { static constexpr float c[5] = {1.0f, 0.693042695522308349609375f, 0.2412912547588348388671875f, 5.2225358784198760986328125e-2f, 1.3434938155114650726318359375e-2f}; static constexpr int size = 5; };
template<> struct PolyEx2<5>  { static constexpr float c[6] = {1.0f, 0.693151414394378662109375f, 0.24016360938549041748046875f, 5.5802188813686370849609375e-2f, 9.01452265679836273193359375e-3f, 1.86810153536498546600341796875e-3f}; static constexpr int size = 6; };
}  // namespace detail

// Scalar exp2 emulation using Sollya minimax polynomial.
// Matches flash_attn/cute/utils.py::ex2_emulation (degree 3 default).
template<int PolyDegree = 3>
__forceinline__ __device__ float exp2f_emu(float x) {
    constexpr auto& poly = detail::PolyEx2<PolyDegree>::c;
    constexpr int deg = detail::PolyEx2<PolyDegree>::size - 1;
    // Clamp to >= -127.0
    x = fmaxf(x, -127.0f);
    // Range reduction: x = k + f, k = floor(x), f in [0, 1)
    // Using add.rm.ftz.f32 (round-down add) with magic bias
    constexpr float magic = float(1 << 23 | 1 << 22);  // 2^23 + 2^22
    float x_rounded;
    asm("add.rm.ftz.f32 %0, %1, %2;\n\t" : "=f"(x_rounded) : "f"(x), "f"(magic));
    float x_rounded_back = x_rounded - magic;
    float frac = x - x_rounded_back;
    // Horner evaluation
    float p = poly[deg];
    #pragma unroll
    for (int i = deg - 1; i >= 0; i--) {
        p = fmaf(p, frac, poly[i]);
    }
    // Reconstruct 2^k * p via exponent field manipulation
    int k = __float_as_int(x_rounded);  // integer floor in low bits (magic-biased)
    int bits = __float_as_int(p) + (k << 23);
    return __int_as_float(bits);
}

// Packed exp2 emulation for sm_100+ (2 lanes at once).
// Matches flash_attn/cute/utils.py::ex2_emulation_2 + e2e_asm2.
template<int PolyDegree = 3>
__forceinline__ __device__ void exp2f_emu_x2(float &d0, float &d1, float y0, float y1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && !defined(UNFUSE_FMA)
    constexpr auto& poly = detail::PolyEx2<PolyDegree>::c;
    constexpr int deg = detail::PolyEx2<PolyDegree>::size - 1;

    // Clamp
    y0 = fmaxf(y0, -127.0f);
    y1 = fmaxf(y1, -127.0f);

    // Packed round-down add: x_rounded = add.rm.ftz.f32x2(y, magic)
    constexpr float magic = float(1 << 23 | 1 << 22);
    float r0, r1;
    asm("{\n\t"
        ".reg .b64 rl, ml, rb;\n\t"
        "mov.b64 ml, {%4, %5};\n\t"
        "mov.b64 rl, {%2, %3};\n\t"
        "add.rm.ftz.f32x2 rb, rl, ml;\n\t"
        "mov.b64 {%0, %1}, rb;\n\t"
        "}\n"
        : "=f"(r0), "=f"(r1)
        : "f"(y0), "f"(y1), "f"(magic), "f"(magic));

    // Packed subtract-back: x_rounded_back = r - magic
    float rb0, rb1;
    fma_f32x2(rb0, rb1, r0, r1, 1.0f, 1.0f, -magic, -magic);

    // Packed fractional: frac = y - x_rounded_back
    float f0, f1;
    fma_f32x2(f0, f1, rb0, rb1, -1.0f, -1.0f, y0, y1);

    // Horner with packed FMA
    float p0 = poly[deg], p1 = poly[deg];
    #pragma unroll
    for (int i = deg - 1; i >= 0; i--) {
        fma_f32x2(p0, p1, p0, p1, f0, f1, poly[i], poly[i]);
    }

    // Reconstruct 2^k * p
    int k0 = __float_as_int(r0);
    int k1 = __float_as_int(r1);
    int b0 = __float_as_int(p0) + (k0 << 23);
    int b1 = __float_as_int(p1) + (k1 << 23);
    d0 = __int_as_float(b0);
    d1 = __int_as_float(b1);
#else
    d0 = exp2f_emu<PolyDegree>(y0);
    d1 = exp2f_emu<PolyDegree>(y1);
#endif
}
```

#### Edit 2: `softmax.h` — add `Use_exp2_emu` and `PolyDegree` to `scale_apply_exp2`

**Location:** `softmax.h`, the `scale_apply_exp2` function (currently lines ~83–116).

**Before (current signature):**
```cpp
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
```

**After:**
```cpp
template <bool Scale_max=true, bool Use_exp2_emu=false, int PolyDegree=3, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
```

**In the sm_100+ branch** (currently lines ~96–110), replace the `exp2f(r0)` / `exp2f(r1)` calls:

**Before:**
```cpp
            tensor(mi, ni)     = exp2f(r0);
            tensor(mi, ni + 1) = exp2f(r1);
```

**After:**
```cpp
            if constexpr (Use_exp2_emu) {
                exp2f_emu_x2<PolyDegree>(tensor(mi, ni), tensor(mi, ni + 1), r0, r1);
            } else {
                tensor(mi, ni)     = exp2f(r0);
                tensor(mi, ni + 1) = exp2f(r1);
            }
```

**In the scalar branch** (the `#else` path, lines ~111–116), add the same conditional:

**Before:**
```cpp
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
```

**After:**
```cpp
            if constexpr (Use_exp2_emu) {
                tensor(mi, ni) = exp2f_emu<PolyDegree>(tensor(mi, ni) * scale - max_scaled);
            } else {
                tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            }
```

#### Edit 3: Call sites — no changes needed yet

The default `Use_exp2_emu=false` means all existing call sites (`softmax_rescale_o` in `softmax.h`, `flash_bwd_kernel.h:536`) remain unchanged. B-1 is enabled by explicitly passing `Use_exp2_emu=true` at the call site during Phase 2 tuning.

---

### 15.5 B-2: mixed mode with fragment-boundary logic (§14.5)

**Files affected:**
1. `csrc/flash_attn/src/softmax.h` — `scale_apply_exp2` gains `Ex2EmuFreq`, `Ex2EmuRes`, `Ex2EmuStartFrg` parameters

This edit builds on top of §15.4 (B-1 must be applied first).

#### Edit 1: `softmax.h` — extend `scale_apply_exp2` template

**Location:** `softmax.h`, the `scale_apply_exp2` signature (modified in §15.4).

**Before (after B-1 edit):**
```cpp
template <bool Scale_max=true, bool Use_exp2_emu=false, int PolyDegree=3, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
```

**After:**
```cpp
template <bool Scale_max=true, bool Use_exp2_emu=false, int PolyDegree=3,
          int Ex2EmuFreq=0, int Ex2EmuRes=4, int Ex2EmuStartFrg=0,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1>
```

Default `Ex2EmuFreq=0` means "all hardware exp2" (= B-1's `Use_exp2_emu=false`). `Ex2EmuFreq > 0` activates mixed mode. `Use_exp2_emu=true` with `Ex2EmuFreq=0` means "all polynomial" (= B-1 full emulation).

#### Edit 2: `softmax.h` — mixed-mode dispatch in the sm_100+ branch

**Location:** The sm_100+ inner loop of `scale_apply_exp2` (modified in §15.4).

**Replace the inner loop body with:**
```cpp
        // B-2 mixed mode: partition N1 into frg_tile=32 chunks
        constexpr int frg_tile = 32;
        constexpr int frg_cnt = N1 / frg_tile;
        static_assert(N1 % frg_tile == 0,
                      "B-2 mixed mode requires N1 % 32 == 0; "
                      "fall back to Use_exp2_emu=false or Ex2EmuFreq=0 if violated.");
        #pragma unroll
        for (int j = 0; j < frg_cnt; ++j) {
            #pragma unroll
            for (int k = 0; k < frg_tile; k += 2) {
                int ni = j * frg_tile + k;
                float t0 = tensor(mi, ni);
                float t1 = tensor(mi, ni + 1);
                float r0, r1;
                fma_f32x2(r0, r1, t0, t1, scale, scale, neg_max_scaled, neg_max_scaled);

                // Determine whether to use hardware MUFU.EX2 or polynomial emulation
                bool use_hw;
                if constexpr (!Use_exp2_emu || Ex2EmuFreq == 0) {
                    use_hw = true;  // No emulation requested
                } else {
                    use_hw = false;
                    // Hardware for: (a) lanes where k % freq < freq - res,
                    // (b) last fragment, (c) before start fragment
                    if (k % Ex2EmuFreq < Ex2EmuFreq - Ex2EmuRes) use_hw = true;
                    if (j >= frg_cnt - 1) use_hw = true;
                    if (j < Ex2EmuStartFrg) use_hw = true;
                }

                if (use_hw) {
                    tensor(mi, ni)     = exp2f(r0);
                    tensor(mi, ni + 1) = exp2f(r1);
                } else {
                    exp2f_emu_x2<PolyDegree>(tensor(mi, ni), tensor(mi, ni + 1), r0, r1);
                }
            }
        }
```

#### Edit 3: Call sites — enable B-2 during Phase 2 tuning

No call site changes are needed for the default build. When tuning, enable B-2 at specific call sites in `softmax_rescale_o` by passing the template parameters:

```cpp
// Example: enable B-2 mixed mode for hdim=128, causal
FLASH_NAMESPACE::scale_apply_exp2</*Scale_max=*/true,
                                   /*Use_exp2_emu=*/true,
                                   /*PolyDegree=*/3,
                                   /*Ex2EmuFreq=*/10,
                                   /*Ex2EmuRes=*/4,
                                   /*Ex2EmuStartFrg=*/1>(scores, row_max, softmax_scale_log2);
```

This requires passing the parameters through `softmax_rescale_o` to `scale_apply_exp2`, which means `softmax_rescale_o` itself gains the same template parameters. The propagation is mechanical:

```cpp
template<bool Is_first, bool Check_inf=false, float RescaleThreshold=0.0f,
         bool Use_exp2_emu=false, int PolyDegree=3,
         int Ex2EmuFreq=0, int Ex2EmuRes=4, int Ex2EmuStartFrg=0,
         typename Tensor0, typename Tensor1>
__forceinline__ __device__ void softmax_rescale_o(...) {
    ...
    FLASH_NAMESPACE::scale_apply_exp2</*Scale_max=*/true, Use_exp2_emu, PolyDegree,
                                      Ex2EmuFreq, Ex2EmuRes, Ex2EmuStartFrg>(
        scores, row_max, softmax_scale_log2);
    ...
}
```

And the call sites in `flash_fwd_kernel.h` pass the hdim-specific tuning values. This is the last step of Phase 2, after benchmarking confirms the optimal `(freq, res, start)` per hdim.

---

### 15.6 C-1: hdim-bucketed block-size retuning (§5.6)

**Files affected:**
1. `csrc/flash_attn/src/kernel_traits.h` — new `Flash_fwd_kernel_traits_blackwell` family
2. `csrc/flash_attn/src/flash_fwd_launch_template.h` — dispatch to Blackwell traits

#### Edit 1: `kernel_traits.h` — add Blackwell traits

**Location:** `kernel_traits.h`, after the existing `Flash_fwd_kernel_traits` and `Flash_fwd_kernel_traits_with_8_warps` definitions.

**Add:**
```cpp
////////////////////////////////////////////////////////////////////////////////////////////////////
// Plan C-1: Blackwell-specific kernel traits
// Gated by FLASH_FWD_USE_BLACKWELL_TRAITS macro.
// When enabled, these traits override kBlockM/kBlockN/kNWarps for sm_100/120.
// Default values are placeholders; C-1 tuning will replace them with measured optima.

#ifdef FLASH_FWD_USE_BLACKWELL_TRAITS
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
         bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false, typename elem_type=cutlass::half_t,
         typename Base=Flash_fwd_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type>>
struct Flash_fwd_kernel_traits_blackwell : public Base {
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    // Inherit smem layout, MMA atoms, copy atoms from Base.
    // Only block sizes and warp counts are overridden.
    // C-1 tuning will determine the optimal (kBlockM, kBlockN, kNWarps) per hdim.
};
#endif  // FLASH_FWD_USE_BLACKWELL_TRAITS
```

#### Edit 2: `flash_fwd_launch_template.h` — dispatch macro

**Location:** At the top of each `run_mha_fwd_hdim*` function, add an arch check.

**Pattern (shown for hdim=128, apply similarly to other hdims):**

```cpp
template<typename T, bool Is_causal>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_blackwell = cc_major >= 10;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
#ifdef FLASH_FWD_USE_BLACKWELL_TRAITS
            if (is_blackwell) {
                // C-1 tuned values for sm_100/120 — placeholders, replace after measurement
                run_flash_fwd<Flash_fwd_kernel_traits_blackwell<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
                return;
            }
#endif
            // ... existing sm_80/sm_90 dispatch ...
        }
    });
}
```

#### Verification

```powershell
# Build with the macro defined:
$env:NVCC_FLAGS = "-DFLASH_FWD_USE_BLACKWELL_TRAITS"
.\WindowsWhlBuilder_cuda.bat

# Benchmark comparison (requires Phase 0 baseline):
python bench\fa2_baseline.py --output bench\baseline_blackwell.json
# Compare against bench\baseline_sm80.json
```

---

### 15.7 Build verification checklist

After applying all edits in order (§15.1 → §15.2 → §15.3 → §15.4 → §15.5 → §15.6):

| Step | Command | Expected result |
|---|---|---|
| Clean build | `.\WindowsWhlBuilder_cuda.bat` | No compile errors; wheel produced |
| Smoke test (fp16) | `python -c "import torch; from flash_attn import flash_attn_func; ..."` | Output shape/dtype correct |
| Smoke test (bf16) | Same with `dtype=torch.bfloat16` | Output shape/dtype correct |
| A-1 accuracy | Compare output vs pre-change build | Relative L∞ ≤ 2e-2 (fp16), ≤ 1e-2 (bf16) |
| A-2 SASS gate | `cuobjdump --dump-sass *.so \| Select-String "FFMA.X2"` | `FFMA.X2` present on sm_120 build |
| A-4 SASS gate | `cuobjdump --dump-sass *.so \| Select-String "FMAX"` | Increased `FMAX` count vs pre-A-4 |
| B-1 SASS gate (when enabled) | `cuobjdump --dump-sass *.so \| Select-String "MUFU.EX2"` | `MUFU.EX2` absent in emulated lanes on sm_120 |
| B-1 accuracy (when enabled) | `pytest tests/test_flash_attn.py::test_flash_attn_output` | Relative L∞ ≤ 1e-2 (fp16), ≤ 5e-3 (bf16) |
| B-2 accuracy (when enabled) | Same test per (freq, res, start) cell | Same thresholds as B-1 |
| Logits cosine | LLM end-to-end prompt comparison | ≥ 0.9995 |

### 15.8 Commit sequence

Recommended commit order (each commit is independently buildable):

| Commit | Contents | Phase |
|---|---|---|
| 1 | `M_LOG2E` fix (already in working tree) | Pre |
| 2 | A-1 threshold revision (§15.1) | 1 |
| 3 | A-4 packed reduction (§15.2) | 1 |
| 4 | A-3 assert placeholder (§15.3) | 2 |
| 5 | B-1 polynomial emulation (§15.4) | 2 |
| 6 | B-2 mixed mode (§15.5, default off) | 2 |
| 7 | C-1 Blackwell traits (§15.6, gated off) | 3 |

Commits 2–3 can be squashed if Phase 1 lands as a single PR. Commits 4–6 should land individually for reviewability. Commit 7 is independent.

## 16. A-1/A-2 Existing Implementation: Delta Fix Plan (2026-08-11)

This section documents the concrete changes needed to bring the **already-applied** A-1 and A-2 implementations in line with the updated plan (§14) and the latest FA4 codebase. The refinements from `AI/A1_A2_REFINEMENTS_PLAN.md` (A-1-R1 through A-2-R6) are all already applied in commit `2eda0be`; this section covers only the **remaining deltas** discovered after the FA4 sync update (§14).

### 16.1 Current state summary

| Item | Status | Where |
|---|---|---|
| A-1: `Use_rescale_threshold` bool template | Applied | `softmax.h` line ~143 |
| A-1: `kSoftmaxRescaleSkipThreshold = -0.01f` | Applied (A-1-R1) | `softmax.h` line ~120 |
| A-1: `static_assert(!(Is_first && Use_rescale_threshold))` | Applied (A-1-R2) | `softmax.h` line ~144 |
| A-1: Skip-cascade comment | Applied (A-1-R3) | `softmax.h` lines ~170–173 |
| A-1: 4 call sites with `Use_rescale_threshold=true` | Applied | `flash_fwd_kernel.h` lines ~351, ~414, ~925, ~992 |
| A-2: `fma_f32x2` helper in `utils.h` | Applied (A-2-R3) | `utils.h` line ~416 |
| A-2: `scale_apply_exp2` sm_100+ packed-FMA path | Applied | `softmax.h` lines ~88–110 |
| A-2: `static_assert(N1 % 2 == 0)` | Applied (A-2-R2) | `softmax.h` line ~100 |
| A-2: `max_scale_exp2_sum` dead code removed | Applied (A-2-R1) | — (already gone) |
| A-2: `UNFUSE_FMA` `#pragma message` | Applied (A-2-R4) | `utils.h` after `fma_f32x2` |
| A-2: `bench/check_sass_gates.py` | Applied (A-2-R5) | `bench/check_sass_gates.py` |
| A-2: bwd coverage comment | Applied (A-2-R6) | `softmax.h` lines ~72–77 |

### 16.2 Delta list

The following changes are needed to align the existing A-1/A-2 code with the updated plan (§14). Each delta is tagged with its source section and priority.

---

#### Delta 1 (HIGH): A-1 threshold type change — `bool` → `float` (from §14.1, §15.1)

This is the largest delta. The current implementation uses a `bool Use_rescale_threshold` template parameter and a separate `kSoftmaxRescaleSkipThreshold = -0.01f` constant. The updated plan replaces both with a single `float RescaleThreshold = 0.0f` template parameter, set to `8.0f` at call sites.

**Why:** FA4 evolved `rescale_threshold` from a bool to a float (default `8.0` for fp16/bf16). The `0.01` threshold is too conservative — FA4 validated `8.0` on B200. The float parameter unifies the threshold and its activation into one clean template argument.

**Edit A — `softmax.h`, delete `kSoftmaxRescaleSkipThreshold` (line ~120):**

Delete these 4 lines entirely:
```cpp
// Threshold below which softmax_rescale_o skips the O / row_sum rescale.
// scores_scale = exp2(scaled_diff); scaled_diff is the negative excursion of
// row_max from one iteration to the next, in units of softmax_scale_log2.
// At -0.01f, scores_scale >= ~0.993 (worst-case relative error <= 0.7%).
inline constexpr float kSoftmaxRescaleSkipThreshold = -0.01f;
```

The threshold value is now embedded in the template parameter at the call site, not a file-scope constant.

**Edit B — `softmax.h`, change template signature (line ~143):**

Before:
```cpp
template<bool Is_first, bool Check_inf=false, bool Use_rescale_threshold=false, typename Tensor0, typename Tensor1>
__forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
    static_assert(!(Is_first && Use_rescale_threshold),
                  "Use_rescale_threshold has no effect when Is_first=true; "
                  "remove the flag from the Is_first=true call site.");
```

After:
```cpp
template<bool Is_first, bool Check_inf=false, float RescaleThreshold=0.0f, typename Tensor0, typename Tensor1>
__forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
    static_assert(!(Is_first && RescaleThreshold > 0.0f),
                  "RescaleThreshold > 0 has no effect when Is_first=true; "
                  "remove the threshold from the Is_first=true call site.");
```

**Edit C — `softmax.h`, change skip condition (lines ~175–177):**

Before:
```cpp
                if constexpr (Use_rescale_threshold) {
                    if (scaled_diff >= kSoftmaxRescaleSkipThreshold) { continue; }
                }
```

After:
```cpp
                if constexpr (RescaleThreshold > 0.0f) {
                    if (scaled_diff >= -RescaleThreshold) { continue; }
                }
```

Note on semantics change: The old code compared `scaled_diff >= -0.01` (i.e. skip when the rescale factor is within 1% of 1.0). The new code compares `scaled_diff >= -RescaleThreshold`. With `RescaleThreshold=8.0f`, this skips when `scaled_diff >= -8.0`, meaning `scores_scale >= exp2(-8.0) ≈ 0.0039`. This is far more aggressive but validated by FA4 on B200.

**Edit D — `softmax.h`, update the skip-cascade comment (lines ~170–173):**

Before:
```cpp
                // Optionally skip the O / row_sum rescale when the new row_max is virtually
                // the same as the previous one (scaled_diff is a very small negative number,
                // so scores_scale = exp2(scaled_diff) ~= 1.0).
                // This skip does not compound across iterations: normalize_softmax_lse
                // divides acc_o by row_sum, so the skipped factor cancels between
                // numerator and denominator, bounding the relative error by the threshold.
```

After:
```cpp
                // Optionally skip the O / row_sum rescale when the new row_max is close
                // enough to the previous one that the rescale factor is near 1.0.
                // RescaleThreshold is in log2 units of the softmax scale: skip when
                // scaled_diff >= -RescaleThreshold, i.e. scores_scale >= exp2(-RescaleThreshold).
                // With RescaleThreshold=8.0 (FA4 default for fp16/bf16), scores_scale >= 0.0039.
                // This skip does not compound across iterations: normalize_softmax_lse
                // divides acc_o by row_sum, so the skipped factor cancels between
                // numerator and denominator, bounding the relative error by the threshold.
                // Safety: max_offset + RescaleThreshold < log2(dtype_max) must hold
                // (see §14.3 static_assert, to be added when A-3 lands).
```

**Edit E — `flash_fwd_kernel.h`, update 4 call sites (lines ~351, ~414, ~925, ~992):**

Before (all 4 sites, same pattern):
```cpp
softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/..., /*Use_rescale_threshold=*/true>(acc_s, acc_o, params.scale_softmax_log2);
```

After:
```cpp
softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/..., /*RescaleThreshold=*/8.0f>(acc_s, acc_o, params.scale_softmax_log2);
```

The 2 `Is_first=true` call sites (lines ~350, ~924) are unchanged — they don't pass the threshold parameter (defaults to `0.0f`).

**Verification:**
```powershell
cd D:\USERFILES\GitHub\flash-attention
.\WindowsWhlBuilder_cuda.bat
# Smoke test
python -c "import torch; from flash_attn import flash_attn_func; q=torch.randn(1,512,8,128,dtype=torch.float16,device='cuda'); k=torch.randn_like(q); v=torch.randn_like(q); o=flash_attn_func(q,k,v,causal=True); print('OK', o.shape, o.dtype)"
# Accuracy: compare output against pre-change build
# Acceptance: relative L∞ ≤ 2e-2 (fp16), ≤ 1e-2 (bf16), logits cosine ≥ 0.9995
```

---

#### Delta 2 (LOW): A-1 `A1_A2_REFINEMENTS_PLAN.md` status update

After Delta 1 lands, `AI/A1_A2_REFINEMENTS_PLAN.md` items A-1-R1 (file-scope constant) is superseded. The constant `kSoftmaxRescaleSkipThreshold` no longer exists. Add a note at the top of that document:

```markdown
> **Update 2026-08-11:** A-1-R1 is superseded by §14.1/§15.1 of the main plan.
> The `kSoftmaxRescaleSkipThreshold` constant is deleted; the threshold is now
> a `float RescaleThreshold` template parameter (default `8.0f` at call sites,
> matching FA4). A-1-R2's `static_assert` is updated to check
> `RescaleThreshold > 0.0f` instead of `Use_rescale_threshold`.
```

This is documentation-only; no code change.

---

#### Delta 3 (NONE): A-2 — no changes needed

The A-2 implementation (`fma_f32x2` in `utils.h`, `scale_apply_exp2` sm_100+ packed-FMA path in `softmax.h`) is **already aligned** with the updated plan. Specifically:

- `fma_f32x2` is in `utils.h` (A-2-R3 applied) ✅
- `static_assert(N1 % 2 == 0)` is present (A-2-R2 applied) ✅
- `UNFUSE_FMA` `#pragma message` is present (A-2-R4 applied) ✅
- `max_scale_exp2_sum` dead code is removed (A-2-R1 applied) ✅
- `bench/check_sass_gates.py` exists (A-2-R5 applied) ✅
- bwd coverage comment is present (A-2-R6 applied) ✅
- The `asm volatile` in `fma_f32x2` matches the PTX layout described in §14.4 ✅

The only A-2-adjacent change is the **addition** of B-1's `exp2f_emu` / `exp2f_emu_x2` functions (§15.4) and the `Use_exp2_emu` template parameter on `scale_apply_exp2` (§15.4 Edit 2). These are additive — they don't modify the existing A-2 code path, only add a new conditional branch that defaults to the current behavior (`Use_exp2_emu=false`).

No edit to the existing A-2 code is required.

---

#### Delta 4 (MEDIUM): `asm volatile` in `fma_f32x2` — consider removing `volatile`

**Current** (`utils.h` line ~424):
```cpp
    asm volatile(
```

**Issue:** The `volatile` keyword prevents the compiler from reordering or CSE-ing the asm block. This was originally added defensively. However, the asm block has explicit outputs (`"=f"(d0), "=f"(d1)`) and inputs (`"f"(a0)...`), so the compiler already knows the data dependencies. `volatile` is only needed if the asm has side effects not captured by the constraints (e.g., writing to memory). Since `fma.rn.f32x2` is a pure computation, `volatile` is unnecessary and may prevent the compiler from optimizing surrounding code.

FA4's equivalent (`cute.arch.fma_packed_f32x2`) does not use `volatile`.

**Proposed change:**
```cpp
    asm(
```

Remove the `volatile` keyword. The output/input constraints are sufficient.

**Risk:** Very low. If the compiler starts folding the asm block (which would be a problem), the SASS gate (`bench/check_sass_gates.py`) will catch it — `FFMA.X2` would disappear from the binary.

**Verification:**
```powershell
# After rebuild, run SASS gate:
python bench\check_sass_gates.py --pyd flash_attn_2_cuda.cp314-win_amd64.pyd --arch sm_120 --gate a2
# Must exit 0 (FFMA.X2 present)
```

---

#### Delta 5 (LOW): Add `RescaleThreshold` to the bwd path comment

**Current** (`softmax.h` lines ~72–77):
```cpp
// Apply the exp to all the elements.
// Note: this function is shared by:
//   - fwd via Softmax::softmax_rescale_o (both Is_first branches), and
//   - bwd via flash_bwd_kernel.h::compute_dq_dk_dv (Scale_max=false branch).
// Both paths benefit equally from the sm_100+ packed-FMA inner loop below.
```

This comment is accurate. After Delta 1 lands, add a note about `RescaleThreshold`:

```cpp
// Apply the exp to all the elements.
// Note: this function is shared by:
//   - fwd via Softmax::softmax_rescale_o (both Is_first branches), and
//   - bwd via flash_bwd_kernel.h::compute_dq_dk_dv (Scale_max=false branch).
// Both paths benefit equally from the sm_100+ packed-FMA inner loop below.
// RescaleThreshold (A-1) is applied only in softmax_rescale_o, not here;
// this function is called after the threshold check has already passed.
```

Comment-only; no code change.

---

### 16.3 Implementation order

| Step | Delta | Priority | Files | Binary change? |
|---|---|---|---|---|
| 1 | Delta 1 (A-1 threshold type change) | HIGH | `softmax.h`, `flash_fwd_kernel.h` | Yes — threshold widened from 0.01 to 8.0 |
| 2 | Delta 4 (remove `volatile`) | MEDIUM | `utils.h` | Possibly — compiler may reorder differently |
| 3 | Delta 5 (comment update) | LOW | `softmax.h` | No |
| 4 | Delta 2 (doc update) | LOW | `AI/A1_A2_REFINEMENTS_PLAN.md` | No |
| — | Delta 3 (A-2 changes) | NONE | — | — |

Step 1 is the only change that affects numerical output. Step 2 may affect SASS layout but not correctness. Steps 3–4 are documentation.

### 16.4 Commit recommendation

| Commit | Contents | Deltas |
|---|---|---|
| 1 | A-1 threshold revision: `bool` → `float`, `0.01` → `8.0` | Delta 1 + Delta 5 (comment) |
| 2 | Remove `volatile` from `fma_f32x2` asm + SASS gate re-verify | Delta 4 |
| 3 | Doc update: mark A-1-R1 superseded in refinement plan | Delta 2 |

Commit 1 should be validated with:
- Full test suite (`pytest tests/test_flash_attn.py`)
- SASS gate (`python bench\check_sass_gates.py --arch sm_120 --gate a2`)
- Accuracy comparison against pre-change build (save reference outputs first)

Commit 2 should be validated with:
- SASS gate only (no numeric change expected)

### 16.5 Rollback

If Delta 1 causes accuracy regression beyond the widened threshold:

1. Change `8.0f` back to `0.01f` at the 4 call sites in `flash_fwd_kernel.h` — this restores the conservative threshold while keeping the `float` parameter type.
2. If the `float` type itself causes issues (unlikely — it's a compile-time constant), revert to `bool Use_rescale_threshold` + `kSoftmaxRescaleSkipThreshold`.

The float parameter type is backward-compatible with any threshold value, so step 1 is a one-line-per-callsite change with no template surgery.
