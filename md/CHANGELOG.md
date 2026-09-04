# Changelog

## v1.8 — 2026-09-04

- **Summary:** Fork release **v1.8** — PyTorch **2.14.0+cu132** Windows compatibility verification and upstream sync.
  - **PyTorch 2.14.0 C++ ABI compatibility:** Validated MSVC linking against PyTorch 2.14.0's updated `c10::cuda::c10_cuda_check_implementation` (6-argument signature with `CUDAErrorLogCapture* = nullptr`). Clean-build verification resolves LNK2001/LNK1120 unresolved externals caused by outdated object files.
  - **Upstream sync (`ce088ab`):** Merged official upstream improvements for FA4 / CuTe DSL (SM100 scalar mask-mod compilation speedup `#2819`, CUTLASS DSL `>=4.6.2` floor `#2798`, Quack packed subtraction compatibility `#2787`).
  - **Fork preservation:** FA2 core kernel implementations (`csrc/flash_attn/`), 24 `split_align` kernels, MSVC 2GB-limit bypass (bat2/bat3 SM split), and workflow-removal policy (`tools/ci` suppression) are 100% preserved.
  - **Build scripts:** Adjusted `MAX_JOBS=5` in `WindowsWhlBuilder_cuda_3.bat`.
- **Release notes:** [v1.8_RELEASE.md](v1.8_RELEASE.md)

## v1.7 — 2026-08-12

- **Summary:** Fork release **v1.7** / package **`flash_attn` 2.9.2.post1** — patch on the 2.9.2 `split_align` line. **A-1:** `softmax_rescale_o` uses float template `RescaleThreshold` (call sites `8.0f`, FA4 16-bit default) in place of a bool / `-0.01f` skip; on skip, FA4-style max lagging restores `row_max` to the previous value so running O, `row_sum`, and P share one base (exact output and LSE). **A-2:** `fma_f32x2` packed `fma.rn.f32x2` on sm_100+ drops `volatile` so the compiler may schedule the instruction. Same Blackwell / legacy wheel split as v1.6 (`bat2` `100;120;121`, `bat3` `80;89;90`).
- **Release (GitHub):** https://github.com/ussoewwin/flash-attention/releases/tag/v1.7

## v1.6 — 2026-07-13

- **Summary:** Fork release **v1.6** / package **`flash_attn` 2.9.2** — Official merge of the `split_align` architecture. Extracted the `num_splits == 1` forward pass alignment kernels into 24 independent compilation units (`flash_fwd_split_align_*.cu`) to resolve NVIDIA `ptxas` compiler timeouts and serial compilation crashes. To bypass the Windows MSVC 2GB linker limit (LNK1189) caused by compiling these massive object files, the build scripts were physically split into `bat2` (Blackwell: `100;120;121`) and `bat3` (Ampere/Hopper: `80;89;90`). Dynamic runtime versioning implemented in `__init__.py` to correctly branch between `2.9.1` and `2.9.2`.
- **Release notes:** [v1.6_RELEASE.md](v1.6_RELEASE.md) | **Technical Spec:** [split_align_kernels_explanation.md](split_align_kernels_explanation.md)

## v1.5 — 2026-07-10

- **Summary:** Fork release **v1.5** — Windows build fixes for PyTorch **2.13.0+cu13.2** under MSVC. Resolved the MSVC compiler overload resolution ambiguity (C2666) by introducing exact-match `operator==` and `operator!=` for `c10::ArrayRef<T>` template in namespace `c10`. Upgraded default C++ standard flags to C++20 (`-std=c++20` / `/std:c++20`) in `setup.py` to support C++20 modern syntax features (such as designated initializers and bitfield member initializers) used in PyTorch 2.13.0's headers.
- **Release notes:** [v1.5_RELEASE.md](v1.5_RELEASE.md)

## v1.4 — 2026-05-23

- **Summary:** Fork release **v1.4** / package **`flash_attn` 2.9.1** — patch release on the 2.9.0 feature line (Plan A-1 / A-2). **A-1:** file-scope `kSoftmaxRescaleSkipThreshold` and skip rescale in `softmax_rescale_o` when `scaled_diff > -0.01f`, with `static_assert(!(Is_first && Use_rescale_threshold))`. **A-2:** `scale_apply_exp2` uses packed `fma.rn.f32x2` on sm_100+ via `fma_f32x2` in `utils.h`, `static_assert(N1 % 2 == 0)`, optional `#pragma message` when `UNFUSE_FMA` is set. SASS gate check: `bench/check_sass_gates.py`. Same multi-arch wheel policy as v1.2 (**`80;90;100;120`**). GitHub Release tag and wheel: *to be published*.
- **Release notes:** [FA2 2.9.0 → 2.9.1](FA2_2.9.0_to_2.9.1_RELEASE.md)

## v1.3 — 2026-05-15

- **Summary:** Fork release **v1.3** — Windows FA2 **CUDA 13.2** build (PyTorch **2.12.0+cu132**, CUDA toolkit **13.2**, MSVC `/Zc:preprocessor`, default arch **`80;90;100;120`**, wheel tag **`cu132torch2.12.0`**). Smoke tests under `tests/`.
- **Release (GitHub):** https://github.com/ussoewwin/flash-attention/releases/tag/v1.3

## v1.2 — 2026-05-15

- **Summary:** Fork release **v1.2** / package **`flash_attn` 2.9.0** — FA2 CUDA for **sm_80+** on Windows (PyTorch **2.10+**, CUDA **13+**). Forward backports **A-1** (rescale threshold skip in `softmax_rescale_o`) and **A-2** (packed `fma.rn.f32x2` in `scale_apply_exp2` on sm_100 / sm_120). Also: Triton `flash_attn_triton.py` fix; split-KV launch countermeasure 1 in `flash_fwd_launch_template.h`. Multi-arch wheel build default **`FLASH_ATTN_CUDA_ARCHS=80;90;100;120`** only (Ampere, Hopper, Blackwell datacenter/consumer — **no Thor / sm_101 / sm_110**); see `setup.py` `cuda_archs()` / `add_cuda_gencodes()`.
- **Release (GitHub):** https://github.com/ussoewwin/flash-attention/releases/tag/v1.2

## v1.1 — 2026-05-13

- **Summary:** Fork **v2.8.4** vs **v2.8.3** — upstream v2.9.0-equivalent merge + fork-only fixes (split-KV `num_splits==1` branch removed for `sm_80` ptxas stability; CUTLASS bump).
- **Release (GitHub):** https://github.com/ussoewwin/flash-attention/releases/tag/v1.1

## v1.0 — 2026-05-13

- Fork baseline; later fork releases append under new headings.
- CUDA install path: `setup.py` sets `install_requires` to include `torch>=2.10`.
- ROCm Triton install path: `install_requires` does not list `torch` (users supply PyTorch separately).
