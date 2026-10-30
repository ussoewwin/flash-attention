## Overview

**FlashAttention Windows Fork v1.8.3** (Package version: `flash_attn` **2.9.2.post2**) delivers native compilation and execution support for **PyTorch 2.15.0+cu134** (Python 3.14, CUDA 13.4 Toolkit) under MSVC 2022 on Windows x64. In addition, it integrates 16 upstream commits from `Dao-AILab/flash-attention` (`76d93310` -> `a3a9dfa9`), incorporating critical CUDA graph capture safety for FA2 paged KV caches, SM100 hd256 backward optimizations, and bounded TMA ragged load guards, while maintaining 100% preservation of all fork customizations.

---

## 1. PyTorch 2.15.0+cu134 & CUDA 13.4 MSVC Integration (`setup2.py`)

### C++20 Standard & MSVC Permissive Flag Alignment
- **Dynamic C++ Standard Detection:** Starting with PyTorch 2.13+, ATen C++ headers enforce C++20 compliance (`#error C++20 or later ... required`). `setup2.py` dynamically evaluates PyTorch version attributes and assigns:
  ```python
  cxx_standard = "c++20" if (TORCH_MAJOR, TORCH_MINOR) >= (2, 13) else "c++17"
  ```
- **MSVC Permissive Conformance Adjustment (`-Xcompiler /permissive`):**
  When `/std:c++20` is passed to MSVC 2022, strict conformance mode (`/permissive-`) is activated by default. Under CUDA 13.x toolkits, nvcc host compilation fails during CuTe template metaprogramming instantiation—specifically throwing `C3545: 'stride_type': an expression that produces an object of type '...' is required` within `cute/stride.hpp`. To resolve this without breaking PyTorch ATen C++20 headers, `setup2.py` conditionally appends `["-Xcompiler", "/permissive"]` to `nvcc_flags` during host compilation on Windows.

### Mathematical Constants Exposure (`-D_USE_MATH_DEFINES`)
- Upstream PR #2669 removed transitive ATen header inclusions that previously exposed `M_LOG2E` via `CUDAGeneratorImpl.h`.
- On MSVC, standard C++ `<cmath>` suppresses POSIX math constants (`M_LOG2E`, `M_PI`, etc.) unless `_USE_MATH_DEFINES` is explicitly defined prior to header inclusion. Added `-D_USE_MATH_DEFINES` to both MSVC CXX flags and nvcc host compilation flags in `setup2.py`, preventing unresolved identifier errors across `softmax.h` and `flash_api.cpp`.

### CUDA Architecture PTX Generation Safety (`add_cuda_gencodes`)
- Refactored virtual PTX emission in `add_cuda_gencodes()`: nvcc targets are strictly mapped from verified `supported_archs` (`80`, `89`, `90`, `100`, `120`, `121`).
- The logic determines `newest = max(supported_archs, key=int)` and generates `arch=compute_{newest},code=compute_{newest}`. This prevents nvcc build crashes caused by passing non-existent virtual PTX tokens (such as `compute_121`).

### PEP 427 Automated Wheel Packaging Pipeline
- Validated `WindowsWhlBuilder_cuda_2.bat` and `WindowsWhlBuilder_cuda_3.bat` parse routines:
  - Correctly maps `torch.version.cuda='13.4'` -> `cu134`.
  - Correctly maps `torch.__version__='2.15.0+cu134'` -> `torch2.15.0`.
  - Produces PEP 427-compliant binary wheels (e.g., `flash_attn-2.9.2.post2+cu134torch2.15.0cxx11abiTRUE.blackwell-cp314-cp314-win_amd64.whl`).

---

## 2. Upstream Sync Merge Details (`76d93310` -> `a3a9dfa9`, 16 Commits)

| Commit Hash | Topic / PR | Description |
|---|---|---|
| `a3a9dfa9` | [CuTe, SM100] #3058 | Unify hd256 backward (5/7): cheaper deterministic dQ semaphore (overlapped release on M=64 serial schedule, early acquire, relaxed release) |
| `98c1cee0` | [CuTe, SM100] #3034 | Fix CLC tile coordinates and make forward work-tile coordinates logical |
| `bc52a40a` | [CuTe, SM100] #2967 | Unify hd256 backward (4/7): pack query heads of a KV head into Q tile for Grouped Query Attention (GQA) |
| `10b74d98` | [CuTe] #3071 | Bound ragged K/V TMA loads with per-CTA descriptor updates instead of wrap-around descriptor |
| `767386d8` | [CuTe, SM90] #3013 | Bound K/V TMA loads by seqlen_k (fwd + bwd, packed varlen and dense seqused_k) |
| `2bdeddd1` | [CuTe, Bwd, SM100] #3012 | Bound K/V TMA loads by seqlen_k so unused rows cannot poison dQ accumulators |
| `1326ab1d` | [CuTe] #3067 | Run ruff format on config.py for pre-commit hook compliance |
| `a6fcdfdb` | [CuTe, Fwd, SM100] #2983 | Prevent NaN contamination from unused V cache entries |
| `e7a6402a` | [CuTe] #2732 | Add typed forward configurations |
| `965c9212` | [CuTe, SM100] #2966 | Unify hd256 backward (3/5): hdim 256 on FlashAttentionBackwardSm100 (opt-in) |
| `e9d8944c` | [Tests] #3007 | CUDA graph capture and replay test coverage for FA4 (CuTe) and FA2 entry points |
| `2b00d76f` | [CuTe, SM100] #2965 | Unify hd256 backward (2/5): 64-row KV tile per CTA (M=64 TMEM layout) with GQA, varlen and deterministic support |
| `73019aff` | [FA2] #2984 | Fix paged KV-cache CUDA graph capture by removing CPU-GPU host synchronization in `mha_fwd_kvcache` |
| `1fc02211` | [CuTe, SM120] #2758 | Support automatic `num_splits` on SM120 |
| `bb25ba3c` | [Version] #2992 | Bump upstream package reference version to 3.0.1 |
| `94e22c90` | [CuTe, SM100] #2978 | Key MLA forward compilation on the KV head count |

### Upstream Highlight Details:
1. **FA2 Paged KV-Cache CUDA Graph Safety (`flash_api.cpp`, #2984):**
   Prior implementations suffered from CPU-GPU host synchronization triggers within `mha_fwd_kvcache` during CUDA graph capture. Upstream #2984 completely eliminated host-device sync points in the paged KV pathway, enabling seamless capture and replay of inference pipelines utilizing FlashAttention paged KV-cache.
2. **CuTe / FA4 SM100 Architecture Unification (#2965, #2966, #2967, #3058):**
   Provides comprehensive unification for head dimension 256 backward execution on SM100 architectures. Implements 64-row KV tiles per CTA (M=64 TMEM layout) with deterministic GQA support, packs query heads into Q tiles, and introduces overlapped semaphore release on serial schedules to minimize sync overhead.
3. **Bounded Ragged TMA Loads (#3012, #3013, #3071):**
   In packed varlen and dense configurations with variable sequence lengths, unaligned TMA loads could read beyond valid memory regions and poison dQ accumulators. Upstream replaced wrap-around descriptors with dynamic per-CTA descriptor updates strictly bounded by `seqlen_k`.
4. **V Cache NaN Guard (#2983):**
   Added isolation masks for unallocated slots in the V cache to prevent floating-point NaN contamination from propagating through the attention accumulator.

---

## 3. Preserved Fork Customizations

- **24 Split-Align Compilation Units (`flash_fwd_split_align_*.cu`):**
  Isolates `num_splits == 1` forward kernels into 24 distinct translation units, completely resolving NVIDIA `ptxas` compiler stalls, out-of-memory errors, and compilation timeouts on Windows.
- **MSVC 2GB Linker Limit Bypass (Architecture Partitioning):**
  Due to the massive symbol table produced by FlashAttention kernels under MSVC, attempting to build all architectures in a single DLL exceeds the 2GB COFF linker boundary (LNK1189). The fork maintains architectural wheel partitioning:
  - `WindowsWhlBuilder_cuda_2.bat`: Blackwell targets (`sm_100`, `sm_120`, `sm_121`).
  - `WindowsWhlBuilder_cuda_3.bat`: Legacy targets (`sm_80`, `sm_89`, `sm_90`).
- **Zero-CI Policy:**
  Full suppression of remote GitHub Actions CI pipelines (`tools/ci`) to ensure strict local verification and zero unintended workflow triggers.

---

## 4. Verification & Testing Matrix

- **Forward Pass Verification:** `test_fa2.py` passed with `causal=True` and `causal=False` across diverse batch sizes and head dimensions.
- **Backward Gradient Verification:** Numerical accuracy validated against standard PyTorch attention reference outputs (`torch.allclose` at `atol=1e-3`, `rtol=1e-3` for FP16/BF16).
- **Environment:**
  - OS: Windows 11 x64
  - Compiler: MSVC 2022 (v19.4x)
  - Python: 3.14.0
  - PyTorch: `2.15.0+cu134`
  - CUDA Toolkit: 13.4.0
