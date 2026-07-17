# CUDA 13.2 + PyTorch 2.10+ build fix (official FA2 2.8.4 fork, Windows)

This document records changes on the **`official`** branch: **Dao-AILab** `flash_attn` **2.8.4** (`upstream/main` @ `ddfec5d9`) plus **minimal fork patches** for:

1. **PyTorch 2.10+** (extension header, faster split-KV compile path)
2. **CUDA 13.2** wheels (MSVC preprocessor + supported SASS arch list)

This branch **does not** include 2.9.0, SA4, or A-1/A-2 backports.

## Environment

| Item | Example |
|------|---------|
| PyTorch | `2.12.0+cu132` or `2.13.0+cu132` |
| CUDA toolkit (`CUDA_HOME`) | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2` |
| Package version | `2.8.4` |
| Default SASS archs | `80;90;100;120` (no Thor `110`) |

## Fork diffs vs `upstream/main`

| File | Purpose |
|------|---------|
| `csrc/flash_attn/flash_api.cpp` | `#include <torch/extension.h>` (PyTorch 2.10+); `CHECK_SHAPE` uses `.equals()` for PyTorch 2.13 + MSVC |
| `csrc/flash_attn/src/flash_fwd_launch_template.h` | Remove `num_splits==1` alignment kernel tree (shorter compile) |
| `setup.py` | Build info prints; `FORK_SUPPORTED_CUDA_ARCHS` + `cuda_archs()` filter; no Thor gencode; Windows `/Zc:preprocessor` + spawn hook; host/cuda **c++20** on Windows |
| `WindowsWhlBuilder_cuda.bat` | Wheel helper (`DISTUTILS_USE_SDK=1`, arch override) |
| `md/CUDA_13.0_TO_13.2_BUILD_FIX.md` | This document |

## Problem A — CCCL / MSVC preprocessor (CUDA 13.2)

When `DISTUTILS_USE_SDK=1`, add `/Zc:preprocessor` alongside `/Zc:__cplusplus` in `setup.py` (Windows block). `NinjaBuildExtension` also sets `CL` and patches `cl.exe` / `nvcc.exe` command lines. PyTorch 2.13 still omits `/Zc:preprocessor` from `COMMON_MSVC_FLAGS` (upstream [PR #188761](https://github.com/pytorch/pytorch/pull/188761)).

## Problem C — `IntArrayRef` / MSVC C2666 (PyTorch 2.13.0)

PyTorch 2.13.0 refactored `c10::ArrayRef<T>` to inherit `HeaderOnlyArrayRef<T>`. `ArrayRef` had no exact-match `operator==`, so MSVC ADL could not pick among `HeaderOnlyArrayRef`, `OptionalArrayRef`, `IValue`, etc. when `flash_api.cpp` used `tensor.sizes() == torch::IntArrayRef({...})` in `CHECK_SHAPE`.

**Fork fix (preferred):** use `.equals()` instead of `==` in `csrc/flash_attn/flash_api.cpp` (survives `pip install`).

**Optional venv-only workaround:** hidden-friend `operator==` / `operator!=` inside `ArrayRef` in `torch/include/c10/util/ArrayRef.h` — lost on torch reinstall; not required if the fork fix is applied.

## Problem B — Thor `sm_110` gencode

Default upstream arch list includes `110`. This fork builds **80/90/100/120** only. `cuda_archs()` drops `101`/`110` with a warning. Override with `FLASH_ATTN_CUDA_ARCHS` if needed.

## Build recipe

```bat
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set PATH=%CUDA_HOME%\bin;%PATH%
set MAX_JOBS=4
WindowsWhlBuilder_cuda.bat
```

Inplace:

```bat
set DISTUTILS_USE_SDK=1
set FLASH_ATTN_CUDA_ARCHS=80;90;100;120
set FLASH_ATTENTION_FORCE_BUILD=TRUE
python setup.py build_ext --inplace
```

Confirm: no C1189 preprocessor error; no MSVC C2666 on `IntArrayRef`; no fatal `compute_110f.cpp1.ii` when using the arch list above.
