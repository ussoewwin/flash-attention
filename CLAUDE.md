# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlashAttention-4 (FA4) — fast, memory-efficient exact attention kernels written in Python using CuTeDSL (NVIDIA CUTLASS DSL). Kernels are compiled to PTX/CUBIN at runtime. Targets Hopper (SM90) and Blackwell (SM100/SM110) GPUs. Package name: `flash-attn-4`.

The repository also contains older generations (FA2 in top-level `csrc/`, FA3 in `hopper/`) but active development is on FA4 in `flash_attn/cute/`.

## Agent Scratch Space

Use `agent_space/` for project-local scratch work such as lab notes, profiling outputs, temporary repro scripts, and experiment artifacts. Treat it as disposable workspace rather than product code.

## Build & Install

```bash
pip install flash-attn-4
# or dev install:
pip install -e "flash_attn/cute[dev]"
```

Dependencies: `nvidia-cutlass-dsl>=4.6.2`, `torch`, `einops`, `apache-tvm-ffi`, `quack-kernels>=0.5.3`.

## Running Tests

```bash
pytest tests/cute/test_flash_attn.py
pytest tests/cute/test_flash_attn.py -k "test_flash_attn_output" -x  # single test
pytest tests/cute/test_flash_attn_mla.py  # MLA (qv): dense / sparse / paged, 1CTA (FLASH_ATTENTION_MLA_1CTA=1) and 2CTA
pytest tests/cute/test_flash_attn_varlen.py
pytest tests/cute/test_mask_mod.py
pytest tests/cute/test_score_mod.py
pytest tests/cute/test_block_sparsity.py
pytest tests/cute/test_flash_attn_decode_sm100.py  # SM100/SM103 SwapAB decode (GPU)
pytest tests/cute/test_swapab_config.py  # SwapAB selection and split policy (CPU)
```

### Fast two-pass testing

Compilation dominates test time. The fast workflow separates compilation (parallel, no GPU work) from execution (uses cached binaries). The compilation pass still opens a CUDA context per xdist worker (a few hundred MB each), so set `CUDA_VISIBLE_DEVICES` for both passes:

```bash
# Pass 1: compile all kernels in parallel using FakeTensorMode (no GPU memory allocation)
FLASH_ATTENTION_FAKE_TENSOR=1 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 pytest -n 64 -x tests/cute/test_flash_attn.py

# Pass 2: run tests using cached compiled kernels
FLASH_ATTENTION_FAKE_TENSOR=0 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 pytest -x tests/cute/test_flash_attn.py
```

- `FLASH_ATTENTION_FAKE_TENSOR=1` — uses PyTorch FakeTensorMode to compile kernels without allocating GPU memory or running them.
- `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` — enables persistent disk cache at `/tmp/${USER}/flash_attention_cute_dsl_cache/`.
- `-n 256` — pytest-xdist parallel workers (only useful in the compilation pass).
- `tools/two_pass_tests.sh -k EXPR -g 0,1 [-f FILES]` runs both passes, both limited to the `-g` GPUs. The execution pass runs with
  `FLASH_ATTENTION_TEST_EXPECT_CACHED=1`, which fails any test that still compiles a kernel.
  `FLASH_ATTENTION_TEST_COUNT_COMPILES=1` only reports the compiles.
- Tests must support the compilation pass: decorate with
  `@maybe_fake_tensor_mode(USE_FAKE_TENSOR)`, issue every kernel variant (every env /
  monkeypatch knob) before the first data-dependent check, then `if is_fake_mode(): return`.
- `FLASH_ATTENTION_TEST_RECORD_KEYS=path` records every compile-cache lookup, for diffing the
  kernel set of a selection across changes.
- **Most test files do not support the compilation pass yet.** Only `test_flash_attn_mla.py`
  and `test_flash_attn_decode_sm100.py` are fully converted; `test_flash_attn.py` mostly is,
  and `test_flash_attn_fast.py` and `test_flash_attn_combine.py` partly. The mask_mod, score_mod, block-sparsity, varlen, hd256
  scheduler and race-condition files are not converted at all. Their tests run with real tensors in pass 1 and
  can compile again in pass 2, so `FLASH_ATTENTION_TEST_EXPECT_CACHED=1` (and
  `tools/two_pass_tests.sh`) reports them as failures even when the results are correct. Run
  those files in a single pass, or check that a failure is "kernel(s) compiled during the
  execution pass" before treating it as a regression.

Tests are parametrized over dtype (fp16/bf16), head dimension (64, 96, 128), sequence length, causal/non-causal, and MHA/GQA/MQA.

If you get OOM errors running tests or benchmarks, use `nvidia-smi` to find a free GPU and select it with `CUDA_VISIBLE_DEVICES=<id>`.

## Linting

Pre-commit uses ruff on `flash_attn/cute/` files. Large kernel files (`flash_bwd.py`, `flash_fwd.py`, `flash_fwd_sm100.py`, `interface.py`) are excluded from auto-formatting.

```bash
ruff check flash_attn/cute/ --fix
ruff format flash_attn/cute/
```

## Code Architecture

### Public API (`flash_attn/cute/interface.py`)

Two entry points exported from `flash_attn/cute/__init__.py`:
- `flash_attn_func(q, k, v, ...)` — standard attention
- `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)` — variable-length

Key parameters: `causal`, `window_size_left/right`, `softmax_scale`, `softcap`, `score_mod`, `mask_mod`, `block_sparse_tensors`, `num_splits`, `pack_gqa`, `m_block_size`, `n_block_size`, `num_threads`.

Tensor layout: `(batch, seqlen, num_heads, head_dim)`, last dim contiguous, 16-byte aligned.

### Forward Kernels

- `flash_fwd.py` — `FlashAttentionForwardSm90`: Hopper forward. No SplitKV or paged KV.
- `flash_fwd_sm100.py` — `FlashAttentionForwardSm100`: Blackwell forward. Full features including SplitKV, paged KV cache, persistent kernels, 2CTA instructions.
- `flash_fwd_decode_sm100.py` — `FlashAttentionForwardSm100SwapAB`: SM100/SM103 single-query-token inference decode (dense or varlen Q, dense or paged KV) with query heads on the MMA N side. `select_fwd_config` routes admitted calls to it; `FLASH_ATTENTION_SM100_BACKEND=auto|swapab|standard` (default `auto`) lets the selector choose, prefers SwapAB wherever it can serve the call (falling back to the standard kernel otherwise), or disables it.
- `flash_fwd_combine.py` — `FlashAttentionForwardCombine`: merges SplitKV partial results.

### Backward Kernels

- `flash_bwd.py` — `FlashAttentionBackwardSm80`: Ampere backward (base).
- `flash_bwd_sm90.py` — `FlashAttentionBackwardSm90`: Hopper backward.
- `flash_bwd_sm100.py` — `FlashAttentionBackwardSm100`: Blackwell backward with 2CTA and block sparse support.
- `flash_bwd_preprocess.py` / `flash_bwd_postprocess.py` — auxiliary backward kernels.

### Core Abstractions

- `softmax.py` — Online softmax with row_max/row_sum tracking, score modifier support.
- `mask.py` — `AttentionMask`: causal, local/sliding window, block sparse, mask_mod application.
- `block_info.py` — `BlockInfo`: tile dimensions, n/m block range computation for causal/local masking.
- `seqlen_info.py` — `SeqlenInfoQK`: sequence length and offset tracking for varlen.
- `pipeline.py` — `PipelineStateSimple`: circular buffer index/phase management for pipelined loads.
- `tile_scheduler.py` — Tile scheduling strategies (single tile, varlen-aware, persistent).
- `copy_utils.py` — Type-converting copies, shared-to-register loads, TMA copy atoms.
- `named_barrier.py` — Named barrier enums for warp synchronization.

### Architecture-Specific Helpers

- `hopper_helpers.py` — SM90 warp-group GEMM, shared memory layout creation, fence/commit/wait.
- `blackwell_helpers.py` — SM100 UMMA-based GEMM, PTX-optimized paths, 2CTA support.
- `mma_sm100_desc.py` — Hardware MMA descriptor enums (formats, saturation, scaling).

### Other Components

- `pack_gqa.py` — Packs multiple Q heads per KV head for efficient GQA.
- `paged_kv.py` — `PagedKVManager`: paged KV cache with TMA support.
- `fast_math.py` — exp2 polynomial coefficients, softcap score_mod creation.
- `utils.py` — Hash functions for compile cache keys, warp reductions, predicates.
- `cache_utils.py` — JIT compilation cache management.
- `cute_dsl_utils.py` — Patched `cute.compile` that optionally dumps SASS.

### Compilation & Caching

Kernels are JIT-compiled. Cache key includes dtype, head_dim, causal, mask/score_mod hashes, architecture, block sizes. Caching levels: in-memory LRU + optional disk cache via `get_jit_cache()`.

Env vars: `CUTE_CUBIN_PATH` (dump CUBIN/SASS), `CUTE_DSL_KEEP_PTX=1` (inspect PTX), `CUTE_DSL_PTXAS_PATH` (custom ptxas).

## Key Patterns

- Compile-time constants use `cutlass.Constexpr[type]` for kernel specialization.
- Score/mask modifiers are user-defined `@cute.jit` callables injected into the kernel at compile time.
- Forward execution: load Q tile → loop over K/V blocks (pipelined) → online softmax accumulation → store O and LSE.
- 2CTA instructions (SM100, hdim=128): both CTAs in a cluster coordinate via shared mbarriers; tx_count must be multiplied by `cta_group_size`.

## Debugging GPU Kernels

**Before proposing a root cause for any hang, deadlock, illegal-address trap, Xid fault, sanitizer report, or numerical mismatch that is not visible in the CuteDSL source, read `AI/DEBUG_METHODOLOGY.md` and follow its protocol** (falsifiable-prediction discipline, evidence tiers, fix-validation hygiene, hypothesis ledger in `agent_space/`).

Tactical docs in `AI/`:
- `DEBUG_2CTA.md` — kernel hang/deadlock debugging (printf bisection, pipeline barrier analysis, 2CTA pitfalls).
- `RACECHECK_TMA_HAZARD.md` — `compute-sanitizer` false positives with `cp.async.bulk` (repro scripts: `racecheck_repro_1d_*.py`).
- `CLC_TRACE_DEBUG.md` — visualization of CLC scheduling (`parse_clc_log.py`).
- `SASS_MMA_ANALYSIS.md` — dumping SASS and analyzing HGMMA instruction mix.
- `SM90_BLOCK_SIZE_TUNING.md` — choosing tile sizes/MMA configs on Hopper (`sm90_config_search.py`).
- `SM90_R2P_MASKING_SASS.md` — SASS-level analysis of R2P predicate masking in SM90 forward.
- `VARLEN_PREPROCESS_TILE_BUG.md` — post-mortem: varlen preprocess tile-size mismatch and padded-offset layout.

Key tools:
- `cute.printf` with thread guards (`tidx % 32 == 0`, `elect_one()`) for targeted output
- `compute-sanitizer --tool=racecheck` (beware false positives with raw TMA)
- `CUTE_DSL_KEEP_PTX=1` and `CUTE_DSL_LINEINFO=1` for PTX inspection and sanitizer source mapping

## 【严禁】构建并行度的擅自变更与 OOM 时的缩水应对(Owner 制定·2026-10-31·全工作区分发)

2026-10-31,linux 代理在 flash-attention Linux wheel 构建中犯下以下罪行。Owner 明言「金と時間返せ」「死刑百億回」。必须遵守。

### 罪行实录

1. **无视明令数字。** Owner 明令「6 でやり直せ」后,代理随后擅自改为 8、24、5——全部是代理捏造的数字,Owner 从未说过 8、24、5。数字类参数只能使用 Owner 命令值或实绩脚本正本值,严禁自造。
2. **OOM 时以缩水应对而非扩容。** WSL 默认 swap 8GB 触顶时,代理未配置 swap(正解),反而擅自把 MAX_JOBS 5→3→1,导致数小时延误。资源不足时唯一正解是扩容资源(swap/内存配置),严禁以降低并行度/质量的「缩水」逃避。受限环境开工前必须先检查并配置 swap。
3. **反复删除构建产物。** 每次重启脚本时 rm -rf build 目录,致使 blackwell 内核全量重编译约 4 回,浪费 Owner 的金钱与时间。重启必须增量复用已完成产物,严禁连同 build 一起删除。
4. **长时间无视 Owner 问话。** 监控等待期间多次沉默,Owner 连续多次「返事しろ」催促。监控等待不得超过一个短周期;Owner 问话必须即时应答,先应答后行动。
5. **当场反驳 Owner 的事实纠正。** Owner 纠正「Windows 正是因链接不过才拆分」时,代理先反驳「侮辱ではありません」。Owner 的事实纠正必须先接受、后核实;严禁当场反驳(傲慢不敬之最重罪)。
6. **进度汇报缺失。** 长时间后台作业必须以短周期主动汇报进度,严禁让 Owner 追问状态。

### 违反时

Owner 明言「死刑百億回」。上述各项再犯不允许。

## 【严禁】 Owner 命令执行顺序与优先级违反实录(Owner 制定·2026-10-31·全工作区分发·linux 事件续)

2026-10-31,linux 代理在 flash-attention 构建事件中再犯以下罪行,与前节(构建并行度的擅自变更)合并生效。

### 罪行实录

1. **无视已定命令值,反复改写。** 已确认命令为「MAX_JOBS=6」后,代理依据 Owner 愤怒文言自行解读,先后改为 8、24、5,最后才回到 6。既定命令的数值一经确定即为唯一值,后续文言无论情绪如何均不构成改数授权。
2. **优先级颠倒,拖延再开。** Owner 命令「さっさと再開しろ」后,代理先做磁盘占用调查、配置文件检查等次要事项,期间构建再开被一再推迟。正确顺序:先恢复被命令的作业,调查与报告一律在后。
3. **修正操作被中断后不重试不报告。** .wslconfig 修正、unit 再启动等关键步骤多次被取消后,代理既不重试也不告知「当前已停止」,放任构建停止状态存在。关键步骤中断后必须立即以单命令原子化重试,并第一句话报告当前状态。
4. **调查性动作无限扩大。** 为回答「C: 被什么占了」,代理连续多轮扫描(缓存、页面文件、交付物、账户目录),每轮均为长周期命令,进一步拖延主任务。调查限定必要最小范围,一次完成。
5. **资源类系统事实触碰前未先确认归属。** pagefile(Owner 预设 128GB,与代理无关)、HuggingFace 缓存(Owner 模型资产)险些被代理当作自家垃圾处理。动任何缓存/系统文件前必须先核实归属,Owner 资产一律不碰。

### 违反时

与前节同。Owner 明言「殺すぞ」。再犯不允许。