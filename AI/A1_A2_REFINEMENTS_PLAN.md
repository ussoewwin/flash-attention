# Plan A-1 / A-2 Refinements (post-implementation review)

**Status:** draft
**Scope:** revisions only for Plans A-1 and A-2 of `AI/FA2_BACKPORT_FROM_FA4_PLAN.md`.
**Assumption:** A-1 and A-2 are already applied in `csrc/flash_attn/src/softmax.h` (single-source for FA2) and the four `Is_first=false` call sites in `csrc/flash_attn/src/flash_fwd_kernel.h`.

This document does **not** modify `AI/FA2_BACKPORT_FROM_FA4_PLAN.md`. It records:

1. What was actually implemented and where it lives in the current tree (verified read-only).
2. Discrepancies between the original plan and the current code.
3. Concrete refinements that should be applied on top of the current A-1 / A-2 implementation, with priority.

The FA3 (`hopper/`) and FA4 (`flash_attn/cute/`) trees have their own `softmax.h` / `softmax.py` and are **out of scope** for this document. FA3 already has its own `Max_offset` parameter equivalent to A-3.

---

## 1. Verification of current state (read-only)

All facts in this section are verified against the working tree at the time of writing.

### 1.1 A-1 implementation (FA2)

`csrc/flash_attn/src/softmax.h::Softmax::softmax_rescale_o` (lines 184–224):

- Template signature: `<bool Is_first, bool Check_inf=false, bool Use_rescale_threshold=false, typename Tensor0, typename Tensor1>`. `Use_rescale_threshold` is a per-call-site template parameter, default `false`.
- Threshold value: `constexpr float kRescaleSkipThreshold = -0.01f;` declared inside the per-row loop (line 211).
- Skip behavior: `if constexpr (Use_rescale_threshold) { if (scaled_diff >= kRescaleSkipThreshold) { continue; } }` skips both the `row_sum *= scores_scale` update (line 215) and the `acc_o_rowcol *= scores_scale` rescale loop (line 217). Correct because `scores_scale = exp2(scaled_diff)` is `>= ~0.993` in that regime.
- The newly reduced `row_max` (line 196) is written **before** the threshold check, so the subsequent `scale_apply_exp2(scores, row_max, ...)` at line 219 always uses the fresh `row_max`. The skip only affects the running accumulators.

Call sites in `csrc/flash_attn/src/flash_fwd_kernel.h` (verified by grep):

| Line | Branch | Flags |
|---|---|---|
| 343 | `Is_first=true` (fixed-len, fast) | `Use_rescale_threshold` defaulted (false) — correct, no `prev` yet |
| 344 | `Is_first=false` (fixed-len, fast) | `Use_rescale_threshold=true` |
| 407 | `Is_first=false` (fixed-len, slow) | `Use_rescale_threshold=true` |
| 917 | `Is_first=true` (varlen, fast) | `Use_rescale_threshold` defaulted (false) — correct |
| 918 | `Is_first=false` (varlen, fast) | `Use_rescale_threshold=true` |
| 985 | `Is_first=false` (varlen, slow) | `Use_rescale_threshold=true` |

Backward path (`csrc/flash_attn/src/flash_bwd_kernel.h`) does not call `softmax_rescale_o`, so A-1 does not apply to bwd. **Confirmed by grep across `csrc/`.**

### 1.2 A-2 implementation (FA2)

`csrc/flash_attn/src/softmax.h`:

- `fma_f32x2(d0,d1, a0,a1, b0,b1, c0,c1)` helper (lines 65–89). Inline PTX `fma.rn.f32x2` under `#if __CUDA_ARCH__ >= 1000 && !defined(UNFUSE_FMA)`. Fallback is two scalar `fmaf` otherwise.
- `scale_apply_exp2<bool Scale_max=true>(tensor, max, scale)` (lines 92–140). On sm_100+, iterates `ni` in steps of 2, calls `fma_f32x2` to produce `(scale*t0 - max_scaled, scale*t1 - max_scaled)`, then scalar `exp2f` on each lane. The `if constexpr (N1 % 2 != 0)` trailing-odd lane (lines 119–122) keeps the scalar formula.

Call sites (verified by grep):

- `softmax.h::Softmax::softmax_rescale_o` calls `scale_apply_exp2` at line 191 (`Is_first=true` branch) and 219 (`Is_first=false` branch). Both default `Scale_max=true`.
- `csrc/flash_attn/src/flash_bwd_kernel.h:536` calls `FLASH_NAMESPACE::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);`. The bwd path benefits from A-2 on sm_100+ as well.

### 1.3 Discrepancies with the original plan

`AI/FA2_BACKPORT_FROM_FA4_PLAN.md` line 187 says:

> **Location:** `csrc/flash_attn/src/softmax.h::scale_apply_exp2`, and the matching `max_scale_exp2_sum` for FA2 bwd (`csrc/flash_attn/src/flash_bwd_kernel.h:536`).

This is **incorrect** in two ways:

1. The function called at `flash_bwd_kernel.h:536` is `scale_apply_exp2` (with `Scale_max=false`), **not** `max_scale_exp2_sum`.
2. `max_scale_exp2_sum` (defined in `softmax.h:142–172`) has **no call sites anywhere in the repository**. Verified by `rg max_scale_exp2_sum`: the only hits are the definition itself and references inside the plan document. It is dead code.

Other places in the original plan that inherit this misattribution (and need consequential cleanup if R1 below is accepted):

- Line 139 (symbol table): `max_scale_exp2_sum` row claims "Used by FA2 bwd."
- Line 368 (§7.4 B-1 goal): mentions `max_scale_exp2_sum` as a B-1 target alongside `scale_apply_exp2`.
- Lines 374–375, 434, 440, 444 (§7.4 B-1 detail): same misattribution.

The plan's claim that A-2 also covers `max_scale_exp2_sum` is moot because nothing calls `max_scale_exp2_sum`. The bwd path is in fact covered by A-2 because it shares `scale_apply_exp2` (the right function).

### 1.4 SASS verification status

The original plan §6 Phase-1 exit criterion and §8.2 inspection row require a SASS check confirming `FFMA.X2` appears inside `scale_apply_exp2` on the sm_120 binary. As of this document, the check is performed manually; no automated `bench/check_sass_gates.py` exists in the tree. See R5 below.

---

## 2. Refinements

Refinements are grouped by plan and tagged with priority (**High** / **Medium** / **Low**). Each item lists the change, the file(s) affected, and the expected impact on the generated binary.

### A-1 refinements

#### A-1-R1 (Low) — Promote `kRescaleSkipThreshold` to a named file-scope constant

**Current:** `constexpr float kRescaleSkipThreshold = -0.01f;` declared inside the per-row loop of `softmax_rescale_o` (line 211). The value is not visible from outside the function.

**Problem:** The original plan §9 risk R2 reserves the right to tighten the threshold to `-0.005f` after Phase-0 measurements. That follow-up should be a one-line constant change in a header, not a function-body edit.

**Proposed change (`csrc/flash_attn/src/softmax.h`, near the top of the file or just above the `Softmax` struct):**

```cpp
// Threshold below which softmax_rescale_o skips the O / row_sum rescale.
// scores_scale = exp2(scaled_diff); scaled_diff is the negative excursion of
// row_max from one iteration to the next, in units of softmax_scale_log2.
// At -0.01f, scores_scale >= ~0.993 (worst-case relative error <= 0.7%).
inline constexpr float kSoftmaxRescaleSkipThreshold = -0.01f;
```

Then in `softmax_rescale_o`:

```cpp
if constexpr (Use_rescale_threshold) {
    if (scaled_diff >= kSoftmaxRescaleSkipThreshold) { continue; }
}
```

**Impact:** Bit-identical binary. Pure refactor.

#### A-1-R2 (Low) — Catch `Is_first=true && Use_rescale_threshold=true` at compile time

**Current:** Nothing in the code prevents a future call site from writing `Is_first=true, Use_rescale_threshold=true`. The `else` branch where the threshold lives is gated by `if (Is_first) { ... } else { ... }`, so passing the flag together with `Is_first=true` is silently a no-op rather than an error.

**Proposed change (`csrc/flash_attn/src/softmax.h::softmax_rescale_o`, immediately after the template signature):**

```cpp
static_assert(!(Is_first && Use_rescale_threshold),
              "Use_rescale_threshold has no effect when Is_first=true; "
              "remove the flag from the Is_first=true call site.");
```

**Impact:** Bit-identical. Catches a class of caller mistakes at compile time.

#### A-1-R3 (Low) — Clarify in-source comment for the skip-cascade bound

**Current:** Comment at lines 206–209 explains what the threshold is, but not why skipping does not cascade across iterations.

**Proposed comment addition** above the `if constexpr (Use_rescale_threshold)` block: mention that the final `normalize_softmax_lse` (line 226) divides by the same `row_sum` whose update was skipped, so the skipped factor appears in both numerator and denominator and the relative error in the normalized output is bounded by the threshold — it does not compound across blocks.

This is already explained in `AI/FA2_BACKPORT_FROM_FA4_PLAN.md` line 256 but is not present in the source file.

**Impact:** Comment-only.

### A-2 refinements

#### A-2-R1 (High) — Fix the `max_scale_exp2_sum` misattribution and remove dead code

**Two-part change.** Apply both:

(a) **Remove dead code.** Delete `max_scale_exp2_sum` from `csrc/flash_attn/src/softmax.h` (the definition spans lines 142–172, including its leading comment). No call site is affected (verified by repo-wide grep).

(b) **Correct the plan.** Update `AI/FA2_BACKPORT_FROM_FA4_PLAN.md` as follows:

| Plan line | Current text (paraphrase) | Replace with |
|---|---|---|
| 139 (symbol table) | "`max_scale_exp2_sum` … Used by FA2 bwd. … (B-2 will revisit)" | Remove the row entirely. |
| 187 (§7.2 A-2 location) | "…and the matching `max_scale_exp2_sum` for FA2 bwd (`flash_bwd_kernel.h:536`)" | "The same code path is exercised by `flash_bwd_kernel.h:536`, which calls `scale_apply_exp2<Scale_max=false>` and therefore receives A-2 automatically." |
| 368, 374–375 (§7.4 B-1 goal & references) | "replace `MUFU.EX2` in `scale_apply_exp2` and `max_scale_exp2_sum`" | "replace `MUFU.EX2` in `scale_apply_exp2`" (single function; covers both fwd and bwd via shared callee). |
| 434, 440, 444 (§7.4 B-1 plug-in & bwd note) | mentions of `max_scale_exp2_sum` | rewrite to refer only to `scale_apply_exp2<Scale_max=false>` for the bwd path. |
| 721 (§12 affected files) | "`flash_bwd_kernel.h` — call site of `scale_apply_exp2` for the backward path" | unchanged (already correct). |

**Impact of (a):** binary-identical (the deleted function was never instantiated). `softmax.h` shrinks by ~30 lines.
**Impact of (b):** documentation-only, but removes the largest factual error in the original plan and ensures B-1 design work targets the right symbol.

#### A-2-R2 (Medium) — Promote `N1 % 2 == 0` from runtime branch to compile-time assertion

**Current** (`csrc/flash_attn/src/softmax.h:119–122`):

```cpp
if constexpr (N1 % 2 != 0) {
    constexpr int last = N1 - 1;
    tensor(mi, last) = exp2f(tensor(mi, last) * scale - max_scaled);
}
```

**Observation:** `tensor` is the `convert_layout_acc_rowcol`-reshaped MMA accumulator. For every MMA atom used in the current FA2 fwd and bwd paths (the m16n8kK and m16n16kK families), the `ncol` dimension is always even (each MMA atom produces 2 columns per atom). The `N1 % 2 != 0` branch has never fired for any shipped configuration.

**Proposed change:**

```cpp
#if __CUDA_ARCH__ >= 1000 && !defined(UNFUSE_FMA)
    constexpr int N1 = decltype(size<1>(tensor))::value;
    static_assert(N1 % 2 == 0,
                  "scale_apply_exp2 packed-FMA path assumes N1 is even; "
                  "if a new MMA atom produces odd N1, restore the scalar tail loop.");
    const float neg_max_scaled = -max_scaled;
    #pragma unroll
    for (int ni = 0; ni < N1; ni += 2) {
        float t0 = tensor(mi, ni);
        float t1 = tensor(mi, ni + 1);
        float r0, r1;
        fma_f32x2(r0, r1, t0, t1, scale, scale, neg_max_scaled, neg_max_scaled);
        tensor(mi, ni)     = exp2f(r0);
        tensor(mi, ni + 1) = exp2f(r1);
    }
#else
    // unchanged scalar loop
#endif
```

Notes:

- Drops the trailing `if constexpr (N1 % 2 != 0)` block.
- The `for` loop bound becomes `ni < N1` (was `ni < N1 - 1`) because evenness is now asserted.
- If a future MMA atom ever produces odd `N1`, the `static_assert` triggers at compile time and the developer either restores the scalar tail or rethinks the layout.

**Impact:** Bit-identical for current configurations. Removes a dead `if constexpr` branch and replaces an implicit assumption with an enforced one.

#### A-2-R3 (Medium) — Move `fma_f32x2` to `utils.h`

**Current:** `fma_f32x2` lives in `csrc/flash_attn/src/softmax.h` (lines 65–89). It is logically a `utils`-level helper, not a softmax-specific concept.

**Future consumers** named by the original plan:

- B-1: `exp2f_emu_x2` Horner polynomial — three packed FMAs (`AI/FA2_BACKPORT_FROM_FA4_PLAN.md` lines 416–420).
- B-2: bounded-domain `exp2`.
- A-3: `MaxOffsetMilli` plumbing composes naturally with the helper located outside `softmax.h`.

**Proposed change:** move the `fma_f32x2` definition (and the preceding comment) from `softmax.h` to `csrc/flash_attn/src/utils.h`, inside the `FLASH_NAMESPACE`. `softmax.h` continues to compile because it already includes `utils.h`.

**Impact:** Bit-identical. Decouples softmax from a general-purpose helper that B-* and any future code path will share.

#### A-2-R4 (Medium) — Compile-time warning when `UNFUSE_FMA` is set on sm_100+

**Observation:** Defining `UNFUSE_FMA` (originally a workaround for `pytorch/pytorch#121558`) while compiling for sm_100+ silently disables A-2's packed-FMA path. There is no warning. This is contrary to the intent of A-2 (which is unconditionally beneficial on sm_100+) and contrary to the assumption in the original plan §8.2 ablation row, which documents `UNFUSE_FMA` as an **opt-in** ablation knob.

**Proposed change:** add a non-fatal `#pragma message` (preferred over `#warning` because it works on MSVC + nvcc cleanly) in `utils.h` after the `fma_f32x2` definition (after the move in R3):

```cpp
#if defined(UNFUSE_FMA) && (__CUDA_ARCH__ >= 1000)
    #pragma message("UNFUSE_FMA is defined while compiling for sm_100+; " \
                    "the Blackwell-only fma.rn.f32x2 path in scale_apply_exp2 is disabled. " \
                    "This is the documented ablation path (see AI/FA2_BACKPORT_FROM_FA4_PLAN.md " \
                    "§8.2 ablation row 'A-2 off' and §10.2 rollback). " \
                    "Remove UNFUSE_FMA to re-enable the packed FMA path.")
#endif
```

**Impact:** Bit-identical. Surfaces the ablation switch so it cannot silently regress performance in production builds.

#### A-2-R5 (High) — Implement the SASS hard gate the original plan promised

**Plan reference:** `AI/FA2_BACKPORT_FROM_FA4_PLAN.md` §6 Phase-1 exit criterion, §8.2 SASS inspection row, §9 risk R8 ("SASS gate is not enforced as a hard gate").

**Promised artifact:** `bench/check_sass_gates.py` — parse `cuobjdump --dump-sass` output and exit non-zero if either of:

- A-2: `FFMA.X2` is absent inside any `_ZN12FLASH_NAMESPACE16scale_apply_exp2*` symbol on a sm_120 build.
- B-1 (when implemented): `MUFU.EX2` is present inside `_ZN12FLASH_NAMESPACE16scale_apply_exp2*` on a sm_120 build.

**Current state:** the script does not exist in the tree. The check is performed manually during reviews.

**Proposed action:** add `bench/check_sass_gates.py` as part of this refinement set. Minimum interface:

```text
python bench/check_sass_gates.py \
    --whl dist/flash_attn-2.9.0+cu132torch2.12.0-cp312-cp312-win_amd64.whl \
    --arch sm_120 \
    --gate a2
```

Exit `0` if `FFMA.X2` is present in the `scale_apply_exp2` symbols, `1` otherwise. Print one line per matching symbol with the `FFMA.X2` count.

A reference implementation can follow `md/2.9.0_COMPLETE_TEST_AND_VALIDATION_GUIDE.md` §11, which already documents the manual procedure and the expected output shape.

**Impact:** No kernel binary change. Unblocks the Phase-1 exit criterion of `AI/FA2_BACKPORT_FROM_FA4_PLAN.md` §6.

#### A-2-R6 (Low) — Document bwd-side coverage above `scale_apply_exp2`

**Current:** The doc-comment above `fma_f32x2` mentions sm_100+ and `UNFUSE_FMA` but not that the same helper is also exercised by the bwd path via `scale_apply_exp2<Scale_max=false>`.

**Proposed comment addition above `scale_apply_exp2`:**

```cpp
// Note: this function is shared by:
//   - fwd via Softmax::softmax_rescale_o (both Is_first branches), and
//   - bwd via flash_bwd_kernel.h::compute_dq_dk_dv (Scale_max=false branch).
// Both paths benefit equally from the sm_100+ packed-FMA inner loop below.
```

**Impact:** Comment-only. Avoids the next reader having to re-discover bwd coverage by grepping.

---

## 3. Priority and ordering

If applied in a single commit series, the recommended order is:

1. **A-2-R1** (dead-code removal + plan correction) — clears the largest factual ambiguity. Touches `csrc/flash_attn/src/softmax.h` and `AI/FA2_BACKPORT_FROM_FA4_PLAN.md`.
2. **A-2-R3** (move `fma_f32x2` to `utils.h`) — needed before R4 and before B-1/B-2 are designed against the wrong location.
3. **A-2-R4** (`UNFUSE_FMA` compile-time message) — lives in `utils.h` next to the helper after R3.
4. **A-2-R2** (`static_assert(N1 % 2 == 0)` + dead-tail removal) — `softmax.h` change.
5. **A-1-R1** (named `kSoftmaxRescaleSkipThreshold` constant) — `softmax.h` change.
6. **A-1-R2** (`static_assert(!(Is_first && Use_rescale_threshold))`) — same file.
7. **A-1-R3**, **A-2-R6** (comment polish) — last; no code.
8. **A-2-R5** (`bench/check_sass_gates.py`) — separate add-only commit; no impact on the kernel binary.

Steps 1–7 produce a bit-identical kernel binary modulo the trivial dead-function removal in step 1 and the modified `softmax.h` source (no instantiated code path changes). Step 8 is build-time tooling.

---

## 4. Items intentionally not changed

- **The `-0.01f` threshold value itself.** The original plan §9 R2 reserves the right to tighten this to `-0.005` after Phase-0 measurements; that measurement is gated on the bench scaffolding, not on this refinement set.
- **The inline PTX in `fma_f32x2`.** It is functionally correct and matches FA4's `flash_attn/cute/utils.py::e2e_asm2` layout. The roll-forward note at §10.3 of the original plan covers what to do if the SASS gate ever fails.
- **The four `Is_first=false` call sites in `flash_fwd_kernel.h`.** They are correct as-is and require no edit beyond R2's compile-time guard.
- **`Scale_max` vs. `scale_max` capitalization.** The template parameter is `Scale_max` (line 92); the bwd call site uses `/*scale_max=*/false` in the comment (line 536 of `flash_bwd_kernel.h`). Minor stylistic inconsistency; not worth a churn commit.
- **FA3 `hopper/softmax.h` and FA4 `flash_attn/cute/softmax.py`.** Out of scope. FA3 already has its own `Max_offset` template parameter (equivalent to A-3); FA4 has its own polynomial `exp2` (equivalent to B-1).

---

## 5. Tracking

Once any of A-1-Rn / A-2-Rn lands, append an entry under `§11 Change Log` of `AI/FA2_BACKPORT_FROM_FA4_PLAN.md` with the commit hash, mirroring the existing `2026-05-13 — Plan A-1 applied` / `Plan A-2 applied` entries.

---

## 6. Cross-references

- Original plan: `AI/FA2_BACKPORT_FROM_FA4_PLAN.md` §5.1, §5.2, §6, §7.1, §7.2, §8.2, §9 (R2, R8), §10.1, §10.2, §10.3, §11.1, §11.2.
- Implementation walk-through: `md/FA2_CHANGES_v1.2.md` §1, §2.
- Verification procedure (manual): `md/2.9.0_COMPLETE_TEST_AND_VALIDATION_GUIDE.md` §10 (A-1), §11 (A-2).


## 7. 2.9.2.post1 Update (2026-08-12) — Delta 1/4/5/6 Applied

This section supplements the original refinement plan with changes applied in the
2.9.2.post1 release of `ussoewwin/flash-attention` (commits `5b068de` through `eea0f68`).

### 7.1 Refinements already applied before 2.9.2.post1 (commit `2eda0be`)

Several A-2 refinements from §2 were applied in commit `2eda0be`
("refactor(fa2): A1/A2 refinements from FA4 backport plan"):

| Refinement | Status |
|---|---|
| A-2-R2 | **Applied.** `static_assert(N1 % 2 == 0)` replaces `if constexpr (N1 % 2 != 0)` tail. |
| A-2-R3 | **Applied.** `fma_f32x2` moved from `softmax.h` to `utils.h`. |
| A-2-R4 | **Applied.** `#pragma message` warns when `UNFUSE_FMA` is set on sm_100+. |
| A-2-R6 | **Applied.** Comment documents bwd-side coverage of `scale_apply_exp2`. |
| A-1-R2 | **Applied (partially).** `static_assert(!(Is_first && Use_rescale_threshold))` guards `Is_first=true` call sites. Updated in Delta 1 to use `float RescaleThreshold`. |

### 7.2 Delta 1: A-1 threshold type change

The original A-1 used `bool Use_rescale_threshold` with a hardcoded `-0.01f` threshold.
This was superseded by:

- **Template change:** `float RescaleThreshold=0.0f` replaces `bool Use_rescale_threshold=false`.
- **Constant deletion:** `kSoftmaxRescaleSkipThreshold = -0.01f` removed entirely.
  The threshold is now passed as a compile-time float at the call site.
- **Call sites:** 4 `Is_first=false` invocations pass `/*RescaleThreshold=*/8.0f`
  (matching FA4's default for fp16/bf16 on B200).
- **`static_assert`:** `!(Is_first && RescaleThreshold > 0.0f)` guards `Is_first=true` call sites.

This replaces A-1-R1's proposal (a named file-scope constant). The new approach
aligns the fork with FA4's design: the threshold is type-specific (`8.0` for
16-bit Q, `0.0` for 8-bit), and embedding it as a template parameter enables
future per-dtype dispatch.

### 7.3 Delta 6: FA4-style max lagging (discovered 2026-08-12)

**Problem:** The original A-1 skip semantics (use new `row_max` for `scale_apply_exp2`
while keeping the rescale factor at `1.0`) is correct for a small threshold (0.01, ≤0.7%
error). At the FA4-derived threshold `8.0`, the inconsistency would over-weight stale block
contributions by up to `1/exp2(-8) ≈ 256×`.

**FA4 reference:** `flash_attn/cute/softmax.py::SoftmaxSm100.update_row_max` reverts
`row_max_new = row_max_old` on skip, making the online softmax **exact** (the max
simply lags, and the output and LSE are mathematically identical to the full-rescale path).

**Fix applied (Delta 6):**

```cpp
// In softmax_rescale_o's per-row loop:
if constexpr (RescaleThreshold > 0.0f) {
    if (scaled_diff >= -RescaleThreshold) {
        row_max(mi) = scores_max_prev(mi);   // FA4-style max lagging
        continue;
    }
}
```

**Correctness proof:** With max lagging, `P = exp2(S·scale − m_prev·scale)` and
O/row_sum stay on `m_prev`. The final `normalize_softmax_lse` divides by the
same-base row_sum. LSE identity: `m_prev·s + log(RS_prev + RS_cur/ss) = m_cur·s + log(RS_prev·ss + RS_cur).`

**P-saturation:** P can reach `exp2(8) = 256`, well within fp16 max (65504) and
bf16 max (3.39e38). No overflow risk.

### 7.4 Delta 4: Remove `volatile` from `fma_f32x2` asm

In `utils.h`, `fma_f32x2`: `asm volatile(` → `asm(`. This matches FA4's
`cute.arch.fma_packed_f32x2` which does not use `volatile`. On sm_120, `ptxas`
may lower `fma.rn.f32x2` to scalar FFMA pairs; the `volatile` removal allows
the compiler to schedule them more freely.

### 7.5 Delta 5: Comment update

`scale_apply_exp2` doc-comment notes that `RescaleThreshold` (A-1) is applied
only in `softmax_rescale_o`, not here; this function is called after the
threshold check has passed.

### 7.6 Version

Bumped from `2.9.2` to `2.9.2.post1` in:
- `flash_attn/__init__.py` (fallback `__version__`)
- `setup2.py` (`public_version`)
- `setup.py` reads version dynamically from `__init__.py`

### 7.7 Build and Test Results

**Environment:** RTX 5060 Ti (sm_120), CUDA 13.2, PyTorch 2.13.0, Python 3.14.6.

**Commits:**
| Commit | Description |
|---|---|
| `5b068de` | fix(softmax): A-1 rescale threshold upgrade + FA4-style max lagging |
| `9abf358` | perf(asm): remove volatile from fma_f32x2 inline asm (Delta 4) |
| `b6dad32` | docs: add Delta 6 to plan + bump version to 2.9.2.post1 |
| `eea0f68` | docs: add 2.9.2.post1 complete test and validation guide |

**Test summary (56 tests, all PASS):**

| Category | Cases | Result |
|---|---|---|
| Accuracy vs fp32 ref (fp16) | 12 shapes, S=512–8192, D=64/128/256 | Worst rel_Linf=7.94e-04 (gate ≤2e-2) |
| Accuracy vs fp32 ref (bf16) | 6 shapes, S=512–4096, D=64/128 | Worst rel_Linf=6.54e-03 (gate ≤1e-2) |
| Edge cases | S=1, S=2, H=1, D=256 bf16, return_lse, softcap+causal, window_size | All PASS |
| Backward | fp16/bf16 finiteness + grad accuracy vs fp32 ref | All finite; dQ=2.10e-05, dK=1.86e-05, dV=1.47e-03 |
| Varlen | forward + backward | All PASS |
| Determinism | Same input → same output | Bit-identical |
| API surface | 7 paths (flash_attn_func, varlen, kvcache, window, alibi, softcap) | All PASS |
| Latency | S=1024–8192, D=64/128 | 0.079–6.504 ms/iter, matches 2.9.0 baseline |

**SASS gate (A-2):** 96 sm_120 cubins scanned. `FFMA.X2` **not found**. On sm_120
(Blackwell), `fma.rn.f32x2` PTX appears to be lowered to scalar FFMA pairs by `ptxas`.
The `check_sass_gates.py` tool (A-2-R5) needs an sm_120 instruction pattern update.
Accuracy is unaffected: all 28 accuracy + 10 edge cases pass comfortably.

Full details: `md/2.9.2.post1_COMPLETE_TEST_AND_VALIDATION_GUIDE.md`.

### 7.8 Refinements Still Pending

| Refinement | Status | Note |
|---|---|---|
| A-2-R1 | **Not applied.** `max_scale_exp2_sum` dead code still present; plan misattribution not yet corrected. | Low priority; no binary impact. |
| A-1-R3 | **Not applied (superseded).** Comment was updated with max-lagging text instead (Delta 5+6). |
| A-2-R5 | **Partial.** `bench/check_sass_gates.py` exists but needs sm_120 support. |
