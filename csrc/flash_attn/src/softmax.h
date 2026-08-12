/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "philox.cuh"
#include "utils.h"

// MSVC does not expose M_LOG2E from <cmath> unless _USE_MATH_DEFINES is set.
// Upstream #2669 stopped pulling ATen CUDAGeneratorImpl via flash.h, which had
// previously defined this constant transitively on Windows builds.
#ifndef M_LOG2E
#define M_LOG2E 1.44269504088896340736
#endif

namespace FLASH_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

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

// Apply the exp to all the elements.
// Note: this function is shared by:
//   - fwd via Softmax::softmax_rescale_o (both Is_first branches), and
//   - bwd via flash_bwd_kernel.h::compute_dq_dk_dv (Scale_max=false branch).
// Both paths benefit equally from the sm_100+ packed-FMA inner loop below.
// RescaleThreshold (A-1) is applied only in softmax_rescale_o, not here;
// this function is called after the threshold check has already passed.
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
#if __CUDA_ARCH__ >= 1000 && !defined(UNFUSE_FMA)
        // Plan A-2: on sm_100+ (Blackwell), pair-process columns via fma.rn.f32x2.
        // The pre-exp2f term is fma(t, scale, -max_scaled); fma.rn.f32x2 does two
        // such FMAs in one instruction. exp2f itself stays scalar (MUFU.EX2 has
        // no f32x2 form). Rounding mode .rn matches scalar fmaf default.
        const float neg_max_scaled = -max_scaled;
        constexpr int N1 = decltype(size<1>(tensor))::value;
        static_assert(N1 % 2 == 0,
                      "scale_apply_exp2 packed-FMA path assumes N1 is even; "
                      "if a new MMA atom produces odd N1, restore the scalar tail loop.");
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
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            // The following macro will disable the use of fma.
            // See: https://github.com/pytorch/pytorch/issues/121558 for more details
            // This macro is set in PyTorch and not FlashAttention
            #ifdef UNFUSE_FMA
                tensor(mi, ni) = exp2f(__fmul_rn(tensor(mi, ni), scale) - max_scaled);
            #else
                tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            #endif
        }
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNRows>
struct Softmax {

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;

    __forceinline__ __device__ Softmax() {};

    template<bool Is_first, bool Check_inf=false, float RescaleThreshold=0.0f, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
        static_assert(!(Is_first && RescaleThreshold > 0.0f),
                      "RescaleThreshold > 0 has no effect when Is_first=true; "
                      "remove the threshold from the Is_first=true call site.");
        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);
        if (Is_first) {
            FLASH_NAMESPACE::template reduce_max</*zero_init=*/true>(scores, row_max);
            FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            FLASH_NAMESPACE::reduce_sum</*zero_init=*/true>(scores, row_sum);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            FLASH_NAMESPACE::template reduce_max</*zero_init=*/false>(scores, row_max);
            // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
            static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                float scaled_diff = (scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2;
                // Optionally skip the O / row_sum rescale when the new row_max is close
                // enough to the previous one that the rescale factor is near 1.0.
                // RescaleThreshold is in log2 units of the softmax scale: skip when
                // scaled_diff >= -RescaleThreshold, i.e. scores_scale >= exp2(-RescaleThreshold).
                // With RescaleThreshold=8.0 (FA4 default for fp16/bf16), scores_scale >= 0.0039.
                // FA4-style max lagging: on skip, revert row_max to the previous value so that
                // scale_apply_exp2 below computes P on the same base as the (unrescaled) running
                // O and row_sum. The output and LSE are then exact -- normalize_softmax_lse
                // divides by a row_sum on the same base. The rescale (and base advance) only
                // happens when the row max moves by more than RescaleThreshold log2 units.
                // Safety: max_offset + RescaleThreshold < log2(dtype_max) must hold
                // (see section 14.3 static_assert, to be added when A-3 lands).
                if constexpr (RescaleThreshold > 0.0f) {
                    if (scaled_diff >= -RescaleThreshold) {
                        row_max(mi) = scores_max_prev(mi);
                        continue;
                    }
                }
                float scores_scale = exp2f(scaled_diff);
                row_sum(mi) *= scores_scale;
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
            }
            FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            // We don't do the reduce across threads here since we don't need to use the row_sum.
            // We do that reduce at the end when we need to normalize the softmax.
            FLASH_NAMESPACE::reduce_sum</*zero_init=*/false>(scores, row_sum);
        }
    };

    template<bool Is_dropout=false, bool Split=false, typename Tensor0>
    __forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout=1.0) {
        SumOp<float> sum_op;
        quad_allreduce_(row_sum, row_sum, sum_op);
        TensorT lse = make_fragment_like(row_sum);
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
        }
        return lse;
    };
};

}  // namespace FLASH_NAMESPACE
