/*
 * zdnn_sim_stubs.c - Stub implementations for zdnn-sim extension functions
 *
 * These functions are used by llama.cpp's zdnn backend but are not part of
 * the IBM zDNN API. They are implemented in zdnn-sim for macOS development.
 *
 * On s390x with real AIU hardware, these operations fall back to CPU or
 * return ZDNN_FUNC_RC_F000 (function not available).
 *
 * TODO: Implement optimized versions using NNPA instructions where possible.
 */

#include "zdnn.h"
#include <string.h>
#include <math.h>

/*
 * zdnn_rmsnorm - RMS Normalization (zdnn-sim extension)
 *
 * Computes: output = (input / rms(input)) * weight
 * where rms(x) = sqrt(mean(x^2) + epsilon)
 *
 * This is a key operation for LLaMA-style models.
 */
zdnn_status zdnn_rmsnorm(const zdnn_ztensor *input,
                         const zdnn_ztensor *weight,
                         float epsilon,
                         zdnn_ztensor *output) {
    (void)input;
    (void)weight;
    (void)epsilon;
    (void)output;
    /* TODO: Implement using NNPA or fall back to software */
    return ZDNN_FUNC_RC_F000; /* Function not available */
}

/*
 * zdnn_get_rows - Extract rows using indices (zdnn-sim extension)
 *
 * Used for embedding table lookups. Given a 2D source tensor and an array
 * of row indices, copies the selected rows to the output.
 */
zdnn_status zdnn_get_rows(const float *src,
                          int64_t src_ne0,
                          int64_t src_ne1,
                          const int32_t *indices,
                          int64_t num_indices,
                          float *output) {
    (void)src;
    (void)src_ne0;
    (void)src_ne1;
    (void)indices;
    (void)num_indices;
    (void)output;
    /* TODO: Implement - this is a simple gather operation */
    return ZDNN_FUNC_RC_F000; /* Function not available */
}

/*
 * zdnn_get_rows_batched - Batched row extraction (zdnn-sim extension)
 *
 * Like zdnn_get_rows but handles 4D tensors for batched operations.
 */
zdnn_status zdnn_get_rows_batched(const float *src,
                                   int64_t src_ne0,
                                   int64_t src_ne1,
                                   int64_t src_ne2,
                                   int64_t src_ne3,
                                   const int32_t *indices,
                                   int64_t idx_ne0,
                                   int64_t idx_ne1,
                                   int64_t idx_ne2,
                                   float *output) {
    (void)src;
    (void)src_ne0;
    (void)src_ne1;
    (void)src_ne2;
    (void)src_ne3;
    (void)indices;
    (void)idx_ne0;
    (void)idx_ne1;
    (void)idx_ne2;
    (void)output;
    /* TODO: Implement batched gather operation */
    return ZDNN_FUNC_RC_F000; /* Function not available */
}

/*
 * zdnn_rope - Rotary Position Embedding (zdnn-sim extension)
 *
 * Applies rotary position embeddings to input tensor.
 * This is a key operation for transformer attention mechanisms.
 */
zdnn_status zdnn_rope(const float *input,
                      const int32_t *positions,
                      int64_t ne0,
                      int64_t ne1,
                      int64_t ne2,
                      int64_t ne3,
                      int n_dims,
                      int mode,
                      float freq_base,
                      float freq_scale,
                      float *output) {
    (void)input;
    (void)positions;
    (void)ne0;
    (void)ne1;
    (void)ne2;
    (void)ne3;
    (void)n_dims;
    (void)mode;
    (void)freq_base;
    (void)freq_scale;
    (void)output;
    /* TODO: Implement RoPE - compute sin/cos and apply rotation */
    return ZDNN_FUNC_RC_F000; /* Function not available */
}
