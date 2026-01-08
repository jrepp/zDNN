/*
 * zdnn_sim_stubs.c - Extension function implementations for llama.cpp
 *
 * These functions are used by llama.cpp's zdnn backend but are not part of
 * the IBM zDNN API. They provide CPU implementations optimized for z/Architecture.
 *
 * Target: IBM z16 with NNPA
 *
 * Functions:
 *   - zdnn_rmsnorm: RMS Normalization (LLaMA-style)
 *   - zdnn_get_rows: Embedding table lookup (gather)
 *   - zdnn_get_rows_batched: Batched embedding lookup
 *   - zdnn_rope: Rotary Position Embedding
 */

#include "zdnn.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

/* ============================================================================
 * zdnn_get_rows - Extract rows using indices (embedding lookup)
 *
 * This is the fundamental operation for token embedding lookups.
 * Given a 2D embedding matrix and an array of token indices,
 * copies the selected rows to the output.
 *
 * Parameters:
 *   src         - Source embedding matrix [src_ne1 x src_ne0] (row-major)
 *   src_ne0     - Embedding dimension (row width)
 *   src_ne1     - Vocabulary size (number of rows)
 *   indices     - Array of row indices to gather
 *   num_indices - Number of indices (sequence length)
 *   output      - Output buffer [num_indices x src_ne0]
 *
 * Returns: ZDNN_OK on success
 * ============================================================================
 */
zdnn_status zdnn_get_rows(const float *src,
                          int64_t src_ne0,
                          int64_t src_ne1,
                          const int32_t *indices,
                          int64_t num_indices,
                          float *output) {
    /* Validate inputs */
    if (!src || !indices || !output) {
        return ZDNN_INVALID_BUFFER;
    }
    if (src_ne0 <= 0 || src_ne1 <= 0 || num_indices <= 0) {
        return ZDNN_INVALID_SHAPE;
    }

    const size_t row_bytes = (size_t)src_ne0 * sizeof(float);

    /* Gather rows - optimized for sequential output access */
    for (int64_t i = 0; i < num_indices; i++) {
        int32_t row_idx = indices[i];

        /* Bounds check */
        if (row_idx < 0 || row_idx >= src_ne1) {
            return ZDNN_INVALID_STATE;
        }

        const float *src_row = src + (int64_t)row_idx * src_ne0;
        float *dst_row = output + i * src_ne0;

        memcpy(dst_row, src_row, row_bytes);
    }

    return ZDNN_OK;
}

/* ============================================================================
 * zdnn_get_rows_batched - Batched row extraction for 4D tensors
 *
 * Extends zdnn_get_rows to handle batched operations with 4D tensors.
 * Used when processing multiple sequences with different embeddings.
 *
 * Parameters:
 *   src         - Source tensor [src_ne3 x src_ne2 x src_ne1 x src_ne0]
 *   src_ne0-3   - Source tensor dimensions
 *   indices     - Index tensor [idx_ne2 x idx_ne1 x idx_ne0]
 *   idx_ne0-2   - Index tensor dimensions
 *   output      - Output tensor
 *
 * Returns: ZDNN_OK on success
 * ============================================================================
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
    /* Validate inputs */
    if (!src || !indices || !output) {
        return ZDNN_INVALID_BUFFER;
    }
    if (src_ne0 <= 0 || src_ne1 <= 0 || src_ne2 <= 0 || src_ne3 <= 0) {
        return ZDNN_INVALID_SHAPE;
    }
    if (idx_ne0 <= 0 || idx_ne1 <= 0 || idx_ne2 <= 0) {
        return ZDNN_INVALID_SHAPE;
    }

    const size_t row_bytes = (size_t)src_ne0 * sizeof(float);
    const int64_t src_stride1 = src_ne0;                          /* stride between rows */
    const int64_t src_stride2 = src_ne0 * src_ne1;                /* stride between 2D slices */
    const int64_t src_stride3 = src_ne0 * src_ne1 * src_ne2;      /* stride between 3D blocks */

    const int64_t idx_stride1 = idx_ne0;                          /* stride between index rows */
    const int64_t idx_stride2 = idx_ne0 * idx_ne1;                /* stride between index slices */

    const int64_t out_stride1 = src_ne0;                          /* output row stride */
    const int64_t out_stride2 = src_ne0 * idx_ne0;                /* output slice stride */
    const int64_t out_stride3 = src_ne0 * idx_ne0 * idx_ne1;      /* output block stride */

    /* Iterate over batch dimensions */
    for (int64_t i3 = 0; i3 < idx_ne2; i3++) {
        /* Source batch index - broadcast if src_ne3 == 1 */
        int64_t s3 = (src_ne3 == 1) ? 0 : i3;
        if (s3 >= src_ne3) s3 = src_ne3 - 1;

        for (int64_t i2 = 0; i2 < idx_ne1; i2++) {
            /* Source slice index - broadcast if src_ne2 == 1 */
            int64_t s2 = (src_ne2 == 1) ? 0 : i2;
            if (s2 >= src_ne2) s2 = src_ne2 - 1;

            const float *src_slice = src + s3 * src_stride3 + s2 * src_stride2;
            const int32_t *idx_slice = indices + i3 * idx_stride2 + i2 * idx_stride1;
            float *out_slice = output + i3 * out_stride3 + i2 * out_stride2;

            /* Gather rows for this slice */
            for (int64_t i1 = 0; i1 < idx_ne0; i1++) {
                int32_t row_idx = idx_slice[i1];

                /* Bounds check */
                if (row_idx < 0 || row_idx >= src_ne1) {
                    return ZDNN_INVALID_STATE;
                }

                const float *src_row = src_slice + (int64_t)row_idx * src_stride1;
                float *dst_row = out_slice + i1 * out_stride1;

                memcpy(dst_row, src_row, row_bytes);
            }
        }
    }

    return ZDNN_OK;
}

/* ============================================================================
 * zdnn_rmsnorm - RMS Normalization (CPU Reference Implementation)
 *
 * Computes: output = (input / rms(input)) * weight
 * where rms(x) = sqrt(mean(x^2) + epsilon)
 *
 * This is the normalization layer used in LLaMA, Mistral, and modern LLMs.
 * It's simpler than LayerNorm (no mean subtraction).
 *
 * Parameters:
 *   input   - Input tensor (must be transformed/stickified)
 *   weight  - Per-element scaling weights (gamma), can be NULL
 *   epsilon - Small constant for numerical stability (typically 1e-5 or 1e-6)
 *   output  - Output tensor (same shape as input)
 *
 * This is a CPU reference implementation that:
 *   1. Extracts raw float data from stickified tensors
 *   2. Computes RMSNorm in plain C
 *   3. Stores result back to stickified format
 *
 * Returns: ZDNN_OK on success
 * ============================================================================
 */
zdnn_status zdnn_rmsnorm(const zdnn_ztensor *input,
                         const zdnn_ztensor *weight,
                         float epsilon,
                         zdnn_ztensor *output) {
    zdnn_status status;

    /* Validate inputs */
    if (!input || !output) {
        return ZDNN_INVALID_BUFFER;
    }
    if (!input->buffer || !output->buffer) {
        return ZDNN_INVALID_BUFFER;
    }
    if (!input->pre_transformed_desc || !output->pre_transformed_desc) {
        return ZDNN_INVALID_BUFFER;
    }

    /* Get tensor dimensions from pre-transformed descriptor */
    const zdnn_tensor_desc *desc = input->pre_transformed_desc;
    uint32_t dim1 = desc->dim1;  /* Innermost dimension (normalization axis) */
    uint32_t dim2 = desc->dim2 > 0 ? desc->dim2 : 1;
    uint32_t dim3 = desc->dim3 > 0 ? desc->dim3 : 1;
    uint32_t dim4 = desc->dim4 > 0 ? desc->dim4 : 1;

    if (dim1 == 0) {
        return ZDNN_INVALID_SHAPE;
    }

    uint64_t total_elements = (uint64_t)dim4 * dim3 * dim2 * dim1;
    uint64_t outer_count = (uint64_t)dim4 * dim3 * dim2;

    /* Allocate temporary buffers for raw float data */
    float *in_raw = (float *)malloc(total_elements * sizeof(float));
    float *out_raw = (float *)malloc(total_elements * sizeof(float));
    float *w_raw = NULL;

    if (!in_raw || !out_raw) {
        free(in_raw);
        free(out_raw);
        return ZDNN_ALLOCATION_FAILURE;
    }

    /* Extract raw data from input tensor */
    status = zdnn_transform_origtensor(input, in_raw);
    if (status != ZDNN_OK) {
        free(in_raw);
        free(out_raw);
        return status;
    }

    /* Extract weight data if provided */
    if (weight && weight->buffer) {
        w_raw = (float *)malloc(dim1 * sizeof(float));
        if (!w_raw) {
            free(in_raw);
            free(out_raw);
            return ZDNN_ALLOCATION_FAILURE;
        }
        status = zdnn_transform_origtensor(weight, w_raw);
        if (status != ZDNN_OK) {
            free(in_raw);
            free(out_raw);
            free(w_raw);
            return status;
        }
    }

    /* Compute RMSNorm for each slice along dim1 */
    for (uint64_t idx = 0; idx < outer_count; idx++) {
        const float *slice_in = in_raw + idx * dim1;
        float *slice_out = out_raw + idx * dim1;

        /* Compute mean of squares */
        float sum_sq = 0.0f;
        for (uint32_t i = 0; i < dim1; i++) {
            sum_sq += slice_in[i] * slice_in[i];
        }
        float mean_sq = sum_sq / (float)dim1;

        /* Compute inverse RMS: 1 / sqrt(mean(x^2) + epsilon) */
        float inv_rms = 1.0f / sqrtf(mean_sq + epsilon);

        /* Normalize and optionally apply weight */
        if (w_raw) {
            for (uint32_t i = 0; i < dim1; i++) {
                slice_out[i] = slice_in[i] * inv_rms * w_raw[i];
            }
        } else {
            for (uint32_t i = 0; i < dim1; i++) {
                slice_out[i] = slice_in[i] * inv_rms;
            }
        }
    }

    /* Transform result back to stickified format */
    status = zdnn_transform_ztensor(output, out_raw);

    /* Cleanup */
    free(in_raw);
    free(out_raw);
    free(w_raw);

    return status;
}

/* ============================================================================
 * zdnn_rope - Rotary Position Embedding
 *
 * Applies rotary position embeddings to input tensor.
 * This encodes position information directly into the attention computation.
 *
 * Algorithm:
 *   For dimension pairs (i, i+1) at position p:
 *     theta = p * freq_base^(-2i/n_dims) * freq_scale
 *     out[i]   = in[i] * cos(theta) - in[i+1] * sin(theta)
 *     out[i+1] = in[i] * sin(theta) + in[i+1] * cos(theta)
 *
 * Parameters:
 *   input      - Input tensor data [ne3 x ne2 x ne1 x ne0]
 *   positions  - Position indices for each token
 *   ne0-ne3    - Tensor dimensions
 *   n_dims     - Number of dimensions to apply rotation to
 *   mode       - RoPE variant (0=standard, 2=GPT-NeoX)
 *   freq_base  - Base frequency, typically 10000.0
 *   freq_scale - Scaling factor for extended context (typically 1.0)
 *   output     - Output tensor data
 *
 * Returns: ZDNN_OK on success
 * ============================================================================
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
    /* Validate inputs */
    if (!input || !positions || !output) {
        return ZDNN_INVALID_BUFFER;
    }
    if (ne0 <= 0 || ne1 <= 0 || ne2 <= 0 || ne3 <= 0) {
        return ZDNN_INVALID_SHAPE;
    }
    if (n_dims <= 0 || n_dims > ne0) {
        return ZDNN_INVALID_SHAPE;
    }

    /* Only support standard RoPE (mode 0) and GPT-NeoX (mode 2) */
    if (mode != 0 && mode != 2) {
        return ZDNN_FUNC_RC_F000;
    }

    const float theta_scale = freq_scale;
    const int half_dims = n_dims / 2;

    /* Iterate over all positions */
    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            /* Get position for this token */
            int32_t pos = positions[i2];

            for (int64_t i1 = 0; i1 < ne1; i1++) {
                const float *src = input + i3 * ne2 * ne1 * ne0 + i2 * ne1 * ne0 + i1 * ne0;
                float *dst = output + i3 * ne2 * ne1 * ne0 + i2 * ne1 * ne0 + i1 * ne0;

                if (mode == 0) {
                    /* Standard RoPE: pairs are (0,1), (2,3), (4,5), ... */
                    for (int i = 0; i < half_dims; i++) {
                        /* Compute rotation angle */
                        float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)n_dims);
                        float theta = (float)pos * freq * theta_scale;
                        float cos_theta = cosf(theta);
                        float sin_theta = sinf(theta);

                        int idx0 = 2 * i;
                        int idx1 = 2 * i + 1;

                        float x0 = src[idx0];
                        float x1 = src[idx1];

                        dst[idx0] = x0 * cos_theta - x1 * sin_theta;
                        dst[idx1] = x0 * sin_theta + x1 * cos_theta;
                    }
                } else if (mode == 2) {
                    /* GPT-NeoX: pairs are (0, n/2), (1, n/2+1), ... */
                    for (int i = 0; i < half_dims; i++) {
                        float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)n_dims);
                        float theta = (float)pos * freq * theta_scale;
                        float cos_theta = cosf(theta);
                        float sin_theta = sinf(theta);

                        int idx0 = i;
                        int idx1 = i + half_dims;

                        float x0 = src[idx0];
                        float x1 = src[idx1];

                        dst[idx0] = x0 * cos_theta - x1 * sin_theta;
                        dst[idx1] = x0 * sin_theta + x1 * cos_theta;
                    }
                }

                /* Copy remaining dimensions unchanged */
                for (int64_t i0 = n_dims; i0 < ne0; i0++) {
                    dst[i0] = src[i0];
                }
            }
        }
    }

    return ZDNN_OK;
}
