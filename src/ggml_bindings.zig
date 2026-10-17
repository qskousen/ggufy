const c = @import("ggml_c");

// Re-export only what we need
pub const ggml_quantize_chunk = c.ggml_quantize_chunk;

// Block geometry, so the hand-written tables can be pinned against ggml's own.
pub const ggml_blck_size = c.ggml_blck_size;
pub const ggml_type_size = c.ggml_type_size;

// Which encoders abort without importance weights, asked of ggml rather than
// hardcoded: the list has changed before (iq1_m is commented out upstream).
pub const ggml_quantize_requires_imatrix = c.ggml_quantize_requires_imatrix;

pub const ggml_type         = c.ggml_type;
pub const enum_ggml_type    = c.enum_ggml_type;

// Scalar types
pub const ggml_fp16_t = c.ggml_fp16_t; // = uint16_t
pub const ggml_bf16_t = c.ggml_bf16_t; // = struct { uint16_t bits; }

// SIMD-optimized row conversion functions
pub const ggml_fp16_to_fp32_row = c.ggml_fp16_to_fp32_row;
pub const ggml_fp32_to_fp16_row = c.ggml_fp32_to_fp16_row;
pub const ggml_bf16_to_fp32_row = c.ggml_bf16_to_fp32_row;
pub const ggml_fp32_to_bf16_row = c.ggml_fp32_to_bf16_row;
pub const ggml_get_type_traits   = c.ggml_get_type_traits;