const c = @cImport({
    @cDefine("NDEBUG", "1");
    @cInclude("ggml.h");
    @cInclude("gguf.h");
});

// Re-export only what we need
pub const ggml_quantize_chunk = c.ggml_quantize_chunk;

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