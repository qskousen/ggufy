const c = @cImport({
    @cDefine("NDEBUG", "1");
    @cInclude("ggml.h");
    @cInclude("gguf.h");
});

// Re-export only what we need
pub const ggml_quantize_chunk = c.ggml_quantize_chunk;

pub const ggml_type         = c.ggml_type;
pub const enum_ggml_type    = c.enum_ggml_type;