//! ConvertCallbacks — progress and cancel hooks passed into the conversion
//! pipeline so the GUI can track per-tensor progress and support cancellation.
//!
//! Kept in its own file so Gguf.zig and Safetensor.zig can import it without
//! creating a circular dependency with Convert.zig or types.zig.

/// Called after each tensor is written.  Invoked on the convert thread.
/// `done` counts from 1;  `total` is the full (filtered) tensor count.
pub const ProgressFn = *const fn (
    ctx: ?*anyopaque,
    done: u32,
    total: u32,
    name: []const u8,
    src_type: []const u8,
    dst_type: []const u8,
    n_elements: u64,
) void;

/// Return true to cancel the conversion.  Invoked on the convert thread at
/// the start of each tensor write.  The caller cleans up the partial output.
pub const CancelFn = *const fn (ctx: ?*anyopaque) bool;

pub const ConvertCallbacks = struct {
    progress_fn: ?ProgressFn = null,
    progress_ctx: ?*anyopaque = null,
    cancel_fn: ?CancelFn = null,
    cancel_ctx: ?*anyopaque = null,

    pub fn reportProgress(
        self: ConvertCallbacks,
        done: u32,
        total: u32,
        name: []const u8,
        src_type: []const u8,
        dst_type: []const u8,
        n_elements: u64,
    ) void {
        if (self.progress_fn) |f|
            f(self.progress_ctx, done, total, name, src_type, dst_type, n_elements);
    }

    pub fn isCancelled(self: ConvertCallbacks) bool {
        if (self.cancel_fn) |f| return f(self.cancel_ctx);
        return false;
    }
};
