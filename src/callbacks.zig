//! Progress and cancel hooks passed into the long-running pipelines so the GUI
//! can track progress and support cancellation.
//!
//! Kept in its own file so Gguf.zig and Safetensor.zig can import it without
//! creating a circular dependency with Convert.zig or types.zig.
//!
//! Each pipeline gets its own callback type rather than sharing one: convert
//! progress is tensor-shaped, capture progress is (prompt, step)-shaped, and
//! overloading a single struct would mean passing meaningless fields.

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

/// Called as an activation capture run advances. `prompt` counts from 0 and
/// `step` from 1 (a step that has completed), so the overall fraction done is
/// `(prompt * steps + step) / (prompts * steps)`. Invoked on the capture thread.
pub const CaptureProgressFn = *const fn (
    ctx: ?*anyopaque,
    prompt: u32,
    prompts: u32,
    step: u32,
    steps: u32,
) void;

/// Progress + cancel for `Calibrate.run`. Cancel is polled by TensorPencil
/// throughout a forward — between encoder layers, between DiT blocks, inside the
/// CPU GEMM — so a stop lands within a fraction of a step even on CPU, which is
/// what makes a slow capture interruptible from the GUI.
pub const CaptureCallbacks = struct {
    progress_fn: ?CaptureProgressFn = null,
    progress_ctx: ?*anyopaque = null,
    cancel_fn: ?CancelFn = null,
    cancel_ctx: ?*anyopaque = null,

    pub fn reportProgress(self: CaptureCallbacks, prompt: u32, prompts: u32, step: u32, steps: u32) void {
        if (self.progress_fn) |f| f(self.progress_ctx, prompt, prompts, step, steps);
    }

    pub fn isCancelled(self: CaptureCallbacks) bool {
        if (self.cancel_fn) |f| return f(self.cancel_ctx);
        return false;
    }
};
