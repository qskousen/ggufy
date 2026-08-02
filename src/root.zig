//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

pub const convert = @import("Convert.zig");
pub const dataTransform = @import("DataTransform.zig");
pub const gguf = @import("Gguf.zig");
pub const imageArch = @import("ImageArch.zig");
pub const tensorClusters = @import("TensorClusters.zig");
pub const safetensor = @import("Safetensor.zig");
pub const types = @import("types.zig");
pub const fileLoader = @import("FileLoader.zig");
pub const callbacks = @import("callbacks.zig");
/// The 1–100 sensitivity-score encoding shared by the two measurement levels that
/// emit a `sensitivities/*.json` and read by `Convert.calculateQuantizationLevel`.
pub const ladderScore = @import("LadderScore.zig");

/// Activation-aware quantization support (see ACTIVATION_AWARE_PLAN.md). Needs the
/// TensorPencil umbrella (inference), unlike everything above.
pub const activations = @import("Activations.zig");
pub const calibrationCache = @import("CalibrationCache.zig");
pub const calibrate = @import("Calibrate.zig");
pub const sensitivity = @import("Sensitivity.zig");
pub const imatrix = @import("Imatrix.zig");
pub const equalize = @import("Equalize.zig");
pub const gptq = @import("Gptq.zig");
pub const gptqPlan = @import("GptqPlan.zig");
pub const verdict = @import("Verdict.zig");
pub const divergence = @import("Divergence.zig");
/// Weight-space dispersion screen — the free half of the routing question, and the
/// only cross-architecture measurement available before TensorPencil can run one.
pub const heterogeneity = @import("Heterogeneity.zig");
