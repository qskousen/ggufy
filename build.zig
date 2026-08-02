const std = @import("std");
const cimgui = @import("cimgui");

pub fn build(b: *std.Build) void {
    const git_version = get_git_version(b.allocator, b.graph.io) catch "dev";

    const options = b.addOptions();
    options.addOption([]const u8, "version", git_version);
    // One shared module instance for the generated options file; adding it via multiple
    // addOptions() calls would root the same file in several modules and fail to compile.
    const options_mod = options.createModule();

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // --- Shared modules ---

    // ggml comes from TensorPencil, not a local vendored copy: quantize/dequantize
    // kernels are inference-side concerns and live there (see ACTIVATION_AWARE_PLAN.md
    // §2). `tp_core` is the bottom layer — dtypes, ggml quant/dequant, safetensors and
    // GGUF parsing — with no GPU, models or pipeline, so it stays a cheap dependency.
    // It links the ggml static lib (always built ReleaseFast) into every artifact that
    // imports it.
    const tp_dep = b.dependency("TensorPencil", .{
        .target = target,
        .optimize = optimize,
    });
    const tp_core = tp_dep.module("tp_core");

    // The umbrella module adds what `tp_core` deliberately leaves out: the CPU/GPU
    // GEMMs, the architectures, and the diffusion pipeline. Activation capture and
    // sensitivity measurement are user-facing features that run from both the CLI and
    // the GUI, and CPU-only would make them unusably slow (minutes on a GPU vs many
    // hours), so the backends have to be reachable from the shipping binaries — see
    // ACTIVATION_AWARE_PLAN.md §2.4 and §6.
    //
    // This does NOT add a hard runtime dependency on a GPU: TensorPencil `dlopen`s
    // libvulkan / libcuda, embeds its SPIR-V kernels at compile time, and falls back
    // to CPU with a log line when device init fails. It does cost build time and
    // binary size, both measured and recorded in the plan.
    const tp = tp_dep.module("TensorPencil");

    const mod = b.addModule("ggufy", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "tp_core", .module = tp_core },
            // The inference-backed measurement code (activation capture, sensitivity)
            // lives in this library so the CLI and the GUI share one implementation.
            .{ .name = "TensorPencil", .module = tp },
            .{ .name = "build_options", .module = options_mod },
        },
    });

    // --- CLI ---

    const cli = b.addExecutable(.{
        .name = "ggufy",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "ggufy",  .module = mod },
                .{ .name = "tp_core", .module = tp_core },
            },
        }),
    });

    const clap = b.dependency("clap", .{
        .target = target,
        .optimize = optimize,
    });
    cli.root_module.addImport("clap", clap.module("clap"));
    cli.root_module.addImport("build_options", options_mod);


    const cli_install = b.addInstallArtifact(cli, .{});
    const cli_step = b.step("cli", "Build the CLI");
    cli_step.dependOn(&cli_install.step);

    const run_cmd = b.addRunArtifact(cli);
    run_cmd.step.dependOn(&cli_install.step);
    if (b.args) |args| run_cmd.addArgs(args);
    b.step("run", "Run the CLI").dependOn(&run_cmd.step);

    // --- GUI ---

    const gui = b.addExecutable(.{
        .name = "ggufy-gui",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/gui/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "ggufy",  .module = mod },
                .{ .name = "tp_core", .module = tp_core },
            },
        }),
    });


    const dvui_dep = b.dependency("dvui", .{ .target = target, .optimize = optimize, .backend = .sdl3, .@"tree-sitter" = false });
    gui.root_module.addImport("dvui", dvui_dep.module("dvui_sdl3"));
    gui.root_module.addImport("backend", dvui_dep.module("sdl3"));
    gui.root_module.addImport("build_options", options_mod);

    // When cross-compiling for macOS with an explicit sysroot (e.g. CI),
    // Zig does not automatically add the SDK's framework search path, so we must wire it up ourselves.
    if (target.result.os.tag == .macos) {
        if (b.sysroot) |sysroot| {
            const fw_path = b.pathJoin(&.{ sysroot, "System/Library/Frameworks" });
            gui.root_module.addFrameworkPath(.{ .cwd_relative = fw_path });
        }
    }

    const gui_install = b.addInstallArtifact(gui, .{});
    const gui_step = b.step("gui", "Build the GUI");
    gui_step.dependOn(&gui_install.step);

    const run_gui_cmd = b.addRunArtifact(gui);
    run_gui_cmd.step.dependOn(&gui_install.step);
    if (b.args) |args| run_gui_cmd.addArgs(args);
    b.step("run-gui", "Run the GUI").dependOn(&run_gui_cmd.step);

    // --- Default: build both ---

    b.getInstallStep().dependOn(&cli_install.step);
    b.getInstallStep().dependOn(&gui_install.step);

    // --- Benchmarks ---

    const bench = b.addExecutable(.{
        .name = "bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "tp_core", .module = tp_core },
            },
        }),
    });
    const run_bench = b.addRunArtifact(bench);
    b.step("bench", "Run F8 benchmarks").dependOn(&run_bench.step);

    // --- Precision report ---

    const precision = b.addExecutable(.{
        .name = "precision",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/precision_main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "tp_core", .module = tp_core },
            },
        }),
    });
    const run_precision = b.addRunArtifact(precision);
    if (b.args) |args| run_precision.addArgs(args);
    b.step("precision", "Run the quantization precision report").dependOn(&run_precision.step);

    const bench_eff = b.addExecutable(.{
        .name = "bench-efficiency",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench_efficiency.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "tp_core", .module = tp_core },
            },
        }),
    });
    const run_bench_eff = b.addRunArtifact(bench_eff);
    b.step("bench-efficiency", "Run quantization efficiency benchmarks").dependOn(&run_bench_eff.step);

    // --- Tests ---

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mod })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cli.root_module })).step);

    const arch_detect_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/arch_detection_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "tp_core", .module = tp_core },
            },
        }),
    });
    test_step.dependOn(&b.addRunArtifact(arch_detect_test).step);

    const data_transform_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/DataTransform.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "tp_core", .module = tp_core },
            },
        }),
    });
    test_step.dependOn(&b.addRunArtifact(data_transform_test).step);

    const tensor_clusters_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/TensorClusters.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "tp_core", .module = tp_core },
            },
        })
    });
    test_step.dependOn(&b.addRunArtifact(tensor_clusters_test).step);

    // Activation capture: runs real GEMMs through TensorPencil's matmul probe, so it
    // needs the umbrella (not just tp_core).
    const activations_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/Activations.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "TensorPencil", .module = tp },
            },
        }),
    });
    test_step.dependOn(&b.addRunArtifact(activations_test).step);

    // Calibration cache: reads/writes the safetensors container and drives the same
    // matmul probe, so it needs the umbrella too.
    const calib_cache_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/CalibrationCache.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "TensorPencil", .module = tp },
            },
        }),
    });
    test_step.dependOn(&b.addRunArtifact(calib_cache_test).step);

    // Capture driver: the `run` path needs real checkpoints and is exercised by
    // hand, but the prompt-set parsing and the model identity hash are testable
    // here and both decide whether a cache is usable.
    const calibrate_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/Calibrate.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "TensorPencil", .module = tp },
            },
        }),
    });
    test_step.dependOn(&b.addRunArtifact(calibrate_test).step);

    // Level-1 sensitivity harness: needs the umbrella (it runs GEMMs) and
    // build_options (it pulls in Convert.zig's provenance stamp transitively).
    const sensitivity_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/Sensitivity.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "TensorPencil", .module = tp },
                .{ .name = "tp_core", .module = tp_core },
                .{ .name = "build_options", .module = options_mod },
            },
        }),
    });
    test_step.dependOn(&b.addRunArtifact(sensitivity_test).step);

    // Activation-aware weighting: builds ggml's imatrix from a calibration cache,
    // so it needs the umbrella (the cache reader and the probe both live there).
    const imatrix_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/Imatrix.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "TensorPencil", .module = tp },
                .{ .name = "tp_core", .module = tp_core },
            },
        }),
    });
    test_step.dependOn(&b.addRunArtifact(imatrix_test).step);

    // §8B equalization. Pure policy — it only needs the umbrella because its tests
    // reuse Imatrix's floor constant, which reaches the cache reader.
    const equalize_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/Equalize.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "TensorPencil", .module = tp },
                .{ .name = "tp_core", .module = tp_core },
            },
        }),
    });
    test_step.dependOn(&b.addRunArtifact(equalize_test).step);

    // §8C GPTQ. Pure numerics, but its tests pin the sweep against the ComfyUI
    // int4/int8 quantizers via the precision harness, which reaches ggml.
    const gptq_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/Gptq.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "TensorPencil", .module = tp },
                .{ .name = "tp_core", .module = tp_core },
            },
        }),
    });
    test_step.dependOn(&b.addRunArtifact(gptq_test).step);

    // §8C convert-side policy: reads activation rows out of a calibration cache,
    // so it needs the umbrella like the other cache consumers.
    const gptq_plan_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/GptqPlan.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "TensorPencil", .module = tp },
                .{ .name = "tp_core", .module = tp_core },
            },
        }),
    });
    test_step.dependOn(&b.addRunArtifact(gptq_plan_test).step);

    // Image verdict: decodes PNGs and computes metrics through TensorPencil.
    const verdict_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/Verdict.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "TensorPencil", .module = tp },
            },
        }),
    });
    test_step.dependOn(&b.addRunArtifact(verdict_test).step);

    // Level 2 divergence: drives TensorPencil's pipeline stages, so it needs the
    // umbrella. (The measurement itself needs checkpoints and self-skips without
    // them; these tests cover the metric arithmetic and the option guards.)
    const divergence_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/Divergence.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "TensorPencil", .module = tp },
                // The per-tensor arm quantizes through `precision_harness`, which
                // reaches ggml via tp_core.
                .{ .name = "tp_core", .module = tp_core },
            },
        }),
    });
    test_step.dependOn(&b.addRunArtifact(divergence_test).step);

    // The 1–100 sensitivity-score encoding, shared by both measurement levels and
    // read by Convert.calculateQuantizationLevel. Pure arithmetic plus ImageArch's
    // name rules, so it needs no dependency at all.
    const ladder_score_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/LadderScore.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_step.dependOn(&b.addRunArtifact(ladder_score_test).step);

    // Weight-space heterogeneity screen: quantizes through precision_harness (ggml via
    // tp_core) and reads checkpoints through the umbrella's safetensors.
    const heterogeneity_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/Heterogeneity.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "TensorPencil", .module = tp },
                .{ .name = "tp_core", .module = tp_core },
            },
        }),
    });
    test_step.dependOn(&b.addRunArtifact(heterogeneity_test).step);

    const convert_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/Convert.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "tp_core", .module = tp_core },
                // Convert.zig now reaches a calibration cache (--calib), which is
                // read through TensorPencil — so this standalone module needs the
                // umbrella as well as tp_core.
                .{ .name = "TensorPencil", .module = tp },
                // Convert.zig imports build_options (stampConverterProvenance uses the version
                // string); the standalone test module must provide it too, or a test that
                // exercises the write path fails to compile with "no module named build_options".
                .{ .name = "build_options", .module = options_mod },
            },
        }),
    });
    test_step.dependOn(&b.addRunArtifact(convert_test).step);

    const precision_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/precision_harness.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "tp_core", .module = tp_core },
            },
        }),
    });
    test_step.dependOn(&b.addRunArtifact(precision_test).step);

    const precision_metrics_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/PrecisionMetrics.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_step.dependOn(&b.addRunArtifact(precision_metrics_test).step);
}

fn get_git_version(allocator: std.mem.Allocator, io: std.Io) ![]const u8 {
    const version_result = try std.process.run(allocator, io, .{ .argv = &.{ "git", "describe", "--tags", "--always" } });
    defer allocator.free(version_result.stdout);
    defer allocator.free(version_result.stderr);

    if (version_result.stdout.len == 0) return error.GitDescribeFailed;

    const status_result = try std.process.run(allocator, io, .{ .argv = &.{ "git", "status", "--porcelain" } });
    defer allocator.free(status_result.stdout);
    defer allocator.free(status_result.stderr);

    const trimmed_version = std.mem.trimEnd(u8, version_result.stdout, "\n");

    if (status_result.stdout.len > 0) {
        return try std.fmt.allocPrint(allocator, "{s}(DIRTY)", .{trimmed_version});
    }
    return try allocator.dupe(u8, trimmed_version);
}