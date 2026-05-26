const std = @import("std");
const ggml = @import("build_ggml.zig");
const cimgui = @import("cimgui");

pub fn build(b: *std.Build) void {
    const git_version = get_git_version(b.allocator, b.graph.io) catch "dev";

    const options = b.addOptions();
    options.addOption([]const u8, "version", git_version);

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // --- Shared modules ---

    const ggml_h_module = b.addModule("ggml.h", .{
        .root_source_file = b.path("src/ggml_bindings.zig"),
        .target = target,
        .optimize = optimize,
    });
    ggml_h_module.addIncludePath(b.path("vendor/ggml/include"));

    const mod = b.addModule("ggufy", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "ggml.h", .module = ggml_h_module },
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
                .{ .name = "ggml.h", .module = ggml_h_module },
            },
        }),
    });

    const clap = b.dependency("clap", .{
        .target = target,
        .optimize = optimize,
    });
    cli.root_module.addImport("clap", clap.module("clap"));
    cli.root_module.addOptions("build_options", options);

    ggml.link(b, cli, target, optimize);

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
                .{ .name = "ggml.h", .module = ggml_h_module },
            },
        }),
    });

    ggml.link(b, gui, target, optimize);

    const dvui_dep = b.dependency("dvui", .{ .target = target, .optimize = optimize, .backend = .sdl3, .@"tree-sitter" = false });
    gui.root_module.addImport("dvui", dvui_dep.module("dvui_sdl3"));
    gui.root_module.addImport("backend", dvui_dep.module("sdl3"));
    gui.root_module.addOptions("build_options", options);

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
                .{ .name = "ggml.h", .module = ggml_h_module },
            },
        }),
    });
    ggml.link(b, bench, target, .ReleaseFast);
    const run_bench = b.addRunArtifact(bench);
    b.step("bench", "Run F8 benchmarks").dependOn(&run_bench.step);

    const bench_eff = b.addExecutable(.{
        .name = "bench-efficiency",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench_efficiency.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "ggml.h", .module = ggml_h_module },
            },
        }),
    });
    ggml.link(b, bench_eff, target, .ReleaseFast);
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
                .{ .name = "ggml.h", .module = ggml_h_module },
            },
        }),
    });
    ggml.link(b, arch_detect_test, target, optimize);
    test_step.dependOn(&b.addRunArtifact(arch_detect_test).step);

    const data_transform_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/DataTransform.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "ggml.h", .module = ggml_h_module },
            },
        }),
    });
    ggml.link(b, data_transform_test, target, optimize);
    test_step.dependOn(&b.addRunArtifact(data_transform_test).step);

    const scaled_quant_test = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/ScaledQuant.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "ggml.h", .module = ggml_h_module },
            },
        })
    });
    ggml.link(b, scaled_quant_test, target, optimize);
    test_step.dependOn(&b.addRunArtifact(scaled_quant_test).step);
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