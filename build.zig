const std = @import("std");
const ggml = @import("build_ggml.zig");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

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

    const exe = b.addExecutable(.{
        .name = "ggufy",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "ggufy",  .module = mod },
                .{ .name = "ggml.h", .module = ggml_h_module },
            },
        }),
    });

    const clap = b.dependency("clap", .{});
    exe.root_module.addImport("clap", clap.module("clap"));

    ggml.link(b, exe, target, optimize);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    b.step("run", "Run the app").dependOn(&run_cmd.step);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mod })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = exe.root_module })).step);

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
}