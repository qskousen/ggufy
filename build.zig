const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const ggml_h_module = b.addTranslateC(.{
        .root_source_file = b.path("vendor/ggml/include/ggml.h"),
        .target = target,
        .optimize = optimize,
    }).createModule();
    const imports: []const std.Build.Module.Import = &.{
        .{
            .name = "ggml.h",
            .module = ggml_h_module,
        },
    };

    const mod = b.addModule("ggufy", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .imports = imports,
    });

    const exe = b.addExecutable(.{
        .name = "ggufy",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "ggufy", .module = mod },
                .{ .name = "ggml.h", .module = ggml_h_module},
            },
        }),
    });

    // GGML libraries must already be compiled!
    const os = target.result.os.tag;
    const arch = target.result.cpu.arch;

    switch (os) {
        .linux => {
            exe.root_module.addObjectFile(b.path("artifacts/ggml-linux-x86_64/libggml.a"));
            exe.root_module.addObjectFile(b.path("artifacts/ggml-linux-x86_64/libggml-base.a"));
            exe.root_module.addObjectFile(b.path("artifacts/ggml-linux-x86_64/libggml-cpu.a"));
            exe.linkSystemLibrary("pthread");
            exe.linkSystemLibrary("m");
            exe.linkSystemLibrary("c++");
            exe.linkSystemLibrary("c++abi");
        },
        .macos => {
            switch (arch) {
                .aarch64 => {
                    exe.root_module.addObjectFile(b.path("artifacts/ggml-macos-arm64/libggml.a"));
                    exe.root_module.addObjectFile(b.path("artifacts/ggml-macos-arm64/libggml-base.a"));
                    exe.root_module.addObjectFile(b.path("artifacts/ggml-macos-arm64/libggml-cpu.a"));
                },
                .x86_64 => {
                    exe.root_module.addObjectFile(b.path("artifacts/ggml-macos-x86_64//libggml.a"));
                    exe.root_module.addObjectFile(b.path("artifacts/ggml-macos-x86_64/libggml-base.a"));
                    exe.root_module.addObjectFile(b.path("artifacts/ggml-macos-x86_64/libggml-cpu.a"));
                },
                else => {
                    unreachable;
                }
            }
            exe.linkSystemLibrary("c++");
            exe.linkSystemLibrary("c++abi");
        },
        .windows => {
            exe.root_module.addObjectFile(b.path("artifacts/ggml-windows-x86_64/ggml.lib"));
            exe.root_module.addObjectFile(b.path("artifacts/ggml-windows-x86_64/ggml-base.lib"));
            exe.root_module.addObjectFile(b.path("artifacts/ggml-windows-x86_64/ggml-cpu.lib"));
            exe.linkSystemLibrary("kernel32");
            exe.linkSystemLibrary("user32");
            exe.linkSystemLibrary("bcrypt");
            exe.linkSystemLibrary("advapi32");
        },
        else => {
            unreachable;
        }
    }

    const clap = b.dependency("clap", .{});
    exe.root_module.addImport("clap", clap.module("clap"));

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });

    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
}
