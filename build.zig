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

    switch (os) {
        .linux => {
            exe.root_module.addObjectFile(b.path("vendor/ggml/build/src/libggml.a"));
            exe.root_module.addObjectFile(b.path("vendor/ggml/build/src/libggml-base.a"));
            exe.root_module.addObjectFile(b.path("vendor/ggml/build/src/libggml-cpu.a"));
            exe.linkSystemLibrary("pthread");
            exe.linkSystemLibrary("m");
            exe.linkSystemLibrary("c++");
            exe.linkSystemLibrary("c++abi");
        },
        .macos => {
            exe.root_module.addObjectFile(b.path("vendor/ggml/build/src/libggml.a"));
            exe.root_module.addObjectFile(b.path("vendor/ggml/build/src/libggml-base.a"));
            exe.root_module.addObjectFile(b.path("vendor/ggml/build/src/libggml-cpu.a"));
            exe.linkSystemLibrary("c++");
            exe.linkSystemLibrary("c++abi");
        },
        .windows => {
            exe.root_module.addObjectFile(b.path("vendor/ggml/build/src/libggml.lib"));
            exe.root_module.addObjectFile(b.path("vendor/ggml/build/src/libggml-base.lib"));
            exe.root_module.addObjectFile(b.path("vendor/ggml/build/src/libggml-cpu.lib"));
        },
        else => {
            unreachable;
        }
    }

    const clap = b.dependency("clap", .{});
    exe.root_module.addImport("clap", clap.module("clap"));

    // disabling ggml direct building because zig's build of ggml is ~8x slower than clang, etc.
    // sadly this means we have to build and distribute the ggml libraries ourselves
    //ggmlc.link(exe);

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
