const std = @import("std");
const Builder = std.Build;
const Target = std.Build.ResolvedTarget;
const OptimizeMode = std.builtin.OptimizeMode;
const CompileStep = std.Build.Step.Compile;
const LazyPath = std.Build.LazyPath;
const Module = std.Build.Module;

pub const Options = struct {
    target: Target,
    optimize: OptimizeMode,
    shared: bool, // static or shared lib
    build_number: usize = 0, // number that will be writen in build info
    metal_ndebug: bool = false,
    metal_use_bf16: bool = false,
};

// Build context
pub const Context = struct {
    b: *Builder,
    options: Options,
    build_info: *CompileStep,
    path_prefix: []const u8 = "",
    lib: ?*CompileStep = null,

    pub fn init(b: *Builder, op: Options) Context {
        const path_prefix = b.pathJoin(&.{ thisPath(), "/vendor/ggml" });
        const zig_version = @import("builtin").zig_version_string;
        const commit_hash = std.process.Child.run(
            .{ .allocator = b.allocator, .argv = &.{ "git", "rev-parse", "HEAD" } },
        ) catch |err| {
            std.log.err("Can't get git commit hash! err: {}", .{err});
            unreachable;
        };

        const build_info_zig = true; // use cpp or zig file for build-info
        const build_info_path = b.pathJoin(&.{ "common", "build-info." ++ if (build_info_zig) "zig" else "cpp" });
        const build_info = b.fmt(if (build_info_zig)
            \\pub export var LLAMA_BUILD_NUMBER : c_int = {};
            \\pub export var LLAMA_COMMIT = "{s}";
            \\pub export var LLAMA_COMPILER = "Zig {s}";
            \\pub export var LLAMA_BUILD_TARGET = "{s}_{s}";
            \\
        else
            \\int LLAMA_BUILD_NUMBER = {};
            \\char const *LLAMA_COMMIT = "{s}";
            \\char const *LLAMA_COMPILER = "Zig {s}";
            \\char const *LLAMA_BUILD_TARGET = "{s}_{s}";
            \\
        , .{ op.build_number, commit_hash.stdout[0 .. commit_hash.stdout.len - 1], zig_version, op.target.result.zigTriple(b.allocator) catch unreachable, @tagName(op.optimize) });

        const root_module = b.createModule(.{
            .root_source_file = b.addWriteFiles().add(build_info_path, build_info),
            .target = op.target,
            .optimize = op.optimize,
        });
        return .{
            .b = b,
            .options = op,
            .path_prefix = path_prefix,
            .build_info = b.addObject(.{ .name = "llama-build-info", .root_module = root_module }),
        };
    }

    /// just builds everything needed and links it to your target
    pub fn link(ctx: *Context, comp: *CompileStep) void {
        const lib = ctx.library();
        comp.linkLibrary(lib);
        if (ctx.options.shared) ctx.b.installArtifact(lib);
    }

    pub fn library(ctx: *Context) *CompileStep {
        if (ctx.lib) |l| return l;

        const linkage: std.builtin.LinkMode = if (ctx.options.shared) .dynamic else .static;

        const root_module = ctx.b.createModule(.{
            .target = ctx.options.target,
            .optimize = ctx.options.optimize,
        });

        const lib = ctx.b.addLibrary(.{
            .root_module = root_module,
            .name = "ggml",
            .linkage = linkage,
        });

        if (ctx.options.shared) {
            lib.root_module.addCMacro("GGML_SHARED", "");
            lib.root_module.addCMacro("GGML_BUILD", "");
            lib.root_module.addCMacro("GGML_COMMIT", "");
            lib.root_module.addCMacro("GGML_VERSION", "");
        }

        ctx.addBuildInfo(lib);
        ctx.addGgml(lib);

        if (ctx.options.target.result.abi != .msvc)
            lib.root_module.addCMacro("_GNU_SOURCE", "");

        ctx.lib = lib;
        return lib;
    }

    /// link everything directly to target
    pub fn addAll(ctx: *Context, compile: *CompileStep) void {
        ctx.addBuildInfo(compile);
        ctx.addGgml(compile);
        ctx.addLLama(compile);
    }

    /// zig module with translated headers
    pub fn moduleLlama(ctx: *Context) *Module {
        const tc = ctx.b.addTranslateC(.{
            .root_source_file = ctx.includePath("llama.h"),
            .target = ctx.options.target,
            .optimize = ctx.options.optimize,
        });
        if (ctx.options.shared) tcDefineCMacro(tc, "LLAMA_SHARED", null);
        tc.addSystemIncludePath(ctx.path(&.{ "ggml", "include" }));
        tcDefineCMacro(tc, "NDEBUG", null); // otherwise zig is unhappy about c ASSERT macro
        return tc.addModule("llama.h");
    }

    /// zig module with translated headers
    pub fn moduleGgml(ctx: *Context) *Module {
        const tc = ctx.b.addTranslateC(.{
            .root_source_file = ctx.path(&.{ "include", "ggml.h" }),
            .target = ctx.options.target,
            .optimize = ctx.options.optimize,
        });

        tcDefineCMacro(tc, "LLAMA_SHARED", null);
        tcDefineCMacro(tc, "NDEBUG", null);

        return tc.addModule("ggml.h");
    }

    pub fn addBuildInfo(ctx: *Context, compile: *CompileStep) void {
        compile.addObject(ctx.build_info);
    }

    pub fn addGgml(ctx: *Context, compile: *CompileStep) void {
        ctx.common(compile);
        compile.addIncludePath(ctx.path(&.{"include"}));
        compile.addIncludePath(ctx.path(&.{"src"}));

        if (ctx.options.target.result.os.tag == .windows) {
            compile.root_module.addCMacro("GGML_ATTRIBUTE_FORMAT(...)", "");
        }

        // Define GGML_VERSION and GGML_COMMIT macros
        const ggml_version = "1"; // or get from a version file if available
        const commit_hash = std.process.Child.run(
            .{ .allocator = ctx.b.allocator, .argv = &.{ "git", "-C", ctx.path_prefix, "rev-parse", "HEAD" } },
        ) catch {
            // If git fails, use a default value
            compile.root_module.addCMacro("GGML_VERSION", "\"unknown\"");
            compile.root_module.addCMacro("GGML_COMMIT", "\"unknown\"");
            std.log.warn("Could not get git commit hash for GGML, using 'unknown'", .{});
            // Continue with rest of function
            return;
        };
        defer ctx.b.allocator.free(commit_hash.stdout);
        defer ctx.b.allocator.free(commit_hash.stderr);

        const commit_str = std.mem.trim(u8, commit_hash.stdout, &std.ascii.whitespace);
        compile.root_module.addCMacro("GGML_VERSION", ctx.b.fmt("\"{s}\"", .{ggml_version}));
        compile.root_module.addCMacro("GGML_COMMIT", ctx.b.fmt("\"{s}\"", .{commit_str}));

        var sources = std.ArrayList(LazyPath).initCapacity(ctx.b.allocator, 20) catch unreachable;
        sources.appendSlice(ctx.b.allocator, &.{
            ctx.path(&.{ "src", "ggml-alloc.c" }),
            ctx.path(&.{ "src", "ggml-backend-reg.cpp" }),
            ctx.path(&.{ "src", "ggml-backend.cpp" }),
            ctx.path(&.{ "src", "ggml-opt.cpp" }),
            ctx.path(&.{ "src", "ggml-quants.c" }),
            ctx.path(&.{ "src", "ggml-threading.cpp" }),
            ctx.path(&.{ "src", "ggml.c" }),
            ctx.path(&.{ "src", "gguf.cpp" }),
        }) catch unreachable;

        compile.addIncludePath(ctx.path(&.{ "src", "ggml-cpu" }));
        compile.linkLibCpp();
        sources.appendSlice(ctx.b.allocator, &.{
            ctx.path(&.{ "src", "ggml-cpu", "ggml-cpu.c" }),
            ctx.path(&.{ "src", "ggml-cpu", "ggml-cpu.cpp" }),
            //ctx.path(&.{ "src", "ggml-cpu", "ggml-cpu-aarch64.cpp" }),
            //ctx.path(&.{ "src", "ggml-cpu", "ggml-cpu-hbm.cpp" }),
            //ctx.path(&.{ "src", "ggml-cpu", "ggml-cpu-quants.c" }),
            ctx.path(&.{ "src", "ggml-cpu", "arch", "x86", "cpu-feats.cpp" }),
            ctx.path(&.{ "src", "ggml-cpu", "arch", "x86", "quants.c" }),
            ctx.path(&.{ "src", "ggml-cpu", "amx/amx.cpp" }),
            ctx.path(&.{ "src", "ggml-cpu", "amx/mmq.cpp" }),
        }) catch unreachable;

        for (sources.items) |src| compile.addCSourceFile(.{ .file = src, .flags = ctx.flags() });
    }

    pub fn addLLama(ctx: *Context, compile: *CompileStep) void {
        ctx.common(compile);
        compile.addIncludePath(ctx.path(&.{"include"}));
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-adapter.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-arch.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-batch.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-chat.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-context.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-grammar.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-hparams.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-impl.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-kv-cache.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-mmap.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-model-loader.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-model.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-sampling.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-vocab.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama-vocab.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("llama.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("unicode-data.cpp"), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.srcPath("unicode.cpp"), .flags = ctx.flags() });

        compile.addCSourceFile(.{ .file = ctx.path(&.{ "common", "common.cpp" }), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.path(&.{ "common", "sampling.cpp" }), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.path(&.{ "common", "console.cpp" }), .flags = ctx.flags() });

        compile.addCSourceFile(.{ .file = ctx.path(&.{ "common", "json-schema-to-grammar.cpp" }), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.path(&.{ "common", "speculative.cpp" }), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.path(&.{ "common", "ngram-cache.cpp" }), .flags = ctx.flags() });

        compile.addCSourceFile(.{ .file = ctx.path(&.{ "common", "log.cpp" }), .flags = ctx.flags() });
        compile.addCSourceFile(.{ .file = ctx.path(&.{ "common", "arg.cpp" }), .flags = ctx.flags() });
    }

    pub fn samples(ctx: *Context, install: bool) !void {
        const b = ctx.b;
        const examples = [_][]const u8{
            "main",
            "simple",
            // "perplexity",
            // "embedding",
            // "finetune",
            // "train-text-from-scratch",
            // "lookahead",
            // "speculative",
            // "parallel",
        };

        for (examples) |ex| {
            const exe = b.addExecutable(.{ .name = ex, .target = ctx.options.target, .optimize = ctx.options.optimize });
            exe.addIncludePath(ctx.path(&.{"include"}));
            exe.addIncludePath(ctx.path(&.{"common"}));
            exe.addIncludePath(ctx.path(&.{ "ggml", "include" }));
            exe.addIncludePath(ctx.path(&.{ "ggml", "src" }));

            exe.want_lto = false; // TODO: review, causes: error: lld-link: undefined symbol: __declspec(dllimport) _create_locale
            if (install) b.installArtifact(exe);
            { // add all c/cpp files from example dir
                const rpath = b.pathJoin(&.{ ctx.path_prefix, "examples", ex });
                exe.addIncludePath(.{ .cwd_relative = rpath });
                var dir = if (@hasDecl(std.fs, "openIterableDirAbsolute")) try std.fs.openIterableDirAbsolute(b.pathFromRoot(rpath), .{}) else try std.fs.openDirAbsolute(b.pathFromRoot(rpath), .{ .iterate = true }); // zig 11 vs nightly compatibility
                defer dir.close();
                var dir_it = dir.iterate();
                while (try dir_it.next()) |f| switch (f.kind) {
                    .file => if (std.ascii.endsWithIgnoreCase(f.name, ".c") or std.ascii.endsWithIgnoreCase(f.name, ".cpp")) {
                        const src = b.pathJoin(&.{ ctx.path_prefix, "examples", ex, f.name });
                        exe.addCSourceFile(.{ .file = .{ .cwd_relative = src }, .flags = ctx.flags() });
                    },
                    else => {},
                };
            }
            ctx.common(exe);
            ctx.link(exe);

            const run_exe = b.addRunArtifact(exe);
            if (b.args) |args| run_exe.addArgs(args); // passes on args like: zig build run -- my fancy args
            run_exe.step.dependOn(b.default_step); // allways copy output, to avoid confusion
            b.step(b.fmt("run-cpp-{s}", .{ex}), b.fmt("Run llama.cpp example: {s}", .{ex})).dependOn(&run_exe.step);
        }
    }

    fn flags(ctx: Context) []const []const u8 {
        _ = ctx;
        return &.{
            "-fno-sanitize=undefined",
            "-mavx",
            "-mavx2",
            "-mfma",
            "-mf16c",
            "-mavx512f",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
            "-U__AVX512BF16__", // Undefine this to prevent using unsupported intrinsics -- causing compile issues
        };
    }

    fn common(ctx: Context, lib: *CompileStep) void {
        lib.linkLibCpp();
        lib.addIncludePath(ctx.path(&.{"common"}));
        if (ctx.options.optimize != .Debug) lib.root_module.addCMacro("NDEBUG", "");
    }

    pub fn path(self: Context, subpath: []const []const u8) LazyPath {
        const sp = self.b.pathJoin(subpath);
        return .{ .cwd_relative = self.b.pathJoin(&.{ self.path_prefix, sp }) };
    }

    pub fn srcPath(self: Context, p: []const u8) LazyPath {
        return .{ .cwd_relative = self.b.pathJoin(&.{ self.path_prefix, "src", p }) };
    }

    pub fn includePath(self: Context, p: []const u8) LazyPath {
        return .{ .cwd_relative = self.b.pathJoin(&.{ self.path_prefix, "include", p }) };
    }
};

fn thisPath() []const u8 {
    return std.fs.path.dirname(@src().file) orelse ".";
}

// TODO: idk, root_module.addCMacro returns: TranslateC.zig:110:28: error: root struct of file 'Build' has no member named 'constructranslate_cMacro'
// use raw macro for now
fn tcDefineCMacro(tc: *std.Build.Step.TranslateC, comptime name: []const u8, comptime value: ?[]const u8) void {
    tc.defineCMacroRaw(name ++ "=" ++ (value orelse "1"));
}
