const std = @import("std");

const ggml_root = thisDir() ++ "/vendor/ggml";

// ── Compiler flag arrays ─────────────────────────────────────────────────────

const base_flags = [_][]const u8{
    "-O3",
    "-fno-sanitize=undefined",
    "-ffast-math",
    "-fno-finite-math-only", // keeps inf/nan behaviour sane despite ffast-math
    "-Wno-macro-redefined",  // _GNU_SOURCE is set both by Zig and by GGML headers
    // Suppress any AVX512 paths Zig's native CPU detection might activate.
    // These are unset globally because ggml.c and other base files have
    // AVX512 code paths that require AVX512-specific headers we're not linking.
    "-U__AVX512F__",
    "-U__AVX512BF16__",
    "-U__AVX512VNNI__",
    "-U__AVX512VBMI__",
    "-U__AVX512CD__",
    "-U__AVX512BW__",
    "-U__AVX512DQ__",
    "-U__AVX512VL__",
};

const base_c_flags   = base_flags ++ [_][]const u8{ "-std=c11" };
const base_cpp_flags = base_flags ++ [_][]const u8{ "-std=c++17" };

// Compiler ISA flags — tell the compiler it may EMIT these instructions
const x86_isa_flags = [_][]const u8{
    "-msse4.2",
    "-mavx",
    "-mavx2",
    "-mfma",
    "-mf16c",
    "-mbmi2",
};

// GGML's own feature defines — tell GGML's #ifdefs which paths to compile
// Both sets are required: ISA flags alone are not enough.
const x86_ggml_defines = [_][]const u8{
    "-DGGML_SSE42",
    "-DGGML_AVX",
    "-DGGML_AVX2",
    "-DGGML_FMA",
    "-DGGML_F16C",
    "-DGGML_BMI2",
};

const x86_c_flags   = base_c_flags   ++ x86_isa_flags ++ x86_ggml_defines;
const x86_cpp_flags = base_cpp_flags ++ x86_isa_flags ++ x86_ggml_defines;

// ── Source file lists ────────────────────────────────────────────────────────

// Group 1 — ggml-base: pure logic, no SIMD
const base_c_sources = [_][]const u8{
    ggml_root ++ "/src/ggml.c",
    ggml_root ++ "/src/ggml-alloc.c",
    ggml_root ++ "/src/ggml-quants.c",
};
const base_cpp_sources = [_][]const u8{
    ggml_root ++ "/src/ggml.cpp",
    ggml_root ++ "/src/ggml-backend.cpp",
    ggml_root ++ "/src/ggml-opt.cpp",
    ggml_root ++ "/src/ggml-threading.cpp",
    ggml_root ++ "/src/gguf.cpp",
};

// Group 2 — ggml-backend-reg: needs C++ exceptions, no SIMD
// Intentionally separate so we never pass -fno-exceptions to it.
const reg_cpp_sources = [_][]const u8{
    ggml_root ++ "/src/ggml-backend-reg.cpp",
};

// Group 3 — ggml-cpu: shared CPU kernel files (get SIMD flags on x86)
const cpu_c_sources = [_][]const u8{
    ggml_root ++ "/src/ggml-cpu/ggml-cpu.c",
    ggml_root ++ "/src/ggml-cpu/quants.c",
};
const cpu_cpp_sources = [_][]const u8{
    ggml_root ++ "/src/ggml-cpu/ggml-cpu.cpp",
    ggml_root ++ "/src/ggml-cpu/repack.cpp",
    ggml_root ++ "/src/ggml-cpu/hbm.cpp",
    ggml_root ++ "/src/ggml-cpu/traits.cpp",
    ggml_root ++ "/src/ggml-cpu/binary-ops.cpp",
    ggml_root ++ "/src/ggml-cpu/unary-ops.cpp",
    ggml_root ++ "/src/ggml-cpu/vec.cpp",
    ggml_root ++ "/src/ggml-cpu/ops.cpp",
    // AMX files compile fine without AMX flags — they just won't activate.
    // Include them so the symbol table is complete on x86 builds.
    ggml_root ++ "/src/ggml-cpu/amx/amx.cpp",
    ggml_root ++ "/src/ggml-cpu/amx/mmq.cpp",
};

// Group 3 — arch-specific CPU sources
const cpu_x86_c_sources = [_][]const u8{
    ggml_root ++ "/src/ggml-cpu/arch/x86/quants.c",
};
const cpu_x86_cpp_sources = [_][]const u8{
    ggml_root ++ "/src/ggml-cpu/arch/x86/repack.cpp",
};

const cpu_arm_c_sources = [_][]const u8{
    ggml_root ++ "/src/ggml-cpu/arch/arm/quants.c",
};
const cpu_arm_cpp_sources = [_][]const u8{
    ggml_root ++ "/src/ggml-cpu/arch/arm/repack.cpp",
};

// ── Public API ───────────────────────────────────────────────────────────────

/// Build a static GGML library and link it into `exe`.
pub fn link(
    b: *std.Build,
    exe: *std.Build.Step.Compile,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) void {
    const lib = buildLib(b, target, optimize);
    exe.linkLibrary(lib);
}

/// Build and return the static GGML library.
/// Use this if you need to inspect or further configure the library step.
pub fn buildLib(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step.Compile {
    const lib = b.addLibrary(.{
        .name = "ggml",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
        }),
    });

    addSources(b, lib, target);
    return lib;
}

// ── Internal implementation ──────────────────────────────────────────────────

fn addSources(
    b: *std.Build,
    lib: *std.Build.Step.Compile,
    target: std.Build.ResolvedTarget,
) void {
    const arch = target.result.cpu.arch;
    const os   = target.result.os.tag;
    const is_x86 = arch == .x86_64 or arch == .x86;
    const is_arm = arch == .aarch64;

    // Include paths
    lib.addIncludePath(b.path(ggml_root ++ "/include"));
    lib.addIncludePath(b.path(ggml_root ++ "/src"));
    lib.addIncludePath(b.path(ggml_root ++ "/src/ggml-cpu"));

    // Platform macros — use defineCMacro rather than -D flags to avoid
    // the "macro redefined" warning from Zig's built-in definitions.
    switch (os) {
        .linux   => lib.root_module.addCMacro("_GNU_SOURCE",            ""),
        .windows => lib.root_module.addCMacro("_CRT_SECURE_NO_WARNINGS",""),
        .macos   => lib.root_module.addCMacro("_DARWIN_C_SOURCE",       ""),
        else     => {},
    }
    // Required by GGML to emit version/commit strings
    lib.root_module.addCMacro("GGML_VERSION", "\"0.0.0\"");
    lib.root_module.addCMacro("GGML_COMMIT",  "\"unknown\"");
    // Activates weight repacking optimisation in the CPU backend
    lib.root_module.addCMacro("GGML_USE_CPU_REPACK", "");

    // Windows-specific: GGML uses __attribute__((format(...))) which MSVC
    // doesn't understand; the macro definition below disables it.
    if (os == .windows) {
        lib.root_module.addCMacro("GGML_ATTRIBUTE_FORMAT(...)", "");
    }

    // All C++ files need libc++ (satisfies exception/RTTI runtime symbols,
    // including the SEH glue on Windows that was causing link errors).
    lib.linkLibCpp();

    // ── Group 1: ggml-base (no SIMD) ────────────────────────────────────────
    for (base_c_sources)   |f| lib.addCSourceFile(.{ .file = b.path(f), .flags = &base_c_flags });
    for (base_cpp_sources) |f| lib.addCSourceFile(.{ .file = b.path(f), .flags = &base_cpp_flags });

    // ── Group 2: backend registry (C++ exceptions required, no SIMD) ────────
    for (reg_cpp_sources)  |f| lib.addCSourceFile(.{ .file = b.path(f), .flags = &base_cpp_flags });

    // ── Group 3: CPU backend (SIMD flags on x86) ────────────────────────────
    const c_flags_cpu   = if (is_x86) &x86_c_flags   else &base_c_flags;
    const cpp_flags_cpu = if (is_x86) &x86_cpp_flags  else &base_cpp_flags;

    for (cpu_c_sources)   |f| lib.addCSourceFile(.{ .file = b.path(f), .flags = c_flags_cpu });
    for (cpu_cpp_sources) |f| lib.addCSourceFile(.{ .file = b.path(f), .flags = cpp_flags_cpu });

    // Arch-specific sources
    if (is_x86) {
        for (cpu_x86_c_sources)   |f| lib.addCSourceFile(.{ .file = b.path(f), .flags = &x86_c_flags });
        for (cpu_x86_cpp_sources) |f| lib.addCSourceFile(.{ .file = b.path(f), .flags = &x86_cpp_flags });
    } else if (is_arm) {
        for (cpu_arm_c_sources)   |f| lib.addCSourceFile(.{ .file = b.path(f), .flags = &base_c_flags });
        for (cpu_arm_cpp_sources) |f| lib.addCSourceFile(.{ .file = b.path(f), .flags = &base_cpp_flags });
    }
}

fn thisDir() []const u8 {
    return std.fs.path.dirname(@src().file) orelse ".";
}