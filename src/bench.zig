const std = @import("std");
const DataTransform = @import("DataTransform.zig");
const Quantizer = DataTransform.Quantizer;

const N = 1 << 20; // 1M elements
const RUNS = 5;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Generate random-ish f32 data in a range that exercises all code paths
    const input = try allocator.alloc(f32, N);
    defer allocator.free(input);
    const output = try allocator.alloc(u8, N);
    defer allocator.free(output);
    const output_lut = try allocator.alloc(f32, N);
    defer allocator.free(output_lut);

    var rng = std.Random.DefaultPrng.init(0xDEADBEEF);
    const rand = rng.random();
    for (input) |*v| {
        // Values spanning subnormals, normals, and overflow range
        v.* = (rand.float(f32) * 2.0 - 1.0) * 1000.0;
    }

    const print = std.debug.print;

    // -------------------------------------------------------------------------
    // F8 E4M3 encode: scalar vs SIMD
    // -------------------------------------------------------------------------
    print("\n=== F8 E4M3 Encode ({d}M f32 → f8) ===\n", .{N / (1 << 20)});

    {
        var best_ns: u64 = std.math.maxInt(u64);
        for (0..RUNS) |_| {
            var timer = try std.time.Timer.start();
            for (input, output) |v, *o| o.* = Quantizer.f32_to_fp8_e4m3(v);
            const ns = timer.read();
            std.mem.doNotOptimizeAway(output.ptr);
            best_ns = @min(best_ns, ns);
        }
        const gb_per_s = @as(f64, @floatFromInt(N * @sizeOf(f32))) / @as(f64, @floatFromInt(best_ns));
        print("  scalar:  {d:7.2} ms  ({d:.2} GB/s input)\n", .{ @as(f64, @floatFromInt(best_ns)) / 1e6, gb_per_s });
    }

    {
        var best_ns: u64 = std.math.maxInt(u64);
        for (0..RUNS) |_| {
            var timer = try std.time.Timer.start();
            const W = 8;
            var i: usize = 0;
            while (i + W <= N) : (i += W) {
                const chunk: @Vector(W, f32) = input[i..][0..W].*;
                const result: @Vector(W, u8) = Quantizer.f32_to_fp8_e4m3_chunk(chunk);
                output[i..][0..W].* = result;
            }
            while (i < N) : (i += 1) output[i] = Quantizer.f32_to_fp8_e4m3(input[i]);
            const ns = timer.read();
            std.mem.doNotOptimizeAway(output.ptr);
            best_ns = @min(best_ns, ns);
        }
        const gb_per_s = @as(f64, @floatFromInt(N * @sizeOf(f32))) / @as(f64, @floatFromInt(best_ns));
        print("  SIMD:    {d:7.2} ms  ({d:.2} GB/s input)\n", .{ @as(f64, @floatFromInt(best_ns)) / 1e6, gb_per_s });
    }

    // -------------------------------------------------------------------------
    // F8 E5M2 encode: scalar vs SIMD
    // -------------------------------------------------------------------------
    print("\n=== F8 E5M2 Encode ({d}M f32 → f8) ===\n", .{N / (1 << 20)});

    {
        var best_ns: u64 = std.math.maxInt(u64);
        for (0..RUNS) |_| {
            var timer = try std.time.Timer.start();
            for (input, output) |v, *o| o.* = Quantizer.f32_to_fp8_e5m2(v);
            const ns = timer.read();
            std.mem.doNotOptimizeAway(output.ptr);
            best_ns = @min(best_ns, ns);
        }
        const gb_per_s = @as(f64, @floatFromInt(N * @sizeOf(f32))) / @as(f64, @floatFromInt(best_ns));
        print("  scalar:  {d:7.2} ms  ({d:.2} GB/s input)\n", .{ @as(f64, @floatFromInt(best_ns)) / 1e6, gb_per_s });
    }

    {
        var best_ns: u64 = std.math.maxInt(u64);
        for (0..RUNS) |_| {
            var timer = try std.time.Timer.start();
            const W = 8;
            var i: usize = 0;
            while (i + W <= N) : (i += W) {
                const chunk: @Vector(W, f32) = input[i..][0..W].*;
                const result: @Vector(W, u8) = Quantizer.f32_to_fp8_e5m2_chunk(chunk);
                output[i..][0..W].* = result;
            }
            while (i < N) : (i += 1) output[i] = Quantizer.f32_to_fp8_e5m2(input[i]);
            const ns = timer.read();
            std.mem.doNotOptimizeAway(output.ptr);
            best_ns = @min(best_ns, ns);
        }
        const gb_per_s = @as(f64, @floatFromInt(N * @sizeOf(f32))) / @as(f64, @floatFromInt(best_ns));
        print("  SIMD:    {d:7.2} ms  ({d:.2} GB/s input)\n", .{ @as(f64, @floatFromInt(best_ns)) / 1e6, gb_per_s });
    }

    // -------------------------------------------------------------------------
    // F8 E4M3 decode: scalar fn vs LUT
    // -------------------------------------------------------------------------
    // Fill output buffer with all 256 values repeated
    for (output, 0..) |*b, i| b.* = @intCast(i % 256);

    print("\n=== F8 E4M3 Decode ({d}M f8 → f32) ===\n", .{N / (1 << 20)});

    {
        var best_ns: u64 = std.math.maxInt(u64);
        for (0..RUNS) |_| {
            var timer = try std.time.Timer.start();
            for (output, output_lut) |b, *o| o.* = Quantizer.fp8_e4m3_to_f32(b);
            const ns = timer.read();
            std.mem.doNotOptimizeAway(output_lut.ptr);
            best_ns = @min(best_ns, ns);
        }
        const gb_per_s = @as(f64, @floatFromInt(N)) / @as(f64, @floatFromInt(best_ns));
        print("  scalar:  {d:7.2} ms  ({d:.2} GB/s input)\n", .{ @as(f64, @floatFromInt(best_ns)) / 1e6, gb_per_s });
    }

    {
        var best_ns: u64 = std.math.maxInt(u64);
        for (0..RUNS) |_| {
            var timer = try std.time.Timer.start();
            for (output, output_lut) |b, *o| o.* = Quantizer.lut_e4m3[b];
            const ns = timer.read();
            std.mem.doNotOptimizeAway(output_lut.ptr);
            best_ns = @min(best_ns, ns);
        }
        const gb_per_s = @as(f64, @floatFromInt(N)) / @as(f64, @floatFromInt(best_ns));
        print("  LUT:     {d:7.2} ms  ({d:.2} GB/s input)\n", .{ @as(f64, @floatFromInt(best_ns)) / 1e6, gb_per_s });
    }

    // -------------------------------------------------------------------------
    // F8 E5M2 decode: scalar fn vs LUT
    // -------------------------------------------------------------------------
    print("\n=== F8 E5M2 Decode ({d}M f8 → f32) ===\n", .{N / (1 << 20)});

    {
        var best_ns: u64 = std.math.maxInt(u64);
        for (0..RUNS) |_| {
            var timer = try std.time.Timer.start();
            for (output, output_lut) |b, *o| o.* = Quantizer.fp8_e5m2_to_f32(b);
            const ns = timer.read();
            std.mem.doNotOptimizeAway(output_lut.ptr);
            best_ns = @min(best_ns, ns);
        }
        const gb_per_s = @as(f64, @floatFromInt(N)) / @as(f64, @floatFromInt(best_ns));
        print("  scalar:  {d:7.2} ms  ({d:.2} GB/s input)\n", .{ @as(f64, @floatFromInt(best_ns)) / 1e6, gb_per_s });
    }

    {
        var best_ns: u64 = std.math.maxInt(u64);
        for (0..RUNS) |_| {
            var timer = try std.time.Timer.start();
            for (output, output_lut) |b, *o| o.* = Quantizer.lut_e5m2[b];
            const ns = timer.read();
            std.mem.doNotOptimizeAway(output_lut.ptr);
            best_ns = @min(best_ns, ns);
        }
        const gb_per_s = @as(f64, @floatFromInt(N)) / @as(f64, @floatFromInt(best_ns));
        print("  LUT:     {d:7.2} ms  ({d:.2} GB/s input)\n", .{ @as(f64, @floatFromInt(best_ns)) / 1e6, gb_per_s });
    }

    print("\n", .{});
}
