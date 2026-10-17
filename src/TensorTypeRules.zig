//! llama-quantize's --tensor-type-file, read as tools/quantize/quantize.cpp
//! reads it: whitespace-separated pattern=type entries split at the first '=',
//! the pattern lowercased, and the first one found anywhere in a name wins
//! (llama_tensor_get_type in src/llama-quant.cpp breaks on the first match).

const std = @import("std");
const types = @import("types.zig");
const Regex = @import("Regex.zig");

const TensorTypeRules = @This();

pub const Rule = struct {
    pattern: []const u8,
    regex: Regex,
    type: types.DataType,
    hits: usize = 0,
};

rules: []Rule,

pub const ParseError = error{
    NoTensorTypeRules,
    MalformedTensorTypeRule,
    InvalidTensorTypeRulePattern,
    UnsupportedTensorTypeRulePattern,
    TensorTypeRulePatternTooLarge,
    UnknownTensorTypeRuleType,
    TensorTypeRuleTypeNotInFormat,
    OutOfMemory,
};

/// Whether `content` is a llama.cpp type file rather than a ggufy JSON template.
pub fn isRuleFile(content: []const u8) bool {
    const body = std.mem.trimStart(u8, content, " \t\r\n");
    return body.len == 0 or body[0] != '{';
}

/// Rules from a type file's text, with their types spelled for `filetype`, or
/// as written when it is null. On error, `bad` holds the entry that caused it.
/// A file with no entries is an error: converting with it would quietly ignore it.
pub fn parse(arena: std.mem.Allocator, content: []const u8, filetype: ?types.FileType, bad: *[]const u8) ParseError!TensorTypeRules {
    var list: std.ArrayList(Rule) = .empty;
    var it = std.mem.tokenizeAny(u8, content, " \t\r\n\x0b\x0c");
    while (it.next()) |entry| {
        bad.* = entry;
        const eq = std.mem.findScalar(u8, entry, '=') orelse return error.MalformedTensorTypeRule;
        if (eq == 0 or eq + 1 == entry.len) return error.MalformedTensorTypeRule;
        const pattern = try std.ascii.allocLowerString(arena, entry[0..eq]);
        const dt = parseType(entry[eq + 1 ..]) orelse return error.UnknownTensorTypeRuleType;
        const resolved = if (filetype) |ft| dt.forFormat(ft) catch return error.TensorTypeRuleTypeNotInFormat else dt;
        if (resolved.formatType() == .gguf and !writableGguf(resolved)) return error.UnknownTensorTypeRuleType;
        const re = Regex.compile(arena, pattern) catch |e| return switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            error.InvalidPattern => error.InvalidTensorTypeRulePattern,
            error.UnsupportedPattern => error.UnsupportedTensorTypeRulePattern,
            error.PatternTooLarge => error.TensorTypeRulePatternTooLarge,
        };
        try list.append(arena, .{ .pattern = pattern, .regex = re, .type = resolved });
    }
    if (list.items.len == 0) return error.NoTensorTypeRules;
    return .{ .rules = list.items };
}

/// The first rule whose pattern occurs in `name`. The caller counts its `hits`.
pub fn find(self: TensorTypeRules, name: []const u8) ?*Rule {
    return &self.rules[self.findIndex(name) orelse return null];
}

pub fn findIndex(self: TensorTypeRules, name: []const u8) ?usize {
    for (self.rules, 0..) |*r, i| {
        if (r.regex.search(name)) return i;
    }
    return null;
}

/// Case-insensitive, as ggml's names are (q4_K), for SafeTensors types too.
fn parseType(s: []const u8) ?types.DataType {
    var buf: [32]u8 = undefined;
    if (s.len > buf.len) return null;
    if (types.DataType.fromString(std.ascii.lowerString(&buf, s))) |dt| return dt else |_| {}
    if (types.DataType.fromString(std.ascii.upperString(&buf, s))) |dt| return dt else |_| {}
    return null;
}

/// GGUF types the converter has an encoder for.
fn writableGguf(dt: types.DataType) bool {
    return switch (dt) {
        .f32, .f16, .bf16, .q4_0, .q4_1, .q5_0, .q5_1, .q8_0 => true,
        .q2_k, .q3_k, .q4_k, .q5_k, .q6_k => true,
        .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq1_s, .iq1_m, .iq4_nl, .iq4_xs, .mxfp4 => true,
        else => false,
    };
}

const testing = std.testing;

fn parseFor(arena: std.mem.Allocator, text: []const u8, filetype: ?types.FileType) ParseError!TensorTypeRules {
    var bad: []const u8 = "";
    return parse(arena, text, filetype, &bad);
}

test "the first matching rule wins and patterns search anywhere in the name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rules = try parseFor(a,
        \\blk\.0\.ffn_down=q8_0
        \\ffn_down=q6_K   attn_v=Q5_K
        \\ffn_.*_exps=iq4_xs
        \\token_embd=f16
    , .gguf);
    try testing.expectEqual(@as(usize, 5), rules.rules.len);

    try testing.expectEqual(types.DataType.q8_0, rules.find("blk.0.ffn_down.weight").?.type);
    try testing.expectEqual(types.DataType.q6_k, rules.find("blk.10.ffn_down.weight").?.type);
    try testing.expectEqual(types.DataType.q5_k, rules.find("blk.3.attn_v.weight").?.type);
    // The earlier, broader ffn_down rule shadows the experts rule for down_exps.
    try testing.expectEqual(types.DataType.q6_k, rules.find("blk.3.ffn_down_exps.weight").?.type);
    try testing.expectEqual(types.DataType.iq4_xs, rules.find("blk.3.ffn_gate_exps.weight").?.type);
    try testing.expectEqual(types.DataType.f16, rules.find("token_embd.weight").?.type);
    try testing.expect(rules.find("blk.3.attn_q.weight") == null);
}

test "patterns are lowercased as llama-quantize lowercases them" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    // Lowercasing the whole pattern turns \D into \d, as in llama.cpp.
    var rules = try parseFor(arena.allocator(), "ATTN_Q=q8_0 blk\\.\\D+\\.ffn=q4_0", .gguf);
    try testing.expect(rules.find("blk.0.attn_q.weight") != null);
    try testing.expectEqual(types.DataType.q4_0, rules.find("blk.12.ffn_up.weight").?.type);
}

test "malformed entries, unknown types and bad patterns are refused with the entry named" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var bad: []const u8 = "";
    try testing.expectError(error.MalformedTensorTypeRule, parse(a, "attn_q=q8_0 # comment", .gguf, &bad));
    try testing.expectEqualStrings("#", bad);
    try testing.expectError(error.MalformedTensorTypeRule, parse(a, "=q8_0", .gguf, &bad));
    try testing.expectError(error.MalformedTensorTypeRule, parse(a, "attn_q=", .gguf, &bad));
    try testing.expectError(error.UnknownTensorTypeRuleType, parse(a, "attn_q=q9_9", .gguf, &bad));
    try testing.expectEqualStrings("attn_q=q9_9", bad);
    // Real ggml types the converter has no encoder for.
    try testing.expectError(error.UnknownTensorTypeRuleType, parse(a, "attn_q=tq1_0", .gguf, &bad));
    try testing.expectError(error.InvalidTensorTypeRulePattern, parse(a, "blk\\.(0=q8_0", .gguf, &bad));
    try testing.expectError(error.UnsupportedTensorTypeRulePattern, parse(a, "(?!attn)q=q8_0", .gguf, &bad));
    try testing.expectError(error.TensorTypeRulePatternTooLarge, parse(a, "attn.{0,100000}=q8_0", .gguf, &bad));
    try testing.expectError(error.NoTensorTypeRules, parse(a, " \n\t", .gguf, &bad));
    try testing.expectError(error.TensorTypeRuleTypeNotInFormat, parse(a, "attn_q=q4_k", .safetensors, &bad));
}

test "a type file names types for the output container" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const st = try parseFor(a, "attn=bf16 mlp=int8_convrot", .safetensors);
    try testing.expectEqual(types.DataType.BF16, st.rules[0].type);
    try testing.expectEqual(types.DataType.INT8_CONVROT, st.rules[1].type);
    const g = try parseFor(a, "attn=BF16", .gguf);
    try testing.expectEqual(types.DataType.bf16, g.rules[0].type);
}

test "only a file opening with a brace is a JSON template" {
    try testing.expect(!isRuleFile("  \n{\"tensors\":{}}"));
    try testing.expect(isRuleFile("attn_q=q8_0\n"));
    try testing.expect(isRuleFile(""));
}

test "a type file parsed for no container keeps its types as written" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try parseFor(a, "attn=Q8_0 mlp=int8_convrot", null);
    try testing.expectEqual(types.DataType.q8_0, r.rules[0].type);
    try testing.expectEqual(types.DataType.INT8_CONVROT, r.rules[1].type);
    var bad: []const u8 = "";
    try testing.expectError(error.UnknownTensorTypeRuleType, parse(a, "attn=tq1_0", null, &bad));
}
