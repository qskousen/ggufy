//! HuggingFace LLM checkpoint sidecars: config.json, tokenizer.json,
//! tokenizer_config.json, generation_config.json. Turns them into the
//! llama.cpp-flavored GGUF metadata and native tensor names that make the
//! output loadable by llama.cpp. The tensor payload itself comes from
//! Safetensor; this module only reads the small sibling JSON files.

const std = @import("std");
const imagearch = @import("ImageArch.zig");
const types = @import("types.zig");

const HfLlm = @This();

pub const GGUF_TOKEN_TYPE_NORMAL: u32 = 1;
pub const GGUF_TOKEN_TYPE_UNKNOWN: u32 = 2;
pub const GGUF_TOKEN_TYPE_CONTROL: u32 = 3;
pub const GGUF_TOKEN_TYPE_USER_DEFINED: u32 = 4;
pub const GGUF_TOKEN_TYPE_UNUSED: u32 = 5;
pub const GGUF_TOKEN_TYPE_BYTE: u32 = 6;

pub const CardEntry = struct {
    name: []const u8,
    organization: ?[]const u8,
    repo_url: ?[]const u8,
};

/// config.json's rope_scaling block. `kind` is HF's rope_type verbatim: only
/// "linear" and "yarn" have a GGUF metadata spelling, and "llama3" lives in a
/// generated rope_freqs tensor instead, which ggufy does not write.
pub const RopeScaling = struct {
    kind: []const u8,
    factor: f32,
    orig_ctx: ?u32,
};

/// The qwen35 hybrid's own dimensions. Separate from Model because nothing
/// else reads them, and because their absence is what tells the writer this is
/// not a Gated DeltaNet checkpoint.
pub const Qwen35 = struct {
    /// ssm.* in GGUF, linear_* in config.json. The names do not line up: state
    /// size is the key head dim, group count the key head count, and time step
    /// rank the value head count.
    conv_kernel: u32,
    state_size: u32,
    group_count: u32,
    time_step_rank: u32,
    inner_size: u32,
    /// Needed by the V-head reorder, not written to the file.
    value_head_dim: u32,
    nextn_layers: u32,
    full_attention_interval: u32,
    /// One entry per GGUF block, the MTP block included: true where the block
    /// runs linear attention rather than full.
    recurrent: []const bool,
    /// M-RoPE section widths, padded to four.
    rope_sections: [4]i64,
    /// head_dim * partial_rotary_factor, which is not head_dim here.
    rope_dim_count: u32,

    /// How many V heads share one K head. 1 means the two counts agree and the
    /// reorder is a no-op.
    pub fn vPerK(self: Qwen35) u32 {
        return if (self.group_count == 0) 1 else self.time_step_rank / self.group_count;
    }
};

pub const Model = struct {
    arch: *const imagearch.Arch,
    dir: []const u8,

    block_count: u32,
    embedding_length: u32,
    /// 0 = config.json names none, which an MoE config may do: llama.cpp reads
    /// the key as optional and sizes the experts from their own key.
    feed_forward_length: u32,
    /// 0 = dense. The rest of the expert fields mean nothing then.
    expert_count: u32 = 0,
    expert_used_count: u32 = 0,
    expert_feed_forward_length: ?u32 = null,
    expert_shared_feed_forward_length: ?u32 = null,
    head_count: u32,
    head_count_kv: u32,
    context_length: u32,
    rms_eps: f32,
    rope_theta: f32,
    /// null = config.json carries no rope_scaling block.
    rope_scaling: ?RopeScaling,
    vocab_size: u32,
    head_dim: ?u32,
    /// null = not a Gated DeltaNet hybrid.
    qwen35: ?Qwen35 = null,
    /// M-RoPE section widths padded to four, for an arch that is not qwen35
    /// (which carries its own). null = plain RoPE.
    rope_sections: ?[4]i64 = null,
    /// Vision layers whose features the text tower adds back in (Qwen3-VL).
    deepstack_layers: ?u32 = null,
    /// null = no vision tower config, or one this tool cannot describe.
    vision: ?Vision = null,

    /// True when the checkpoint ships a SentencePiece tokenizer.model; the
    /// vocab then comes from its protobuf, not tokenizer.json.
    is_spm: bool,
    /// Per-token logprob; the spm path carries them, the BPE path does not.
    scores: ?[]const f32,

    /// Indexed by token id; holes are "[PAD]" + id, matching what llama.cpp's
    /// own converter writes when vocab_size exceeds the defined vocabulary.
    /// Empty = no tokenizer file anywhere (a text encoder shipped with only its
    /// config.json), and addMetadata writes no tokenizer.ggml.* keys.
    tokens: []const []const u8,
    token_types: []const u32,
    merges: []const []const u8,
    tokenizer_pre: []const u8,

    chat_template: ?[]const u8,
    /// null = the checkpoint gives no value and the key stays unwritten.
    add_bos_token: ?bool,
    add_eos_token: ?bool,
    add_sep_token: ?bool,
    bos_id: ?u32,
    eos_id: ?u32,
    pad_id: ?u32,
    unk_id: ?u32,
    sampling_temp: ?f32,
    sampling_top_k: ?u32,
    sampling_top_p: ?f32,

    license: ?[]const u8,
    license_link: ?[]const u8,
    tags: []const []const u8,
    languages: []const []const u8,
    datasets: []const CardEntry,
    base_models: []const CardEntry,
};

fn readJson(io: std.Io, dir: std.Io.Dir, name: []const u8, alloc: std.mem.Allocator) !std.json.Parsed(std.json.Value) {
    const bytes = try dir.readFileAlloc(io, name, alloc, .limited64(256 * 1024 * 1024));
    return try std.json.parseFromSlice(std.json.Value, alloc, bytes, .{});
}

fn obj(v: std.json.Value) ?std.json.ObjectMap {
    return switch (v) {
        .object => |o| o,
        else => null,
    };
}

fn getU64(o: std.json.ObjectMap, key: []const u8) ?u64 {
    const v = o.get(key) orelse return null;
    return switch (v) {
        .integer => |i| if (i >= 0) @intCast(i) else null,
        // config.json is user input, and @intFromFloat out of range is illegal
        // behavior. 2^63 holds it to what the integer arm can produce.
        .float => |f| if (f >= 0 and f < 9223372036854775808.0) @intFromFloat(f) else null,
        else => null,
    };
}

/// The config fields land in u32s. A number too large is an unusable config,
/// not a field to wrap around.
fn getU32(o: std.json.ObjectMap, key: []const u8) ?u32 {
    return std.math.cast(u32, getU64(o, key) orelse return null);
}

fn getF64(o: std.json.ObjectMap, key: []const u8) ?f64 {
    const v = o.get(key) orelse return null;
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}

fn getStr(o: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = o.get(key) orelse return null;
    return switch (v) {
        .string => |s| s,
        else => null,
    };
}

fn getBool(o: std.json.ObjectMap, key: []const u8, default: bool) bool {
    const v = o.get(key) orelse return default;
    return switch (v) {
        .bool => |b| b,
        else => default,
    };
}

/// Token ids land in u32s, so the same bound getU32 applies: a number too large
/// is an unusable config, not a field to wrap around.
fn idFromItem(item: std.json.Value) ?u32 {
    return switch (item) {
        .integer => |i| std.math.cast(u32, i),
        else => null,
    };
}

/// eos_token_id may be a scalar or a list; llama.cpp's converter keeps the first.
fn getIdOrFirst(o: std.json.ObjectMap, key: []const u8) ?u32 {
    const v = o.get(key) orelse return null;
    return switch (v) {
        .array => |a| if (a.items.len > 0) idFromItem(a.items[0]) else null,
        else => idFromItem(v),
    };
}

/// Newer checkpoints spell the kind "rope_type", older ones "type". A block
/// without a factor scales nothing, so it reads as absent.
fn readRopeScaling(c: std.json.ObjectMap) ?RopeScaling {
    const rs = obj(c.get("rope_scaling") orelse return null) orelse return null;
    const kind = getStr(rs, "rope_type") orelse getStr(rs, "type") orelse return null;
    const factor = getF64(rs, "factor") orelse return null;
    return .{
        .kind = kind,
        .factor = @floatCast(factor),
        .orig_ctx = getU32(rs, "original_max_position_embeddings"),
    };
}

/// Newer Qwen configs carry the rope base inside a rope_parameters block
/// instead of at the top level, and the two differ by three orders of
/// magnitude on a long-context model - a silent fallback to 10000 would cap the
/// usable context without anything failing.
fn readRopeTheta(c: std.json.ObjectMap) ?f64 {
    if (getF64(c, "rope_theta")) |v| return v;
    const rp = obj(c.get("rope_parameters") orelse return null) orelse return null;
    return getF64(rp, "rope_theta");
}

/// The Gated DeltaNet dimensions, or null when this config describes no hybrid.
/// `text_blocks` is num_hidden_layers; the MTP block is appended past it and is
/// a full-attention block.
fn readQwen35(c: std.json.ObjectMap, text_blocks: u32, head_dim: u32, alloc: std.mem.Allocator) !?Qwen35 {
    const num_v = getU32(c, "linear_num_value_heads") orelse return null;
    const num_k = nonZero(getU32(c, "linear_num_key_heads")) orelse return null;
    const v_head_dim = getU32(c, "linear_value_head_dim") orelse return null;
    const nextn = getU32(c, "mtp_num_hidden_layers") orelse 0;

    // layer_types names each text block; the MTP block past them is full
    // attention. A config without the list falls back to the interval, which is
    // how the same layout is spelled when the list is omitted.
    const interval = nonZero(getU32(c, "full_attention_interval")) orelse 4;
    const total = text_blocks + nextn;
    const recurrent = try alloc.alloc(bool, total);
    if (c.get("layer_types")) |lt| switch (lt) {
        .array => |arr| {
            for (recurrent, 0..) |*r, i| {
                r.* = if (i < arr.items.len) switch (arr.items[i]) {
                    .string => |s| std.mem.eql(u8, s, "linear_attention"),
                    else => false,
                } else false;
            }
        },
        else => for (recurrent, 0..) |*r, i| {
            r.* = i < text_blocks and (i + 1) % interval != 0;
        },
    } else for (recurrent, 0..) |*r, i| {
        r.* = i < text_blocks and (i + 1) % interval != 0;
    }

    var sections = [4]i64{ 0, 0, 0, 0 };
    var partial: f64 = 1.0;
    if (obj(c.get("rope_parameters") orelse std.json.Value{ .null = {} })) |rp| {
        if (getF64(rp, "partial_rotary_factor")) |p| partial = p;
        if (rp.get("mrope_section")) |ms| switch (ms) {
            .array => |arr| for (arr.items, 0..) |v, i| {
                if (i >= 3) break;
                sections[i] = switch (v) {
                    .integer => |n| n,
                    else => 0,
                };
            },
            else => {},
        };
    }

    return .{
        .conv_kernel = getU32(c, "linear_conv_kernel_dim") orelse return null,
        .state_size = getU32(c, "linear_key_head_dim") orelse return null,
        .group_count = num_k,
        .time_step_rank = num_v,
        .inner_size = v_head_dim * num_v,
        .value_head_dim = v_head_dim,
        .nextn_layers = nextn,
        .full_attention_interval = interval,
        .recurrent = recurrent,
        .rope_sections = sections,
        .rope_dim_count = @intFromFloat(@as(f64, @floatFromInt(head_dim)) * partial),
    };
}

/// rope_parameters.mrope_section, or the older rope_scaling spelling, padded to four.
fn readMropeSections(c: std.json.ObjectMap) ?[4]i64 {
    for ([_][]const u8{ "rope_parameters", "rope_scaling" }) |k| {
        const rp = obj(c.get(k) orelse continue) orelse continue;
        const ms = rp.get("mrope_section") orelse continue;
        if (ms != .array) continue;
        var out = [4]i64{ 0, 0, 0, 0 };
        for (ms.array.items, 0..) |v, i| {
            if (i >= 4) break;
            out[i] = switch (v) {
                .integer => |n| n,
                else => return null,
            };
        }
        return out;
    }
    return null;
}

fn isQwen35Family(arch_name: []const u8) bool {
    return std.mem.eql(u8, arch_name, "qwen35") or std.mem.eql(u8, arch_name, "qwen35moe");
}

/// Model types whose tensors and metadata are another arch's, as llama.cpp's
/// converter writes them.
const model_type_aliases = [_]struct { hf: []const u8, arch: *const imagearch.Arch }{
    .{ .hf = "mistral", .arch = &imagearch.llama },
};

pub fn archForModelType(model_type: []const u8) ?*const imagearch.Arch {
    for (model_type_aliases) |al| if (std.mem.eql(u8, al.hf, model_type)) return al.arch;
    for (imagearch.arch_list) |arch| {
        if (arch.hf_model_type.len > 0 and std.mem.eql(u8, arch.hf_model_type, model_type)) return arch;
    }
    return null;
}

fn nonZero(v: ?u32) ?u32 {
    return if (v) |n| (if (n == 0) null else n) else null;
}

/// The arch config.json's model_type names, without reading the rest of the checkpoint.
pub fn configArch(io: std.Io, dir: std.Io.Dir, alloc: std.mem.Allocator) ?*const imagearch.Arch {
    const cfg = readJson(io, dir, "config.json", alloc) catch return null;
    const c = obj(cfg.value) orelse return null;
    return archForModelType(getStr(c, "model_type") orelse return null);
}

/// Why `load` returned null. "Not an HF checkpoint at all" and "an HF
/// checkpoint whose tokenizer we cannot read" send the user to different
/// files, so the caller needs to tell them apart to say anything useful.
pub const LoadFailure = enum { no_config, unknown_model_type, incomplete_config, no_tokenizer };

/// Look for an HF checkpoint whose sidecars we understand. `dir` holds
/// config.json (and normally model.safetensors). null = not an HF LLM
/// checkpoint (missing files, unknown model_type, non-BPE tokenizer),
/// which routes the caller to the plain safetensors pipeline; `reason`, when
/// given, receives which of those it was.
pub fn load(io: std.Io, dir: std.Io.Dir, name_path: []const u8, alloc: std.mem.Allocator, reason: ?*LoadFailure) !?Model {
    if (reason) |r| r.* = .no_config;
    const cfg = readJson(io, dir, "config.json", alloc) catch return null;
    const c = obj(cfg.value) orelse return null;
    const model_type = getStr(c, "model_type") orelse return null;
    if (reason) |r| r.* = .unknown_model_type;
    const arch = archForModelType(model_type) orelse return null;
    if (reason) |r| r.* = .incomplete_config;

    // A ForConditionalGeneration checkpoint nests the language tower's
    // dimensions under text_config and leaves only the wrapper at the top, so
    // every lookup below would miss. Flatten with text_config winning, which is
    // what llama.cpp's converter does. A config without one merges to itself.
    //
    // model_type is read above, off the top level: text_config names the tower
    // ("qwen3_5_text") rather than the checkpoint, and the arch table is keyed
    // by the latter.
    var merged: std.json.ObjectMap = .empty;
    {
        var it = c.iterator();
        while (it.next()) |e| try merged.put(alloc, e.key_ptr.*, e.value_ptr.*);
        if (c.get("text_config")) |tcv| if (obj(tcv)) |tc| {
            var it2 = tc.iterator();
            while (it2.next()) |e| try merged.put(alloc, e.key_ptr.*, e.value_ptr.*);
        };
    }
    const cm = merged;

    const num_key_value_heads = getU32(cm, "num_key_value_heads") orelse getU32(cm, "num_attention_heads");

    var model = Model{
        .arch = arch,
        .dir = name_path,
        .block_count = getU32(cm, "num_hidden_layers") orelse return null,
        .embedding_length = getU32(cm, "hidden_size") orelse return null,
        .feed_forward_length = getU32(cm, "intermediate_size") orelse
            (if (getU32(cm, "num_experts") orelse getU32(cm, "num_local_experts")) |_| 0 else return null),
        // Zero heads is an incomplete config, not a model: the rope dimension
        // count divides by this, and the Q/K row permute groups by it.
        .head_count = nonZero(getU32(cm, "num_attention_heads")) orelse return null,
        .head_count_kv = nonZero(num_key_value_heads) orelse return null,
        // llama.cpp's loader requires <arch>.context_length, so a config that
        // names it under none of these spellings cannot produce a file that
        // loads. Refuse here rather than write one.
        .context_length = nonZero(getU32(cm, "max_position_embeddings") orelse
            getU32(cm, "n_ctx") orelse getU32(cm, "n_positions")) orelse return null,
        .rms_eps = @floatCast(getF64(cm, "rms_norm_eps") orelse return null),
        .rope_theta = @floatCast(readRopeTheta(cm) orelse 10000),
        .rope_scaling = readRopeScaling(cm),
        .vocab_size = getU32(cm, "vocab_size") orelse return null,
        .head_dim = getU32(cm, "head_dim"),
        .is_spm = false,
        .scores = null,
        .tokens = &.{},
        .token_types = &.{},
        .merges = &.{},
        // Read off the tokenizer, not guessed from the model type: model_type
        // does not determine the pre-tokenizer, and a wrong tag here is a
        // silently mis-tokenizing GGUF.
        .tokenizer_pre = "",
        .chat_template = null,
        .add_bos_token = null,
        .add_eos_token = null,
        .add_sep_token = null,
        .bos_id = null,
        .eos_id = null,
        .pad_id = null,
        .unk_id = null,
        .sampling_temp = null,
        .sampling_top_k = null,
        .sampling_top_p = null,
        .license = null,
        .license_link = null,
        .tags = &.{},
        .languages = &.{},
        .datasets = &.{},
        .base_models = &.{},
    };

    if (nonZero(getU32(cm, "num_experts") orelse getU32(cm, "num_local_experts"))) |n| {
        model.expert_count = n;
        // llama.cpp refuses an MoE file with no used count.
        model.expert_used_count = nonZero(getU32(cm, "num_experts_per_tok")) orelse return null;
        model.expert_feed_forward_length = getU32(cm, "moe_intermediate_size");
        model.expert_shared_feed_forward_length = getU32(cm, "shared_expert_intermediate_size");
    }

    model.qwen35 = try readQwen35(
        cm,
        model.block_count,
        model.head_dim orelse (model.embedding_length / model.head_count),
        alloc,
    );
    // The hybrid's dimensions are not optional for an arch that is one: without
    // them the writer would emit a qwen35 file with no ssm.* keys, which
    // llama.cpp refuses to load.
    if (isQwen35Family(arch.name) and model.qwen35 == null) return null;

    if (std.mem.eql(u8, arch.name, "qwen3vl") or std.mem.eql(u8, arch.name, "qwen2vl")) {
        // llama.cpp requires the sections for these archs; without them the
        // file does not load.
        model.rope_sections = readMropeSections(cm) orelse return null;
        if (obj(cm.get("vision_config") orelse std.json.Value{ .null = {} })) |vc| {
            if (vc.get("deepstack_visual_indexes")) |v| if (v == .array) {
                model.deepstack_layers = @intCast(v.array.items.len);
            };
        }
    }

    model.vision = try readVision(io, dir, model_type, c, cm, model.embedding_length, alloc);

    // "model.safetensors" or "." names the checkpoint directory nowhere in the
    // path the user typed, and general.name is what llama.cpp prints: ask the
    // open handle where it actually is. Failure leaves the name parts out,
    // which beats calling the model ".".
    if (checkpointDirName(name_path) == null) {
        var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        if (dir.realPath(io, &buf)) |n| {
            model.dir = try alloc.dupe(u8, buf[0..n]);
        } else |_| {}
    }

    // An unusable tokenizer means we cannot emit loadable metadata, not that
    // this is not a checkpoint: fall back to the plain pipeline, as documented.
    if (reason) |r| r.* = .no_tokenizer;
    var sibling = if (hasTokenizer(io, dir)) null else try siblingTokenizerDir(io, dir, alloc);
    defer if (sibling) |*s| s.dir.close(io);
    const tok_dir = if (sibling) |s| blk: {
        std.log.info("no tokenizer beside {s}; reading it from {s}", .{ name_path, s.path });
        break :blk s.dir;
    } else dir;
    // Nothing to read is not the same as something unreadable: the first is a
    // text encoder that never had a tokenizer, the second a broken checkpoint.
    const vocabless = sibling == null and !anyTokenizerFile(io, dir);
    if (!vocabless) loadTokenizer(io, tok_dir, alloc, &model, c) catch return null;
    loadCard(io, dir, alloc, &model);

    if (readJson(io, dir, "generation_config.json", alloc)) |gc| {
        if (obj(gc.value)) |g| {
            // Only fill ids tokenizer_config/config.json left unset.
            model.eos_id = model.eos_id orelse getIdOrFirst(g, "eos_token_id");
            model.bos_id = model.bos_id orelse getIdOrFirst(g, "bos_token_id");
            model.pad_id = model.pad_id orelse getIdOrFirst(g, "pad_token_id");
            model.unk_id = model.unk_id orelse getIdOrFirst(g, "unk_token_id");
            if (getBool(g, "do_sample", false)) {
                if (getF64(g, "temperature")) |t| model.sampling_temp = @floatCast(t);
                if (getU32(g, "top_k")) |k| model.sampling_top_k = k;
                if (getF64(g, "top_p")) |p| model.sampling_top_p = @floatCast(p);
            }
        }
    } else |_| {}

    if (model.tokens.len == 0 and !vocabless) return null;
    return model;
}

fn hasAnyOf(io: std.Io, dir: std.Io.Dir, names: []const []const u8) bool {
    for (names) |n| {
        if (dir.statFile(io, n, .{})) |_| return true else |_| {}
    }
    return false;
}

/// The files loadTokenizer can build a vocabulary from.
fn hasTokenizer(io: std.Io, dir: std.Io.Dir) bool {
    return hasAnyOf(io, dir, &.{ "tokenizer.json", "tokenizer.model" });
}

/// Any trace of a tokenizer, readable or not.
fn anyTokenizerFile(io: std.Io, dir: std.Io.Dir) bool {
    return hasAnyOf(io, dir, &.{ "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json", "merges.txt" });
}

pub const SiblingDir = struct { dir: std.Io.Dir, path: []const u8 };

/// A diffusers pipeline keeps text_encoder<suffix>/ and tokenizer<suffix>/ side
/// by side. null = `dir` is not named text_encoder*, or its paired tokenizer dir
/// holds no tokenizer file at all. The suffix must match: SDXL's tokenizer_2 is
/// not tokenizer.
pub fn siblingTokenizerDir(io: std.Io, dir: std.Io.Dir, alloc: std.mem.Allocator) !?SiblingDir {
    var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const n = dir.realPath(io, &buf) catch return null;
    const real = buf[0..n];
    const base = std.Io.Dir.path.basename(real);
    const prefix = "text_encoder";
    if (!std.mem.startsWith(u8, base, prefix)) return null;
    const parent = std.Io.Dir.path.dirname(real) orelse return null;
    const path = try alloc.print("{s}/tokenizer{s}", .{ parent, base[prefix.len..] });
    const sd = std.Io.Dir.cwd().openDir(io, path, .{}) catch return null;
    if (!anyTokenizerFile(io, sd)) {
        sd.close(io);
        return null;
    }
    return .{ .dir = sd, .path = path };
}

/// SentencePiece pieces from tokenizer.model's protobuf: repeated field 1
/// of ModelProto, each { string piece = 1; float score = 2; Type type = 3 }.
const SpmPiece = struct { piece: []const u8, score: f32, ptype: u64 };

/// llama.cpp's LlamaModel.permute: within each head, the (2, half) row split
/// and its inverse swap the two RoPE halves. ggufy writes half-split order
/// (`to_hf = false`, HF source interleaved -> GGUF) and undoes it on read
/// back to HF (`to_hf = true`). The two maps are positional inverses, not
/// the same permutation.
pub fn ropePermuteInPlace(
    alloc: std.mem.Allocator,
    bytes: []u8,
    elem_size: usize,
    rows: usize,
    cols: usize,
    groups: usize,
    to_hf: bool,
) !void {
    // The caller has already decided this tensor carries RoPE rows, so a shape
    // that cannot be permuted is a bug upstream, not a tensor to pass through:
    // skipping it silently writes Q/K rows in the wrong order into a file that
    // claims the other one, which nothing downstream can detect.
    if (groups == 0 or rows % (groups * 2) != 0) return error.RopeRowsNotDivisible;
    const row_size = cols * elem_size;
    if (bytes.len != rows * row_size) return error.RopeSizeMismatch;
    const half = rows / groups / 2;
    if (half == 0) return;
    const span = 2 * half;

    // One row of scratch, not a copy of the tensor: an 8B attn_q.weight in f32
    // would be half a gigabyte on top of the dequant buffer. Each group is
    // permuted by walking its cycles, filling every position from its source
    // and carrying the row the cycle started on in `tmp`.
    const tmp = try alloc.alloc(u8, row_size);
    defer alloc.free(tmp);
    const seen = try alloc.alloc(bool, span);
    defer alloc.free(seen);

    for (0..groups) |g| {
        const base = g * span;
        @memset(seen, false);
        for (0..span) |start| {
            if (seen[start]) continue;
            @memcpy(tmp, bytes[(base + start) * row_size ..][0..row_size]);
            var dst = start;
            while (true) {
                seen[dst] = true;
                // Position dst takes the row that sat at src: interleaved
                // a*half+b on the way out, half-split b*2+a on the way back.
                const src = if (to_hf) (dst % half) * 2 + dst / half else (dst % 2) * half + dst / 2;
                if (src == start) {
                    @memcpy(bytes[(base + dst) * row_size ..][0..row_size], tmp);
                    break;
                }
                @memcpy(bytes[(base + dst) * row_size ..][0..row_size], bytes[(base + src) * row_size ..][0..row_size]);
                dst = src;
            }
        }
    }
}

/// Gated DeltaNet linear attention stores V heads grouped under their K head;
/// ggml's binary ops broadcast tiled, so llama.cpp's converter transposes the
/// two. Reading the rows as [k_heads][v_per_k][head_dim], row block (k, v)
/// moves to (v, k) and the head_dim rows inside a block keep their order.
///
/// Only meaningful where k_heads != v_heads. `in_proj_qkv` carries the q and k
/// rows ahead of the v rows, so that caller passes the v slice alone.
///
/// Getting this wrong produces a file that loads and generates nonsense, which
/// is why it is a hard error rather than a skip: nothing downstream can tell
/// grouped rows from tiled ones.
/// `to_hf` runs it backwards, tiled -> grouped. The two are positional
/// inverses, not the same permutation: k_heads and v_per_k differ, so calling
/// the forward direction twice scrambles the heads instead of restoring them.
pub fn vReorderInPlace(
    alloc: std.mem.Allocator,
    bytes: []u8,
    elem_size: usize,
    rows: usize,
    cols: usize,
    k_heads: usize,
    v_per_k: usize,
    head_dim: usize,
    to_hf: bool,
) !void {
    if (k_heads == 0 or v_per_k == 0 or head_dim == 0) return error.VReorderBadShape;
    if (rows != k_heads * v_per_k * head_dim) return error.VReorderRowsNotDivisible;
    const row_size = cols * elem_size;
    if (bytes.len != rows * row_size) return error.VReorderSizeMismatch;
    if (v_per_k == 1) return;

    const tmp = try alloc.alloc(u8, row_size);
    defer alloc.free(tmp);
    const seen = try alloc.alloc(bool, rows);
    defer alloc.free(seen);
    @memset(seen, false);

    for (0..rows) |start| {
        if (seen[start]) continue;
        @memcpy(tmp, bytes[start * row_size ..][0..row_size]);
        var dst = start;
        while (true) {
            seen[dst] = true;
            // Going out, dst splits as v*(k_heads*head_dim) + k*head_dim + d and
            // the row that belongs there sat at k*(v_per_k*head_dim) + v*head_dim
            // + d. Coming back the two groupings swap roles.
            const d = dst % head_dim;
            const src = if (to_hf) blk: {
                const v = (dst / head_dim) % v_per_k;
                const k = dst / (head_dim * v_per_k);
                break :blk v * (k_heads * head_dim) + k * head_dim + d;
            } else blk: {
                const k = (dst / head_dim) % k_heads;
                const v = dst / (head_dim * k_heads);
                break :blk k * (v_per_k * head_dim) + v * head_dim + d;
            };
            if (src == start) {
                @memcpy(bytes[dst * row_size ..][0..row_size], tmp);
                break;
            }
            @memcpy(bytes[dst * row_size ..][0..row_size], bytes[src * row_size ..][0..row_size]);
            dst = src;
        }
    }
}

pub const RopeHeads = struct { head_count: usize, head_count_kv: usize };

pub fn ropePermuteGroups(heeds: RopeHeads, name: []const u8) ?usize {
    if (std.mem.endsWith(u8, name, "attn_q.weight") or std.mem.endsWith(u8, name, "attn_q.bias") or
        std.mem.endsWith(u8, name, "self_attn.q_proj.weight") or std.mem.endsWith(u8, name, "self_attn.q_proj.bias"))
    {
        return heeds.head_count;
    }
    if (std.mem.endsWith(u8, name, "attn_k.weight") or std.mem.endsWith(u8, name, "attn_k.bias") or
        std.mem.endsWith(u8, name, "self_attn.k_proj.weight") or std.mem.endsWith(u8, name, "self_attn.k_proj.bias"))
    {
        return heeds.head_count_kv;
    }
    return null;
}

fn readVarint(bytes: []const u8, pos: *usize) !u64 {
    var value: u64 = 0;
    var shift: u6 = 0;
    while (pos.* < bytes.len) {
        const b = bytes[pos.*];
        pos.* += 1;
        value |= @as(u64, b & 0x7f) << shift;
        if (b < 0x80) return value;
        if (shift < 63) shift += 7 else return error.BadProtobuf;
    }
    return error.BadProtobuf;
}

fn parseSpmModel(alloc: std.mem.Allocator, bytes: []const u8) ![]SpmPiece {
    var pieces: std.ArrayList(SpmPiece) = .empty;
    var pos: usize = 0;
    while (pos < bytes.len) {
        const tag = try readVarint(bytes, &pos);
        const field = tag >> 3;
        switch (@as(u3, @intCast(tag & 7))) {
            0 => _ = try readVarint(bytes, &pos),
            1 => pos += 8,
            2 => {
                const len = try readVarint(bytes, &pos);
                if (len > bytes.len - pos) return error.BadProtobuf;
                const end = pos + len;
                if (field == 1) {
                    var ep: usize = 0;
                    const entry = bytes[pos..end];
                    var piece: []const u8 = "";
                    var score: f32 = 0;
                    var ptype: u64 = 1;
                    while (ep < entry.len) {
                        const etag = try readVarint(entry, &ep);
                        switch (@as(u3, @intCast(etag & 7))) {
                            0 => {
                                const v = try readVarint(entry, &ep);
                                if (etag >> 3 == 3) ptype = v;
                            },
                            2 => {
                                const elen = try readVarint(entry, &ep);
                                if (elen > entry.len - ep) return error.BadProtobuf;
                                const eend = ep + elen;
                                if (etag >> 3 == 1) piece = entry[ep..eend];
                                ep = eend;
                            },
                            5 => {
                                if (ep + 4 > entry.len) return error.BadProtobuf;
                                if (etag >> 3 == 2) {
                                    const bits = std.mem.readInt(u32, entry[ep..][0..4], .little);
                                    score = @bitCast(bits);
                                }
                                ep += 4;
                            },
                            1 => ep += 8,
                            else => return error.BadProtobuf,
                        }
                    }
                    try pieces.append(alloc, .{ .piece = piece, .score = score, .ptype = ptype });
                    pos = end;
                } else pos = end;
            },
            5 => pos += 4,
            else => return error.BadProtobuf,
        }
    }
    return pieces.items;
}

fn spmTokenType(piece: []const u8, ptype: u64) u32 {
    // <0xNN> byte placeholders take the BYTE tag regardless of proto type.
    if (piece.len == 6 and std.mem.startsWith(u8, piece, "<0x") and piece[piece.len - 1] == '>') {
        return GGUF_TOKEN_TYPE_BYTE;
    }
    // spm's Type enum and gguf's token types share their numbering, and
    // llama.cpp's converter maps them straight across. Keeping UNKNOWN distinct
    // from CONTROL is what lets the -H path find the unk piece again.
    return switch (ptype) {
        0, 1 => GGUF_TOKEN_TYPE_NORMAL,
        2 => GGUF_TOKEN_TYPE_UNKNOWN,
        3 => GGUF_TOKEN_TYPE_CONTROL,
        4 => GGUF_TOKEN_TYPE_USER_DEFINED,
        5 => GGUF_TOKEN_TYPE_UNUSED,
        6 => GGUF_TOKEN_TYPE_BYTE,
        else => GGUF_TOKEN_TYPE_UNUSED,
    };
}

/// llama.cpp's does_token_look_special: checkpoints routinely leave a token
/// that has to be a control token marked as not special, so the bracketed
/// spellings count as special on their own.
fn looksSpecial(s: []const u8) bool {
    for ([_][]const u8{ "<pad>", "<mask>", "<2mass>", "[@BOS@]" }) |known| {
        if (std.mem.eql(u8, s, known)) return true;
    }
    if (std.mem.startsWith(u8, s, "<|") and std.mem.endsWith(u8, s, "|>")) return true;
    if (std.mem.startsWith(u8, s, "<\u{ff5c}") and std.mem.endsWith(u8, s, "\u{ff5c}>")) return true;
    if (std.mem.startsWith(u8, s, "<unused") and std.mem.endsWith(u8, s, ">")) return true;
    return false;
}

const AddedToken = struct { id: u32, content: []const u8, special: bool };

/// Tokens a checkpoint adds on top of its trained vocabulary, from
/// tokenizer.json, added_tokens.json and tokenizer_config's
/// added_tokens_decoder. Returned in that order, later entries overriding
/// earlier ones for the same id, which is the order llama.cpp reads them in.
fn addedTokens(
    io: std.Io,
    dir: std.Io.Dir,
    alloc: std.mem.Allocator,
    t_root: ?std.json.ObjectMap,
    tcc: ?std.json.ObjectMap,
) ![]const AddedToken {
    var list: std.ArrayList(AddedToken) = .empty;
    if (t_root) |root| if (root.get("added_tokens")) |v| {
        if (v == .array) for (v.array.items) |item| {
            const a = obj(item) orelse continue;
            const content = getStr(a, "content") orelse continue;
            try list.append(alloc, .{
                .id = getU32(a, "id") orelse continue,
                .content = content,
                .special = getBool(a, "special", false) or looksSpecial(content),
            });
        };
    };
    if (readJson(io, dir, "added_tokens.json", alloc)) |aj| {
        if (obj(aj.value)) |o| {
            var it = o.iterator();
            while (it.next()) |e| {
                const id = switch (e.value_ptr.*) {
                    .integer => |i| if (i >= 0) std.math.cast(u32, i) orelse continue else continue,
                    else => continue,
                };
                try list.append(alloc, .{
                    .id = id,
                    .content = e.key_ptr.*,
                    .special = looksSpecial(e.key_ptr.*),
                });
            }
        }
    } else |_| {}
    if (tcc) |t| if (t.get("added_tokens_decoder")) |v| {
        if (v == .object) {
            var it = v.object.iterator();
            while (it.next()) |e| {
                const id = std.fmt.parseInt(u32, e.key_ptr.*, 10) catch continue;
                const a = obj(e.value_ptr.*) orelse continue;
                const content = getStr(a, "content") orelse continue;
                try list.append(alloc, .{
                    .id = id,
                    .content = content,
                    .special = getBool(a, "special", false) or looksSpecial(content),
                });
            }
        }
    };
    return list.items;
}

fn loadTokenizer(io: std.Io, dir: std.Io.Dir, alloc: std.mem.Allocator, model: *Model, cfg: std.json.ObjectMap) !void {
    var t_root: ?std.json.ObjectMap = null;
    var t_model: ?std.json.ObjectMap = null;
    if (readJson(io, dir, "tokenizer.json", alloc)) |tj| {
        if (obj(tj.value)) |root| {
            t_root = root;
            t_model = obj(root.get("model") orelse std.json.Value{ .null = {} });
        }
    } else |_| {}

    var chat_template: ?[]const u8 = null;
    var tcc: ?std.json.ObjectMap = null;
    if (readJson(io, dir, "tokenizer_config.json", alloc)) |tc| {
        if (obj(tc.value)) |tcv| {
            tcc = tcv;
            chat_template = getStr(tcv, "chat_template");
        }
    } else |_| {}
    // Newer checkpoints keep the template in its own file.
    if (chat_template == null) {
        chat_template = dir.readFileAlloc(io, "chat_template.jinja", alloc, .limited64(16 * 1024 * 1024)) catch null;
    }
    model.chat_template = chat_template;

    // tokenizer.model wins over tokenizer.json for the vocab, as in llama.cpp's
    // converter (sentencepiece first, gpt2 fallback).
    const spm_bytes = dir.readFileAlloc(io, "tokenizer.model", alloc, .limited64(256 * 1024 * 1024)) catch null;
    if (spm_bytes) |bytes| {
        const pieces = parseSpmModel(alloc, bytes) catch return error.BadTokenizer;
        if (pieces.len == 0) return error.BadTokenizer;
        // A tuned checkpoint adds tokens the proto never had (a ChatML pair on a
        // 32000-piece base), with rows of their own in token_embd; take the
        // highest id anything defines, as the BPE branch below does.
        const added = try addedTokens(io, dir, alloc, t_root, tcc);
        var n = @max(pieces.len, model.vocab_size);
        for (added) |at| n = @max(n, @as(usize, at.id) + 1);
        var tokens = try alloc.alloc([]const u8, n);
        var types_ = try alloc.alloc(u32, n);
        var scores = try alloc.alloc(f32, n);
        for (pieces, 0..) |p, i| {
            tokens[i] = p.piece;
            types_[i] = spmTokenType(p.piece, p.ptype);
            // llama.cpp keeps the trained score for every piece in the proto;
            // the added-token overlay below flattens the ones it touches.
            scores[i] = p.score;
        }
        for (pieces.len..n) |i| {
            tokens[i] = try alloc.print("[PAD{d}]", .{i});
            types_[i] = GGUF_TOKEN_TYPE_UNUSED;
            scores[i] = -10000;
        }
        // Without this overlay a ChatML tune's <|im_end|> stays a [PAD]
        // placeholder, the chat template's markers tokenize as text, and
        // nothing stops generation. A piece the proto already spells the same
        // way keeps its proto type unless a plain one is marked special:
        // UNKNOWN and BYTE mean things "special" does not, and -H reads them.
        for (added) |at| {
            if (at.id < pieces.len and std.mem.eql(u8, tokens[at.id], at.content)) {
                if (at.special and types_[at.id] == GGUF_TOKEN_TYPE_NORMAL) {
                    types_[at.id] = GGUF_TOKEN_TYPE_CONTROL;
                }
                continue;
            }
            tokens[at.id] = at.content;
            types_[at.id] = if (at.special) GGUF_TOKEN_TYPE_CONTROL else GGUF_TOKEN_TYPE_USER_DEFINED;
            // llama.cpp flattens every added token's score, trained or not.
            scores[at.id] = -1000;
        }
        model.tokens = tokens;
        model.token_types = types_;
        model.scores = scores;
        model.is_spm = true;
        model.tokenizer_pre = "default";
    } else {
        const root = t_root orelse return error.BadTokenizer;
        const tmod = t_model orelse return error.BadTokenizer;
        if (!std.mem.eql(u8, getStr(tmod, "type") orelse "", "BPE")) return error.BadTokenizer;
        // Empty = splitting no tag describes. Whether that stops the conversion
        // or writes llama.cpp's "default" is the caller's call, not this one's.
        model.tokenizer_pre = detectPretok(root) orelse "";
        const vocab = obj(tmod.get("vocab") orelse return error.BadTokenizer) orelse return error.BadTokenizer;

        // A tuned checkpoint can add tokens above config.json's vocab_size (a
        // ChatML pair on a 32000-token base). Their rows are in token_embd, and
        // llama.cpp rejects a token list shorter than it, so take the highest id
        // either table defines.
        // Every source a fast tokenizer takes added tokens from, as the SPM
        // branch reads them: a marker declared only in added_tokens.json or
        // tokenizer_config's added_tokens_decoder would otherwise stay a [PAD]
        // hole, and the chat template's markers tokenize as text.
        const added = try addedTokens(io, dir, alloc, t_root, tcc);

        var n: usize = model.vocab_size;
        var nit = vocab.iterator();
        while (nit.next()) |entry| {
            if (entry.value_ptr.* == .integer and entry.value_ptr.*.integer >= 0)
                n = @max(n, @as(usize, @intCast(entry.value_ptr.*.integer)) + 1);
        }
        for (added) |at| n = @max(n, @as(usize, at.id) + 1);

        var tokens = try alloc.alloc(?[]const u8, n);
        @memset(tokens, null);
        var types_ = try alloc.alloc(u32, n);
        @memset(types_, GGUF_TOKEN_TYPE_UNUSED);

        var it = vocab.iterator();
        while (it.next()) |entry| {
            const id: usize = switch (entry.value_ptr.*) {
                .integer => |i| if (i >= 0) @intCast(i) else continue,
                else => continue,
            };
            tokens[id] = entry.key_ptr.*;
            types_[id] = GGUF_TOKEN_TYPE_NORMAL;
        }

        for (added) |at| {
            // llama.cpp's rule: the special flag (or a spelling that looks like
            // one) is CONTROL, every other added token USER_DEFINED. The split
            // matters at runtime: llama.cpp drops CONTROL tokens from generated
            // text unless the caller asks for specials, so a <think> typed
            // CONTROL vanishes from the output stream.
            if (at.special) {
                tokens[at.id] = at.content;
                types_[at.id] = GGUF_TOKEN_TYPE_CONTROL;
            } else {
                // User-defined tokens are pre-normalized: the metaspace marker
                // becomes a plain space, as llama.cpp's converter writes it.
                tokens[at.id] = try std.mem.replaceOwned(u8, alloc, at.content, "\u{2581}", " ");
                types_[at.id] = GGUF_TOKEN_TYPE_USER_DEFINED;
            }
        }

        // Holes in the defined vocab become "[PAD]<id>", as llama.cpp's converter writes.
        var final_tokens = try alloc.alloc([]const u8, n);
        for (tokens, 0..) |maybe_tok, i| {
            final_tokens[i] = maybe_tok orelse blk: {
                types_[i] = GGUF_TOKEN_TYPE_UNUSED;
                break :blk try alloc.print("[PAD{d}]", .{i});
            };
        }
        model.tokens = final_tokens;
        model.token_types = types_;

        loadMerges(tmod, io, dir, alloc, model);
    }

    resolveSpecialTokens(tcc, cfg, model);
    applyPostProcessor(t_root, tcc, model);
}

/// tokenizer.json merges entries are either strings or [left, right] pairs;
/// pairs join with a space, matching the raw merges.txt form. With no merges
/// in tokenizer.json the raw merges.txt lines are used.
fn loadMerges(tmod: std.json.ObjectMap, io: std.Io, dir: std.Io.Dir, alloc: std.mem.Allocator, model: *Model) void {
    if (tmod.get("merges")) |v| {
        if (v == .array and v.array.items.len > 0) {
            var list: std.ArrayList([]const u8) = .empty;
            var skipped: usize = 0;
            for (v.array.items) |m| {
                const entry: ?[]const u8 = switch (m) {
                    .string => |s| s,
                    .array => |pair| if (pair.items.len == 2 and
                        pair.items[0] == .string and pair.items[1] == .string)
                        alloc.print("{s} {s}", .{ pair.items[0].string, pair.items[1].string }) catch return
                    else
                        null,
                    else => null,
                };
                // An entry in neither form is not a rule. Keeping its place with an
                // empty string writes a blank line into the GGUF merge table, which
                // llama.cpp's BPE loader reads as a rule over two empty pieces.
                list.append(alloc, entry orelse {
                    skipped += 1;
                    continue;
                }) catch return;
            }
            if (skipped > 0) {
                std.log.warn("tokenizer.json: {d} merges entries are neither a string nor a [left, right] pair; leaving them out", .{skipped});
            }
            // All of them dropped is the same as none: fall through to
            // merges.txt rather than write a BPE vocabulary with no merge table,
            // which llama.cpp refuses to load.
            if (list.items.len > 0) {
                model.merges = list.items;
                return;
            }
        }
    }
    const txt = dir.readFileAlloc(io, "merges.txt", alloc, .limited64(256 * 1024 * 1024)) catch return;
    var lines = std.mem.splitScalar(u8, txt, '\n');
    var list: std.ArrayList([]const u8) = .empty;
    var first = true;
    while (lines.next()) |line| {
        if (first) {
            first = false;
            if (std.mem.startsWith(u8, line, "#version")) continue;
        }
        const trimmed = std.mem.trim(u8, line, "\r");
        if (trimmed.len > 0) list.append(alloc, trimmed) catch return;
    }
    if (list.items.len > 0) model.merges = list.items;
}

/// tokenizer_config's "<name>_token" strings name the special tokens; ids are
/// looked up in the vocabulary. config.json numeric ids fill what is missing.
fn resolveSpecialTokens(tcc: ?std.json.ObjectMap, cfg: std.json.ObjectMap, model: *Model) void {
    const Special = struct { key: []const u8, id_key: []const u8 };
    const specials = [_]Special{
        .{ .key = "bos_token", .id_key = "bos_token_id" },
        .{ .key = "eos_token", .id_key = "eos_token_id" },
        .{ .key = "pad_token", .id_key = "pad_token_id" },
        .{ .key = "unk_token", .id_key = "unk_token_id" },
    };
    for (specials) |s| {
        var id: ?u32 = null;
        if (tcc) |t| {
            var name: ?[]const u8 = null;
            if (t.get(s.key)) |v| switch (v) {
                .string => |str| name = str,
                .object => |o| name = getStr(o, "content"),
                else => {},
            };
            if (name) |nm| for (model.tokens, 0..) |tok, i| {
                if (std.mem.eql(u8, tok, nm)) {
                    id = @intCast(i);
                    break;
                }
            };
        }
        // Only now config.json: a tune that moves eos to <|im_end|> says so in
        // tokenizer_config and leaves the base model's eos_token_id in place.
        // A list here (Llama 3 spells eos as one) keeps its first entry, as in
        // llama.cpp's converter.
        if (id == null) id = getIdOrFirst(cfg, s.id_key);
        if (id) |v| {
            if (std.mem.eql(u8, s.key, "bos_token")) model.bos_id = v
            else if (std.mem.eql(u8, s.key, "eos_token")) model.eos_id = v
            else if (std.mem.eql(u8, s.key, "pad_token")) model.pad_id = v
            else model.unk_id = v;
        }
    }
}

/// tokenizer.json post_processor TemplateProcessing decides whether BOS/EOS/SEP
/// are prepended/appended, mirroring llama.cpp's crude parsing (simple
/// templates only).
fn applyPostProcessor(t_root: ?std.json.ObjectMap, tcc: ?std.json.ObjectMap, model: *Model) void {
    // Explicit tokenizer_config add_<typ>_token bools win outright.
    if (tcc) |t| {
        if (t.get("add_bos_token")) |v| {
            if (v == .bool) model.add_bos_token = v.bool;
        }
        if (t.get("add_eos_token")) |v| {
            if (v == .bool) model.add_eos_token = v.bool;
        }
        if (t.get("add_sep_token")) |v| {
            if (v == .bool) model.add_sep_token = v.bool;
        }
    }

    const ppv = (t_root orelse return).get("post_processor") orelse return;
    var pp = obj(ppv) orelse return;
    if (pp.get("processors")) |plist| {
        if (plist != .array) return;
        pp = std.json.ObjectMap.empty;
        for (plist.array.items) |item| {
            const o = obj(item) orelse continue;
            if (std.mem.eql(u8, getStr(o, "type") orelse "", "TemplateProcessing")) {
                pp = o;
                break;
            }
        }
    }
    if (!std.mem.eql(u8, getStr(pp, "type") orelse "", "TemplateProcessing")) return;

    var bos_str: ?[]const u8 = null;
    var eos_str: ?[]const u8 = null;
    var sep_str: ?[]const u8 = null;
    if (tcc) |t| {
        bos_str = specialStr(t, "bos_token");
        eos_str = specialStr(t, "eos_token");
        sep_str = specialStr(t, "sep_token");
    }

    const single = if (pp.get("single")) |v| (if (v == .array) v.array.items else null) else null;
    var special_first: ?[]const u8 = null;
    var special_last: ?[]const u8 = null;
    if (single) |s| if (s.len > 1) {
        special_first = specTok(s[0]);
        special_last = specTok(s[s.len - 1]);
        if (special_first) |f| model.add_bos_token = model.add_bos_token orelse std.mem.eql(u8, f, bos_str orelse "");
        if (special_last) |l| model.add_eos_token = model.add_eos_token orelse std.mem.eql(u8, l, eos_str orelse "");
    };

    if (pp.get("pair")) |pv| {
        if (pv != .array) return;
        const items = pv.array.items;
        if (items.len == 0) return;
        const start: usize = if (special_first) |f| @intFromBool(eqSpec(items[0], f)) else 0;
        const stop = if (special_last) |l|
            items.len - @as(usize, @intFromBool(eqSpec(items[items.len - 1], l)))
        else
            items.len;
        // One entry that is both the leading and the trailing special token
        // (a tokenizer whose BOS and EOS are the same) crosses the two ends.
        if (start >= stop) return;
        const tr = items[start..stop];
        if (tr.len > 2 and eqSeq(tr[0], "A") and eqSeq(tr[tr.len - 1], "B")) {
            const mid = specTok(tr[1]) orelse "";
            model.add_sep_token = model.add_sep_token orelse (special_last == null and
                (std.mem.eql(u8, mid, eos_str orelse "\xff") or std.mem.eql(u8, mid, sep_str orelse "\xff")));
        }
    }
}

fn specialStr(o: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = o.get(key) orelse return null;
    return switch (v) {
        .string => |s| s,
        .object => |obj_| getStr(obj_, "content"),
        else => null,
    };
}

fn specTok(item: std.json.Value) ?[]const u8 {
    const o = obj(item) orelse return null;
    const st = obj(o.get("SpecialToken") orelse return null) orelse return null;
    return getStr(st, "id");
}

fn eqSpec(item: std.json.Value, want: []const u8) bool {
    const id = specTok(item) orelse return false;
    return std.mem.eql(u8, id, want);
}

fn eqSeq(item: std.json.Value, want: []const u8) bool {
    const o = obj(item) orelse return false;
    const st = obj(o.get("Sequence") orelse return false) orelse return false;
    const id = getStr(st, "id") orelse return false;
    return std.mem.eql(u8, id, want);
}

/// HF state-dict name -> llama.cpp native name. null = not a tensor llama.cpp
/// consumes (rotary caches and friends); the caller drops those for GGUF output.
/// Tensors llama.cpp's converter drops on purpose: rotary caches and the
/// attention masks it rebuilds itself. Anything else mapName cannot place is a
/// hole in the table, and dropping it would write an incomplete model.
pub fn isIgnoredTensor(name: []const u8) bool {
    const ignored = [_][]const u8{
        "rotary_emb.inv_freq",
        "attention.masked_bias",
        "attention.bias",
    };
    for (ignored) |suffix| if (std.mem.endsWith(u8, name, suffix)) return true;
    return false;
}

/// The same, for the GGUF->HF direction: native tensors transformers has no slot
/// for because it recomputes them from config.json at load time.
pub fn isIgnoredNativeTensor(name: []const u8) bool {
    return std.mem.endsWith(u8, name, "rope_freqs.weight");
}

/// Per-layer suffixes shared by the qwen35 hybrid: the full-attention blocks
/// carry the self_attn set, the Gated DeltaNet blocks the linear_attn set, and
/// both carry the MLP and the two norms. A_log and dt_bias are the only ones
/// with no .weight on either side.
const NamePair = struct { from: []const u8, to: []const u8 };

const qwen35_suffix = [_]NamePair{
    .{ .from = "input_layernorm.weight", .to = "attn_norm.weight" },
    .{ .from = "post_attention_layernorm.weight", .to = "post_attention_norm.weight" },
    .{ .from = "self_attn.q_proj.weight", .to = "attn_q.weight" },
    .{ .from = "self_attn.k_proj.weight", .to = "attn_k.weight" },
    .{ .from = "self_attn.v_proj.weight", .to = "attn_v.weight" },
    .{ .from = "self_attn.o_proj.weight", .to = "attn_output.weight" },
    .{ .from = "self_attn.q_norm.weight", .to = "attn_q_norm.weight" },
    .{ .from = "self_attn.k_norm.weight", .to = "attn_k_norm.weight" },
    .{ .from = "mlp.gate_proj.weight", .to = "ffn_gate.weight" },
    .{ .from = "mlp.up_proj.weight", .to = "ffn_up.weight" },
    .{ .from = "mlp.down_proj.weight", .to = "ffn_down.weight" },
    .{ .from = "linear_attn.in_proj_qkv.weight", .to = "attn_qkv.weight" },
    .{ .from = "linear_attn.in_proj_z.weight", .to = "attn_gate.weight" },
    .{ .from = "linear_attn.in_proj_a.weight", .to = "ssm_alpha.weight" },
    .{ .from = "linear_attn.in_proj_b.weight", .to = "ssm_beta.weight" },
    .{ .from = "linear_attn.conv1d.weight", .to = "ssm_conv1d.weight" },
    .{ .from = "linear_attn.norm.weight", .to = "ssm_norm.weight" },
    .{ .from = "linear_attn.out_proj.weight", .to = "ssm_out.weight" },
    .{ .from = "linear_attn.A_log", .to = "ssm_a" },
    .{ .from = "linear_attn.dt_bias", .to = "ssm_dt.bias" },
} ++ moe_suffix;

/// The MoE block's single tensors. The experts themselves stack into one tensor
/// per projection, which a rename cannot express: see expertPart.
const moe_suffix = [_]NamePair{
    .{ .from = "mlp.gate.weight", .to = "ffn_gate_inp.weight" },
    .{ .from = "mlp.shared_expert.gate_proj.weight", .to = "ffn_gate_shexp.weight" },
    .{ .from = "mlp.shared_expert.up_proj.weight", .to = "ffn_up_shexp.weight" },
    .{ .from = "mlp.shared_expert.down_proj.weight", .to = "ffn_down_shexp.weight" },
    .{ .from = "mlp.shared_expert_gate.weight", .to = "ffn_gate_inp_shexp.weight" },
};

pub const ExpertProj = enum { gate, up, down, gate_up };

/// One HF expert weight. `expert` null = the fused 3-D form newer transformers
/// write, one tensor holding every expert.
pub const ExpertPart = struct { block: u32, expert: ?u32, proj: ExpertProj };

/// Parse an HF MoE expert weight name: model.layers.N.mlp.experts.M.gate_proj.weight,
/// or the fused model.layers.N.mlp.experts.gate_up_proj. The qwen35 text tower
/// and MTP head prefixes are accepted too; `mtp_base` places the latter, and
/// null leaves its names unparsed.
pub fn expertPart(name: []const u8, mtp_base: ?u32) ?ExpertPart {
    var block: u32 = undefined;
    var rest: []const u8 = undefined;
    if (std.mem.startsWith(u8, name, "mtp.layers.0.")) {
        block = mtp_base orelse return null;
        rest = name["mtp.layers.0.".len..];
    } else {
        const body = if (std.mem.startsWith(u8, name, "model.language_model.layers."))
            name["model.language_model.layers.".len..]
        else if (std.mem.startsWith(u8, name, "model.layers."))
            name["model.layers.".len..]
        else
            return null;
        const dot = std.mem.findScalar(u8, body, '.') orelse return null;
        block = std.fmt.parseInt(u32, body[0..dot], 10) catch return null;
        rest = body[dot + 1 ..];
    }
    const pre = "mlp.experts.";
    if (!std.mem.startsWith(u8, rest, pre)) return null;
    rest = rest[pre.len..];

    const Fused = struct { from: []const u8, proj: ExpertProj };
    for ([_]Fused{
        .{ .from = "gate_up_proj", .proj = .gate_up },
        .{ .from = "down_proj", .proj = .down },
    }) |f| {
        if (std.mem.eql(u8, rest, f.from) or
            (std.mem.startsWith(u8, rest, f.from) and std.mem.eql(u8, rest[f.from.len..], ".weight")))
            return .{ .block = block, .expert = null, .proj = f.proj };
    }

    const dot = std.mem.findScalar(u8, rest, '.') orelse return null;
    const expert = std.fmt.parseInt(u32, rest[0..dot], 10) catch return null;
    const Per = struct { from: []const u8, proj: ExpertProj };
    for ([_]Per{
        .{ .from = "gate_proj.weight", .proj = .gate },
        .{ .from = "up_proj.weight", .proj = .up },
        .{ .from = "down_proj.weight", .proj = .down },
    }) |p| {
        if (std.mem.eql(u8, rest[dot + 1 ..], p.from)) return .{ .block = block, .expert = expert, .proj = p.proj };
    }
    return null;
}

/// Plan the stacked expert tensors for every block: `tensors` are the HF expert
/// weights expertPart accepts, in any order. Refuses (after naming what is
/// wrong) a block missing an expert, experts that disagree in shape or type, and
/// a fused tensor whose leading dimension is not `n_expert`: any of those would
/// write a tensor llama.cpp routes tokens into the wrong rows of.
pub fn planExpertStacks(
    alloc: std.mem.Allocator,
    tensors: []const types.Tensor,
    n_expert: u32,
    mtp_base: ?u32,
) ![]types.ExpertStack {
    if (tensors.len == 0) return &.{};
    if (n_expert == 0) {
        std.log.warn("the checkpoint carries MoE expert weights ({s}, ...) but its config.json names no num_experts", .{tensors[0].name});
        return error.IncompleteExperts;
    }
    // Keyed by block and output projection; gate_up feeds both gate and up.
    const Key = struct { block: u32, proj: ExpertProj };
    const Slot = struct { per: []?types.Tensor, fused: ?types.Tensor = null, fused_half: u1 = 0 };
    var slots: std.array_hash_map.Auto(Key, Slot) = .empty;

    for (tensors) |t| {
        const part = expertPart(t.name, mtp_base) orelse continue;
        const outs: []const ExpertProj = if (part.proj == .gate_up) &.{ .gate, .up } else &.{part.proj};
        for (outs, 0..) |proj, half| {
            const gop = try slots.getOrPut(alloc, .{ .block = part.block, .proj = proj });
            if (!gop.found_existing) {
                const per = try alloc.alloc(?types.Tensor, n_expert);
                @memset(per, null);
                gop.value_ptr.* = .{ .per = per };
            }
            if (part.expert) |e| {
                if (e >= n_expert) {
                    std.log.warn("{s}: expert {d} is past config.json's num_experts ({d})", .{ t.name, e, n_expert });
                    return error.IncompleteExperts;
                }
                gop.value_ptr.per[e] = t;
            } else {
                gop.value_ptr.fused = t;
                gop.value_ptr.fused_half = @intCast(half);
            }
        }
    }

    var stacks: std.ArrayList(types.ExpertStack) = .empty;
    var it = slots.iterator();
    while (it.next()) |e| {
        const key = e.key_ptr.*;
        const slot = e.value_ptr.*;
        const name = try alloc.print("blk.{d}.ffn_{s}_exps.weight", .{ key.block, @tagName(key.proj) });
        var any_per = false;
        for (slot.per) |p| any_per = any_per or p != null;
        if (slot.fused) |f| {
            if (any_per) {
                std.log.warn("{s}: block {d} carries both fused and per-expert weights", .{ name, key.block });
                return error.IncompleteExperts;
            }
            if (f.dims.len != 3 or f.dims[0] != n_expert or (key.proj != .down and f.dims[1] % 2 != 0)) {
                std.log.warn("{s} has shape {any}, not [{d}, rows, cols]", .{ f.name, f.dims, n_expert });
                return error.IncompleteExperts;
            }
            if (key.proj == .down) {
                try stacks.append(alloc, .{ .name = name, .dims = f.dims, .dtype = f.type, .segments = try alloc.dupe(types.StackSegment, &.{
                    .{ .tensor = f, .elem_offset = 0, .elem_count = f.dims[0] * f.dims[1] * f.dims[2] },
                }) });
                continue;
            }
            // gate is the first half of each expert's rows, up the second, as
            // llama.cpp's converter splits them.
            const rows = f.dims[1] / 2;
            const cols = f.dims[2];
            const segs = try alloc.alloc(types.StackSegment, n_expert);
            for (segs, 0..) |*sg, x| sg.* = .{
                .tensor = f,
                .elem_offset = (x * 2 + slot.fused_half) * rows * cols,
                .elem_count = rows * cols,
            };
            try stacks.append(alloc, .{ .name = name, .dims = try alloc.dupe(usize, &.{ n_expert, rows, cols }), .dtype = f.type, .segments = segs });
            continue;
        }
        const first = slot.per[0] orelse {
            std.log.warn("{s}: expert 0 is missing", .{name});
            return error.IncompleteExperts;
        };
        if (first.dims.len != 2) {
            std.log.warn("{s} has shape {any}, not [rows, cols]", .{ first.name, first.dims });
            return error.IncompleteExperts;
        }
        const segs = try alloc.alloc(types.StackSegment, n_expert);
        for (slot.per, 0..) |maybe, x| {
            const p = maybe orelse {
                std.log.warn("{s}: expert {d} of {d} is missing", .{ name, x, n_expert });
                return error.IncompleteExperts;
            };
            if (!std.mem.eql(usize, p.dims, first.dims) or !std.mem.eql(u8, p.type, first.type)) {
                std.log.warn("{s} is {s} {any}, but expert 0 is {s} {any}", .{ p.name, p.type, p.dims, first.type, first.dims });
                return error.IncompleteExperts;
            }
            segs[x] = .{ .tensor = p, .elem_offset = 0, .elem_count = p.dims[0] * p.dims[1] };
        }
        try stacks.append(alloc, .{ .name = name, .dims = try alloc.dupe(usize, &.{ n_expert, first.dims[0], first.dims[1] }), .dtype = first.type, .segments = segs });
    }
    return stacks.items;
}

/// The vision tower of a multimodal checkpoint. llama.cpp puts these in a
/// separate mmproj file, so they are not part of the text GGUF and their
/// absence is not a hole in the table.
pub fn isVisionTensor(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "model.visual.") or
        std.mem.startsWith(u8, name, "visual.");
}

/// The clip projector a tower becomes. The layouts differ: qwen25vl splits
/// qkv, gates its MLP and uses RMSNorm; qwen3vl keeps qkv fused, uses
/// LayerNorm with biases and adds deepstack mergers.
pub const VisionProjector = enum {
    qwen25vl,
    qwen3vl,

    pub fn clipName(self: VisionProjector) []const u8 {
        return switch (self) {
            .qwen25vl => "qwen2.5vl_merger",
            .qwen3vl => "qwen3vl_merger",
        };
    }

    pub fn fromClipName(s: []const u8) ?VisionProjector {
        inline for (@typeInfo(VisionProjector).@"enum".field_names) |name| {
            const p = @field(VisionProjector, name);
            if (std.mem.eql(u8, s, p.clipName())) return p;
        }
        return null;
    }
};

/// config.json's vision_config, in the terms llama.cpp's clip.* keys use.
pub const Vision = struct {
    projector: VisionProjector,
    image_size: u32,
    patch_size: u32,
    embedding_length: u32,
    feed_forward_length: u32,
    block_count: u32,
    head_count: u32,
    /// The text tower's width, which the merger projects into.
    projection_dim: u32,
    spatial_merge_size: u32,
    layer_norm_eps: f32,
    image_mean: [3]f32,
    image_std: [3]f32,
    /// qwen3vl: the blocks feeding a deepstack merger, in merger order.
    deepstack_indexes: []const u32 = &.{},
    /// qwen25vl: every n-th block attends to the whole image, the rest to windows.
    n_wa_pattern: u32 = 0,
    /// qwen3vl: the position table's row count, which image_size is derived from.
    num_position_embeddings: u32 = 0,
};

fn visionProjectorFor(model_type: []const u8) ?VisionProjector {
    if (std.mem.eql(u8, model_type, "qwen2_5_vl")) return .qwen25vl;
    for ([_][]const u8{ "qwen3_vl", "qwen3_5", "qwen3_5_moe" }) |t| {
        if (std.mem.eql(u8, model_type, t)) return .qwen3vl;
    }
    return null;
}

// What the published checkpoints' preprocessor_config.json says, for a
// checkpoint that ships without one (a pipeline's text_encoder directory).
const clip_mean = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
const clip_std = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };

fn readTriple(v: ?std.json.Value) ?[3]f32 {
    const arr = v orelse return null;
    if (arr != .array or arr.array.items.len != 3) return null;
    var out: [3]f32 = undefined;
    for (arr.array.items, 0..) |item, i| out[i] = switch (item) {
        .float => |f| @floatCast(f),
        .integer => |n| @floatFromInt(n),
        else => return null,
    };
    return out;
}

/// The image normalization, from preprocessor_config.json or processor_config.json
/// as llama.cpp's converter reads them (the latter nests it under image_processor).
fn readImageNorm(io: std.Io, dir: std.Io.Dir, alloc: std.mem.Allocator) ?[2][3]f32 {
    if (readJson(io, dir, "preprocessor_config.json", alloc)) |p| {
        if (obj(p.value)) |o| {
            if (readTriple(o.get("image_mean"))) |m| if (readTriple(o.get("image_std"))) |s| return .{ m, s };
        }
    } else |_| {}
    if (readJson(io, dir, "processor_config.json", alloc)) |p| {
        if (obj(p.value)) |o| if (obj(o.get("image_processor") orelse std.json.Value{ .null = {} })) |ip| {
            if (readTriple(ip.get("image_mean"))) |m| if (readTriple(ip.get("image_std"))) |s| return .{ m, s };
        };
    } else |_| {}
    return null;
}

/// null = no vision_config, or one missing a dimension the clip keys need.
fn readVision(io: std.Io, dir: std.Io.Dir, model_type: []const u8, c: std.json.ObjectMap, cm: std.json.ObjectMap, embedding_length: u32, alloc: std.mem.Allocator) !?Vision {
    const projector = visionProjectorFor(model_type) orelse return null;
    const vc = obj(c.get("vision_config") orelse return null) orelse return null;
    const depth = getU32(vc, "depth") orelse return null;
    var v = Vision{
        .projector = projector,
        .image_size = 0,
        .patch_size = getU32(vc, "patch_size") orelse return null,
        .embedding_length = getU32(vc, "hidden_size") orelse return null,
        .feed_forward_length = getU32(vc, "intermediate_size") orelse return null,
        .block_count = depth,
        .head_count = getU32(vc, "num_heads") orelse return null,
        .projection_dim = embedding_length,
        .spatial_merge_size = getU32(vc, "spatial_merge_size") orelse 2,
        .layer_norm_eps = @floatCast(getF64(cm, "rms_norm_eps") orelse 1e-6),
        .image_mean = undefined,
        .image_std = undefined,
    };
    // llama.cpp's clip graph splits the patch conv in two and assumes that.
    if ((getU32(vc, "temporal_patch_size") orelse 2) != 2) {
        std.log.warn("vision_config.temporal_patch_size is not 2; llama.cpp's clip loader handles only 2", .{});
        return null;
    }
    switch (projector) {
        .qwen3vl => {
            v.num_position_embeddings = getU32(vc, "num_position_embeddings") orelse 2304;
            const side = std.math.sqrt(@as(f64, @floatFromInt(v.num_position_embeddings)));
            v.image_size = @intFromFloat(side * @as(f64, @floatFromInt(v.patch_size)));
            var ds: std.ArrayList(u32) = .empty;
            if (vc.get("deepstack_visual_indexes")) |arr| if (arr == .array) for (arr.array.items) |item| {
                const idx: u32 = switch (item) {
                    .integer => |n| if (n >= 0 and n < depth) @intCast(n) else return null,
                    else => return null,
                };
                try ds.append(alloc, idx);
            };
            v.deepstack_indexes = ds.items;
        },
        .qwen25vl => {
            v.image_size = getU32(vc, "image_size") orelse 560;
            // Full-attention blocks must fall every n blocks, the one pattern
            // clip.vision.n_wa_pattern can say.
            const fa = vc.get("fullatt_block_indexes") orelse return null;
            if (fa != .array or fa.array.items.len == 0) return null;
            var prev: i64 = -1;
            var n: i64 = 0;
            for (fa.array.items, 0..) |item, i| {
                const idx = switch (item) {
                    .integer => |x| x,
                    else => return null,
                };
                if (i == 0) n = idx + 1 else if (idx - prev != n) return null;
                prev = idx;
            }
            v.n_wa_pattern = @intCast(n);
        },
    }
    if (readImageNorm(io, dir, alloc)) |norm| {
        v.image_mean = norm[0];
        v.image_std = norm[1];
    } else {
        std.log.warn("no image_mean/image_std beside the checkpoint (preprocessor_config.json); using the values its model family publishes", .{});
        switch (projector) {
            .qwen3vl => {
                v.image_mean = .{ 0.5, 0.5, 0.5 };
                v.image_std = .{ 0.5, 0.5, 0.5 };
            },
            .qwen25vl => {
                v.image_mean = clip_mean;
                v.image_std = clip_std;
            },
        }
    }
    return v;
}

/// Per-block HF suffixes and their clip names. qkv is absent for qwen25vl,
/// which llama.cpp splits: see planVision.
fn visionBlockPairs(p: VisionProjector) []const NamePair {
    const common = [_]NamePair{
        .{ .from = "norm1.weight", .to = "ln1.weight" },
        .{ .from = "norm1.bias", .to = "ln1.bias" },
        .{ .from = "norm2.weight", .to = "ln2.weight" },
        .{ .from = "norm2.bias", .to = "ln2.bias" },
        .{ .from = "attn.proj.weight", .to = "attn_out.weight" },
        .{ .from = "attn.proj.bias", .to = "attn_out.bias" },
    };
    const q3 = common ++ [_]NamePair{
        .{ .from = "attn.qkv.weight", .to = "attn_qkv.weight" },
        .{ .from = "attn.qkv.bias", .to = "attn_qkv.bias" },
        .{ .from = "mlp.linear_fc1.weight", .to = "ffn_up.weight" },
        .{ .from = "mlp.linear_fc1.bias", .to = "ffn_up.bias" },
        .{ .from = "mlp.linear_fc2.weight", .to = "ffn_down.weight" },
        .{ .from = "mlp.linear_fc2.bias", .to = "ffn_down.bias" },
    };
    const q25 = common ++ [_]NamePair{
        .{ .from = "mlp.gate_proj.weight", .to = "ffn_gate.weight" },
        .{ .from = "mlp.gate_proj.bias", .to = "ffn_gate.bias" },
        .{ .from = "mlp.up_proj.weight", .to = "ffn_up.weight" },
        .{ .from = "mlp.up_proj.bias", .to = "ffn_up.bias" },
        .{ .from = "mlp.down_proj.weight", .to = "ffn_down.weight" },
        .{ .from = "mlp.down_proj.bias", .to = "ffn_down.bias" },
    };
    return switch (p) {
        .qwen3vl => &q3,
        .qwen25vl => &q25,
    };
}

fn visionTopPairs(p: VisionProjector) []const NamePair {
    const q3 = [_]NamePair{
        .{ .from = "pos_embed.weight", .to = "v.position_embd.weight" },
        .{ .from = "patch_embed.proj.bias", .to = "v.patch_embd.bias" },
        .{ .from = "merger.norm.weight", .to = "v.post_ln.weight" },
        .{ .from = "merger.norm.bias", .to = "v.post_ln.bias" },
        // A two-layer MLP with the activation between, so llama.cpp numbers
        // its layers 0 and 2.
        .{ .from = "merger.linear_fc1.weight", .to = "mm.0.weight" },
        .{ .from = "merger.linear_fc1.bias", .to = "mm.0.bias" },
        .{ .from = "merger.linear_fc2.weight", .to = "mm.2.weight" },
        .{ .from = "merger.linear_fc2.bias", .to = "mm.2.bias" },
    };
    const q25 = [_]NamePair{
        .{ .from = "patch_embed.proj.bias", .to = "v.patch_embd.bias" },
        .{ .from = "merger.ln_q.weight", .to = "v.post_ln.weight" },
        .{ .from = "merger.ln_q.bias", .to = "v.post_ln.bias" },
        .{ .from = "merger.mlp.0.weight", .to = "mm.0.weight" },
        .{ .from = "merger.mlp.0.bias", .to = "mm.0.bias" },
        .{ .from = "merger.mlp.2.weight", .to = "mm.2.weight" },
        .{ .from = "merger.mlp.2.bias", .to = "mm.2.bias" },
    };
    return switch (p) {
        .qwen3vl => &q3,
        .qwen25vl => &q25,
    };
}

const deepstack_pairs = [_]NamePair{
    .{ .from = "norm.weight", .to = "norm.weight" },
    .{ .from = "norm.bias", .to = "norm.bias" },
    .{ .from = "linear_fc1.weight", .to = "fc1.weight" },
    .{ .from = "linear_fc1.bias", .to = "fc1.bias" },
    .{ .from = "linear_fc2.weight", .to = "fc2.weight" },
    .{ .from = "linear_fc2.bias", .to = "fc2.bias" },
};

/// Split "<prefix><n>.<rest>" into n and rest.
fn indexed(body: []const u8, prefix: []const u8) ?struct { n: u32, rest: []const u8 } {
    if (!std.mem.startsWith(u8, body, prefix)) return null;
    const after = body[prefix.len..];
    const dot = std.mem.findScalar(u8, after, '.') orelse return null;
    const n = std.fmt.parseInt(u32, after[0..dot], 10) catch return null;
    return .{ .n = n, .rest = after[dot + 1 ..] };
}

/// The tower's HF names mapped into llama.cpp's clip namespace. null for
/// anything outside the tower, and for the tensors that split into several
/// (see planVision), which a name map cannot express.
pub fn mapVisionName(alloc: std.mem.Allocator, v: Vision, name: []const u8) !?[]const u8 {
    if (!isVisionTensor(name)) return null;
    const body = name[std.mem.find(u8, name, "visual.").? + "visual.".len ..];

    for (visionTopPairs(v.projector)) |p| if (std.mem.eql(u8, body, p.from)) return p.to;

    if (indexed(body, "blocks.")) |b| {
        for (visionBlockPairs(v.projector)) |p| {
            if (std.mem.eql(u8, b.rest, p.from)) return try alloc.print("v.blk.{d}.{s}", .{ b.n, p.to });
        }
        return null;
    }
    // The mergers are numbered in list order; llama.cpp names each after the
    // block whose features it takes.
    if (indexed(body, "deepstack_merger_list.")) |d| {
        if (d.n >= v.deepstack_indexes.len) return null;
        for (deepstack_pairs) |p| {
            if (std.mem.eql(u8, d.rest, p.from)) {
                return try alloc.print("v.deepstack.{d}.{s}", .{ v.deepstack_indexes[d.n], p.to });
            }
        }
    }
    return null;
}

/// mapVisionName read backwards, onto `prefix` ("model.visual." or "visual.").
pub fn mapVisionNameReverse(alloc: std.mem.Allocator, v: Vision, prefix: []const u8, name: []const u8) !?[]const u8 {
    for (visionTopPairs(v.projector)) |p| {
        if (std.mem.eql(u8, name, p.to)) return try alloc.print("{s}{s}", .{ prefix, p.from });
    }
    if (indexed(name, "v.blk.")) |b| {
        for (visionBlockPairs(v.projector)) |p| {
            if (std.mem.eql(u8, b.rest, p.to)) return try alloc.print("{s}blocks.{d}.{s}", .{ prefix, b.n, p.from });
        }
        return null;
    }
    if (indexed(name, "v.deepstack.")) |d| {
        const slot = std.mem.findScalar(u32, v.deepstack_indexes, d.n) orelse return null;
        for (deepstack_pairs) |p| {
            if (std.mem.eql(u8, d.rest, p.to)) return try alloc.print("{s}deepstack_merger_list.{d}.{s}", .{ prefix, slot, p.from });
        }
    }
    return null;
}

/// True for the conv whose HF form is [out, in_c, 2, kh, kw] and whose GGUF
/// form is two [out, in_c, kh, kw] tensors, one per temporal step.
pub fn isVisionPatchEmbedWeight(name: []const u8) bool {
    return isVisionTensor(name) and std.mem.endsWith(u8, name, "patch_embed.proj.weight");
}

/// The GGUF names of the patch conv's two temporal halves.
pub const patch_embd_halves = [2][]const u8{ "v.patch_embd.weight", "v.patch_embd.weight.1" };

/// What the mmproj writer needs from a tower's HF tensors.
pub const VisionPlan = struct {
    /// What the writer reads from: renamed copies of the plain renames, and
    /// the split tensors under their HF names for the stacks to read.
    sources: []types.Tensor,
    /// One per GGUF tensor, untyped. The split ones are also stacks.
    outputs: []types.Tensor,
    stacks: []types.ExpertStack,
    unmapped: []const []const u8,
};

/// Plan the tower's tensors onto llama.cpp's clip names, splitting the patch
/// conv by temporal step and, for qwen25vl, qkv into q, k and v, as its
/// converter does.
pub fn planVision(alloc: std.mem.Allocator, v: Vision, tensors: []const types.Tensor) !VisionPlan {
    var sources: std.ArrayList(types.Tensor) = .empty;
    var outputs: std.ArrayList(types.Tensor) = .empty;
    var stacks: std.ArrayList(types.ExpertStack) = .empty;
    var unmapped: std.ArrayList([]const u8) = .empty;

    for (tensors) |t| {
        if (isVisionPatchEmbedWeight(t.name)) {
            if (t.dims.len != 5 or t.dims[2] != 2) {
                std.log.warn("{s} has shape {any}, not [out, in, 2, kh, kw]", .{ t.name, t.dims });
                return error.UnexpectedVisionShape;
            }
            const plane = t.dims[3] * t.dims[4];
            const dims = try alloc.dupe(usize, &.{ t.dims[0], t.dims[1], t.dims[3], t.dims[4] });
            try sources.append(alloc, t);
            for (patch_embd_halves, 0..) |name, half| {
                try appendStack(alloc, &outputs, &stacks, t, name, dims, &.{.{
                    .tensor = t,
                    .elem_offset = half * plane,
                    .elem_count = plane,
                    .runs = t.dims[0] * t.dims[1],
                    .stride = 2 * plane,
                }});
            }
            continue;
        }
        if (v.projector == .qwen25vl) if (indexed(t.name[std.mem.find(u8, t.name, "visual.").? + "visual.".len ..], "blocks.")) |b| {
            const is_w = std.mem.eql(u8, b.rest, "attn.qkv.weight");
            if (is_w or std.mem.eql(u8, b.rest, "attn.qkv.bias")) {
                if (t.dims[0] % 3 != 0 or t.dims.len != @as(usize, if (is_w) 2 else 1)) {
                    std.log.warn("{s} has shape {any}, which does not split in three", .{ t.name, t.dims });
                    return error.UnexpectedVisionShape;
                }
                const rows = t.dims[0] / 3;
                const width: usize = if (is_w) t.dims[1] else 1;
                const dims = if (is_w) try alloc.dupe(usize, &.{ rows, width }) else try alloc.dupe(usize, &.{rows});
                try sources.append(alloc, t);
                for ([_][]const u8{ "q", "k", "v" }, 0..) |which, i| {
                    const name = try alloc.print("v.blk.{d}.attn_{s}.{s}", .{ b.n, which, if (is_w) "weight" else "bias" });
                    try appendStack(alloc, &outputs, &stacks, t, name, dims, &.{.{
                        .tensor = t,
                        .elem_offset = i * rows * width,
                        .elem_count = rows * width,
                    }});
                }
                continue;
            }
        };
        if (try mapVisionName(alloc, v, t.name)) |native| {
            var r = t;
            r.name = native;
            try sources.append(alloc, r);
            try outputs.append(alloc, r);
        } else {
            try unmapped.append(alloc, t.name);
        }
    }
    return .{ .sources = sources.items, .outputs = outputs.items, .stacks = stacks.items, .unmapped = unmapped.items };
}

fn appendStack(
    alloc: std.mem.Allocator,
    outputs: *std.ArrayList(types.Tensor),
    stacks: *std.ArrayList(types.ExpertStack),
    src: types.Tensor,
    name: []const u8,
    dims: []usize,
    segs: []const types.StackSegment,
) !void {
    try stacks.append(alloc, .{ .name = name, .dims = dims, .dtype = src.type, .segments = try alloc.dupe(types.StackSegment, segs) });
    var o = src;
    o.name = name;
    o.dims = dims;
    o.source_path = null;
    try outputs.append(alloc, o);
}

/// The GGUF type llama.cpp's converter gives an mmproj tensor at file type
/// `ftype` (f32, f16, bf16 or q8_0): norms, biases and the position table stay
/// f32, the patch conv is f16 only in an f16 file, and a q8_0 row that is not
/// whole blocks falls back to f16.
pub fn mmprojTensorType(name: []const u8, dims: []const usize, ftype: types.DataType) types.DataType {
    if (dims.len <= 1 or std.mem.endsWith(u8, name, "_norm.weight")) return .f32;
    if (std.mem.startsWith(u8, name, patch_embd_halves[0])) return if (ftype == .f16) .f16 else .f32;
    if (std.mem.eql(u8, name, "v.position_embd.weight") or !std.mem.endsWith(u8, name, ".weight")) return .f32;
    if (ftype == .q8_0 and dims[dims.len - 1] % 32 != 0) return .f16;
    return ftype;
}

/// The file type an mmproj is written at for a text target of `target`.
/// llama.cpp's converter offers f32, f16, bf16 and q8_0 and nothing smaller,
/// so any other target gets f16, the type published mmproj files ship in.
pub fn mmprojFileType(target: ?types.DataType) types.DataType {
    const t = target orelse return .f16;
    return switch (t) {
        .f32, .F32 => .f32,
        .bf16, .BF16 => .bf16,
        .q8_0 => .q8_0,
        else => .f16,
    };
}

/// The vision-language arch a text GGUF's general.architecture names. Its
/// tensors carry the text tower's names, so name detection cannot tell it from
/// the text-only arch. null when it names none.
pub fn visionArchByGgufName(gguf_arch: []const u8) ?*const imagearch.Arch {
    for ([_]*const imagearch.Arch{ &imagearch.qwen3vl, &imagearch.qwen2vl }) |vl| {
        if (std.mem.eql(u8, vl.name, gguf_arch)) return vl;
    }
    return null;
}

/// Where transformers reads `arch`'s vision tower from, and the projector it
/// takes. null for an arch with no tower. qwen2vl uses the legacy top-level
/// layout, which every transformers release loads.
pub fn visionLayout(arch_name: []const u8) ?struct { prefix: []const u8, projector: VisionProjector } {
    if (std.mem.eql(u8, arch_name, "qwen2vl")) return .{ .prefix = "visual.", .projector = .qwen25vl };
    if (std.mem.eql(u8, arch_name, "qwen3vl") or std.mem.eql(u8, arch_name, "qwen35")) return .{ .prefix = "model.visual.", .projector = .qwen3vl };
    return null;
}

/// config.json's vision_config back out of an mmproj's clip.* keys. null when
/// the file is not a vision projector this tool maps, or lacks a dimension.
pub fn visionFromClip(meta: std.json.ObjectMap, tensors: []const types.Tensor, alloc: std.mem.Allocator) !?Vision {
    const projector = VisionProjector.fromClipName(ggStr(meta, "clip.projector_type") orelse return null) orelse return null;
    const u32Of = struct {
        fn f(m: std.json.ObjectMap, key: []const u8) ?u32 {
            const n = ggU64(m, key) orelse return null;
            return std.math.cast(u32, n);
        }
    }.f;
    var v = Vision{
        .projector = projector,
        .image_size = u32Of(meta, "clip.vision.image_size") orelse return null,
        .patch_size = u32Of(meta, "clip.vision.patch_size") orelse return null,
        .embedding_length = u32Of(meta, "clip.vision.embedding_length") orelse return null,
        .feed_forward_length = u32Of(meta, "clip.vision.feed_forward_length") orelse return null,
        .block_count = u32Of(meta, "clip.vision.block_count") orelse return null,
        .head_count = u32Of(meta, "clip.vision.attention.head_count") orelse return null,
        .projection_dim = u32Of(meta, "clip.vision.projection_dim") orelse return null,
        .spatial_merge_size = u32Of(meta, "clip.vision.spatial_merge_size") orelse 2,
        .layer_norm_eps = @floatCast(ggF64(meta, "clip.vision.attention.layer_norm_epsilon") orelse 1e-6),
        .image_mean = undefined,
        .image_std = undefined,
        .n_wa_pattern = u32Of(meta, "clip.vision.n_wa_pattern") orelse 0,
    };
    inline for (.{ .{ "clip.vision.image_mean", &v.image_mean }, .{ "clip.vision.image_std", &v.image_std } }) |kv| {
        const arr = ggArr(meta, kv[0]) orelse return null;
        if (arr.len != 3) return null;
        for (arr, 0..) |item, i| kv[1][i] = switch (item) {
            .float => |x| @floatCast(x),
            .integer => |x| @floatFromInt(x),
            else => return null,
        };
    }
    if (ggArr(meta, "clip.vision.is_deepstack_layers")) |flags| {
        var ds: std.ArrayList(u32) = .empty;
        for (flags, 0..) |flag, i| if (flag == .bool and flag.bool) try ds.append(alloc, @intCast(i));
        v.deepstack_indexes = ds.items;
    }
    if (projector == .qwen25vl and v.n_wa_pattern == 0) return null;
    // The table's own row count, which image_size only implies.
    for (tensors) |t| if (std.mem.eql(u8, t.name, "v.position_embd.weight") and t.dims.len == 2) {
        v.num_position_embeddings = @intCast(t.dims[0]);
    };
    if (v.num_position_embeddings == 0 and v.patch_size > 0) {
        const side = v.image_size / v.patch_size;
        v.num_position_embeddings = side * side;
    }
    return v;
}

/// What the HF writer reads an mmproj through.
pub const VisionReversePlan = struct {
    /// The plain renames under their HF names, and the halves and parts the
    /// stacks join under their GGUF names.
    sources: []types.Tensor,
    stacks: []types.ExpertStack,
    unmapped: []const []const u8,
};

fn findTensor(tensors: []const types.Tensor, name: []const u8) ?types.Tensor {
    for (tensors) |t| if (std.mem.eql(u8, t.name, name)) return t;
    return null;
}

/// planVision read backwards onto `prefix`: the two patch conv halves join
/// into one [out, in, 2, kh, kw] conv, and qwen25vl's q, k and v back into qkv.
/// A tensor already carrying its HF name (an earlier pass renamed it) stays.
pub fn planVisionReverse(alloc: std.mem.Allocator, v: Vision, prefix: []const u8, tensors: []const types.Tensor) !VisionReversePlan {
    var sources: std.ArrayList(types.Tensor) = .empty;
    var stacks: std.ArrayList(types.ExpertStack) = .empty;
    var unmapped: std.ArrayList([]const u8) = .empty;

    for (tensors) |t| {
        if (std.mem.eql(u8, t.name, patch_embd_halves[1])) continue;
        if (std.mem.eql(u8, t.name, patch_embd_halves[0])) {
            const other = findTensor(tensors, patch_embd_halves[1]) orelse {
                std.log.warn("the mmproj has {s} but not {s}", .{ patch_embd_halves[0], patch_embd_halves[1] });
                return error.UnexpectedVisionShape;
            };
            if (t.dims.len != 4 or !std.mem.eql(usize, t.dims, other.dims)) {
                std.log.warn("{s} is {any} and {s} is {any}; they should be the same [out, in, kh, kw]", .{ t.name, t.dims, other.name, other.dims });
                return error.UnexpectedVisionShape;
            }
            const plane = t.dims[2] * t.dims[3];
            const dims = try alloc.dupe(usize, &.{ t.dims[0], t.dims[1], 2, t.dims[2], t.dims[3] });
            var segs: [2]types.StackSegment = undefined;
            for ([_]types.Tensor{ t, other }, 0..) |half, i| segs[i] = .{
                .tensor = half,
                .elem_offset = 0,
                .elem_count = plane,
                .runs = t.dims[0] * t.dims[1],
                .stride = plane,
                .out_offset = i * plane,
                .out_stride = 2 * plane,
            };
            try sources.append(alloc, t);
            try sources.append(alloc, other);
            try stacks.append(alloc, .{
                .name = try alloc.print("{s}patch_embed.proj.weight", .{prefix}),
                .dims = dims,
                .dtype = t.type,
                .segments = try alloc.dupe(types.StackSegment, &segs),
            });
            continue;
        }
        if (v.projector == .qwen25vl) if (indexed(t.name, "v.blk.")) |b| {
            const is_w = std.mem.eql(u8, b.rest, "attn_q.weight");
            const is_b = std.mem.eql(u8, b.rest, "attn_q.bias");
            if (is_w or is_b or std.mem.eql(u8, b.rest, "attn_k.weight") or std.mem.eql(u8, b.rest, "attn_v.weight") or
                std.mem.eql(u8, b.rest, "attn_k.bias") or std.mem.eql(u8, b.rest, "attn_v.bias"))
            {
                // q anchors the join; k and v are read from there.
                if (!is_w and !is_b) continue;
                const kind = if (is_w) "weight" else "bias";
                var parts: [3]types.Tensor = undefined;
                for ([_][]const u8{ "q", "k", "v" }, 0..) |which, i| {
                    const n = try alloc.print("v.blk.{d}.attn_{s}.{s}", .{ b.n, which, kind });
                    parts[i] = findTensor(tensors, n) orelse {
                        std.log.warn("the mmproj has {s} but not {s}", .{ t.name, n });
                        return error.UnexpectedVisionShape;
                    };
                    if (!std.mem.eql(usize, parts[i].dims, t.dims)) {
                        std.log.warn("{s} is {any}, but {s} is {any}", .{ n, parts[i].dims, t.name, t.dims });
                        return error.UnexpectedVisionShape;
                    }
                }
                var n_el: usize = 1;
                for (t.dims) |d| n_el *= d;
                var dims = try alloc.dupe(usize, t.dims);
                dims[0] *= 3;
                var segs: [3]types.StackSegment = undefined;
                for (parts, 0..) |part, i| {
                    segs[i] = .{ .tensor = part, .elem_offset = 0, .elem_count = n_el };
                    try sources.append(alloc, part);
                }
                try stacks.append(alloc, .{
                    .name = try alloc.print("{s}blocks.{d}.attn.qkv.{s}", .{ prefix, b.n, kind }),
                    .dims = dims,
                    .dtype = t.type,
                    .segments = try alloc.dupe(types.StackSegment, &segs),
                });
                continue;
            }
        };
        if (try mapVisionNameReverse(alloc, v, prefix, t.name)) |hf| {
            var r = t;
            r.name = hf;
            try sources.append(alloc, r);
        } else if (isVisionTensor(t.name)) {
            try sources.append(alloc, t);
        } else {
            try unmapped.append(alloc, t.name);
        }
    }
    return .{ .sources = sources.items, .stacks = stacks.items, .unmapped = unmapped.items };
}

/// The general.* and clip.* keys of an mmproj for `m`'s tower, in llama.cpp's
/// spellings. The caller has checked that `m.vision` is set.
pub fn addVisionMetadata(m: Model, metadata: *std.json.ObjectMap, alloc: std.mem.Allocator) !void {
    const v = m.vision.?;
    try putStr(metadata, alloc, "general.architecture", "clip");
    try putStr(metadata, alloc, "general.type", "mmproj");
    try addNameParts(metadata, alloc, checkpointDirName(m.dir) orelse "");
    if (m.sampling_temp) |t| try putFloat(metadata, alloc, "general.sampling.temp", t);
    if (m.sampling_top_k) |k| try putNum(metadata, alloc, "general.sampling.top_k", k);
    if (m.sampling_top_p) |p| try putFloat(metadata, alloc, "general.sampling.top_p", p);
    try metadata.put(alloc, try alloc.dupe(u8, "clip.has_vision_encoder"), .{ .bool = true });
    try putNum(metadata, alloc, "clip.vision.projection_dim", v.projection_dim);
    try putNum(metadata, alloc, "clip.vision.image_size", v.image_size);
    try putNum(metadata, alloc, "clip.vision.patch_size", v.patch_size);
    try putNum(metadata, alloc, "clip.vision.embedding_length", v.embedding_length);
    try putNum(metadata, alloc, "clip.vision.feed_forward_length", v.feed_forward_length);
    try putNum(metadata, alloc, "clip.vision.block_count", v.block_count);
    try putNum(metadata, alloc, "clip.vision.attention.head_count", v.head_count);
    var mean: std.json.Array = std.json.Array.init(alloc);
    var std_: std.json.Array = std.json.Array.init(alloc);
    for (v.image_mean, v.image_std) |mv, sv| {
        try mean.append(.{ .float = mv });
        try std_.append(.{ .float = sv });
    }
    try metadata.put(alloc, try alloc.dupe(u8, "clip.vision.image_mean"), .{ .array = mean });
    try metadata.put(alloc, try alloc.dupe(u8, "clip.vision.image_std"), .{ .array = std_ });
    try putStr(metadata, alloc, "clip.projector_type", v.projector.clipName());
    try putFloat(metadata, alloc, "clip.vision.attention.layer_norm_epsilon", v.layer_norm_eps);
    switch (v.projector) {
        .qwen3vl => {
            try metadata.put(alloc, try alloc.dupe(u8, "clip.use_gelu"), .{ .bool = true });
            try putNum(metadata, alloc, "clip.vision.spatial_merge_size", v.spatial_merge_size);
            const flags = try alloc.alloc(bool, v.block_count);
            @memset(flags, false);
            for (v.deepstack_indexes) |i| flags[i] = true;
            try putBoolArray(metadata, alloc, "clip.vision.is_deepstack_layers", flags);
        },
        .qwen25vl => {
            try metadata.put(alloc, try alloc.dupe(u8, "clip.use_silu"), .{ .bool = true });
            try putNum(metadata, alloc, "clip.vision.n_wa_pattern", v.n_wa_pattern);
        },
    }
}

/// qwen35's HF state dict. `mtp_base` is the block index the MTP head takes,
/// which is the text block count - llama.cpp appends it as one more block
/// rather than giving it its own namespace. null when the caller has no block
/// count to place it at, and the mtp.* names then do not map.
fn mapQwen35(alloc: std.mem.Allocator, name: []const u8, mtp_base: ?u32) !?[]const u8 {
    if (std.mem.eql(u8, name, "lm_head.weight")) return "output.weight";

    if (std.mem.startsWith(u8, name, "mtp.")) {
        const base = mtp_base orelse return null;
        const rest = name["mtp.".len..];
        const head = [_]struct { from: []const u8, to: []const u8 }{
            .{ .from = "fc.weight", .to = "nextn.eh_proj.weight" },
            .{ .from = "pre_fc_norm_embedding.weight", .to = "nextn.enorm.weight" },
            .{ .from = "pre_fc_norm_hidden.weight", .to = "nextn.hnorm.weight" },
            .{ .from = "norm.weight", .to = "nextn.shared_head_norm.weight" },
        };
        for (head) |pair| {
            if (std.mem.eql(u8, rest, pair.from)) {
                return try alloc.print("blk.{d}.{s}", .{ base, pair.to });
            }
        }
        // The head's own transformer layer. Only one is defined today, and a
        // second would land on the same block, so refuse rather than collide.
        if (std.mem.startsWith(u8, rest, "layers.0.")) {
            const suffix = rest["layers.0.".len..];
            for (qwen35_suffix) |pair| {
                if (std.mem.eql(u8, suffix, pair.from)) {
                    return try alloc.print("blk.{d}.{s}", .{ base, pair.to });
                }
            }
        }
        return null;
    }

    // The text tower sits under model.language_model.* because these ship as
    // ForConditionalGeneration checkpoints beside a vision tower.
    const body = if (std.mem.startsWith(u8, name, "model.language_model."))
        name["model.language_model.".len..]
    else if (std.mem.startsWith(u8, name, "model."))
        name["model.".len..]
    else
        return null;

    if (std.mem.eql(u8, body, "embed_tokens.weight")) return "token_embd.weight";
    if (std.mem.eql(u8, body, "norm.weight")) return "output_norm.weight";

    if (!std.mem.startsWith(u8, body, "layers.")) return null;
    const after = body["layers.".len..];
    const dot = std.mem.findScalar(u8, after, '.') orelse return null;
    const layer = std.fmt.parseInt(u32, after[0..dot], 10) catch return null;
    const suffix = after[dot + 1 ..];
    for (qwen35_suffix) |pair| {
        if (std.mem.eql(u8, suffix, pair.from)) {
            return try alloc.print("blk.{d}.{s}", .{ layer, pair.to });
        }
    }
    return null;
}

pub fn mapName(alloc: std.mem.Allocator, arch: *const imagearch.Arch, name: []const u8, mtp_base: ?u32) !?[]const u8 {
    if (isQwen35Family(arch.name)) return mapQwen35(alloc, name, mtp_base);
    if (visionTextTower(arch)) |text| {
        // The language tower is the plain arch, under a prefix in the newer
        // layouts and at model.* in qwen2vl's legacy one.
        for ([_][]const u8{ "model.language_model.", "language_model." }) |pre| {
            if (std.mem.startsWith(u8, name, pre)) {
                const flat = try alloc.print("model.{s}", .{name[pre.len..]});
                return mapName(alloc, text, flat, mtp_base);
            }
        }
        if (std.mem.eql(u8, name, "lm_head.weight")) return "output.weight";
        if (text == &imagearch.qwen2 and std.mem.startsWith(u8, name, "model.")) return mapName(alloc, text, name, mtp_base);
        return null;
    }
    if (std.mem.eql(u8, name, "model.embed_tokens.weight")) return "token_embd.weight";
    if (std.mem.eql(u8, name, "lm_head.weight")) return "output.weight";
    if (std.mem.eql(u8, name, "model.norm.weight")) return "output_norm.weight";

    const prefix = "model.layers.";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    const after = name[prefix.len..];
    const dot = std.mem.findScalar(u8, after, '.') orelse return null;
    const layer = std.fmt.parseInt(u32, after[0..dot], 10) catch return null;
    const suffix = after[dot + 1 ..];

    const map = [_]NamePair{
        .{ .from = "input_layernorm.weight", .to = "attn_norm.weight" },
        .{ .from = "post_attention_layernorm.weight", .to = "ffn_norm.weight" },
        .{ .from = "self_attn.q_proj.weight", .to = "attn_q.weight" },
        .{ .from = "self_attn.q_proj.bias", .to = "attn_q.bias" },
        .{ .from = "self_attn.k_proj.weight", .to = "attn_k.weight" },
        .{ .from = "self_attn.k_proj.bias", .to = "attn_k.bias" },
        .{ .from = "self_attn.v_proj.weight", .to = "attn_v.weight" },
        .{ .from = "self_attn.v_proj.bias", .to = "attn_v.bias" },
        .{ .from = "self_attn.o_proj.weight", .to = "attn_output.weight" },
        .{ .from = "mlp.gate_proj.weight", .to = "ffn_gate.weight" },
        .{ .from = "mlp.up_proj.weight", .to = "ffn_up.weight" },
        .{ .from = "mlp.down_proj.weight", .to = "ffn_down.weight" },
    } ++ moe_suffix;
    for (map) |pair| {
        if (std.mem.eql(u8, suffix, pair.from)) {
            return try alloc.print("blk.{d}.{s}", .{ layer, pair.to });
        }
    }
    // Head norms exist only where llama.cpp models them: qwen3 yes, qwen2/llama no.
    if (hasHeadNorms(arch.name)) {
        if (std.mem.eql(u8, suffix, "self_attn.q_norm.weight")) {
            return try alloc.print("blk.{d}.attn_q_norm.weight", .{layer});
        }
        if (std.mem.eql(u8, suffix, "self_attn.k_norm.weight")) {
            return try alloc.print("blk.{d}.attn_k_norm.weight", .{layer});
        }
    }
    return null;
}

/// The text-only arch whose tensors a vision-language arch's language tower
/// carries. null for anything else.
fn visionTextTower(arch: *const imagearch.Arch) ?*const imagearch.Arch {
    if (std.mem.eql(u8, arch.name, "qwen3vl")) return &imagearch.qwen3;
    if (std.mem.eql(u8, arch.name, "qwen2vl")) return &imagearch.qwen2;
    return null;
}

fn hasHeadNorms(arch_name: []const u8) bool {
    return std.mem.eql(u8, arch_name, "qwen3") or std.mem.eql(u8, arch_name, "qwen3moe");
}

/// mapQwen35 read backwards. `mtp_base` is the block index that came from the
/// MTP head; blocks past the text tower belong to it and go back under mtp.*.
fn mapQwen35Reverse(alloc: std.mem.Allocator, name: []const u8, mtp_base: ?u32) !?[]const u8 {
    if (std.mem.eql(u8, name, "token_embd.weight")) return "model.language_model.embed_tokens.weight";
    if (std.mem.eql(u8, name, "output_norm.weight")) return "model.language_model.norm.weight";
    if (std.mem.eql(u8, name, "output.weight")) return "lm_head.weight";

    if (!std.mem.startsWith(u8, name, "blk.")) return null;
    const after = name["blk.".len..];
    const dot = std.mem.findScalar(u8, after, '.') orelse return null;
    const block = std.fmt.parseInt(u32, after[0..dot], 10) catch return null;
    const suffix = after[dot + 1 ..];

    const is_mtp = if (mtp_base) |b| block >= b else false;
    if (is_mtp) {
        const head = [_]struct { from: []const u8, to: []const u8 }{
            .{ .from = "nextn.eh_proj.weight", .to = "fc.weight" },
            .{ .from = "nextn.enorm.weight", .to = "pre_fc_norm_embedding.weight" },
            .{ .from = "nextn.hnorm.weight", .to = "pre_fc_norm_hidden.weight" },
            .{ .from = "nextn.shared_head_norm.weight", .to = "norm.weight" },
        };
        for (head) |pair| {
            if (std.mem.eql(u8, suffix, pair.from)) {
                return try alloc.print("mtp.{s}", .{pair.to});
            }
        }
        for (qwen35_suffix) |pair| {
            if (std.mem.eql(u8, suffix, pair.to)) {
                return try alloc.print("mtp.layers.0.{s}", .{pair.from});
            }
        }
        return null;
    }

    for (qwen35_suffix) |pair| {
        if (std.mem.eql(u8, suffix, pair.to)) {
            return try alloc.print("model.language_model.layers.{d}.{s}", .{ block, pair.from });
        }
    }
    return null;
}

/// mapName read backwards: native GGUF name -> HF state-dict name. Null when
/// the name is already HF-style or is not part of the mapped LLM set.
pub fn mapNameReverse(alloc: std.mem.Allocator, arch: *const imagearch.Arch, name: []const u8, mtp_base: ?u32) !?[]const u8 {
    if (isQwen35Family(arch.name)) return mapQwen35Reverse(alloc, name, mtp_base);
    // Qwen3-VL has no legacy layout: transformers reads it under
    // model.language_model.* only. qwen2vl's legacy model.* layout loads on
    // every transformers version, so it falls through to qwen2's names.
    if (std.mem.eql(u8, arch.name, "qwen3vl")) {
        const flat = (try mapNameReverse(alloc, &imagearch.qwen3, name, mtp_base)) orelse return null;
        if (!std.mem.startsWith(u8, flat, "model.")) return flat;
        return try alloc.print("model.language_model.{s}", .{flat["model.".len..]});
    }
    if (std.mem.eql(u8, name, "token_embd.weight")) return "model.embed_tokens.weight";
    if (std.mem.eql(u8, name, "output.weight")) return "lm_head.weight";
    if (std.mem.eql(u8, name, "output_norm.weight")) return "model.norm.weight";

    const prefix = "blk.";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    const after = name[prefix.len..];
    const dot = std.mem.findScalar(u8, after, '.') orelse return null;
    const layer = std.fmt.parseInt(u32, after[0..dot], 10) catch return null;
    const suffix = after[dot + 1 ..];

    const Pair = struct { from: []const u8, to: []const u8 };
    const map = [_]Pair{
        .{ .from = "attn_norm.weight", .to = "input_layernorm.weight" },
        .{ .from = "ffn_norm.weight", .to = "post_attention_layernorm.weight" },
        .{ .from = "attn_q.weight", .to = "self_attn.q_proj.weight" },
        .{ .from = "attn_q.bias", .to = "self_attn.q_proj.bias" },
        .{ .from = "attn_k.weight", .to = "self_attn.k_proj.weight" },
        .{ .from = "attn_k.bias", .to = "self_attn.k_proj.bias" },
        .{ .from = "attn_v.weight", .to = "self_attn.v_proj.weight" },
        .{ .from = "attn_v.bias", .to = "self_attn.v_proj.bias" },
        .{ .from = "attn_output.weight", .to = "self_attn.o_proj.weight" },
        .{ .from = "ffn_gate.weight", .to = "mlp.gate_proj.weight" },
        .{ .from = "ffn_up.weight", .to = "mlp.up_proj.weight" },
        .{ .from = "ffn_down.weight", .to = "mlp.down_proj.weight" },
    };
    for (map) |pair| {
        if (std.mem.eql(u8, suffix, pair.from)) {
            return try alloc.print("model.layers.{d}.{s}", .{ layer, pair.to });
        }
    }
    if (hasHeadNorms(arch.name)) {
        if (std.mem.eql(u8, suffix, "attn_q_norm.weight")) {
            return try alloc.print("model.layers.{d}.self_attn.q_norm.weight", .{layer});
        }
        if (std.mem.eql(u8, suffix, "attn_k_norm.weight")) {
            return try alloc.print("model.layers.{d}.self_attn.k_norm.weight", .{layer});
        }
    }
    return null;
}

/// Python str.islower for the ASCII subset we care about.
fn allLower(s: []const u8) bool {
    var any = false;
    for (s) |c| {
        if (std.ascii.isUpper(c)) return false;
        if (std.ascii.isLower(c)) any = true;
    }
    return any;
}

/// Python re.match(r'^(v\d+(?:\.\d+)*|\d.*)$'): a version or number token,
/// kept as-is by id_to_title.
// ^(v\d+(?:\.\d+)*|\d.*)$ . Only reached for all-lowercase words, so no 'V'.
fn isVersionLike(s: []const u8) bool {
    if (s.len == 0) return false;
    if (std.ascii.isDigit(s[0])) return true; // \d.*
    if (s[0] != 'v') return false;
    var i: usize = 1;
    if (i == s.len or !std.ascii.isDigit(s[i])) return false;
    while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
    while (i + 1 < s.len and s[i] == '.' and std.ascii.isDigit(s[i + 1])) {
        i += 1;
        while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
    }
    return i == s.len;
}

fn pyTitle(alloc: std.mem.Allocator, word: []const u8) ![]const u8 {
    var out = try std.ArrayList(u8).initCapacity(alloc, word.len);
    var at_start = true;
    for (word) |c| {
        // Python's title() words on letters only: any non-letter upper-cases
        // the next letter and lower-cases the rest.
        if (std.ascii.isAlphabetic(c)) {
            out.append(alloc, if (at_start) std.ascii.toUpper(c) else std.ascii.toLower(c)) catch unreachable;
            at_start = false;
        } else {
            out.append(alloc, c) catch unreachable;
            at_start = true;
        }
    }
    return out.items;
}

/// Metadata.id_to_title: dashes become spaces; words are title-cased unless
/// they already carry capitals or look like versions.
fn idToTitle(alloc: std.mem.Allocator, s: []const u8) ![]const u8 {
    const spaced = try alloc.dupe(u8, s);
    for (spaced) |*c| {
        if (c.* == '-') c.* = ' ';
    }
    var out: ?[]const u8 = null;
    var list: std.ArrayList([]const u8) = .empty;
    var it = std.mem.tokenizeScalar(u8, spaced, ' ');
    while (it.next()) |w| {
        const t = if (allLower(w) and !isVersionLike(w)) try pyTitle(alloc, w) else w;
        try list.append(alloc, t);
    }
    out = try std.mem.join(alloc, " ", list.items);
    return out.?;
}

fn cardEntryFromId(alloc: std.mem.Allocator, id: []const u8) !CardEntry {
    var org_raw: ?[]const u8 = null;
    var name_raw: []const u8 = id;
    if (std.mem.findScalar(u8, id, '/')) |slash| {
        org_raw = id[0..slash];
        name_raw = id[slash + 1 ..];
    }
    const name = try idToTitle(alloc, name_raw);
    const org = if (org_raw) |o| try idToTitle(alloc, o) else null;
    const url = if (org_raw) |o|
        try alloc.print("https://huggingface.co/{s}/{s}", .{ o, name_raw })
    else
        null;
    return .{ .name = name, .organization = org, .repo_url = url };
}

const CardField = struct { key: []const u8, values: std.ArrayList([]const u8) };

fn findCardField(fields: []const CardField, key: []const u8) ?std.ArrayList([]const u8) {
    for (fields) |f| if (std.mem.eql(u8, f.key, key)) return f.values;
    return null;
}

/// Parses the README.md YAML frontmatter (top-level scalars and "- item"
/// block lists only) for the model-card metadata llama.cpp's converter copies
/// into general.* keys.
fn loadCard(io: std.Io, dir: std.Io.Dir, alloc: std.mem.Allocator, model: *Model) void {
    const txt = dir.readFileAlloc(io, "README.md", alloc, .limited64(16 * 1024 * 1024)) catch return;
    if (!std.mem.startsWith(u8, txt, "---\n")) return;
    const end = std.mem.findPos(u8, txt, 4, "\n---") orelse return;
    var fields: std.ArrayList(CardField) = .empty;
    var lines = std.mem.splitScalar(u8, txt[4..end], '\n');
    var cur: ?*CardField = null;
    while (lines.next()) |line| {
        if (line.len == 0 or line[0] == ' ' or line[0] == '\t') continue;
        if (std.mem.startsWith(u8, line, "- ")) {
            if (cur) |c| {
                const v = std.mem.trim(u8, line[2..], " \r\"'");
                if (v.len > 0) c.values.append(alloc, v) catch return;
            }
            continue;
        }
        const colon = std.mem.findScalar(u8, line, ':') orelse continue;
        const key = std.mem.trim(u8, line[0..colon], " ");
        var value = std.mem.trim(u8, line[colon + 1 ..], " \r\"'");
        if (value.len == 2 and std.mem.startsWith(u8, value, "[") and std.mem.endsWith(u8, value, "]")) value = "";
        cur = null;
        for (fields.items) |*f| if (std.mem.eql(u8, f.key, key)) {
            cur = f;
            break;
        };
        if (cur == null) {
            fields.append(alloc, .{ .key = key, .values = .empty }) catch return;
            cur = &fields.items[fields.items.len - 1];
        }
        if (value.len > 0) {
            // A bare "[" opens a flow sequence continued on the following
            // lines, which this parser does not follow; treat it as no value.
            if (value.len == 1 and value[0] == '[') {
                cur = null;
            } else if (std.mem.startsWith(u8, value, "[")) {
                var parts = std.mem.splitScalar(u8, value[1 .. value.len - 1], ',');
                while (parts.next()) |p| {
                    const v = std.mem.trim(u8, p, " \"'");
                    if (v.len > 0) cur.?.values.append(alloc, v) catch return;
                }
                cur = null; // inline list closes the key
            } else cur.?.values.append(alloc, value) catch return;
        }
    }

    if (findCardField(fields.items, "license")) |v| {
        if (v.items.len > 0) model.license = v.items[0];
    }
    if (findCardField(fields.items, "license_link")) |v| {
        if (v.items.len > 0) model.license_link = v.items[0];
    }

    var tags: std.ArrayList([]const u8) = .empty;
    if (findCardField(fields.items, "tags")) |v| {
        for (v.items) |t| tags.append(alloc, t) catch return;
    }
    if (findCardField(fields.items, "pipeline_tag")) |v| if (v.items.len > 0) tags.append(alloc, v.items[0]) catch return;
    model.tags = tags.items;

    if (findCardField(fields.items, "language") orelse findCardField(fields.items, "languages")) |v| model.languages = v.items;

    if (findCardField(fields.items, "datasets")) |v| {
        var list: std.ArrayList(CardEntry) = .empty;
        for (v.items) |id| list.append(alloc, cardEntryFromId(alloc, id) catch continue) catch return;
        model.datasets = list.items;
    }
    if (findCardField(fields.items, "base_model") orelse findCardField(fields.items, "base_models")) |v| {
        var list: std.ArrayList(CardEntry) = .empty;
        for (v.items) |id| {
            const one = std.mem.trim(u8, id, " \r");
            if (one.len > 0) list.append(alloc, cardEntryFromId(alloc, one) catch continue) catch return;
        }
        model.base_models = list.items;
    }
}

fn putStr(metadata: *std.json.ObjectMap, alloc: std.mem.Allocator, key: []const u8, value: []const u8) !void {
    try metadata.put(alloc, try alloc.dupe(u8, key), .{ .string = try alloc.dupe(u8, value) });
}

fn putNum(metadata: *std.json.ObjectMap, alloc: std.mem.Allocator, key: []const u8, value: anytype) !void {
    try metadata.put(alloc, try alloc.dupe(u8, key), .{ .integer = value });
}

fn putFloat(metadata: *std.json.ObjectMap, alloc: std.mem.Allocator, key: []const u8, value: f32) !void {
    try metadata.put(alloc, try alloc.dupe(u8, key), .{ .float = value });
}

fn putStrArray(metadata: *std.json.ObjectMap, alloc: std.mem.Allocator, key: []const u8, items: []const []const u8) !void {
    var arr: std.json.Array = std.json.Array.init(alloc);
    for (items) |s| try arr.append(.{ .string = try alloc.dupe(u8, s) });
    try metadata.put(alloc, try alloc.dupe(u8, key), .{ .array = arr });
}

fn putBoolArray(metadata: *std.json.ObjectMap, alloc: std.mem.Allocator, key: []const u8, items: []const bool) !void {
    var arr: std.json.Array = std.json.Array.init(alloc);
    for (items) |v| try arr.append(.{ .bool = v });
    try metadata.put(alloc, try alloc.dupe(u8, key), .{ .array = arr });
}

fn putI64Array(metadata: *std.json.ObjectMap, alloc: std.mem.Allocator, key: []const u8, items: []const i64) !void {
    var arr: std.json.Array = std.json.Array.init(alloc);
    for (items) |v| try arr.append(.{ .integer = v });
    try metadata.put(alloc, try alloc.dupe(u8, key), .{ .array = arr });
}

fn putU32Array(metadata: *std.json.ObjectMap, alloc: std.mem.Allocator, key: []const u8, items: []const u32) !void {
    var arr: std.json.Array = std.json.Array.init(alloc);
    for (items) |v| try arr.append(.{ .integer = v });
    try metadata.put(alloc, try alloc.dupe(u8, key), .{ .array = arr });
}

/// Long-context rope scaling. llama.cpp reads linear and yarn from metadata,
/// but carries llama3 scaling in a generated rope_freqs tensor, which needs a
/// synthesized tensor ggufy has no way to write yet: say so rather than leave
/// the user to discover it past the original window.
fn addRopeScaling(self: Model, metadata: *std.json.ObjectMap, a: []const u8, alloc: std.mem.Allocator) !void {
    const rs = self.rope_scaling orelse return;
    const linear = std.mem.eql(u8, rs.kind, "linear");
    const yarn = std.mem.eql(u8, rs.kind, "yarn");
    if (!linear and !yarn) {
        std.log.warn(
            "config.json asks for {s} rope scaling, which this converter cannot express: the model stays correct to {d} tokens and degrades past it",
            .{ rs.kind, rs.orig_ctx orelse self.context_length },
        );
        return;
    }
    try putStr(metadata, alloc, try alloc.print("{s}.rope.scaling.type", .{a}), rs.kind);
    try putFloat(metadata, alloc, try alloc.print("{s}.rope.scaling.factor", .{a}), rs.factor);
    if (rs.orig_ctx) |ctx| {
        try putNum(metadata, alloc, try alloc.print("{s}.rope.scaling.original_context_length", .{a}), ctx);
    }
}

/// Emit the general.*, <arch>.* and tokenizer.ggml.* metadata for an HF checkpoint.
/// `arch_name` prefixes the dimension keys and must be whatever general.architecture
/// ends up saying (-A renames both together, or llama.cpp looks for the keys under a
/// prefix that is not there). The family tests below stay on the detected name: -A
/// relabels the file, it does not turn a llama into something with another key set.
pub fn addMetadata(self: Model, metadata: *std.json.ObjectMap, arch_name: []const u8, alloc: std.mem.Allocator) !void {
    const a = arch_name;
    const family = self.arch.name;
    try putStr(metadata, alloc, "general.type", "model");
    try addNameParts(metadata, alloc, checkpointDirName(self.dir) orelse "");

    // The MTP head is one more block on the end, so the count the file declares
    // is past the text tower's. mapName places it at self.block_count.
    const gguf_blocks = self.block_count + if (self.qwen35) |q| q.nextn_layers else 0;
    try putNum(metadata, alloc, try alloc.print("{s}.block_count", .{a}), gguf_blocks);
    try putNum(metadata, alloc, try alloc.print("{s}.embedding_length", .{a}), self.embedding_length);
    if (self.feed_forward_length > 0) {
        try putNum(metadata, alloc, try alloc.print("{s}.feed_forward_length", .{a}), self.feed_forward_length);
    }
    if (self.expert_count > 0) {
        try putNum(metadata, alloc, try alloc.print("{s}.expert_count", .{a}), self.expert_count);
        try putNum(metadata, alloc, try alloc.print("{s}.expert_used_count", .{a}), self.expert_used_count);
        if (self.expert_feed_forward_length) |n| try putNum(metadata, alloc, try alloc.print("{s}.expert_feed_forward_length", .{a}), n);
        if (self.expert_shared_feed_forward_length) |n| try putNum(metadata, alloc, try alloc.print("{s}.expert_shared_feed_forward_length", .{a}), n);
    }
    try putNum(metadata, alloc, try alloc.print("{s}.attention.head_count", .{a}), self.head_count);
    try putNum(metadata, alloc, try alloc.print("{s}.attention.head_count_kv", .{a}), self.head_count_kv);
    try putNum(metadata, alloc, try alloc.print("{s}.context_length", .{a}), self.context_length);
    try putFloat(metadata, alloc, try alloc.print("{s}.rope.freq_base", .{a}), self.rope_theta);
    try putFloat(metadata, alloc, try alloc.print("{s}.attention.layer_norm_rms_epsilon", .{a}), self.rms_eps);

    // Every architecture needs the rope dimension count: llama.cpp otherwise
    // falls back to embedding_length / head_count, which is wrong exactly when
    // head_dim is explicit and different (Qwen3-0.6B: 128 against a fallback of
    // 64), and applies RoPE over half of each head without complaining.
    const key_len = self.head_dim orelse (self.embedding_length / self.head_count);
    // A partial rotary factor makes the rotated width narrower than the head,
    // so the two are not the same number on this hybrid (64 against 256).
    const rope_dims = if (self.qwen35) |q| q.rope_dim_count else key_len;
    try putNum(metadata, alloc, try alloc.print("{s}.rope.dimension_count", .{a}), rope_dims);

    if (self.qwen35) |q| {
        try putNum(metadata, alloc, try alloc.print("{s}.ssm.conv_kernel", .{a}), q.conv_kernel);
        try putNum(metadata, alloc, try alloc.print("{s}.ssm.state_size", .{a}), q.state_size);
        try putNum(metadata, alloc, try alloc.print("{s}.ssm.group_count", .{a}), q.group_count);
        try putNum(metadata, alloc, try alloc.print("{s}.ssm.time_step_rank", .{a}), q.time_step_rank);
        try putNum(metadata, alloc, try alloc.print("{s}.ssm.inner_size", .{a}), q.inner_size);
        try putNum(metadata, alloc, try alloc.print("{s}.nextn_predict_layers", .{a}), q.nextn_layers);
        try putNum(metadata, alloc, try alloc.print("{s}.full_attention_interval", .{a}), q.full_attention_interval);
        try putBoolArray(metadata, alloc, try alloc.print("{s}.attention.recurrent_layers", .{a}), q.recurrent);
        try putI64Array(metadata, alloc, try alloc.print("{s}.rope.dimension_sections", .{a}), &q.rope_sections);
    }

    // llama.cpp's converter writes key/value length for the llama and qwen3
    // classes (and whenever head_dim is explicit); the llama class also adds
    // vocab_size.
    if (self.head_dim != null or std.mem.eql(u8, family, "llama") or std.mem.eql(u8, family, "qwen3")) {
        try putNum(metadata, alloc, try alloc.print("{s}.attention.key_length", .{a}), key_len);
        try putNum(metadata, alloc, try alloc.print("{s}.attention.value_length", .{a}), key_len);
        if (std.mem.eql(u8, family, "llama")) {
            try putNum(metadata, alloc, try alloc.print("{s}.vocab_size", .{a}), self.vocab_size);
        }
    }

    try addRopeScaling(self, metadata, a, alloc);
    if (self.rope_sections) |rs| {
        try putI64Array(metadata, alloc, try alloc.print("{s}.rope.dimension_sections", .{a}), &rs);
    }
    if (self.deepstack_layers) |n| {
        try putNum(metadata, alloc, try alloc.print("{s}.n_deepstack_layers", .{a}), n);
    }

    if (self.sampling_temp) |t| try putFloat(metadata, alloc, "general.sampling.temp", t);
    if (self.sampling_top_k) |k| try putNum(metadata, alloc, "general.sampling.top_k", k);
    if (self.sampling_top_p) |p| try putFloat(metadata, alloc, "general.sampling.top_p", p);

    if (self.license) |l| try putStr(metadata, alloc, "general.license", l);
    if (self.license_link) |l| try putStr(metadata, alloc, "general.license.link", l);
    if (self.tags.len > 0) try putStrArray(metadata, alloc, "general.tags", self.tags);
    if (self.languages.len > 0) try putStrArray(metadata, alloc, "general.languages", self.languages);
    if (self.datasets.len > 0) {
        try putNum(metadata, alloc, "general.dataset.count", @as(i64, @intCast(self.datasets.len)));
        for (self.datasets, 0..) |d, i| {
            try putStr(metadata, alloc, try alloc.print("general.dataset.{d}.name", .{i}), d.name);
            if (d.organization) |o| try putStr(metadata, alloc, try alloc.print("general.dataset.{d}.organization", .{i}), o);
            if (d.repo_url) |u| try putStr(metadata, alloc, try alloc.print("general.dataset.{d}.repo_url", .{i}), u);
        }
    }
    if (self.base_models.len > 0) {
        try putNum(metadata, alloc, "general.base_model.count", @as(i64, @intCast(self.base_models.len)));
        for (self.base_models, 0..) |bm, i| {
            try putStr(metadata, alloc, try alloc.print("general.base_model.{d}.name", .{i}), bm.name);
            if (bm.organization) |o| try putStr(metadata, alloc, try alloc.print("general.base_model.{d}.organization", .{i}), o);
            if (bm.repo_url) |u| try putStr(metadata, alloc, try alloc.print("general.base_model.{d}.repo_url", .{i}), u);
        }
    }

    if (self.tokens.len == 0) return;
    if (self.eos_id) |id| try putNum(metadata, alloc, "tokenizer.ggml.eos_token_id", id);
    if (self.bos_id) |id| try putNum(metadata, alloc, "tokenizer.ggml.bos_token_id", id);
    if (self.pad_id) |id| try putNum(metadata, alloc, "tokenizer.ggml.padding_token_id", id);
    if (self.unk_id) |id| try putNum(metadata, alloc, "tokenizer.ggml.unknown_token_id", id);
    if (self.add_bos_token) |b| try metadata.put(alloc, try alloc.dupe(u8, "tokenizer.ggml.add_bos_token"), .{ .bool = b });
    if (self.add_eos_token) |b| try metadata.put(alloc, try alloc.dupe(u8, "tokenizer.ggml.add_eos_token"), .{ .bool = b });
    if (self.add_sep_token) |b| try metadata.put(alloc, try alloc.dupe(u8, "tokenizer.ggml.add_sep_token"), .{ .bool = b });
    if (self.chat_template) |ct| try putStr(metadata, alloc, "tokenizer.chat_template", ct);

    if (self.is_spm) {
        try putStr(metadata, alloc, "tokenizer.ggml.model", "llama");
        try putStr(metadata, alloc, "tokenizer.ggml.pre", "default");
    } else {
        // An empty tag means load() could not name the pre-tokenizer and the
        // caller never settled what to do about it. Writing it would produce a
        // GGUF llama.cpp reads as "no tag at all".
        if (self.tokenizer_pre.len == 0) return error.UnknownPretokenizer;
        // loadMerges has already tried tokenizer.json and merges.txt, so there is
        // nothing left to read: a BPE vocabulary with no merge table writes a GGUF
        // llama.cpp refuses to load ("cannot find tokenizer merges in model file").
        if (self.merges.len == 0) return error.MissingMerges;
        try putStr(metadata, alloc, "tokenizer.ggml.model", "gpt2");
        try putStr(metadata, alloc, "tokenizer.ggml.pre", self.tokenizer_pre);
    }
    try putStrArray(metadata, alloc, "tokenizer.ggml.tokens", self.tokens);
    try putU32Array(metadata, alloc, "tokenizer.ggml.token_type", self.token_types);
    if (self.scores) |scores| {
        var arr: std.json.Array = std.json.Array.init(alloc);
        for (scores) |s| try arr.append(.{ .float = s });
        try metadata.put(alloc, try alloc.dupe(u8, "tokenizer.ggml.scores"), .{ .array = arr });
    }
    if (self.merges.len > 0) try putStrArray(metadata, alloc, "tokenizer.ggml.merges", self.merges);
}

/// Directory component holding the checkpoint: the parent for a weights file
/// path, the trailing component (minus any slash) for a directory path. null
/// when the path carries no such component ("model.safetensors", "./x", "."):
/// only the resolved path says what that directory is called, so the caller
/// has to go to the filesystem for it.
fn checkpointDirName(path: []const u8) ?[]const u8 {
    var p = path;
    while (p.len > 1 and p[p.len - 1] == '/') p = p[0 .. p.len - 1];
    const base = std.Io.Dir.path.basename(p);
    const name = if (std.mem.endsWith(u8, base, ".safetensors") or std.mem.endsWith(u8, base, ".json"))
        std.Io.Dir.path.basename(std.Io.Dir.path.dirname(p) orelse return null)
    else
        base;
    if (name.len == 0 or std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) return null;
    return name;
}

/// "Qwen2.5-0.5B-Instruct" style directories yield the name/basename/size_label
/// parts llama.cpp's converter derives from the checkpoint path. Underscores and
/// hyphens are both separators; a size label is digits plus a magnitude suffix.
fn addNameParts(metadata: *std.json.ObjectMap, alloc: std.mem.Allocator, raw_name: []const u8) !void {
    if (raw_name.len == 0) return;
    const name = try alloc.dupe(u8, raw_name);
    for (name) |*ch| {
        if (ch.* == '-' or ch.* == '_') ch.* = ' ';
    }
    try putStr(metadata, alloc, "general.name", name);

    var toks: std.ArrayList([]const u8) = .empty;
    var it = std.mem.tokenizeScalar(u8, name, ' ');
    while (it.next()) |tk| try toks.append(alloc, tk);

    if (toks.items.len == 0) return;
    try putStr(metadata, alloc, "general.basename", toks.items[0]);
    var rest_from: usize = 1;
    if (toks.items.len > 1 and isSizeLabel(toks.items[1])) {
        try putStr(metadata, alloc, "general.size_label", toks.items[1]);
        rest_from = 2;
    }
    if (toks.items.len > rest_from) {
        const finetune = try std.mem.join(alloc, " ", toks.items[rest_from..]);
        try putStr(metadata, alloc, "general.finetune", finetune);
    }
}

fn isSizeLabel(s: []const u8) bool {
    if (s.len < 2) return false;
    const last = std.ascii.toUpper(s[s.len - 1]);
    if (last != 'B' and last != 'M' and last != 'K') return false;
    var any_digit = false;
    for (s[0 .. s.len - 1]) |c| {
        if (std.ascii.isDigit(c)) {
            any_digit = true;
        } else if (c != '.') return false;
    }
    return any_digit;
}

// ============================================================================
// Reverse direction: synthesize HF sidecars from GGUF metadata. A GGUF-only
// user has no HF files, and the -H safetensors are only loadable with
// config.json and tokenizer files next to them. Everything here is derived
// from the keys the forward direction writes.
// ============================================================================

pub fn metaStr(meta: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    return ggStr(meta, key);
}

fn ggStr(meta: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = meta.get(key) orelse return null;
    return switch (v) {
        .string => |s| s,
        else => null,
    };
}

fn ggU64(meta: std.json.ObjectMap, key: []const u8) ?u64 {
    const v = meta.get(key) orelse return null;
    return switch (v) {
        .integer => |i| if (i >= 0) @intCast(i) else null,
        // Same bound as getU64: the metadata comes from the file, and
        // @intFromFloat out of range is illegal behavior.
        .float => |f| if (f >= 0 and f < 9223372036854775808.0) @intFromFloat(f) else null,
        else => null,
    };
}

fn ggF64(meta: std.json.ObjectMap, key: []const u8) ?f64 {
    const v = meta.get(key) orelse return null;
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}

/// null = the key is absent or holds something that is not a flag.
fn boolIf(meta: std.json.ObjectMap, key: []const u8) ?bool {
    const v = meta.get(key) orelse return null;
    return switch (v) {
        .bool => |b| b,
        // Some writers put these flags in as 0/1 rather than a GGUF BOOL.
        .integer => |i| i != 0,
        else => null,
    };
}

fn boolOf(meta: std.json.ObjectMap, key: []const u8, default: bool) bool {
    return boolIf(meta, key) orelse default;
}

fn ggArr(meta: std.json.ObjectMap, key: []const u8) ?[]const std.json.Value {
    const v = meta.get(key) orelse return null;
    return switch (v) {
        .array => |a| a.items,
        else => null,
    };
}

/// Sidecars are written into the caller's output dir, which may already hold
/// another model's HF files; say so before truncating them.
/// Which of the sidecars `writeSidecars` would produce are already in `dir`.
/// Their names are fixed rather than derived from the output name, so with no -o
/// they land on whatever the source directory holds - an HF repo's own
/// config.json included. Ask before writing anything; the caller reports.
/// `weights_name` is the basename the conversion is about to truncate, when it has
/// not been written yet; null once it has, so the check does not find its own output.
pub fn existingSidecars(io: std.Io, dir: std.Io.Dir, meta: std.json.ObjectMap, weights_name: ?[]const u8, alloc: std.mem.Allocator) ![]const []const u8 {
    var names: std.ArrayList([]const u8) = .empty;
    if (weights_name) |w| {
        try names.append(alloc, w);
        // from_pretrained opens model.safetensors before it reads any index, so
        // a -n output written beside another model's weights loads those
        // instead of its own. Only a clash before the write - afterwards the
        // file may well be this conversion's own output.
        if (!std.mem.eql(u8, w, default_weights_name)) try names.append(alloc, default_weights_name);
    }
    // The index is only written for a renamed weights file, but a stale one
    // left in the directory outranks model.safetensors when transformers
    // loads, so it clashes either way.
    // tokenizer.model is listed whatever vocabulary this write produces: only
    // the spm branch writes one, but a leftover outranks tokenizer.json in
    // LlamaTokenizer, so a bpe write clashes with it too.
    try names.appendSlice(alloc, &.{ "config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json", "tokenizer.model", weights_index_name });
    if (try buildCard(meta, alloc) != null) try names.append(alloc, "README.md");

    var found: std.ArrayList([]const u8) = .empty;
    for (names.items) |name| {
        dir.access(io, name, .{}) catch continue;
        try found.append(alloc, name);
    }
    return found.items;
}

pub const weights_index_name = "model.safetensors.index.json";

/// The one weights name transformers opens without an index.
pub const default_weights_name = "model.safetensors";

/// transformers opens model.safetensors and nothing else, unless an index maps
/// tensor names onto the files holding them. A -n output needs that index or
/// the sidecars describe weights from_pretrained will never find.
pub fn writeWeightsIndex(
    io: std.Io,
    dir: std.Io.Dir,
    weights_name: []const u8,
    tensor_names: []const []const u8,
    total_size: u64,
    alloc: std.mem.Allocator,
) !void {
    var weight_map: std.json.ObjectMap = .empty;
    for (tensor_names) |n| try weight_map.put(alloc, try alloc.dupe(u8, n), try jsonStr(alloc, weights_name));

    var meta: std.json.ObjectMap = .empty;
    try meta.put(alloc, "total_size", .{ .integer = @intCast(total_size) });

    var doc: std.json.ObjectMap = .empty;
    try doc.put(alloc, "metadata", .{ .object = meta });
    try doc.put(alloc, "weight_map", .{ .object = weight_map });
    try emitFile(io, dir, weights_index_name, alloc, .{ .object = doc });
}

fn emitFile(io: std.Io, dir: std.Io.Dir, name: []const u8, alloc: std.mem.Allocator, doc: std.json.Value) !void {
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    try std.json.Stringify.value(doc, .{}, &out.writer);
    const f = try dir.createFile(io, name, .{ .truncate = true });
    defer f.close(io);
    try f.writeStreamingAll(io, out.written());
}

// These build the sidecars, so a swallowed allocation failure is a key silently
// written as null in a file the caller then reports as written.
fn jsonStr(alloc: std.mem.Allocator, s: []const u8) !std.json.Value {
    return .{ .string = try alloc.dupe(u8, s) };
}

fn jsonStrOpt(alloc: std.mem.Allocator, s: ?[]const u8) !std.json.Value {
    if (s) |v| return try jsonStr(alloc, v);
    return .null;
}

const KeyStr = struct { k: []const u8, v: std.json.Value };

fn jsonObj(alloc: std.mem.Allocator, entries: []const KeyStr) !std.json.Value {
    var m: std.json.ObjectMap = .empty;
    for (entries) |e| try m.put(alloc, e.k, e.v);
    return .{ .object = m };
}

fn jsonArr(alloc: std.mem.Allocator, items: []const std.json.Value) !std.json.Value {
    var arr: std.json.Array = .init(alloc);
    for (items) |v| try arr.append(v);
    return .{ .array = arr };
}

/// The split regexes tokenizer.json carries, quoted from the "original regex
/// from tokenizer.json" comments in llama.cpp's llama-vocab.cpp. Qwen3.5's
/// differs from Qwen2's in the marks it keeps with a letter, Llama-3's in the
/// number run: \p{N}{1,3} against \p{N}.
const qwen2_regex = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const qwen35_regex = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const llama3_regex = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/// The llama.cpp pre-tokenizer tags whose splitting is one regex we can carry
/// both ways, paired with it. Tags that share a pre-type share a row; the first
/// tag of a row is the one to write.
///
/// "default" is deliberately absent in both directions. It is llama.cpp's "no
/// idea" fallback, and its pre-type splits on four expressions (the `default:`
/// arm of llm_tokenizer_bpe), which no single-Split tokenizer.json reproduces,
/// so neither reading it nor writing it says anything true about the vocabulary.
/// What each family's own tokenizer.json says beyond the regex. `nfc`: Qwen
/// normalizes to NFC before splitting, which llama.cpp does not, so a GGUF
/// cannot say it. `ignore_merges`: a word already in the vocab skips the merges,
/// as llama.cpp does for these tags. `null_affixes`: the tiktoken-derived files
/// write null subword affixes where HF's gpt2-style converters write "".
const BpePretok = struct {
    tags: []const []const u8,
    regex: []const u8,
    nfc: bool = false,
    ignore_merges: bool = false,
    null_affixes: bool = false,
};
const bpe_pretokenizers = [_]BpePretok{
    .{ .tags = &.{ "llama-bpe", "llama3", "llama-v3" }, .regex = llama3_regex, .ignore_merges = true, .null_affixes = true },
    .{ .tags = &.{"qwen2"}, .regex = qwen2_regex, .nfc = true },
    .{ .tags = &.{ "deepseek-r1-qwen", "megrez" }, .regex = qwen2_regex },
    .{ .tags = &.{"qwen35"}, .regex = qwen35_regex, .nfc = true },
};

/// The regex that reproduces the HF tokenizer.json for `model` under `pre`.
/// GGUF stores only the llama.cpp pre-tokenizer tag, so the per-family detail
/// is fixed here; null = the tag names splitting we cannot write out, and
/// tokenizer.json must not be faked.
fn pretokRegex(model: []const u8, pre: []const u8) ?[]const u8 {
    return if (pretokFor(model, pre)) |p| p.regex else null;
}

fn pretokFor(model: []const u8, pre: []const u8) ?*const BpePretok {
    if (!std.mem.eql(u8, model, "gpt2")) return null; // llama = metaspace, no regex
    for (&bpe_pretokenizers) |*p| {
        for (p.tags) |tag| {
            if (std.mem.eql(u8, tag, pre)) return p;
        }
    }
    return null;
}

/// The llama.cpp pre-tokenizer tag for the splitting in a tokenizer.json.
/// llama.cpp's converter picks the tag by hashing a probe string tokenized by
/// the real tokenizer, which we have no way to run; the regex identifies the
/// same families and is right there in the file. null = splitting no tag in the
/// table describes, which the caller must refuse rather than guess at.
fn detectPretok(root: std.json.ObjectMap) ?[]const u8 {
    const rx = splitRegexOf(root.get("pre_tokenizer") orelse return null) orelse return null;
    for (bpe_pretokenizers) |p| {
        if (std.mem.eql(u8, p.regex, rx)) return p.tags[0];
    }
    return null;
}

/// The pattern of the first Split in a pre_tokenizer, which is usually a
/// Sequence of one Split and a ByteLevel that does no splitting of its own.
fn splitRegexOf(v: std.json.Value) ?[]const u8 {
    const o = obj(v) orelse return null;
    const kind = getStr(o, "type") orelse return null;
    if (std.mem.eql(u8, kind, "Sequence")) {
        const list = o.get("pretokenizers") orelse return null;
        if (list != .array) return null;
        for (list.array.items) |item| {
            if (splitRegexOf(item)) |rx| return rx;
        }
        return null;
    }
    if (!std.mem.eql(u8, kind, "Split")) return null;
    const pattern = obj(o.get("pattern") orelse return null) orelse return null;
    return getStr(pattern, "Regex");
}

fn isAddedType(t: i64) bool {
    return t == GGUF_TOKEN_TYPE_CONTROL or t == GGUF_TOKEN_TYPE_USER_DEFINED;
}

/// Synthesize config.json, generation_config.json, tokenizer_config.json,
/// tokenizer.json, README.md and (for spm vocabularies) tokenizer.model into
/// `dir`, alongside the HF-named safetensors already written there. Keep
/// `checkSidecars` in step with whatever this writes.
/// transformers' name for a SafeTensors dtype, or null for the cluster types it
/// has no plain dtype for (a config.json is better off saying nothing than
/// naming a dtype torch cannot allocate).
fn torchDtypeName(type_str: []const u8) ?[]const u8 {
    const dt = types.DataType.fromString(type_str) catch return null;
    return switch (dt) {
        .F16, .f16 => "float16",
        .BF16, .bf16 => "bfloat16",
        .F32, .f32 => "float32",
        .F64, .f64 => "float64",
        .F8_E4M3 => "float8_e4m3fn",
        .F8_E5M2 => "float8_e5m2",
        .I8, .i8 => "int8",
        .U8 => "uint8",
        else => null,
    };
}

/// The transformers dtype most of `tensors`' bytes are in. Counting bytes and
/// not tensors is what keeps the F32 norms from outvoting the model.
pub fn dominantTorchDtype(tensors: []const types.Tensor, alloc: std.mem.Allocator) !?[]const u8 {
    var bytes = std.StringHashMap(u64).init(alloc);
    defer bytes.deinit();
    for (tensors) |t| {
        const e = try bytes.getOrPutValue(t.type, 0);
        e.value_ptr.* += t.size;
    }
    var best: ?[]const u8 = null;
    var best_bytes: u64 = 0;
    var it = bytes.iterator();
    while (it.next()) |e| {
        if (e.value_ptr.* > best_bytes) {
            best_bytes = e.value_ptr.*;
            best = e.key_ptr.*;
        }
    }
    return torchDtypeName(best orelse return null);
}

/// Everything writeSidecars needs from the GGUF metadata, checked without
/// writing anything. The weights go down first and are the gigabytes, so a
/// vocabulary we cannot rebuild has to stop the conversion before it starts
/// rather than leave a finished model.safetensors beside half a layout.
/// It states what is wrong and applies no policy of its own: UnknownPretokenizer
/// is the one its callers are free to go on from, by leaving tokenizer.json out.
/// The architectures writeConfig knows the transformers shape of. Any other
/// name would become a model_type and a ForCausalLM class transformers lacks.
const hf_config_archs = [_][]const u8{ "llama", "qwen2", "qwen3", "qwen35", "qwen3vl", "qwen2vl" };

/// The architectures `hfConfigWritable` takes, for messages: "llama, qwen2, ... and qwen2vl".
pub const hf_config_archs_text = blk: {
    var s: []const u8 = "";
    for (hf_config_archs, 0..) |n, i| s = s ++ (if (i == 0) "" else if (i == hf_config_archs.len - 1) " and " else ", ") ++ n;
    break :blk s;
};

pub fn hfConfigWritable(arch_name: []const u8) bool {
    for (hf_config_archs) |n| if (std.mem.eql(u8, n, arch_name)) return true;
    return false;
}

pub fn sidecarPrecheck(meta: std.json.ObjectMap) !void {
    // Its own error: a missing architecture is config.json's problem, and
    // reporting it as a missing vocabulary sends the user to the wrong key.
    const arch = ggStr(meta, "general.architecture") orelse return error.MissingArchitecture;
    if (!hfConfigWritable(arch)) return error.HfConfigUnsupported;
    // config.json's shape comes off <arch>.*, and every lookup there reads 0 when
    // the prefix is wrong or the keys are absent: a GGUF renamed by -A, or one
    // written by a tool that never wrote them. A layout describing a 0-layer,
    // 0-wide model loads far enough to be confusing, so refuse it here instead.
    var buf: [256]u8 = undefined;
    for ([_][]const u8{ "block_count", "embedding_length", "feed_forward_length", "attention.head_count" }) |suffix| {
        const key = std.mem.print(&buf, "{s}.{s}", .{ arch, suffix }) catch return error.MissingDimensions;
        if (ggU64(meta, key) == null) return error.MissingDimensions;
    }
    if (ggArr(meta, "tokenizer.ggml.tokens") == null) return error.MissingVocabulary;
    if (ggArr(meta, "tokenizer.ggml.token_type") == null) return error.MissingVocabulary;
    const model = ggStr(meta, "tokenizer.ggml.model") orelse "llama";
    if (std.mem.eql(u8, model, "gpt2")) {
        // Ahead of the pre-tokenizer, which is the one the caller may go on from:
        // a BPE tokenizer.json with an empty merges list merges nothing, and
        // segments every prompt differently from the model the weights came from
        // while loading like any other checkpoint.
        const merges = ggArr(meta, "tokenizer.ggml.merges") orelse return error.MissingMerges;
        if (merges.len == 0) return error.MissingMerges;
        const pre = ggStr(meta, "tokenizer.ggml.pre") orelse "default";
        if (pretokRegex(model, pre) == null) return error.UnknownPretokenizer;
    } else if (ggArr(meta, "tokenizer.ggml.scores") == null) return error.MissingVocabulary;
}

pub fn writeSidecars(
    io: std.Io,
    dir: std.Io.Dir,
    meta: std.json.ObjectMap,
    tensor_names: []const []const u8,
    /// transformers dtype name for the weights being written alongside, or null
    /// to leave it out of config.json.
    torch_dtype: ?[]const u8,
    force: bool,
    /// -u: leave tokenizer.json out when no regex reproduces the pre-tokenizer,
    /// instead of refusing to write the layout at all.
    skip_unknown_pretok: bool,
    /// The vision tower written alongside, from its mmproj; null for a text-only layout.
    vision: ?Vision,
    alloc: std.mem.Allocator,
) !void {
    sidecarPrecheck(meta) catch |e| {
        if (e != error.UnknownPretokenizer or !skip_unknown_pretok) return e;
    };
    const arch_name = ggStr(meta, "general.architecture") orelse return error.MissingArchitecture;
    const tokens = ggArr(meta, "tokenizer.ggml.tokens") orelse return error.MissingVocabulary;
    const token_types = ggArr(meta, "tokenizer.ggml.token_type") orelse return error.MissingVocabulary;
    const model = ggStr(meta, "tokenizer.ggml.model") orelse "llama";
    const pre = ggStr(meta, "tokenizer.ggml.pre") orelse "default";
    const is_bpe = std.mem.eql(u8, model, "gpt2");

    var has_output = false;
    var has_attn_bias = false;
    for (tensor_names) |n| {
        if (std.mem.eql(u8, n, "lm_head.weight")) has_output = true;
        if (std.mem.endsWith(u8, n, "self_attn.q_proj.bias")) has_attn_bias = true;
    }

    const bos_id = ggU64(meta, "tokenizer.ggml.bos_token_id");
    const eos_id = ggU64(meta, "tokenizer.ggml.eos_token_id");
    const pad_id = ggU64(meta, "tokenizer.ggml.padding_token_id");
    const unk_id = ggU64(meta, "tokenizer.ggml.unknown_token_id");
    // A GGUF that omits the key says nothing about BOS, and writing false there
    // would tokenize an SPM vocabulary without one. Default it to true for SPM;
    // for BPE leave the key out and let the tokenizer class decide.
    const add_bos: ?bool = if (meta.contains("tokenizer.ggml.add_bos_token"))
        boolOf(meta, "tokenizer.ggml.add_bos_token", true)
    else if (is_bpe) null else true;

    const emb = ggU64(meta, try keyPref(alloc, arch_name, "embedding_length")) orelse 0;
    const heads = ggU64(meta, try keyPref(alloc, arch_name, "attention.head_count")) orelse 0;
    const key_len = ggU64(meta, try keyPref(alloc, arch_name, "attention.key_length"));

    if (!force and (try existingSidecars(io, dir, meta, null, alloc)).len > 0) return error.SidecarExists;
    const card = try buildCard(meta, alloc);

    try writeConfig(io, dir, meta, alloc, arch_name, tokens.len, emb, heads, key_len, has_output, has_attn_bias, torch_dtype, vision);
    if (vision) |v| {
        try writePreprocessorConfig(io, dir, alloc, v);
        // A processor reads its chat template from its own file, not from
        // tokenizer_config.json, and apply_chat_template refuses without one.
        if (ggStr(meta, "tokenizer.chat_template")) |ct| {
            const f = try dir.createFile(io, "chat_template.jinja", .{ .truncate = true });
            defer f.close(io);
            try f.writeStreamingAll(io, ct);
        }
    }
    try writeGenerationConfig(io, dir, meta, alloc);
    if (card) |bytes| {
        const f = try dir.createFile(io, "README.md", .{ .truncate = true });
        defer f.close(io);
        try f.writeStreamingAll(io, bytes);
    }
    try writeTokenizerConfig(io, dir, meta, alloc, arch_name, tokens, token_types, bos_id, eos_id, pad_id, unk_id, add_bos);
    if (is_bpe) {
        // LlamaTokenizer reads tokenizer.model ahead of tokenizer.json, so one
        // left by an earlier spm run in this directory would tokenize with the
        // wrong vocabulary. Without --force existingSidecars already refused;
        // with it, the overwrite has to cover it.
        dir.deleteFile(io, "tokenizer.model") catch {};
        const merges = ggArr(meta, "tokenizer.ggml.merges") orelse return error.MissingMerges;
        // No regex means -u let the precheck through: writing the vocabulary
        // under some other family's splitting is the one thing worse than
        // leaving it out, so leave it out.
        if (pretokFor(model, pre)) |family| {
            try writeTokenizerJsonBpe(io, dir, meta, alloc, tokens, token_types, merges, family.*, unk_id);
        }
    } else {
        const scores = ggArr(meta, "tokenizer.ggml.scores") orelse return error.MissingVocabulary;
        try writeTokenizerJsonUnigram(io, dir, meta, alloc, tokens, token_types, scores, unk_id, bos_id, eos_id);
        // HF's llama fast tokenizer defers to tokenizer.model when present; the
        // GGUF keeps every field spm needs (piece, score, type), so rebuild the
        // protobuf too. Trainer/normalizer specs use the spm defaults the
        // llama-family checkpoints train with.
        try writeSpmProto(io, dir, alloc, tokens, token_types, scores, unk_id);
    }
}

/// GGUF keeps these as f32, and widening prints 1e-6 as 9.999999974752427e-07.
/// The shortest decimal that reads back as the same f32 is what the source
/// config.json most likely said.
fn f32Json(v: f64) std.json.Value {
    var buf: [64]u8 = undefined;
    const s = std.mem.print(&buf, "{d}", .{@as(f32, @floatCast(v))}) catch return .{ .float = v };
    return .{ .float = std.fmt.parseFloat(f64, s) catch v };
}

/// The first transformers release with this model type. null = none that this
/// file can name, and config.json leaves the key out.
fn minTransformersVersion(arch_name: []const u8) ?[]const u8 {
    const table = [_]struct { arch: []const u8, v: []const u8 }{
        .{ .arch = "llama", .v = "4.28.0" },
        .{ .arch = "qwen2", .v = "4.37.0" },
        .{ .arch = "qwen3", .v = "4.51.0" },
        .{ .arch = "qwen3moe", .v = "4.51.0" },
    };
    for (table) |e| if (std.mem.eql(u8, e.arch, arch_name)) return e.v;
    return null;
}

fn keyPref(alloc: std.mem.Allocator, arch: []const u8, suffix: []const u8) ![]const u8 {
    return alloc.print("{s}.{s}", .{ arch, suffix });
}

fn writeConfig(
    io: std.Io,
    dir: std.Io.Dir,
    meta: std.json.ObjectMap,
    alloc: std.mem.Allocator,
    arch_name: []const u8,
    vocab: usize,
    emb: u64,
    heads: u64,
    key_len: ?u64,
    has_output: bool,
    has_attn_bias: bool,
    torch_dtype: ?[]const u8,
    vision: ?Vision,
) !void {
    var entries: std.ArrayList(KeyStr) = .empty;
    const cls = try alloc.print("{s}ForCausalLM", .{try pyTitle(alloc, arch_name)});
    try entries.append(alloc, .{ .k = "architectures", .v = try jsonArr(alloc, &[_]std.json.Value{try jsonStr(alloc, cls)}) });
    try entries.append(alloc, .{ .k = "model_type", .v = try jsonStr(alloc, arch_name) });
    try entries.append(alloc, .{ .k = "hidden_size", .v = .{ .integer = @intCast(emb) } });
    try entries.append(alloc, .{ .k = "intermediate_size", .v = .{ .integer = @intCast(ggU64(meta, try keyPref(alloc, arch_name, "feed_forward_length")) orelse 0) } });
    try entries.append(alloc, .{ .k = "num_hidden_layers", .v = .{ .integer = @intCast(ggU64(meta, try keyPref(alloc, arch_name, "block_count")) orelse 0) } });
    try entries.append(alloc, .{ .k = "num_attention_heads", .v = .{ .integer = @intCast(heads) } });
    try entries.append(alloc, .{ .k = "num_key_value_heads", .v = .{ .integer = @intCast(ggU64(meta, try keyPref(alloc, arch_name, "attention.head_count_kv")) orelse heads) } });
    if (ggU64(meta, try keyPref(alloc, arch_name, "context_length"))) |ctx| {
        try entries.append(alloc, .{ .k = "max_position_embeddings", .v = .{ .integer = @intCast(ctx) } });
    }
    if (ggF64(meta, try keyPref(alloc, arch_name, "attention.layer_norm_rms_epsilon"))) |eps| {
        try entries.append(alloc, .{ .k = "rms_norm_eps", .v = f32Json(eps) });
    }
    if (ggF64(meta, try keyPref(alloc, arch_name, "rope.freq_base"))) |th| {
        try entries.append(alloc, .{ .k = "rope_theta", .v = f32Json(th) });
    }
    if (ggStr(meta, try keyPref(alloc, arch_name, "rope.scaling.type"))) |kind| {
        if (ggF64(meta, try keyPref(alloc, arch_name, "rope.scaling.factor"))) |factor| {
            var rs: std.ArrayList(KeyStr) = .empty;
            try rs.append(alloc, .{ .k = "rope_type", .v = try jsonStr(alloc, kind) });
            try rs.append(alloc, .{ .k = "type", .v = try jsonStr(alloc, kind) });
            try rs.append(alloc, .{ .k = "factor", .v = f32Json(factor) });
            if (ggU64(meta, try keyPref(alloc, arch_name, "rope.scaling.original_context_length"))) |octx| {
                try rs.append(alloc, .{ .k = "original_max_position_embeddings", .v = .{ .integer = @intCast(octx) } });
            }
            try entries.append(alloc, .{ .k = "rope_scaling", .v = try jsonObj(alloc, rs.items) });
        }
    }
    // head_dim only matters when it is not hidden_size / num_heads.
    if (key_len) |kl| {
        if (heads > 0 and emb / heads != kl) {
            try entries.append(alloc, .{ .k = "head_dim", .v = .{ .integer = @intCast(kl) } });
        }
    }
    try entries.append(alloc, .{ .k = "vocab_size", .v = .{ .integer = @intCast(vocab) } });
    // Omitting this makes from_pretrained load the weights at float32, doubling
    // the memory of an f16 model and re-saving it that way.
    if (torch_dtype) |dt| try entries.append(alloc, .{ .k = "torch_dtype", .v = try jsonStr(alloc, dt) });
    try entries.append(alloc, .{ .k = "hidden_act", .v = try jsonStr(alloc, "silu") });
    try entries.append(alloc, .{ .k = "tie_word_embeddings", .v = .{ .bool = !has_output } });
    try entries.append(alloc, .{ .k = "attention_bias", .v = .{ .bool = has_attn_bias } });
    if (ggU64(meta, "tokenizer.ggml.bos_token_id")) |id| try entries.append(alloc, .{ .k = "bos_token_id", .v = .{ .integer = @intCast(id) } });
    if (ggU64(meta, "tokenizer.ggml.eos_token_id")) |id| try entries.append(alloc, .{ .k = "eos_token_id", .v = .{ .integer = @intCast(id) } });
    // transformers 4.57.3 takes a local config.json without this key for a
    // mistral one and warns that the tokenizer regex is broken. The earliest
    // release that loads the model type is the truthful value.
    if (minTransformersVersion(arch_name)) |v| try entries.append(alloc, .{ .k = "transformers_version", .v = try jsonStr(alloc, v) });

    // The hybrid needs its own shape: transformers loads it as a
    // ForConditionalGeneration whose language tower sits under text_config, the
    // MTP block is not one of num_hidden_layers, and none of the linear
    // attention dimensions have a generic spelling to fall out of the loop
    // above.
    if (std.mem.eql(u8, arch_name, "qwen35")) {
        const nextn = ggU64(meta, try keyPref(alloc, arch_name, "nextn_predict_layers")) orelse 0;
        const blocks = ggU64(meta, try keyPref(alloc, arch_name, "block_count")) orelse 0;
        for (entries.items) |*e| {
            if (std.mem.eql(u8, e.k, "num_hidden_layers") and blocks > nextn) {
                e.v = .{ .integer = @intCast(blocks - nextn) };
            }
        }
        const v_heads = ggU64(meta, try keyPref(alloc, arch_name, "ssm.time_step_rank")) orelse 0;
        const inner = ggU64(meta, try keyPref(alloc, arch_name, "ssm.inner_size")) orelse 0;
        const extra = [_]KeyStr{
            .{ .k = "linear_conv_kernel_dim", .v = .{ .integer = @intCast(ggU64(meta, try keyPref(alloc, arch_name, "ssm.conv_kernel")) orelse 0) } },
            .{ .k = "linear_key_head_dim", .v = .{ .integer = @intCast(ggU64(meta, try keyPref(alloc, arch_name, "ssm.state_size")) orelse 0) } },
            .{ .k = "linear_num_key_heads", .v = .{ .integer = @intCast(ggU64(meta, try keyPref(alloc, arch_name, "ssm.group_count")) orelse 0) } },
            .{ .k = "linear_num_value_heads", .v = .{ .integer = @intCast(v_heads) } },
            .{ .k = "linear_value_head_dim", .v = .{ .integer = @intCast(if (v_heads > 0) inner / v_heads else 0) } },
            .{ .k = "full_attention_interval", .v = .{ .integer = @intCast(ggU64(meta, try keyPref(alloc, arch_name, "full_attention_interval")) orelse 4) } },
            .{ .k = "mtp_num_hidden_layers", .v = .{ .integer = @intCast(nextn) } },
        };
        for (extra) |e| try entries.append(alloc, e);

        var outer: std.ArrayList(KeyStr) = .empty;
        try outer.append(alloc, .{ .k = "architectures", .v = try jsonArr(alloc, &[_]std.json.Value{try jsonStr(alloc, "Qwen3_5ForConditionalGeneration")}) });
        try outer.append(alloc, .{ .k = "model_type", .v = try jsonStr(alloc, "qwen3_5") });
        var inner_entries: std.ArrayList(KeyStr) = .empty;
        for (entries.items) |e| {
            if (std.mem.eql(u8, e.k, "architectures")) continue;
            if (std.mem.eql(u8, e.k, "model_type")) {
                try inner_entries.append(alloc, .{ .k = "model_type", .v = try jsonStr(alloc, "qwen3_5_text") });
                continue;
            }
            try inner_entries.append(alloc, e);
        }
        try outer.append(alloc, .{ .k = "text_config", .v = try jsonObj(alloc, inner_entries.items) });
        if (vision) |v| {
            try outer.append(alloc, .{ .k = "vision_config", .v = try visionConfigJson(alloc, v, "qwen3_5") });
            try appendVisionTokenIds(alloc, &outer, meta);
        }
        if (torch_dtype) |dt| try outer.append(alloc, .{ .k = "torch_dtype", .v = try jsonStr(alloc, dt) });
        try emitFile(io, dir, "config.json", alloc, try jsonObj(alloc, outer.items));
        return;
    }

    // The vision-language archs load as ForConditionalGeneration, which needs
    // the tower's config beside the text one. The caller refuses these without
    // a tower: transformers has no causal LM class for their text alone.
    if (std.mem.eql(u8, arch_name, "qwen3vl") or std.mem.eql(u8, arch_name, "qwen2vl")) {
        const v = vision orelse return error.HfConfigUnsupported;
        const q3 = std.mem.eql(u8, arch_name, "qwen3vl");
        const sections = ggArr(meta, try keyPref(alloc, arch_name, "rope.dimension_sections")) orelse return error.MissingDimensions;
        var ms: std.ArrayList(std.json.Value) = .empty;
        for (sections[0..@min(3, sections.len)]) |sec| try ms.append(alloc, sec);
        var rs: std.ArrayList(KeyStr) = .empty;
        if (q3) {
            try rs.append(alloc, .{ .k = "rope_type", .v = try jsonStr(alloc, "default") });
            try rs.append(alloc, .{ .k = "mrope_section", .v = try jsonArr(alloc, ms.items) });
            try rs.append(alloc, .{ .k = "mrope_interleaved", .v = .{ .bool = true } });
        } else {
            try rs.append(alloc, .{ .k = "type", .v = try jsonStr(alloc, "mrope") });
            try rs.append(alloc, .{ .k = "mrope_section", .v = try jsonArr(alloc, ms.items) });
        }
        var text: std.ArrayList(KeyStr) = .empty;
        for (entries.items) |e| {
            if (std.mem.eql(u8, e.k, "architectures") or std.mem.eql(u8, e.k, "model_type") or
                std.mem.eql(u8, e.k, "transformers_version") or std.mem.eql(u8, e.k, "rope_scaling") or
                std.mem.eql(u8, e.k, "head_dim")) continue;
            try text.append(alloc, e);
        }
        try text.append(alloc, .{ .k = "rope_scaling", .v = try jsonObj(alloc, rs.items) });
        // Qwen3's head width is its own number, not hidden_size / heads.
        if (key_len) |kl| try text.append(alloc, .{ .k = "head_dim", .v = .{ .integer = @intCast(kl) } });

        const vl_cls = if (q3) "Qwen3VLForConditionalGeneration" else "Qwen2_5_VLForConditionalGeneration";
        const model_type = if (q3) "qwen3_vl" else "qwen2_5_vl";
        var outer: std.ArrayList(KeyStr) = .empty;
        try outer.append(alloc, .{ .k = "architectures", .v = try jsonArr(alloc, &[_]std.json.Value{try jsonStr(alloc, vl_cls)}) });
        try outer.append(alloc, .{ .k = "model_type", .v = try jsonStr(alloc, model_type) });
        if (q3) {
            try text.append(alloc, .{ .k = "model_type", .v = try jsonStr(alloc, "qwen3_vl_text") });
            try outer.append(alloc, .{ .k = "text_config", .v = try jsonObj(alloc, text.items) });
            try outer.append(alloc, .{ .k = "tie_word_embeddings", .v = .{ .bool = !has_output } });
            if (torch_dtype) |dt| try outer.append(alloc, .{ .k = "torch_dtype", .v = try jsonStr(alloc, dt) });
        } else {
            // Qwen2.5-VL's published configs keep the text keys at the top.
            try outer.appendSlice(alloc, text.items);
        }
        try outer.append(alloc, .{ .k = "vision_config", .v = try visionConfigJson(alloc, v, model_type) });
        try appendVisionTokenIds(alloc, &outer, meta);
        try outer.append(alloc, .{ .k = "transformers_version", .v = try jsonStr(alloc, if (q3) "4.57.0" else "4.49.0") });
        try emitFile(io, dir, "config.json", alloc, try jsonObj(alloc, outer.items));
        return;
    }

    try emitFile(io, dir, "config.json", alloc, try jsonObj(alloc, entries.items));
}

/// vision_config for `v`, in the spelling of the transformers class that
/// `model_type` names.
fn visionConfigJson(alloc: std.mem.Allocator, v: Vision, model_type: []const u8) !std.json.Value {
    var e: std.ArrayList(KeyStr) = .empty;
    try e.append(alloc, .{ .k = "model_type", .v = try jsonStr(alloc, model_type) });
    try e.append(alloc, .{ .k = "depth", .v = .{ .integer = v.block_count } });
    try e.append(alloc, .{ .k = "hidden_size", .v = .{ .integer = v.embedding_length } });
    try e.append(alloc, .{ .k = "intermediate_size", .v = .{ .integer = v.feed_forward_length } });
    try e.append(alloc, .{ .k = "num_heads", .v = .{ .integer = v.head_count } });
    try e.append(alloc, .{ .k = "out_hidden_size", .v = .{ .integer = v.projection_dim } });
    try e.append(alloc, .{ .k = "patch_size", .v = .{ .integer = v.patch_size } });
    try e.append(alloc, .{ .k = "spatial_merge_size", .v = .{ .integer = v.spatial_merge_size } });
    try e.append(alloc, .{ .k = "temporal_patch_size", .v = .{ .integer = 2 } });
    switch (v.projector) {
        .qwen3vl => {
            try e.append(alloc, .{ .k = "in_channels", .v = .{ .integer = 3 } });
            try e.append(alloc, .{ .k = "hidden_act", .v = try jsonStr(alloc, "gelu_pytorch_tanh") });
            try e.append(alloc, .{ .k = "num_position_embeddings", .v = .{ .integer = v.num_position_embeddings } });
            var ds: std.ArrayList(std.json.Value) = .empty;
            for (v.deepstack_indexes) |i| try ds.append(alloc, .{ .integer = i });
            try e.append(alloc, .{ .k = "deepstack_visual_indexes", .v = try jsonArr(alloc, ds.items) });
        },
        .qwen25vl => {
            try e.append(alloc, .{ .k = "in_chans", .v = .{ .integer = 3 } });
            try e.append(alloc, .{ .k = "hidden_act", .v = try jsonStr(alloc, "silu") });
            // Not in the mmproj; every Qwen2.5-VL size ships these.
            try e.append(alloc, .{ .k = "window_size", .v = .{ .integer = 112 } });
            try e.append(alloc, .{ .k = "tokens_per_second", .v = .{ .integer = 2 } });
            var fa: std.ArrayList(std.json.Value) = .empty;
            var i: u32 = v.n_wa_pattern;
            while (i > 0 and i <= v.block_count) : (i += v.n_wa_pattern) try fa.append(alloc, .{ .integer = i - 1 });
            try e.append(alloc, .{ .k = "fullatt_block_indexes", .v = try jsonArr(alloc, fa.items) });
        },
    }
    return jsonObj(alloc, e.items);
}

/// The placeholder token ids the processor and the model agree on, looked up
/// in the vocabulary rather than assumed.
fn appendVisionTokenIds(alloc: std.mem.Allocator, out: *std.ArrayList(KeyStr), meta: std.json.ObjectMap) !void {
    const tokens = ggArr(meta, "tokenizer.ggml.tokens") orelse return;
    const wanted = [_]struct { key: []const u8, tok: []const u8 }{
        .{ .key = "image_token_id", .tok = "<|image_pad|>" },
        .{ .key = "video_token_id", .tok = "<|video_pad|>" },
        .{ .key = "vision_start_token_id", .tok = "<|vision_start|>" },
        .{ .key = "vision_end_token_id", .tok = "<|vision_end|>" },
    };
    for (wanted) |w| for (tokens, 0..) |t, i| {
        if (t == .string and std.mem.eql(u8, t.string, w.tok)) {
            try out.append(alloc, .{ .k = w.key, .v = .{ .integer = @intCast(i) } });
            break;
        }
    };
}

/// The image processor's config: transformers' Qwen2-VL processor serves all
/// three towers, and the mean/std come off the mmproj.
fn writePreprocessorConfig(io: std.Io, dir: std.Io.Dir, alloc: std.mem.Allocator, v: Vision) !void {
    var mean: std.ArrayList(std.json.Value) = .empty;
    var sd: std.ArrayList(std.json.Value) = .empty;
    for (v.image_mean, v.image_std) |mv, sv| {
        try mean.append(alloc, f32Json(mv));
        try sd.append(alloc, f32Json(sv));
    }
    const doc = try jsonObj(alloc, &.{
        .{ .k = "image_processor_type", .v = try jsonStr(alloc, "Qwen2VLImageProcessor") },
        .{ .k = "image_mean", .v = try jsonArr(alloc, mean.items) },
        .{ .k = "image_std", .v = try jsonArr(alloc, sd.items) },
        .{ .k = "patch_size", .v = .{ .integer = v.patch_size } },
        .{ .k = "merge_size", .v = .{ .integer = v.spatial_merge_size } },
        .{ .k = "temporal_patch_size", .v = .{ .integer = 2 } },
    });
    try emitFile(io, dir, "preprocessor_config.json", alloc, doc);
}

fn writeGenerationConfig(io: std.Io, dir: std.Io.Dir, meta: std.json.ObjectMap, alloc: std.mem.Allocator) !void {
    var entries: std.ArrayList(KeyStr) = .empty;
    if (ggU64(meta, "tokenizer.ggml.eos_token_id")) |id| try entries.append(alloc, .{ .k = "eos_token_id", .v = .{ .integer = @intCast(id) } });
    if (ggU64(meta, "tokenizer.ggml.bos_token_id")) |id| try entries.append(alloc, .{ .k = "bos_token_id", .v = .{ .integer = @intCast(id) } });
    if (ggU64(meta, "tokenizer.ggml.padding_token_id")) |id| try entries.append(alloc, .{ .k = "pad_token_id", .v = .{ .integer = @intCast(id) } });
    const temp = ggF64(meta, "general.sampling.temp");
    const top_p = ggF64(meta, "general.sampling.top_p");
    const top_k = ggU64(meta, "general.sampling.top_k");
    if (temp != null or top_p != null or top_k != null) {
        try entries.append(alloc, .{ .k = "do_sample", .v = .{ .bool = true } });
        if (temp) |v| try entries.append(alloc, .{ .k = "temperature", .v = f32Json(v) });
        if (top_p) |v| try entries.append(alloc, .{ .k = "top_p", .v = f32Json(v) });
        if (top_k) |v| try entries.append(alloc, .{ .k = "top_k", .v = .{ .integer = @intCast(v) } });
    }
    try emitFile(io, dir, "generation_config.json", alloc, try jsonObj(alloc, entries.items));
}

fn tokAt(tokens: []const std.json.Value, id: ?u64) ?[]const u8 {
    if (id) |i| {
        if (i < tokens.len) {
            return switch (tokens[i]) {
                .string => |s| s,
                else => null,
            };
        }
    }
    return null;
}

fn strList(alloc: std.mem.Allocator, items: []const std.json.Value) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    for (items) |it| {
        switch (it) {
            .string => |x| try out.append(alloc, x),
            else => {},
        }
    }
    return out.items;
}

/// Model card: the forward reads general.license/tags/language(s)/dataset.*/
/// base_model.* from README.md YAML frontmatter, so write them back. Entry ids
/// come from the stored repo_url (the only exact-casing source kept).
/// Null when there is nothing to say, which is also what keeps README.md out of
/// the overwrite check.
fn buildCard(meta: std.json.ObjectMap, alloc: std.mem.Allocator) !?[]const u8 {
    var w: std.ArrayList(u8) = .empty;
    try w.appendSlice(alloc, "---\n");
    var any = false;

    if (ggStr(meta, "general.license")) |v| {
        try w.appendSlice(alloc, "license: ");
        try w.appendSlice(alloc, v);
        try w.appendSlice(alloc, "\n");
        any = true;
    }
    if (ggStr(meta, "general.license.link")) |v| {
        try w.appendSlice(alloc, "license_link: ");
        try w.appendSlice(alloc, v);
        try w.appendSlice(alloc, "\n");
        any = true;
    }
    const listy = struct {
        fn emit(buf: *std.ArrayList(u8), a: std.mem.Allocator, key: []const u8, items: []const []const u8) !bool {
            var started = false;
            for (items) |sv| {
                if (!started) {
                    try buf.appendSlice(a, key);
                    try buf.appendSlice(a, ":\n");
                    started = true;
                }
                try buf.appendSlice(a, "- ");
                try buf.appendSlice(a, sv);
                try buf.appendSlice(a, "\n");
            }
            return started;
        }
    }.emit;

    if (ggArr(meta, "general.tags")) |items| {
        const tags = try strList(alloc, items);
        if (tags.len > 0 and try listy(&w, alloc, "tags", tags)) any = true;
    }
    if (ggArr(meta, "general.languages")) |items| {
        const langs = try strList(alloc, items);
        if (langs.len > 0 and try listy(&w, alloc, "language", langs)) any = true;
    }
    const entry_ids = struct {
        fn collect(a: std.mem.Allocator, m: std.json.ObjectMap, prefix: []const u8) ![]const []const u8 {
            var ids: std.ArrayList([]const u8) = .empty;
            // An entry whose id carried no "org/" half stored no repo_url, and
            // one from another host stores a url we cannot turn back into an id,
            // so the usable indices are not a contiguous run: walk to count and
            // skip the gaps rather than stopping at the first one.
            const n = ggU64(m, try a.print("{s}.count", .{prefix})) orelse 0;
            for (0..n) |i| {
                const url = ggStr(m, try a.print("{s}.{d}.repo_url", .{ prefix, i })) orelse continue;
                const cut = "https://huggingface.co/";
                if (std.mem.startsWith(u8, url, cut)) try ids.append(a, url[cut.len..]);
            }
            return ids.items;
        }
    }.collect;

    const ds = try entry_ids(alloc, meta, "general.dataset");
    if (ds.len > 0 and try listy(&w, alloc, "datasets", ds)) any = true;
    const bm = try entry_ids(alloc, meta, "general.base_model");
    if (bm.len > 0 and try listy(&w, alloc, "base_model", bm)) any = true;

    if (!any) return null;
    try w.appendSlice(alloc, "---\n");
    return w.items;
}

fn writeTokenizerConfig(
    io: std.Io,
    dir: std.Io.Dir,
    meta: std.json.ObjectMap,
    alloc: std.mem.Allocator,
    arch_name: []const u8,
    tokens: []const std.json.Value,
    token_types: []const std.json.Value,
    bos_id: ?u64,
    eos_id: ?u64,
    pad_id: ?u64,
    unk_id: ?u64,
    add_bos: ?bool,
) !void {
    var entries: std.ArrayList(KeyStr) = .empty;
    const tok_class = if (std.mem.eql(u8, arch_name, "llama"))
        "LlamaTokenizer"
    else if (std.mem.eql(u8, arch_name, "qwen2") or std.mem.eql(u8, arch_name, "qwen3"))
        "Qwen2Tokenizer"
    else
        null;
    if (tok_class) |tc| try entries.append(alloc, .{ .k = "tokenizer_class", .v = try jsonStr(alloc, tc) });
    if (add_bos) |b| try entries.append(alloc, .{ .k = "add_bos_token", .v = .{ .bool = b } });
    // Through boolIf, like add_bos above: a source that wrote these as 0/1 gets
    // a JSON bool here, which is the only form the way back in accepts.
    if (boolIf(meta, "tokenizer.ggml.add_eos_token")) |b| try entries.append(alloc, .{ .k = "add_eos_token", .v = .{ .bool = b } });
    if (boolIf(meta, "tokenizer.ggml.add_space_prefix")) |b| try entries.append(alloc, .{ .k = "add_space_prefix", .v = .{ .bool = b } });
    if (boolIf(meta, "tokenizer.ggml.remove_extra_whitespaces")) |b| try entries.append(alloc, .{ .k = "remove_extra_whitespaces", .v = .{ .bool = b } });
    if (ggStr(meta, "tokenizer.ggml.pre")) |p| try entries.append(alloc, .{ .k = "tokenizer.ggml.pre", .v = try jsonStr(alloc, p) });
    if (tokAt(tokens, bos_id)) |s| try entries.append(alloc, .{ .k = "bos_token", .v = try jsonStr(alloc, s) });
    if (tokAt(tokens, eos_id)) |s| try entries.append(alloc, .{ .k = "eos_token", .v = try jsonStr(alloc, s) });
    if (tokAt(tokens, pad_id)) |s| try entries.append(alloc, .{ .k = "pad_token", .v = try jsonStr(alloc, s) });
    // Written out even when null: some tokenizer classes default it to a
    // literal "<unk>" that this vocabulary may not have.
    try entries.append(alloc, .{ .k = "unk_token", .v = try jsonStrOpt(alloc, tokAt(tokens, unk_id)) });
    if (ggU64(meta, try keyPref(alloc, arch_name, "context_length"))) |ctx| {
        try entries.append(alloc, .{ .k = "model_max_length", .v = .{ .integer = @intCast(ctx) } });
    }
    // Cleanup rewrites " ," to "," on decode, which a byte-level vocabulary
    // never needs; the GGUF cannot say, and false keeps decode lossless.
    try entries.append(alloc, .{ .k = "clean_up_tokenization_spaces", .v = .{ .bool = false } });
    // CONTROL tokens listed here so the forward keeps them CONTROL: a
    // template-referenced added token absent from this list is USER_DEFINED.
    var controls: std.json.Array = .init(alloc);
    for (tokens, 0..) |tok, i| {
        const t: i64 = if (i < token_types.len) switch (token_types[i]) {
            .integer => |v| v,
            else => 1,
        } else 1;
        if (t != GGUF_TOKEN_TYPE_CONTROL) continue;
        const content = switch (tok) {
            .string => |x| x,
            else => continue,
        };
        controls.append(try jsonStr(alloc, content)) catch return error.OutOfMemory;
    }
    if (controls.items.len > 0) {
        try entries.append(alloc, .{ .k = "additional_special_tokens", .v = .{ .array = controls } });
    }
    if (ggStr(meta, "tokenizer.chat_template")) |ct| try entries.append(alloc, .{ .k = "chat_template", .v = try jsonStr(alloc, ct) });
    try emitFile(io, dir, "tokenizer_config.json", alloc, try jsonObj(alloc, entries.items));
}

/// Build added_tokens entries from CONTROL/USER_DEFINED token types; also
/// returns them for the post-processor template's special-token table.
fn collectAdded(alloc: std.mem.Allocator, tokens: []const std.json.Value, token_types: []const std.json.Value) !std.json.Array {
    var added: std.json.Array = .init(alloc);
    for (tokens, 0..) |tok, i| {
        const t: i64 = if (i < token_types.len) switch (token_types[i]) {
            .integer => |v| v,
            else => 1,
        } else 1;
        if (!isAddedType(t)) continue;
        const content = switch (tok) {
            .string => |s| s,
            else => continue,
        };
        // USER_DEFINED added tokens stay special=false so the forward
        // reclassifies template-referenced ones instead of forcing CONTROL.
        try added.append(try jsonObj(alloc, &.{
            .{ .k = "id", .v = .{ .integer = @intCast(i) } },
            .{ .k = "content", .v = try jsonStr(alloc, content) },
            .{ .k = "single_word", .v = .{ .bool = false } },
            .{ .k = "lstrip", .v = .{ .bool = false } },
            .{ .k = "rstrip", .v = .{ .bool = false } },
            .{ .k = "normalized", .v = .{ .bool = false } },
            .{ .k = "special", .v = .{ .bool = t == GGUF_TOKEN_TYPE_CONTROL } },
        }));
    }
    return added;
}

fn writeTokenizerJsonBpe(
    io: std.Io,
    dir: std.Io.Dir,
    meta: std.json.ObjectMap,
    alloc: std.mem.Allocator,
    tokens: []const std.json.Value,
    token_types: []const std.json.Value,
    merges: []const std.json.Value,
    family: BpePretok,
    unk_id: ?u64,
) !void {
    const regex = family.regex;
    const added = try collectAdded(alloc, tokens, token_types);

    var vocab: std.json.ObjectMap = .empty;
    for (tokens, 0..) |tok, i| {
        const s = switch (tok) {
            .string => |x| x,
            else => continue,
        };
        const tt: i64 = if (i < token_types.len) switch (token_types[i]) {
            .integer => |v| v,
            else => 1,
        } else 1;
        if (tt == GGUF_TOKEN_TYPE_UNUSED) continue;
        // The added tokens live in added_tokens alone, as in the files HF writes.
        if (tt == GGUF_TOKEN_TYPE_CONTROL or tt == GGUF_TOKEN_TYPE_USER_DEFINED) continue;
        try vocab.put(alloc, try alloc.dupe(u8, s), .{ .integer = @intCast(i) });
    }
    var merges_arr: std.json.Array = .init(alloc);
    for (merges) |m| merges_arr.append(m) catch return error.OutOfMemory;

    // Byte-level BPE has every byte in its vocab, so there is nothing to fall
    // back to or fuse, and HF's converters write both false.
    const byte_fallback = boolOf(meta, "tokenizer.ggml.byte_fallback", false);
    const affix: std.json.Value = if (family.null_affixes) .null else try jsonStr(alloc, "");
    const model_doc = try jsonObj(alloc, &.{
        .{ .k = "type", .v = try jsonStr(alloc, "BPE") },
        .{ .k = "dropout", .v = .null },
        .{ .k = "unk_token", .v = try jsonStrOpt(alloc, tokAt(tokens, unk_id)) },
        .{ .k = "continuing_subword_prefix", .v = affix },
        .{ .k = "end_of_word_suffix", .v = affix },
        .{ .k = "fuse_unk", .v = .{ .bool = false } },
        .{ .k = "byte_fallback", .v = .{ .bool = byte_fallback } },
        .{ .k = "ignore_merges", .v = .{ .bool = family.ignore_merges } },
        .{ .k = "vocab", .v = .{ .object = vocab } },
        .{ .k = "merges", .v = .{ .array = merges_arr } },
    });

    // Qwen2-style pre-tokenizer: the split regex, then ByteLevel without its
    // own gpt2 regex; add_prefix_space follows the GGUF space flags.
    const add_prefix_space = boolOf(meta, "tokenizer.ggml.add_space_prefix", false);
    // The Split above already segmented the text the way llama.cpp's pre-type
    // does. Letting ByteLevel split again on its gpt2 regex is a second, wrong
    // segmentation, so it stays off whatever the source metadata says.
    const use_regex = false;
    const pre_tok = try jsonObj(alloc, &.{
        .{ .k = "type", .v = try jsonStr(alloc, "Sequence") },
        .{ .k = "pretokenizers", .v = try jsonArr(alloc, &.{
            try jsonObj(alloc, &.{
                .{ .k = "type", .v = try jsonStr(alloc, "Split") },
                .{ .k = "pattern", .v = try jsonObj(alloc, &.{.{ .k = "Regex", .v = try jsonStr(alloc, regex) }}) },
                .{ .k = "behavior", .v = try jsonStr(alloc, "Isolated") },
                .{ .k = "invert", .v = .{ .bool = false } },
            }),
            try jsonObj(alloc, &.{
                .{ .k = "type", .v = try jsonStr(alloc, "ByteLevel") },
                .{ .k = "add_prefix_space", .v = .{ .bool = add_prefix_space } },
                .{ .k = "trim_offsets", .v = .{ .bool = false } },
                .{ .k = "use_regex", .v = .{ .bool = use_regex } },
            }),
        }) },
    });

    const byte_level = try jsonObj(alloc, &.{
        .{ .k = "type", .v = try jsonStr(alloc, "ByteLevel") },
        .{ .k = "add_prefix_space", .v = .{ .bool = add_prefix_space } },
        .{ .k = "trim_offsets", .v = .{ .bool = false } },
        .{ .k = "use_regex", .v = .{ .bool = use_regex } },
    });

    const doc = try jsonObj(alloc, &.{
        .{ .k = "version", .v = try jsonStr(alloc, "1.0") },
        .{ .k = "truncation", .v = .null },
        .{ .k = "padding", .v = .null },
        .{ .k = "added_tokens", .v = .{ .array = added } },
        // The original model's normalizer, not llama.cpp's lack of one: on
        // decomposed input this tokenizes differently from the GGUF, exactly as
        // the checkpoint it came from does.
        .{ .k = "normalizer", .v = if (family.nfc) try jsonObj(alloc, &.{.{ .k = "type", .v = try jsonStr(alloc, "NFC") }}) else .null },
        .{ .k = "pre_tokenizer", .v = pre_tok },
        .{ .k = "post_processor", .v = byte_level },
        .{ .k = "decoder", .v = byte_level },
        .{ .k = "model", .v = model_doc },
    });
    try emitFile(io, dir, "tokenizer.json", alloc, doc);
}

/// SentencePiece pieces as a crate-BPE model, mirroring HF's LlamaConverter:
/// merges are regenerated from the vocab (tokenizers generate_merges), spaces
/// route through the metaspace char, and byte pieces provide fallback.
fn writeTokenizerJsonUnigram(
    io: std.Io,
    dir: std.Io.Dir,
    meta: std.json.ObjectMap,
    alloc: std.mem.Allocator,
    tokens: []const std.json.Value,
    token_types: []const std.json.Value,
    scores: []const std.json.Value,
    unk_id: ?u64,
    bos_id: ?u64,
    eos_id: ?u64,
) !void {
    const added = try collectAdded(alloc, tokens, token_types);

    var names: std.ArrayList([]const u8) = .empty;
    for (tokens) |tok| {
        switch (tok) {
            .string => |x| try names.append(alloc, x),
            else => try names.append(alloc, ""),
        }
    }

    var vocab: std.json.ObjectMap = .empty;
    for (names.items, 0..) |name, i| {
        try vocab.put(alloc, try alloc.dupe(u8, name), .{ .integer = @intCast(i) });
    }
    var piece_scores: std.ArrayList(f64) = .empty;
    for (scores) |v| {
        const f: f64 = switch (v) {
            .float => |x| x,
            .integer => |x| @floatFromInt(x),
            else => 0,
        };
        try piece_scores.append(alloc, f);
    }
    const merges = try generateMerges(alloc, names.items, piece_scores.items);

    const byte_fallback = boolOf(meta, "tokenizer.ggml.byte_fallback", true);
    const model_doc = try jsonObj(alloc, &.{
        .{ .k = "type", .v = try jsonStr(alloc, "BPE") },
        .{ .k = "dropout", .v = .null },
        .{ .k = "unk_token", .v = try jsonStrOpt(alloc, tokAt(tokens, unk_id)) },
        .{ .k = "continuing_subword_prefix", .v = .null },
        .{ .k = "end_of_word_suffix", .v = .null },
        .{ .k = "fuse_unk", .v = .{ .bool = true } },
        .{ .k = "byte_fallback", .v = .{ .bool = byte_fallback } },
        .{ .k = "vocab", .v = .{ .object = vocab } },
        .{ .k = "merges", .v = .{ .array = merges } },
    });

    const metaspace = "\u{2581}";
    const normalizer = try jsonObj(alloc, &.{
        .{ .k = "type", .v = try jsonStr(alloc, "Sequence") },
        .{ .k = "normalizers", .v = try jsonArr(alloc, &.{
            try jsonObj(alloc, &.{
                .{ .k = "type", .v = try jsonStr(alloc, "Prepend") },
                .{ .k = "prepend", .v = try jsonStr(alloc, metaspace) },
            }),
            try jsonObj(alloc, &.{
                .{ .k = "type", .v = try jsonStr(alloc, "Replace") },
                .{ .k = "pattern", .v = try jsonObj(alloc, &.{.{ .k = "String", .v = try jsonStr(alloc, " ") }}) },
                .{ .k = "content", .v = try jsonStr(alloc, metaspace) },
            }),
        }) },
    });
    const decoder = try jsonObj(alloc, &.{
        .{ .k = "type", .v = try jsonStr(alloc, "Sequence") },
        .{ .k = "decoders", .v = try jsonArr(alloc, &.{
            try jsonObj(alloc, &.{
                .{ .k = "type", .v = try jsonStr(alloc, "Replace") },
                .{ .k = "pattern", .v = try jsonObj(alloc, &.{.{ .k = "String", .v = try jsonStr(alloc, metaspace) }}) },
                .{ .k = "content", .v = try jsonStr(alloc, " ") },
            }),
            try jsonObj(alloc, &.{.{ .k = "type", .v = try jsonStr(alloc, "ByteFallback") }}),
            try jsonObj(alloc, &.{.{ .k = "type", .v = try jsonStr(alloc, "Fuse") }}),
            try jsonObj(alloc, &.{
                .{ .k = "type", .v = try jsonStr(alloc, "Strip") },
                .{ .k = "content", .v = try jsonStr(alloc, " ") },
                .{ .k = "start", .v = .{ .integer = 1 } },
                .{ .k = "stop", .v = .{ .integer = 0 } },
            }),
        }) },
    });

    const post = try templateProcessor(alloc, bos_id, eos_id, tokens);
    const doc = try jsonObj(alloc, &.{
        .{ .k = "version", .v = try jsonStr(alloc, "1.0") },
        .{ .k = "truncation", .v = .null },
        .{ .k = "padding", .v = .null },
        .{ .k = "added_tokens", .v = .{ .array = added } },
        .{ .k = "normalizer", .v = normalizer },
        .{ .k = "pre_tokenizer", .v = .null },
        .{ .k = "post_processor", .v = post },
        .{ .k = "decoder", .v = decoder },
        .{ .k = "model", .v = model_doc },
    });
    try emitFile(io, dir, "tokenizer.json", alloc, doc);
}

const MergeRec = struct {
    l: []const u8,
    r: []const u8,
    score: f64,
    l_id: u32,
    r_id: u32,
    idx: usize,

    fn less(_: void, a: MergeRec, b: MergeRec) bool {
        if (a.score != b.score) return a.score > b.score;
        return a.idx < b.idx;
    }
};

/// Port of tokenizers' SentencePieceExtractor.extract: every piece contributes
/// its prefix/suffix splits whose sides are both in-vocab, ordered within the
/// piece by (left id, right id) ascending, then all of them by piece score
/// descending with generation order as the stable tiebreak. The spm trainer's
/// piece scores are the merge ranks HF's derived BPE models were built with.
fn generateMerges(alloc: std.mem.Allocator, names: []const []const u8, scores: []const f64) !std.json.Array {
    var ids = std.StringHashMap(u32).init(alloc);
    for (names, 0..) |name, i| {
        if (name.len == 0) continue;
        try ids.put(name, @intCast(i));
    }

    var list: std.ArrayList(MergeRec) = .empty;
    var offs: std.ArrayList(usize) = .empty;
    var local: std.ArrayList(MergeRec) = .empty;
    for (names, 0..) |piece, i| {
        if (piece.len == 0) continue;
        offs.clearRetainingCapacity();
        var p: usize = 0;
        while (p < piece.len) {
            const l = std.unicode.utf8ByteSequenceLength(piece[p]) catch 1;
            // GGUF token strings are unvalidated bytes, so a piece can end on a
            // multi-byte lead and run the offset past the end.
            p = @min(p + l, piece.len);
            try offs.append(alloc, p);
        }
        local.clearRetainingCapacity();
        const score: f64 = if (i < scores.len) scores[i] else 0;
        for (offs.items) |off| {
            const lr = piece[0..off];
            const rr = piece[off..];
            if (lr.len == 0 or rr.len == 0) continue;
            const li = ids.get(lr) orelse continue;
            const ri = ids.get(rr) orelse continue;
            try local.append(alloc, .{ .l = lr, .r = rr, .score = score, .l_id = li, .r_id = ri, .idx = 0 });
        }
        std.sort.pdq(MergeRec, local.items, {}, struct {
            fn lt(_: void, x: MergeRec, y: MergeRec) bool {
                if (x.l_id != y.l_id) return x.l_id < y.l_id;
                return x.r_id < y.r_id;
            }
        }.lt);
        for (local.items) |rec| {
            var rec2 = rec;
            rec2.idx = list.items.len;
            try list.append(alloc, rec2);
        }
    }

    std.sort.pdq(MergeRec, list.items, {}, MergeRec.less);

    var arr: std.json.Array = .init(alloc);
    for (list.items) |rec| {
        const joined = try alloc.print("{s} {s}", .{ rec.l, rec.r });
        arr.append(try jsonStr(alloc, joined)) catch return error.OutOfMemory;
    }
    return arr;
}

fn wVarint(buf: *std.ArrayList(u8), alloc: std.mem.Allocator, x: u64) !void {
    var val = x;
    while (val >= 0x80) : (val >>= 7) {
        try buf.append(alloc, @intCast((val & 0x7f) | 0x80));
    }
    try buf.append(alloc, @intCast(val));
}

fn wTag(buf: *std.ArrayList(u8), alloc: std.mem.Allocator, field: u64, wire: u64) !void {
    try wVarint(buf, alloc, (field << 3) | wire);
}

fn wLenField(buf: *std.ArrayList(u8), alloc: std.mem.Allocator, field: u64, payload: []const u8) !void {
    try wTag(buf, alloc, field, 2);
    try wVarint(buf, alloc, payload.len);
    try buf.appendSlice(alloc, payload);
}

/// ModelProto: pieces=1{piece=1 str, score=2 f32, type=3 enum},
/// trainer_spec=2{model_type=3 BPE(2), byte_fallback=35 true, unk_id=40},
/// normalizer_spec=3{name=1 "identity", add_dummy_prefix=3 true,
/// escape_whitespaces=5 true}. false bools are proto3 defaults, omitted.
fn writeSpmProto(
    io: std.Io,
    dir: std.Io.Dir,
    alloc: std.mem.Allocator,
    tokens: []const std.json.Value,
    token_types: []const std.json.Value,
    scores: []const std.json.Value,
    unk_id: ?u64,
) !void {
    // sentencepiece refuses to load a model in which no piece has type UNKNOWN
    // ("unk is not defined"), so the unk piece has to be tagged whatever the
    // GGUF calls it: converters disagree, writing <unk> as type 2 or as 3.
    // An id past the end of the piece list is unusable, and writing it into
    // trainer_spec would name a piece that is not there, so treat it as if the
    // GGUF had not named one at all and search for the piece instead.
    const named_unk: ?u64 = if (unk_id) |u| (if (u < tokens.len) u else null) else null;
    var unk_index: ?usize = null;
    if (named_unk) |u| {
        unk_index = @intCast(u);
    } else for (tokens, 0..) |tok, i| {
        if (tok == .string and std.mem.eql(u8, tok.string, "<unk>")) {
            unk_index = i;
            break;
        }
    }

    var buf: std.ArrayList(u8) = .empty;
    for (tokens, 0..) |tok, i| {
        const piece = switch (tok) {
            .string => |x| x,
            else => continue,
        };
        const score: f32 = if (i < scores.len) switch (scores[i]) {
            .float => |f| @floatCast(f),
            .integer => |v| @floatFromInt(v),
            else => 0,
        } else 0;
        const tt: u32 = if (i < token_types.len) switch (token_types[i]) {
            .integer => |v| if (v >= 0) @intCast(v) else 1,
            else => 1,
        } else 1;
        const spm_type: u64 = if (unk_index == i) 2 else switch (tt) {
            2 => 2,
            3 => 3,
            4 => 4,
            5 => 5,
            6 => 6,
            else => 1,
        };
        var entry: std.ArrayList(u8) = .empty;
        try wLenField(&entry, alloc, 1, piece);
        if (score != 0) { // proto3: a zero float is the default and unwritten
            try wTag(&entry, alloc, 2, 5);
            const bits: u32 = @bitCast(score);
            try entry.appendSlice(alloc, std.mem.asBytes(&bits));
        }
        try wTag(&entry, alloc, 3, 0);
        try wVarint(&entry, alloc, spm_type);
        try wLenField(&buf, alloc, 1, entry.items);
    }

    var trainer: std.ArrayList(u8) = .empty;
    try wTag(&trainer, alloc, 3, 0); // model_type = BPE
    try wVarint(&trainer, alloc, 2);
    try wTag(&trainer, alloc, 35, 0); // byte_fallback = true
    try wVarint(&trainer, alloc, 1);
    const trainer_unk: ?u64 = if (unk_index) |u| @as(u64, u) else null;
    if (trainer_unk) |u| {
        try wTag(&trainer, alloc, 40, 0);
        try wVarint(&trainer, alloc, u);
    }
    try wLenField(&buf, alloc, 2, trainer.items);

    var norm: std.ArrayList(u8) = .empty;
    try wLenField(&norm, alloc, 1, "identity");
    try wTag(&norm, alloc, 3, 0); // add_dummy_prefix = true
    try wVarint(&norm, alloc, 1);
    try wTag(&norm, alloc, 5, 0); // escape_whitespaces = true
    try wVarint(&norm, alloc, 1);
    try wLenField(&buf, alloc, 3, norm.items);

    const f = try dir.createFile(io, "tokenizer.model", .{ .truncate = true });
    defer f.close(io);
    try f.writeStreamingAll(io, buf.items);
}

fn specialTokenItem(alloc: std.mem.Allocator, id_str: []const u8, type_id: u8) !std.json.Value {
    return try jsonObj(alloc, &.{
        .{ .k = "SpecialToken", .v = try jsonObj(alloc, &.{
            .{ .k = "id", .v = try jsonStr(alloc, id_str) },
            .{ .k = "type_id", .v = .{ .integer = type_id } },
        }) },
    });
}

fn seqItem(alloc: std.mem.Allocator, name: []const u8, type_id: u8) !std.json.Value {
    return try jsonObj(alloc, &.{
        .{ .k = "Sequence", .v = try jsonObj(alloc, &.{
            .{ .k = "id", .v = try jsonStr(alloc, name) },
            .{ .k = "type_id", .v = .{ .integer = type_id } },
        }) },
    });
}

fn specialEntry(alloc: std.mem.Allocator, piece: []const u8, id: u64) !std.json.Value {
    const ids = try jsonArr(alloc, &[_]std.json.Value{.{ .integer = @intCast(id) }});
    const toks = try jsonArr(alloc, &[_]std.json.Value{try jsonStr(alloc, piece)});
    return try jsonObj(alloc, &.{
        .{ .k = "id", .v = try jsonStr(alloc, piece) },
        .{ .k = "ids", .v = ids },
        .{ .k = "tokens", .v = toks },
    });
}

/// crate-canonical TemplateProcessing: parsed segment lists, llama-style
/// "<s> {0}" / "<s> {0} </s> {1}". null when the GGUF gives no bos.
fn templateProcessor(alloc: std.mem.Allocator, bos_id: ?u64, eos_id: ?u64, tokens: []const std.json.Value) !std.json.Value {
    const bos = tokAt(tokens, bos_id) orelse return .null;
    const eos = tokAt(tokens, eos_id);

    var single_items: std.ArrayList(std.json.Value) = .empty;
    try single_items.append(alloc, try specialTokenItem(alloc, bos, 0));
    try single_items.append(alloc, try seqItem(alloc, "A", 0));
    const single = try jsonArr(alloc, single_items.items);

    var pair_items: std.ArrayList(std.json.Value) = .empty;
    try pair_items.append(alloc, try specialTokenItem(alloc, bos, 0));
    try pair_items.append(alloc, try seqItem(alloc, "A", 0));
    try pair_items.append(alloc, try specialTokenItem(alloc, bos, 1));
    try pair_items.append(alloc, try seqItem(alloc, "B", 1));
    const pair = try jsonArr(alloc, pair_items.items);

    var specials: std.json.ObjectMap = .empty;
    if (bos_id) |id| try specials.put(alloc, try alloc.dupe(u8, bos), try specialEntry(alloc, bos, id));
    if (eos) |e| {
        if (bos_id == null or !std.mem.eql(u8, e, bos)) {
            try specials.put(alloc, try alloc.dupe(u8, e), try specialEntry(alloc, e, eos_id.?));
        }
    }

    return try jsonObj(alloc, &.{
        .{ .k = "type", .v = try jsonStr(alloc, "TemplateProcessing") },
        .{ .k = "single", .v = single },
        .{ .k = "pair", .v = pair },
        .{ .k = "special_tokens", .v = .{ .object = specials } },
    });
}


// ============================================================================
// Tests: synthetic mini-checkpoints written to a temp dir; token content is
// plain ASCII fixtures, ids and types are what the checks look at.
// ============================================================================

const testing = @import("std").testing;

fn writeFile(dir: std.Io.Dir, io: std.Io, name: []const u8, content: []const u8) !void {
    const f = try dir.createFile(io, name, .{ .truncate = true });
    defer f.close(io);
    try f.writeStreamingAll(io, content);
}

const mini_config =
    \\{ "model_type": "qwen2", "num_hidden_layers": 2, "hidden_size": 4,
    \\  "intermediate_size": 8, "num_attention_heads": 2, "num_key_value_heads": 1,
    \\  "max_position_embeddings": 64, "rms_norm_eps": 1e-5, "rope_theta": 1000000.0,
    \\  "vocab_size": 6 }
;

// Qwen2's, which is what mini_config's model_type ships with. load() refuses a
// BPE tokenizer whose splitting matches no tag, so every fixture carries one.
const mini_pretokenizer =
    \\  "pre_tokenizer": { "type": "Sequence", "pretokenizers": [
    \\    { "type": "Split", "behavior": "Isolated", "pattern": { "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" } },
    \\    { "type": "ByteLevel", "add_prefix_space": false, "use_regex": false } ] },
;

// Llama-3's, which differs from Qwen2's only in the number run: \p{N}{1,3}.
const mini_pretokenizer_llama3 =
    \\  "pre_tokenizer": { "type": "Sequence", "pretokenizers": [
    \\    { "type": "Split", "behavior": "Isolated", "pattern": { "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" } },
    \\    { "type": "ByteLevel", "add_prefix_space": false, "use_regex": false } ] },
;

const mini_bpe_body =
    \\  "model": { "type": "BPE", "vocab": { "a": 0, "b": 1, "c": 2, "d": 3 }, "merges": [] } }
;

// ids: 0 a, 1 b, 2 c, 3 d; added: 4 [SP1] special, 5 [SP2] non-special but in
// additional_special_tokens; 6..? would pad — vocab_size is 6 so no padding here.
const mini_tokenizer = "{" ++ mini_pretokenizer ++
    \\  "model": { "type": "BPE",
    \\   "vocab": { "a": 0, "b": 1, "c": 2, "d": 3 },
    \\   "merges": ["a b", "ab c"] },
    \\  "added_tokens": [
    \\    { "id": 4, "content": "[SP1]", "special": true },
    \\    { "id": 5, "content": "[SP2]", "special": false } ] }
;

const mini_tokenizer_config =
    \\{ "add_bos_token": true,
    \\  "additional_special_tokens": ["[SP2]"],
    \\  "chat_template": "tpl" }
;

const unknown_cfg =
    \\{ "model_type": "not-a-real-model" }
;

const mini_generation =
    \\{ "bos_token_id": 4, "pad_token_id": 0, "eos_token_id": [4, 5],
    \\  "do_sample": true, "temperature": 0.7, "top_k": 20, "top_p": 0.8 }
;

fn makeMiniDir(t: *testing.TmpDir, io: std.Io) ![]const u8 {
    const dir = t.dir;
    try writeFile(dir, io, "config.json", mini_config);
    try writeFile(dir, io, "tokenizer.json", mini_tokenizer);
    try writeFile(dir, io, "tokenizer_config.json", mini_tokenizer_config);
    try writeFile(dir, io, "generation_config.json", mini_generation);
    return "Mini-1B-Chat";
}

test "HfLlm.load reads config, tokenizer and generation sidecars" {
    testing.log_level = .err;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const io = std.testing.io;
    var t = testing.tmpDir(.{});
    defer t.cleanup();
    const dir = try makeMiniDir(&t, io);

    const maybe = try load(io, t.dir, dir, arena.allocator(), null);
    try testing.expect(maybe != null);
    const m = maybe.?;
    try testing.expectEqualStrings("qwen2", m.arch.name);
    try testing.expectEqual(@as(u32, 2), m.block_count);
    try testing.expectEqual(@as(u32, 4), m.embedding_length);
    try testing.expectEqual(@as(u32, 1), m.head_count_kv);
    try testing.expectEqual(@as(u32, 6), m.tokens.len);
    try testing.expectEqualStrings("a", m.tokens[0]);
    try testing.expectEqualStrings("[SP1]", m.tokens[4]);
    // special flag -> CONTROL; an added token without it -> USER_DEFINED, as
    // llama.cpp's converter types them, whatever the chat template or
    // additional_special_tokens say.
    try testing.expectEqual(GGUF_TOKEN_TYPE_NORMAL, m.token_types[0]);
    try testing.expectEqual(GGUF_TOKEN_TYPE_CONTROL, m.token_types[4]);
    try testing.expectEqual(GGUF_TOKEN_TYPE_USER_DEFINED, m.token_types[5]);
    try testing.expectEqual(@as(u32, 4), m.eos_id.?); // first of the eos list
    try testing.expectEqual(@as(u32, 4), m.bos_id.?);
    try testing.expectEqual(@as(?bool, true), m.add_bos_token);
    try testing.expectEqual(@as(?u32, 20), m.sampling_top_k);
    try testing.expectEqualStrings("qwen2", m.tokenizer_pre);
}

test "a config.json eos list is read where no other sidecar names one" {
    testing.log_level = .err;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const io = std.testing.io;
    var t = testing.tmpDir(.{});
    defer t.cleanup();

    // Llama 3 spells eos as a list. With no generation_config.json and a
    // tokenizer_config naming no eos_token, config.json is the only source
    // left, and dropping it leaves the GGUF with nothing to stop generation on.
    try writeFile(t.dir, io, "config.json", mini_config[0 .. mini_config.len - 1] ++
        \\, "bos_token_id": 0, "eos_token_id": [2, 3] }
    );
    try writeFile(t.dir, io, "tokenizer.json", mini_tokenizer);
    try writeFile(t.dir, io, "tokenizer_config.json", "{}");

    const m = (try load(io, t.dir, "Mini", arena.allocator(), null)).?;
    try testing.expectEqual(@as(u32, 2), m.eos_id.?);
    try testing.expectEqual(@as(u32, 0), m.bos_id.?);
}

test "a diffusers text encoder takes the tokenizer dir with its own suffix" {
    testing.log_level = .err;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var t = testing.tmpDir(.{});
    defer t.cleanup();

    for ([_][]const u8{ "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "encoder" }) |d| {
        try t.dir.createDir(io, d, .default_dir);
    }
    for ([_][]const u8{ "text_encoder", "text_encoder_2", "encoder" }) |d| {
        var sub = try t.dir.openDir(io, d, .{});
        defer sub.close(io);
        try writeFile(sub, io, "config.json", mini_config);
    }
    {
        var tok = try t.dir.openDir(io, "tokenizer", .{});
        defer tok.close(io);
        try writeFile(tok, io, "tokenizer.json", mini_tokenizer);
        try writeFile(tok, io, "tokenizer_config.json", mini_tokenizer_config);
    }
    // tokenizer_2 holds a different vocabulary: text_encoder_2 must read it and
    // not tokenizer/, and with no tokenizer in it at all must read nothing.
    {
        var tok2 = try t.dir.openDir(io, "tokenizer_2", .{});
        defer tok2.close(io);
        try writeFile(tok2, io, "tokenizer.json", "{" ++ mini_pretokenizer ++ mini_bpe_body[0 .. mini_bpe_body.len - 1] ++
            \\, "added_tokens": [ { "id": 5, "content": "[T2]", "special": true } ] }
        );
    }

    var te = try t.dir.openDir(io, "text_encoder", .{});
    defer te.close(io);
    const m = (try load(io, te, "text_encoder", a, null)).?;
    try testing.expectEqualStrings("[SP1]", m.tokens[4]);
    try testing.expectEqual(@as(?bool, true), m.add_bos_token);

    var te2 = try t.dir.openDir(io, "text_encoder_2", .{});
    defer te2.close(io);
    const m2 = (try load(io, te2, "text_encoder_2", a, null)).?;
    try testing.expectEqualStrings("[T2]", m2.tokens[5]);

    // Only a text_encoder* dir looks sideways.
    var enc = try t.dir.openDir(io, "encoder", .{});
    defer enc.close(io);
    try testing.expectEqual(@as(usize, 0), (try load(io, enc, "encoder", a, null)).?.tokens.len);

    // An empty tokenizer_2 leaves text_encoder_2 without one rather than
    // borrowing tokenizer/.
    try t.dir.deleteFile(io, "tokenizer_2/tokenizer.json");
    try testing.expectEqual(@as(usize, 0), (try load(io, te2, "text_encoder_2", a, null)).?.tokens.len);
}

test "a config with no tokenizer anywhere loads without a vocabulary, a broken one does not" {
    testing.log_level = .err;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var t = testing.tmpDir(.{});
    defer t.cleanup();
    try writeFile(t.dir, io, "config.json", mini_config);
    try writeFile(t.dir, io, "generation_config.json", mini_generation);

    const m = (try load(io, t.dir, "Mini", a, null)).?;
    try testing.expectEqual(@as(usize, 0), m.tokens.len);
    var md: std.json.ObjectMap = .empty;
    try addMetadata(m, &md, "qwen2", a);
    try testing.expectEqual(@as(i64, 2), md.get("qwen2.block_count").?.integer);
    try testing.expect(md.contains("qwen2.context_length"));
    var it = md.iterator();
    while (it.next()) |e| try testing.expect(!std.mem.startsWith(u8, e.key_ptr.*, "tokenizer."));

    // A tokenizer_config.json with nothing to build a vocabulary from is a
    // checkpoint missing a file, not a text encoder.
    try writeFile(t.dir, io, "tokenizer_config.json", mini_tokenizer_config);
    var why: LoadFailure = .no_config;
    try testing.expectEqual(@as(?Model, null), try load(io, t.dir, "Mini", a, &why));
    try testing.expectEqual(LoadFailure.no_tokenizer, why);
}

test "a qwen3_vl config reads its M-RoPE sections, deepstack count and template file" {
    testing.log_level = .err;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var t = testing.tmpDir(.{});
    defer t.cleanup();
    try writeFile(t.dir, io, "config.json",
        \\{"model_type":"qwen3_vl","text_config":{"num_hidden_layers":2,"hidden_size":4,
        \\ "intermediate_size":8,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,
        \\ "max_position_embeddings":64,"rms_norm_eps":1e-6,"vocab_size":6,
        \\ "rope_parameters":{"rope_theta":5000000,"mrope_section":[24,20,20]}},
        \\ "vision_config":{"deepstack_visual_indexes":[5,11,17]}}
    );
    try writeFile(t.dir, io, "tokenizer.json", mini_tokenizer);
    try writeFile(t.dir, io, "tokenizer_config.json",
        \\{"extra_special_tokens":["[SP2]"]}
    );
    try writeFile(t.dir, io, "chat_template.jinja", "tpl [SP2]");

    const m = (try load(io, t.dir, "VL", a, null)).?;
    try testing.expectEqualStrings("qwen3vl", m.arch.name);
    try testing.expectEqual(@as(?[4]i64, .{ 24, 20, 20, 0 }), m.rope_sections);
    try testing.expectEqual(@as(?u32, 3), m.deepstack_layers);
    try testing.expectEqualStrings("tpl [SP2]", m.chat_template.?);
    // Not flagged special, so USER_DEFINED whatever lists name it.
    try testing.expectEqual(GGUF_TOKEN_TYPE_USER_DEFINED, m.token_types[5]);

    var md: std.json.ObjectMap = .empty;
    try addMetadata(m, &md, "qwen3vl", a);
    try testing.expectEqual(@as(i64, 20), md.get("qwen3vl.rope.dimension_sections").?.array.items[1].integer);
    try testing.expectEqual(@as(i64, 3), md.get("qwen3vl.n_deepstack_layers").?.integer);
    try testing.expectEqual(@as(f64, 5000000), md.get("qwen3vl.rope.freq_base").?.float);

    const q = m.arch;
    try testing.expectEqualStrings("blk.3.attn_q_norm.weight", (try mapName(a, q, "language_model.layers.3.self_attn.q_norm.weight", null)).?);
    try testing.expectEqualStrings("token_embd.weight", (try mapName(a, q, "model.language_model.embed_tokens.weight", null)).?);
    try testing.expectEqualStrings("output_norm.weight", (try mapName(a, q, "language_model.norm.weight", null)).?);
    try testing.expectEqual(@as(?[]const u8, null), try mapName(a, q, "visual.blocks.0.attn.qkv.weight", null));

    // Without sections llama.cpp refuses the file, so load does too.
    try writeFile(t.dir, io, "config.json",
        \\{"model_type":"qwen3_vl","text_config":{"num_hidden_layers":2,"hidden_size":4,
        \\ "intermediate_size":8,"num_attention_heads":2,"max_position_embeddings":64,
        \\ "rms_norm_eps":1e-6,"vocab_size":6}}
    );
    try testing.expectEqual(@as(?Model, null), try load(io, t.dir, "VL", a, null));
}

test "mistral converts as llama" {
    try testing.expectEqualStrings("llama", archForModelType("mistral").?.name);
    try testing.expectEqual(@as(?*const imagearch.Arch, null), archForModelType("olmo"));
}

test "HfLlm.load returns null for non-HF and unknown-model directories" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    var t1 = testing.tmpDir(.{});
    defer t1.cleanup();
    try testing.expectEqual(@as(?Model, null), try load(std.testing.io, t1.dir, t1.sub_path[0..], arena.allocator(), null));

    var t2 = testing.tmpDir(.{});
    defer t2.cleanup();
    try writeFile(t2.dir, std.testing.io, "config.json", unknown_cfg);
    try testing.expectEqual(@as(?Model, null), try load(std.testing.io, t2.dir, t2.sub_path[0..], arena.allocator(), null));
}

test "HfLlm.load reads a post-processor whose BOS and EOS are the same token" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const io = std.testing.io;
    var t = testing.tmpDir(.{});
    defer t.cleanup();

    // One entry in `pair` that is both the leading and the trailing special
    // token: the two ends cross, which used to slice items[1..0].
    try writeFile(t.dir, io, "config.json", mini_config);
    try writeFile(t.dir, io, "tokenizer.json", "{" ++ mini_pretokenizer ++
        \\  "model": { "type": "BPE", "vocab": { "a": 0, "b": 1, "c": 2, "d": 3 }, "merges": [] },
        \\  "post_processor": { "type": "TemplateProcessing",
        \\    "single": [ { "SpecialToken": { "id": "<|endoftext|>" } },
        \\                { "Sequence": { "id": "A" } },
        \\                { "SpecialToken": { "id": "<|endoftext|>" } } ],
        \\    "pair": [ { "SpecialToken": { "id": "<|endoftext|>" } } ] } }
    );
    try writeFile(t.dir, io, "tokenizer_config.json",
        \\{ "bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>" }
    );

    const m = (try load(io, t.dir, "Mini", arena.allocator(), null)).?;
    try testing.expectEqual(@as(?bool, true), m.add_bos_token);
    try testing.expectEqual(@as(?bool, true), m.add_eos_token);
}

test "HfLlm.mapName rewrites state dict names and drops unmapped tensors" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    try testing.expectEqualStrings("token_embd.weight", (try mapName(a, &imagearch.qwen2, "model.embed_tokens.weight", null)).?);
    try testing.expectEqualStrings("output_norm.weight", (try mapName(a, &imagearch.qwen2, "model.norm.weight", null)).?);
    try testing.expectEqualStrings("output.weight", (try mapName(a, &imagearch.qwen2, "lm_head.weight", null)).?);
    try testing.expectEqualStrings("blk.7.attn_q.bias", (try mapName(a, &imagearch.qwen2, "model.layers.7.self_attn.q_proj.bias", null)).?);
    try testing.expectEqualStrings("blk.0.ffn_down.weight", (try mapName(a, &imagearch.qwen2, "model.layers.0.mlp.down_proj.weight", null)).?);
    try testing.expectEqualStrings("blk.3.attn_output.weight", (try mapName(a, &imagearch.qwen2, "model.layers.3.self_attn.o_proj.weight", null)).?);
    // head norms only exist where llama.cpp models them
    try testing.expectEqual(@as(?[]const u8, null), try mapName(a, &imagearch.qwen2, "model.layers.0.self_attn.q_norm.weight", null));
    try testing.expectEqualStrings("blk.2.attn_q_norm.weight", (try mapName(a, &imagearch.qwen3, "model.layers.2.self_attn.q_norm.weight", null)).?);
    // rotary caches and friends get dropped
    try testing.expectEqual(@as(?[]const u8, null), try mapName(a, &imagearch.qwen2, "rotary_emb.inv_freq", null));
}

fn spmWv(buf: []u8, len: *usize, x: u64) void {
    var val = x;
    while (val >= 0x80) : (val >>= 7) {
        buf[len.*] = @intCast((val & 0x7f) | 0x80);
        len.* += 1;
    }
    buf[len.*] = @intCast(val);
    len.* += 1;
}

fn spmEntry(buf: *[512]u8, len: *usize, piece: []const u8, score: f32, ptype: ?u64) void {
    var body: [256]u8 = undefined;
    var blen: usize = 0;
    spmWv(&body, &blen, (1 << 3) | 2);
    spmWv(&body, &blen, piece.len);
    @memcpy(body[blen..][0..piece.len], piece);
    blen += piece.len;
    spmWv(&body, &blen, (2 << 3) | 5);
    const bits: u32 = @bitCast(score);
    @memcpy(body[blen..][0..4], std.mem.asBytes(&bits));
    blen += 4;
    if (ptype) |t| {
        spmWv(&body, &blen, (3 << 3) | 0);
        spmWv(&body, &blen, t);
    }
    spmWv(buf, len, (1 << 3) | 2);
    spmWv(buf, len, blen);
    @memcpy(buf[len.*..][0..blen], body[0..blen]);
    len.* += blen;
}

test "HfLlm.parseSpmModel reads pieces, scores and types" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var bytes: [512]u8 = undefined;
    var len: usize = 0;
    spmEntry(&bytes, &len, "ab", 1.5, null);
    spmEntry(&bytes, &len, "unk", -1000, 2);
    spmEntry(&bytes, &len, "ctl", -1000, 3);
    spmEntry(&bytes, &len, "<0x00>", -1000, 6);
    // trailing top-level varint field must not be parsed as an entry
    spmWv(&bytes, &len, (7 << 3) | 0);
    spmWv(&bytes, &len, 9);

    const pieces = try parseSpmModel(arena.allocator(), bytes[0..len]);
    try testing.expectEqual(@as(usize, 4), pieces.len);
    try testing.expectEqualStrings("ab", pieces[0].piece);
    try testing.expectEqual(@as(f32, 1.5), pieces[0].score);
    try testing.expectEqual(@as(u64, 1), pieces[0].ptype);
    try testing.expectEqual(@as(u64, 2), pieces[1].ptype);
    try testing.expectEqual(@as(f32, -1000), pieces[1].score);
    try testing.expectEqual(@as(u32, GGUF_TOKEN_TYPE_UNKNOWN), spmTokenType(pieces[1].piece, pieces[1].ptype));
    try testing.expectEqual(@as(u32, GGUF_TOKEN_TYPE_CONTROL), spmTokenType(pieces[2].piece, pieces[2].ptype));
    try testing.expectEqual(@as(u32, GGUF_TOKEN_TYPE_BYTE), spmTokenType(pieces[3].piece, pieces[3].ptype));
    try testing.expectEqual(@as(u32, GGUF_TOKEN_TYPE_NORMAL), spmTokenType(pieces[0].piece, pieces[0].ptype));
}

test "HfLlm.ropePermuteInPlace swaps the RoPE row halves per head" {
    const p = ropePermuteInPlace;
    var a = [_]u8{ 0, 1, 2, 3, 4, 5 };
    try p(testing.allocator, &a, 1, 6, 1, 1, false);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 3, 1, 4, 2, 5 }, &a);
    try p(testing.allocator, &a, 1, 6, 1, 1, true);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 1, 2, 3, 4, 5 }, &a);

    var two_head = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7 };
    try p(testing.allocator, &two_head, 1, 8, 1, 2, false);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 2, 1, 3, 4, 6, 5, 7 }, &two_head);
    try p(testing.allocator, &two_head, 1, 8, 1, 2, true);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 1, 2, 3, 4, 5, 6, 7 }, &two_head);

    // cols=2 rows travel together
    var wide = [_]u8{ 10, 11, 20, 21, 30, 31, 40, 41 };
    try p(testing.allocator, &wide, 1, 4, 2, 1, false);
    try testing.expectEqualSlices(u8, &[_]u8{ 10, 11, 30, 31, 20, 21, 40, 41 }, &wide);

    // half > 2 is where the row map stops being its own inverse, so the cycles
    // the in-place walk follows are longer than a swap.
    var four = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7 };
    try p(testing.allocator, &four, 1, 8, 1, 1, false);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 4, 1, 5, 2, 6, 3, 7 }, &four);
    try p(testing.allocator, &four, 1, 8, 1, 1, true);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 1, 2, 3, 4, 5, 6, 7 }, &four);

    var two_groups = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    try p(testing.allocator, &two_groups, 1, 16, 1, 2, false);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15 }, &two_groups);
    try p(testing.allocator, &two_groups, 1, 16, 1, 2, true);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }, &two_groups);

    // A shape that cannot be permuted has to be loud: passing the rows through
    // would write them in the order the output file says they are not in.
    var odd = [_]u8{ 1, 2, 3 };
    try testing.expectError(error.RopeRowsNotDivisible, p(testing.allocator, &odd, 1, 3, 1, 2, false));
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 2, 3 }, &odd);
    var short = [_]u8{ 1, 2, 3, 4 };
    try testing.expectError(error.RopeSizeMismatch, p(testing.allocator, &short, 1, 4, 2, 1, false));
}

test "HfLlm.mapName places the qwen35 hybrid, its MTP head and its vision tower" {
    const a = testing.allocator;
    const q = &imagearch.qwen35;
    const m = mapName;

    // The text tower is nested; the bare model.* spelling works too.
    try testing.expectEqualStrings("token_embd.weight", (try m(a, q, "model.language_model.embed_tokens.weight", 64)).?);
    try testing.expectEqualStrings("output_norm.weight", (try m(a, q, "model.language_model.norm.weight", 64)).?);
    try testing.expectEqualStrings("output.weight", (try m(a, q, "lm_head.weight", 64)).?);

    // Gated DeltaNet block.
    const cases = [_]struct { hf: []const u8, gg: []const u8 }{
        .{ .hf = "linear_attn.in_proj_qkv.weight", .gg = "attn_qkv.weight" },
        .{ .hf = "linear_attn.in_proj_z.weight", .gg = "attn_gate.weight" },
        .{ .hf = "linear_attn.in_proj_a.weight", .gg = "ssm_alpha.weight" },
        .{ .hf = "linear_attn.in_proj_b.weight", .gg = "ssm_beta.weight" },
        .{ .hf = "linear_attn.out_proj.weight", .gg = "ssm_out.weight" },
        .{ .hf = "linear_attn.conv1d.weight", .gg = "ssm_conv1d.weight" },
        .{ .hf = "linear_attn.norm.weight", .gg = "ssm_norm.weight" },
        .{ .hf = "linear_attn.A_log", .gg = "ssm_a" },
        .{ .hf = "linear_attn.dt_bias", .gg = "ssm_dt.bias" },
        .{ .hf = "post_attention_layernorm.weight", .gg = "post_attention_norm.weight" },
        .{ .hf = "self_attn.q_proj.weight", .gg = "attn_q.weight" },
        .{ .hf = "self_attn.k_norm.weight", .gg = "attn_k_norm.weight" },
    };
    for (cases) |c| {
        const hf = try a.print("model.language_model.layers.5.{s}", .{c.hf});
        defer a.free(hf);
        const want = try a.print("blk.5.{s}", .{c.gg});
        defer a.free(want);
        const got = (try m(a, q, hf, 64)).?;
        defer a.free(got);
        try testing.expectEqualStrings(want, got);
    }

    // The MTP head lands on one more block past the text tower, and its four
    // own tensors take the nextn names.
    const mtp = [_]struct { hf: []const u8, gg: []const u8 }{
        .{ .hf = "mtp.fc.weight", .gg = "blk.64.nextn.eh_proj.weight" },
        .{ .hf = "mtp.pre_fc_norm_embedding.weight", .gg = "blk.64.nextn.enorm.weight" },
        .{ .hf = "mtp.pre_fc_norm_hidden.weight", .gg = "blk.64.nextn.hnorm.weight" },
        .{ .hf = "mtp.norm.weight", .gg = "blk.64.nextn.shared_head_norm.weight" },
        .{ .hf = "mtp.layers.0.mlp.down_proj.weight", .gg = "blk.64.ffn_down.weight" },
    };
    for (mtp) |c| {
        const got = (try m(a, q, c.hf, 64)).?;
        defer a.free(got);
        try testing.expectEqualStrings(c.gg, got);
    }

    // Without a block count there is nowhere to put the head.
    try testing.expectEqual(@as(?[]const u8, null), try m(a, q, "mtp.fc.weight", null));

    // The vision tower is a separate file, not a hole in the table.
    try testing.expect(isVisionTensor("model.visual.blocks.0.attn.qkv.weight"));
    try testing.expect(!isVisionTensor("model.language_model.layers.0.mlp.up_proj.weight"));
}

test "expertPart reads per-expert and fused HF expert names" {
    const P = ExpertPart;
    try testing.expectEqual(@as(?P, .{ .block = 3, .expert = 17, .proj = .down }), expertPart("model.layers.3.mlp.experts.17.down_proj.weight", null));
    try testing.expectEqual(@as(?P, .{ .block = 0, .expert = null, .proj = .gate_up }), expertPart("model.language_model.layers.0.mlp.experts.gate_up_proj", null));
    try testing.expectEqual(@as(?P, .{ .block = 0, .expert = null, .proj = .down }), expertPart("model.layers.0.mlp.experts.down_proj.weight", null));
    try testing.expectEqual(@as(?P, .{ .block = 40, .expert = 1, .proj = .gate }), expertPart("mtp.layers.0.mlp.experts.1.gate_proj.weight", 40));
    // The MTP head has nowhere to go without a block count.
    try testing.expectEqual(@as(?P, null), expertPart("mtp.layers.0.mlp.experts.1.gate_proj.weight", null));
    // Router, shared expert and dense MLP are renames, not stack parts.
    try testing.expectEqual(@as(?P, null), expertPart("model.layers.0.mlp.gate.weight", null));
    try testing.expectEqual(@as(?P, null), expertPart("model.layers.0.mlp.shared_expert.up_proj.weight", null));
    try testing.expectEqual(@as(?P, null), expertPart("model.layers.0.mlp.gate_proj.weight", null));
    try testing.expectEqual(@as(?P, null), expertPart("model.layers.0.mlp.experts.0.gate_proj.bias", null));
}

fn fakeTensor(name: []const u8, dims: []usize) types.Tensor {
    return .{ .name = name, .type = "BF16", .dims = dims, .size = 0, .offset = 0 };
}

test "planExpertStacks stacks per-expert weights and splits a fused gate_up" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    testing.log_level = .err;

    var d_gate = [_]usize{ 6, 4 };
    var d_down = [_]usize{ 4, 6 };
    // Out of order on purpose: the stack is in expert order, not file order.
    const per = [_]types.Tensor{
        fakeTensor("model.layers.1.mlp.experts.1.gate_proj.weight", &d_gate),
        fakeTensor("model.layers.1.mlp.experts.0.gate_proj.weight", &d_gate),
        fakeTensor("model.layers.1.mlp.experts.0.down_proj.weight", &d_down),
        fakeTensor("model.layers.1.mlp.experts.1.down_proj.weight", &d_down),
    };
    const stacks = try planExpertStacks(a, &per, 2, null);
    try testing.expectEqual(@as(usize, 2), stacks.len);
    for (stacks) |st| {
        if (std.mem.eql(u8, st.name, "blk.1.ffn_gate_exps.weight")) {
            try testing.expectEqualSlices(usize, &.{ 2, 6, 4 }, st.dims);
            try testing.expectEqualStrings("model.layers.1.mlp.experts.0.gate_proj.weight", st.segments[0].tensor.name);
            try testing.expectEqualStrings("model.layers.1.mlp.experts.1.gate_proj.weight", st.segments[1].tensor.name);
        } else {
            try testing.expectEqualStrings("blk.1.ffn_down_exps.weight", st.name);
            try testing.expectEqualSlices(usize, &.{ 2, 4, 6 }, st.dims);
        }
    }

    // A missing expert is refused, not written as a short stack.
    try testing.expectError(error.IncompleteExperts, planExpertStacks(a, per[0..3], 2, null));
    // So is a mismatched shape.
    var d_odd = [_]usize{ 5, 4 };
    const odd = [_]types.Tensor{ per[1], fakeTensor("model.layers.1.mlp.experts.1.gate_proj.weight", &d_odd) };
    try testing.expectError(error.IncompleteExperts, planExpertStacks(a, &odd, 2, null));
    // And expert weights with no expert count to check them against.
    try testing.expectError(error.IncompleteExperts, planExpertStacks(a, &per, 0, null));

    // Fused [E, 2*ff, embd]: gate takes the first ff rows of each expert, up
    // the second, and down passes through whole.
    var d_gu = [_]usize{ 3, 8, 5 };
    var d_fd = [_]usize{ 3, 5, 4 };
    const fused = [_]types.Tensor{
        fakeTensor("model.layers.0.mlp.experts.gate_up_proj", &d_gu),
        fakeTensor("model.layers.0.mlp.experts.down_proj", &d_fd),
    };
    const fs = try planExpertStacks(a, &fused, 3, null);
    try testing.expectEqual(@as(usize, 3), fs.len);
    for (fs) |st| {
        if (std.mem.eql(u8, st.name, "blk.0.ffn_down_exps.weight")) {
            try testing.expectEqualSlices(usize, &.{ 3, 5, 4 }, st.dims);
            try testing.expectEqual(@as(usize, 1), st.segments.len);
            try testing.expectEqual(@as(usize, 60), st.segments[0].elem_count);
            continue;
        }
        try testing.expectEqualSlices(usize, &.{ 3, 4, 5 }, st.dims);
        try testing.expectEqual(@as(usize, 3), st.segments.len);
        const up = std.mem.eql(u8, st.name, "blk.0.ffn_up_exps.weight");
        try testing.expect(up or std.mem.eql(u8, st.name, "blk.0.ffn_gate_exps.weight"));
        for (st.segments, 0..) |sg, x| {
            try testing.expectEqual(@as(usize, 20), sg.elem_count);
            try testing.expectEqual((x * 2 + @intFromBool(up)) * 20, sg.elem_offset);
        }
    }
    // A fused tensor whose leading dimension is not the expert count.
    try testing.expectError(error.IncompleteExperts, planExpertStacks(a, &fused, 4, null));
}

test "HfLlm.mapNameReverse round-trips the qwen35 names it mapped forward" {
    const a = testing.allocator;
    const q = &imagearch.qwen35;

    // Every name mapName produces must come back as the name it came from.
    const names = [_][]const u8{
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
        "model.language_model.layers.5.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.5.linear_attn.in_proj_z.weight",
        "model.language_model.layers.5.linear_attn.A_log",
        "model.language_model.layers.5.linear_attn.dt_bias",
        "model.language_model.layers.5.linear_attn.out_proj.weight",
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "model.language_model.layers.3.self_attn.k_norm.weight",
        "model.language_model.layers.3.post_attention_layernorm.weight",
        "mtp.fc.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight",
        "mtp.layers.0.mlp.down_proj.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
    };
    // Both directions return string literals for the three top-level tensors
    // and allocate for everything layer-indexed, so the frees follow the prefix.
    for (names) |hf| {
        const native = (try mapName(a, q, hf, 64)).?;
        defer if (std.mem.startsWith(u8, native, "blk.")) a.free(native);
        const back = (try mapNameReverse(a, q, native, 64)).?;
        defer if (std.mem.startsWith(u8, back, "model.language_model.layers.") or
            std.mem.startsWith(u8, back, "mtp.")) a.free(back);
        try testing.expectEqualStrings(hf, back);
    }

    // Without a block count the tail blocks are ordinary layers, not the head.
    const no_mtp = (try mapNameReverse(a, q, "blk.64.ffn_down.weight", null)).?;
    defer a.free(no_mtp);
    try testing.expectEqualStrings("model.language_model.layers.64.mlp.down_proj.weight", no_mtp);
}

fn testVision(projector: VisionProjector) Vision {
    return .{
        .projector = projector,
        .image_size = 768,
        .patch_size = 16,
        .embedding_length = 64,
        .feed_forward_length = 128,
        .block_count = 24,
        .head_count = 4,
        .projection_dim = 64,
        .spatial_merge_size = 2,
        .layer_norm_eps = 1e-6,
        .image_mean = .{ 0.5, 0.5, 0.5 },
        .image_std = .{ 0.5, 0.5, 0.5 },
        .deepstack_indexes = &.{ 5, 11, 17 },
        .n_wa_pattern = 8,
    };
}

test "HfLlm.mapVisionName maps both towers into the clip namespace, and back" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const Case = struct { hf: []const u8, gg: []const u8 };
    const q3 = [_]Case{
        .{ .hf = "pos_embed.weight", .gg = "v.position_embd.weight" },
        .{ .hf = "merger.norm.bias", .gg = "v.post_ln.bias" },
        .{ .hf = "merger.linear_fc1.weight", .gg = "mm.0.weight" },
        .{ .hf = "merger.linear_fc2.bias", .gg = "mm.2.bias" },
        .{ .hf = "patch_embed.proj.bias", .gg = "v.patch_embd.bias" },
        .{ .hf = "blocks.9.norm1.weight", .gg = "v.blk.9.ln1.weight" },
        .{ .hf = "blocks.9.norm2.bias", .gg = "v.blk.9.ln2.bias" },
        .{ .hf = "blocks.9.attn.qkv.weight", .gg = "v.blk.9.attn_qkv.weight" },
        .{ .hf = "blocks.9.attn.proj.bias", .gg = "v.blk.9.attn_out.bias" },
        .{ .hf = "blocks.9.mlp.linear_fc1.weight", .gg = "v.blk.9.ffn_up.weight" },
        .{ .hf = "blocks.9.mlp.linear_fc2.bias", .gg = "v.blk.9.ffn_down.bias" },
        // Mergers are listed in order and named after the block they tap.
        .{ .hf = "deepstack_merger_list.0.norm.weight", .gg = "v.deepstack.5.norm.weight" },
        .{ .hf = "deepstack_merger_list.2.linear_fc2.bias", .gg = "v.deepstack.17.fc2.bias" },
    };
    const q25 = [_]Case{
        .{ .hf = "merger.ln_q.weight", .gg = "v.post_ln.weight" },
        .{ .hf = "merger.mlp.0.weight", .gg = "mm.0.weight" },
        .{ .hf = "merger.mlp.2.bias", .gg = "mm.2.bias" },
        .{ .hf = "blocks.3.norm1.weight", .gg = "v.blk.3.ln1.weight" },
        .{ .hf = "blocks.3.mlp.gate_proj.weight", .gg = "v.blk.3.ffn_gate.weight" },
        .{ .hf = "blocks.3.mlp.up_proj.bias", .gg = "v.blk.3.ffn_up.bias" },
        .{ .hf = "blocks.3.mlp.down_proj.weight", .gg = "v.blk.3.ffn_down.weight" },
        .{ .hf = "blocks.3.attn.proj.weight", .gg = "v.blk.3.attn_out.weight" },
    };
    inline for (.{ .{ VisionProjector.qwen3vl, &q3 }, .{ VisionProjector.qwen25vl, &q25 } }) |pc| {
        const v = testVision(pc[0]);
        for (pc[1]) |c| for ([_][]const u8{ "model.visual.", "visual." }) |pre| {
            const hf = try a.print("{s}{s}", .{ pre, c.hf });
            try testing.expectEqualStrings(c.gg, (try mapVisionName(a, v, hf)).?);
            try testing.expectEqualStrings(hf, (try mapVisionNameReverse(a, v, pre, c.gg)).?);
        };
    }

    const v3 = testVision(.qwen3vl);
    const v25 = testVision(.qwen25vl);
    // The split tensors have no single name.
    try testing.expectEqual(@as(?[]const u8, null), try mapVisionName(a, v3, "model.visual.patch_embed.proj.weight"));
    try testing.expectEqual(@as(?[]const u8, null), try mapVisionName(a, v25, "visual.blocks.0.attn.qkv.weight"));
    try testing.expect(isVisionPatchEmbedWeight("model.visual.patch_embed.proj.weight"));
    // A merger past the configured list, a tapped block that is not one, and
    // the other tower's names do not map.
    try testing.expectEqual(@as(?[]const u8, null), try mapVisionName(a, v3, "visual.deepstack_merger_list.3.norm.weight"));
    try testing.expectEqual(@as(?[]const u8, null), try mapVisionNameReverse(a, v3, "visual.", "v.deepstack.6.norm.weight"));
    try testing.expectEqual(@as(?[]const u8, null), try mapVisionName(a, v25, "visual.merger.linear_fc1.weight"));
    try testing.expectEqual(@as(?[]const u8, null), try mapVisionName(a, v3, "visual.merger.ln_q.weight"));
    // Text tensors are not the tower's.
    try testing.expectEqual(@as(?[]const u8, null), try mapVisionName(a, v3, "model.language_model.norm.weight"));
}

test "HfLlm.planVision splits the patch conv by temporal step and qwen25vl's qkv in three" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const t = struct {
        fn f(name: []const u8, dims: []const usize, al: std.mem.Allocator) !types.Tensor {
            return .{ .name = name, .type = "BF16", .dims = try al.dupe(usize, dims), .size = 0, .offset = 0 };
        }
    }.f;
    const src = [_]types.Tensor{
        try t("visual.patch_embed.proj.weight", &.{ 8, 3, 2, 4, 4 }, a),
        try t("visual.blocks.0.attn.qkv.weight", &.{ 12, 4 }, a),
        try t("visual.blocks.0.attn.qkv.bias", &.{12}, a),
        try t("visual.blocks.0.norm1.weight", &.{4}, a),
        try t("visual.blocks.0.what.weight", &.{4}, a),
    };
    const plan = try planVision(a, testVision(.qwen25vl), &src);
    try testing.expectEqual(@as(usize, 1), plan.unmapped.len);
    try testing.expectEqualStrings("visual.blocks.0.what.weight", plan.unmapped[0]);
    // Two halves, three weights, three biases.
    try testing.expectEqual(@as(usize, 8), plan.stacks.len);
    try testing.expectEqual(@as(usize, 9), plan.outputs.len);

    const half1 = plan.stacks[1];
    try testing.expectEqualStrings("v.patch_embd.weight.1", half1.name);
    try testing.expectEqualSlices(usize, &.{ 8, 3, 4, 4 }, half1.dims);
    // One 4x4 plane per (out, in) pair, the second of each pair of planes.
    try testing.expectEqual(@as(usize, 16), half1.segments[0].elem_offset);
    try testing.expectEqual(@as(usize, 16), half1.segments[0].elem_count);
    try testing.expectEqual(@as(usize, 24), half1.segments[0].runs);
    try testing.expectEqual(@as(usize, 32), half1.segments[0].stride);

    const k = plan.stacks[3];
    try testing.expectEqualStrings("v.blk.0.attn_k.weight", k.name);
    try testing.expectEqualSlices(usize, &.{ 4, 4 }, k.dims);
    try testing.expectEqual(@as(usize, 16), k.segments[0].elem_offset);
    try testing.expectEqual(@as(usize, 16), k.segments[0].elem_count);
    const vb = plan.stacks[7];
    try testing.expectEqualStrings("v.blk.0.attn_v.bias", vb.name);
    try testing.expectEqual(@as(usize, 8), vb.segments[0].elem_offset);

    // qwen3vl keeps qkv whole.
    const plan3 = try planVision(a, testVision(.qwen3vl), src[1..2]);
    try testing.expectEqual(@as(usize, 0), plan3.stacks.len);
    try testing.expectEqualStrings("v.blk.0.attn_qkv.weight", plan3.outputs[0].name);

    // A temporal patch other than 2 is not what llama.cpp's clip graph reads.
    const odd = [_]types.Tensor{try t("visual.patch_embed.proj.weight", &.{ 8, 3, 3, 4, 4 }, a)};
    try testing.expectError(error.UnexpectedVisionShape, planVision(a, testVision(.qwen3vl), &odd));
}

test "HfLlm.planVisionReverse joins the patch conv halves and qwen25vl's q, k and v" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const t = struct {
        fn f(name: []const u8, dims: []const usize, al: std.mem.Allocator) !types.Tensor {
            return .{ .name = name, .type = "f16", .dims = try al.dupe(usize, dims), .size = 0, .offset = 0 };
        }
    }.f;
    const src = [_]types.Tensor{
        try t("v.patch_embd.weight", &.{ 8, 3, 4, 4 }, a),
        try t("v.patch_embd.weight.1", &.{ 8, 3, 4, 4 }, a),
        try t("v.blk.2.attn_q.weight", &.{ 4, 4 }, a),
        try t("v.blk.2.attn_k.weight", &.{ 4, 4 }, a),
        try t("v.blk.2.attn_v.weight", &.{ 4, 4 }, a),
        try t("v.blk.2.ffn_gate.bias", &.{4}, a),
        // Renamed by an earlier pass over the same handle.
        try t("visual.merger.ln_q.weight", &.{4}, a),
        try t("v.mystery.weight", &.{4}, a),
    };
    const plan = try planVisionReverse(a, testVision(.qwen25vl), "visual.", &src);
    try testing.expectEqual(@as(usize, 1), plan.unmapped.len);
    try testing.expectEqual(@as(usize, 2), plan.stacks.len);

    const conv = plan.stacks[0];
    try testing.expectEqualStrings("visual.patch_embed.proj.weight", conv.name);
    try testing.expectEqualSlices(usize, &.{ 8, 3, 2, 4, 4 }, conv.dims);
    // Each (out, in) pair's two 4x4 planes interleave, t=0 first.
    for (conv.segments, 0..) |sg, half| {
        try testing.expectEqualStrings(patch_embd_halves[half], sg.tensor.name);
        try testing.expectEqual(@as(usize, 24), sg.runs);
        try testing.expectEqual(@as(usize, 16), sg.stride);
        try testing.expectEqual(@as(?usize, half * 16), sg.out_offset);
        try testing.expectEqual(@as(usize, 32), sg.out_stride);
    }
    const qkv = plan.stacks[1];
    try testing.expectEqualStrings("visual.blocks.2.attn.qkv.weight", qkv.name);
    try testing.expectEqualSlices(usize, &.{ 12, 4 }, qkv.dims);
    try testing.expectEqualStrings("v.blk.2.attn_v.weight", qkv.segments[2].tensor.name);

    var names: std.ArrayList([]const u8) = .empty;
    for (plan.sources) |sv| try names.append(a, sv.name);
    for ([_][]const u8{ "visual.blocks.2.mlp.gate_proj.bias", "visual.merger.ln_q.weight", "v.patch_embd.weight.1", "v.blk.2.attn_k.weight" }) |want| {
        for (names.items) |n| {
            if (std.mem.eql(u8, n, want)) break;
        } else return error.TestExpectedEqual;
    }

    // A half on its own cannot be joined.
    try testing.expectError(error.UnexpectedVisionShape, planVisionReverse(a, testVision(.qwen3vl), "model.visual.", src[0..1]));
}

test "HfLlm.visionFromClip reads the tower back out of the clip keys" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var meta: std.json.ObjectMap = .empty;
    const v0 = testVision(.qwen3vl);
    const m = Model{
        .arch = &imagearch.qwen3vl,
        .dir = "/x/Qwen3-VL-4B",
        .block_count = 1,
        .embedding_length = 64,
        .feed_forward_length = 1,
        .head_count = 1,
        .head_count_kv = 1,
        .context_length = 1,
        .rms_eps = 1e-6,
        .rope_theta = 1,
        .rope_scaling = null,
        .vocab_size = 1,
        .head_dim = null,
        .vision = v0,
        .is_spm = false,
        .scores = null,
        .tokens = &.{},
        .token_types = &.{},
        .merges = &.{},
        .tokenizer_pre = "",
        .chat_template = null,
        .add_bos_token = null,
        .add_eos_token = null,
        .add_sep_token = null,
        .bos_id = null,
        .eos_id = null,
        .pad_id = null,
        .unk_id = null,
        .sampling_temp = null,
        .sampling_top_k = null,
        .sampling_top_p = null,
        .license = null,
        .license_link = null,
        .tags = &.{},
        .languages = &.{},
        .datasets = &.{},
        .base_models = &.{},
    };
    try addVisionMetadata(m, &meta, a);
    try testing.expectEqualStrings("clip", meta.get("general.architecture").?.string);
    try testing.expectEqualStrings("qwen3vl_merger", meta.get("clip.projector_type").?.string);
    try testing.expectEqual(@as(usize, 24), meta.get("clip.vision.is_deepstack_layers").?.array.items.len);

    const pos = [_]types.Tensor{.{ .name = "v.position_embd.weight", .type = "f32", .dims = @constCast(&[_]usize{ 2304, 64 }), .size = 0, .offset = 0 }};
    const v = (try visionFromClip(meta, &pos, a)).?;
    try testing.expectEqual(VisionProjector.qwen3vl, v.projector);
    try testing.expectEqual(v0.block_count, v.block_count);
    try testing.expectEqual(v0.embedding_length, v.embedding_length);
    try testing.expectEqual(v0.projection_dim, v.projection_dim);
    try testing.expectEqualSlices(u32, v0.deepstack_indexes, v.deepstack_indexes);
    try testing.expectEqual(@as(u32, 2304), v.num_position_embeddings);
    try testing.expectEqual(v0.image_mean, v.image_mean);

    // A projector this tool does not map back is not one.
    try putStr(&meta, a, "clip.projector_type", "gemma3");
    try testing.expectEqual(@as(?Vision, null), try visionFromClip(meta, &pos, a));
}

test "HfLlm.mmprojTensorType follows llama.cpp's converter per file type" {
    const T = mmprojTensorType;
    const w2 = &[_]usize{ 64, 64 };
    // What its converter wrote for a Qwen3-VL at each --outtype.
    try testing.expectEqual(types.DataType.f16, T("v.blk.0.attn_qkv.weight", w2, .f16));
    try testing.expectEqual(types.DataType.bf16, T("v.blk.0.attn_qkv.weight", w2, .bf16));
    try testing.expectEqual(types.DataType.q8_0, T("mm.0.weight", w2, .q8_0));
    try testing.expectEqual(types.DataType.f32, T("v.blk.0.ln1.weight", &.{64}, .f16));
    try testing.expectEqual(types.DataType.f32, T("v.blk.0.attn_qkv.bias", &.{192}, .bf16));
    try testing.expectEqual(types.DataType.f32, T("v.position_embd.weight", w2, .f16));
    const conv = &[_]usize{ 64, 3, 16, 16 };
    try testing.expectEqual(types.DataType.f16, T("v.patch_embd.weight", conv, .f16));
    try testing.expectEqual(types.DataType.f16, T("v.patch_embd.weight.1", conv, .f16));
    try testing.expectEqual(types.DataType.f32, T("v.patch_embd.weight.1", conv, .bf16));
    try testing.expectEqual(types.DataType.f32, T("v.patch_embd.weight", conv, .q8_0));
    // A q8_0 row that is not whole blocks falls back to f16.
    try testing.expectEqual(types.DataType.f16, T("mm.2.weight", &.{ 64, 48 }, .q8_0));

    try testing.expectEqual(types.DataType.f16, mmprojFileType(.q4_k));
    try testing.expectEqual(types.DataType.f16, mmprojFileType(null));
    try testing.expectEqual(types.DataType.bf16, mmprojFileType(.bf16));
    try testing.expectEqual(types.DataType.q8_0, mmprojFileType(.q8_0));
    try testing.expectEqual(types.DataType.f32, mmprojFileType(.F32));
}

test "HfLlm.vReorderInPlace tiles V heads that were grouped by K head" {
    const p = vReorderInPlace;

    // k_heads=2, v_per_k=3, head_dim=1: rows arrive k-major as
    // (k0v0 k0v1 k0v2 k1v0 k1v1 k1v2) and leave v-major.
    var a = [_]u8{ 0, 1, 2, 3, 4, 5 };
    try p(testing.allocator, &a, 1, 6, 1, 2, 3, 1, false);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 3, 1, 4, 2, 5 }, &a);
    try p(testing.allocator, &a, 1, 6, 1, 2, 3, 1, true);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 1, 2, 3, 4, 5 }, &a);

    // head_dim=2 keeps the rows inside a head together.
    var hd = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7 };
    try p(testing.allocator, &hd, 1, 8, 1, 2, 2, 2, false);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 1, 4, 5, 2, 3, 6, 7 }, &hd);
    try p(testing.allocator, &hd, 1, 8, 1, 2, 2, 2, true);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 1, 2, 3, 4, 5, 6, 7 }, &hd);

    // cols>1 moves whole rows.
    var wide = [_]u8{ 10, 11, 20, 21, 30, 31, 40, 41 };
    try p(testing.allocator, &wide, 1, 4, 2, 2, 2, 1, false);
    try testing.expectEqualSlices(u8, &[_]u8{ 10, 11, 30, 31, 20, 21, 40, 41 }, &wide);
    try p(testing.allocator, &wide, 1, 4, 2, 2, 2, 1, true);
    try testing.expectEqualSlices(u8, &[_]u8{ 10, 11, 20, 21, 30, 31, 40, 41 }, &wide);

    // The two directions are not the same permutation wherever k_heads and
    // v_per_k differ, so running the forward one twice scrambles rather than
    // restores. This is the mistake the flag exists to prevent.
    var twice = [_]u8{ 0, 1, 2, 3, 4, 5 };
    try p(testing.allocator, &twice, 1, 6, 1, 2, 3, 1, false);
    try p(testing.allocator, &twice, 1, 6, 1, 2, 3, 1, false);
    try testing.expect(!std.mem.eql(u8, &[_]u8{ 0, 1, 2, 3, 4, 5 }, &twice));

    // v_per_k=1 is the k_heads==v_heads case: nothing to tile, either way.
    var same = [_]u8{ 0, 1, 2, 3 };
    try p(testing.allocator, &same, 1, 4, 1, 4, 1, 1, false);
    try p(testing.allocator, &same, 1, 4, 1, 4, 1, 1, true);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 1, 2, 3 }, &same);

    // A shape that cannot be tiled must not pass through: the rows would reach
    // the output in the order the file claims they are not in.
    var odd = [_]u8{ 1, 2, 3 };
    try testing.expectError(error.VReorderRowsNotDivisible, p(testing.allocator, &odd, 1, 3, 1, 2, 3, 1, false));
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 2, 3 }, &odd);
    var short = [_]u8{ 1, 2, 3, 4 };
    try testing.expectError(error.VReorderSizeMismatch, p(testing.allocator, &short, 1, 6, 1, 2, 3, 1, false));
    try testing.expectError(error.VReorderBadShape, p(testing.allocator, &short, 1, 4, 1, 0, 3, 1, false));
}

test "HfLlm.addMetadata emits the loadable key set" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const io = std.testing.io;
    var t = testing.tmpDir(.{});
    defer t.cleanup();
    const dir = try makeMiniDir(&t, io);
    var m = (try load(io, t.dir, dir, arena.allocator(), null)).?;
    m.dir = "/models/Mini-1B-Chat";

    var meta: std.json.ObjectMap = .empty;
    try addMetadata(m, &meta, m.arch.name, arena.allocator());
    try testing.expectEqualStrings("model", meta.get("general.type").?.string);
    try testing.expectEqualStrings("Mini 1B Chat", meta.get("general.name").?.string);
    try testing.expectEqual(@as(i64, 2), meta.get("qwen2.block_count").?.integer);
    try testing.expectApproxEqAbs(@as(f64, 1000000.0), meta.get("qwen2.rope.freq_base").?.float, 1.0);
    try testing.expect(meta.get("tokenizer.ggml.add_bos_token").?.bool);
    try testing.expectEqualStrings("tpl", meta.get("tokenizer.chat_template").?.string);
    try testing.expectEqual(@as(usize, 6), meta.get("tokenizer.ggml.tokens").?.array.items.len);
    try testing.expectEqual(@as(i64, 3), meta.get("tokenizer.ggml.token_type").?.array.items[4].integer);
    try testing.expectEqual(@as(usize, 2), meta.get("tokenizer.ggml.merges").?.array.items.len);
    try testing.expectEqual(@as(i64, 4), meta.get("tokenizer.ggml.eos_token_id").?.integer);
    // Every arch needs the rope dimension count, not just llama.
    try testing.expectEqual(@as(i64, 2), meta.get("qwen2.rope.dimension_count").?.integer);
    try testing.expectEqual(@as(?std.json.Value, null), meta.get("qwen2.rope.scaling.type"));
}

test "a path that names no directory still gets the checkpoint's name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const io = std.testing.io;
    var t = testing.tmpDir(.{});
    defer t.cleanup();
    _ = try makeMiniDir(&t, io);

    // Converting from inside the checkpoint directory. general.name is what
    // llama.cpp prints, so "model.safetensors" and "." are both wrong answers.
    for ([_][]const u8{ "model.safetensors", "./model.safetensors", "." }) |path| {
        try testing.expectEqual(@as(?[]const u8, null), checkpointDirName(path));
        const m = (try load(io, t.dir, path, arena.allocator(), null)).?;
        const name = checkpointDirName(m.dir).?;
        try testing.expect(std.Io.Dir.path.isAbsolute(m.dir));
        try testing.expect(!std.mem.eql(u8, name, "model.safetensors"));
    }

    // A path that names one is left exactly as typed.
    try testing.expectEqualStrings("Mini-1B-Chat", checkpointDirName("/models/Mini-1B-Chat/").?);
    try testing.expectEqualStrings("Mini-1B-Chat", checkpointDirName("/models/Mini-1B-Chat/model.safetensors").?);
    try testing.expectEqualStrings("Mini-1B-Chat", checkpointDirName("Mini-1B-Chat/config.json").?);
}

test "an explicit head_dim sets the rope dimension count" {
    testing.log_level = .err;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var t = testing.tmpDir(.{});
    defer t.cleanup();
    _ = try makeMiniDir(&t, io);
    // hidden_size / num_attention_heads would be 2; head_dim says otherwise,
    // and llama.cpp then rotates half of each head unless this key says so.
    try writeFile(t.dir, io, "config.json",
        \\{ "model_type": "qwen3", "num_hidden_layers": 2, "hidden_size": 4,
        \\  "intermediate_size": 8, "num_attention_heads": 2, "num_key_value_heads": 1,
        \\  "max_position_embeddings": 64, "rms_norm_eps": 1e-5, "head_dim": 8, "vocab_size": 6 }
    );

    const m = (try load(io, t.dir, "Mini", a, null)).?;
    try testing.expectEqualStrings("qwen3", m.arch.name);
    var meta: std.json.ObjectMap = .empty;
    try addMetadata(m, &meta, m.arch.name, a);
    try testing.expectEqual(@as(i64, 8), meta.get("qwen3.rope.dimension_count").?.integer);
    try testing.expectEqual(@as(i64, 8), meta.get("qwen3.attention.key_length").?.integer);
}

test "rope scaling reaches the metadata, or says why it cannot" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;

    const cases = [_]struct { cfg: []const u8, kind: ?[]const u8, factor: f32, orig: ?i64 }{
        .{
            .cfg =
            \\"rope_scaling": { "type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768 },
            ,
            .kind = "yarn",
            .factor = 4.0,
            .orig = 32768,
        },
        .{
            .cfg =
            \\"rope_scaling": { "rope_type": "linear", "factor": 2.0 },
            ,
            .kind = "linear",
            .factor = 2.0,
            .orig = null,
        },
        // llama3 scaling lives in a generated rope_freqs tensor, which this
        // converter does not write: it must warn, not invent a key.
        .{
            .cfg =
            \\"rope_scaling": { "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
            \\                  "high_freq_factor": 4.0, "original_max_position_embeddings": 8192 },
            ,
            .kind = null,
            .factor = 0,
            .orig = null,
        },
    };

    for (cases) |case| {
        testing.log_level = .err;
        var t = testing.tmpDir(.{});
        defer t.cleanup();
        _ = try makeMiniDir(&t, io);
        const cfg = try a.print(
            \\{{ "model_type": "qwen2", "num_hidden_layers": 2, "hidden_size": 4,
            \\   "intermediate_size": 8, "num_attention_heads": 2, "num_key_value_heads": 1,
            \\   "max_position_embeddings": 131072, "rms_norm_eps": 1e-5, {s} "vocab_size": 6 }}
        , .{case.cfg});
        try writeFile(t.dir, io, "config.json", cfg);

        const m = (try load(io, t.dir, "Mini", a, null)).?;
        var meta: std.json.ObjectMap = .empty;
        try addMetadata(m, &meta, m.arch.name, a);
        if (case.kind) |kind| {
            try testing.expectEqualStrings(kind, meta.get("qwen2.rope.scaling.type").?.string);
            try testing.expectApproxEqAbs(case.factor, @as(f32, @floatCast(meta.get("qwen2.rope.scaling.factor").?.float)), 0.001);
            if (case.orig) |o| {
                try testing.expectEqual(o, meta.get("qwen2.rope.scaling.original_context_length").?.integer);
            } else {
                try testing.expectEqual(@as(?std.json.Value, null), meta.get("qwen2.rope.scaling.original_context_length"));
            }
        } else {
            try testing.expectEqual(@as(?std.json.Value, null), meta.get("qwen2.rope.scaling.type"));
        }
    }
}

test "HfLlm.isIgnoredTensor keeps weights and drops rotary caches" {
    try testing.expect(isIgnoredTensor("model.layers.0.self_attn.rotary_emb.inv_freq"));
    try testing.expect(isIgnoredTensor("rotary_emb.inv_freq"));
    try testing.expect(isIgnoredTensor("h.0.attn.attention.masked_bias"));
    try testing.expect(!isIgnoredTensor("model.layers.0.mlp.gate_proj.weight"));
    try testing.expect(!isIgnoredTensor("model.layers.0.self_attn.q_proj.bias"));
    // The reverse direction keeps every weight but the rope cache.
    try testing.expect(isIgnoredNativeTensor("rope_freqs.weight"));
    try testing.expect(isIgnoredNativeTensor("blk.0.rope_freqs.weight"));
    try testing.expect(!isIgnoredNativeTensor("blk.0.ffn_gate_exps.weight"));
}

test "a BPE vocab keeps added tokens above config.json's vocab_size" {
    testing.log_level = .err;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const io = std.testing.io;
    var t = testing.tmpDir(.{});
    defer t.cleanup();

    // A ChatML pair fine-tuned onto a base whose config.json still says 6:
    // those rows are in token_embd, and a shorter token list fails to load.
    try writeFile(t.dir, io, "config.json", mini_config);
    try writeFile(t.dir, io, "tokenizer.json", "{" ++ mini_pretokenizer ++
        \\  "model": { "type": "BPE", "vocab": { "a": 0, "b": 1, "c": 2, "d": 3, "e": 6 }, "merges": [] },
        \\  "added_tokens": [ { "id": 7, "content": "<|im_start|>", "special": true },
        \\                    { "id": 8, "content": "<|im_end|>", "special": true } ] }
    );

    const m = (try load(io, t.dir, "Mini", arena.allocator(), null)).?;
    try testing.expectEqual(@as(usize, 9), m.tokens.len);
    try testing.expectEqualStrings("e", m.tokens[6]);
    try testing.expectEqualStrings("<|im_end|>", m.tokens[8]);
    try testing.expectEqual(GGUF_TOKEN_TYPE_CONTROL, m.token_types[8]);
    // The hole between the two tables still pads.
    try testing.expectEqualStrings("[PAD5]", m.tokens[5]);
    try testing.expectEqual(GGUF_TOKEN_TYPE_UNUSED, m.token_types[5]);
}

test "an spm vocab takes the added tokens its proto never had" {
    testing.log_level = .err;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var t = testing.tmpDir(.{});
    defer t.cleanup();

    // A ChatML tune of a SentencePiece base: the pair exists only in
    // tokenizer_config, and config.json still names the base model's eos.
    var bytes: [512]u8 = undefined;
    var len: usize = 0;
    spmEntry(&bytes, &len, "<unk>", -1000, 2);
    spmEntry(&bytes, &len, "<s>", -1000, 3);
    spmEntry(&bytes, &len, "</s>", -1000, 3);
    spmEntry(&bytes, &len, "a", 1.5, null);
    spmEntry(&bytes, &len, "b", 1.0, null);
    spmEntry(&bytes, &len, "c", 0.5, null);
    try writeFile(t.dir, io, "tokenizer.model", bytes[0..len]);
    try writeFile(t.dir, io, "config.json",
        \\{ "model_type": "llama", "num_hidden_layers": 2, "hidden_size": 4,
        \\  "intermediate_size": 8, "num_attention_heads": 2, "num_key_value_heads": 1,
        \\  "max_position_embeddings": 64, "rms_norm_eps": 1e-5, "vocab_size": 8,
        \\  "eos_token_id": 2 }
    );
    try writeFile(t.dir, io, "tokenizer_config.json",
        \\{ "eos_token": "<|im_end|>",
        \\  "added_tokens_decoder": {
        \\    "6": { "content": "<|im_start|>", "special": true },
        \\    "7": { "content": "<|im_end|>", "special": true } } }
    );

    const m = (try load(io, t.dir, "Mini", a, null)).?;
    try testing.expect(m.is_spm);
    try testing.expectEqual(@as(usize, 8), m.tokens.len);
    try testing.expectEqualStrings("<|im_end|>", m.tokens[7]);
    try testing.expectEqual(GGUF_TOKEN_TYPE_CONTROL, m.token_types[7]);
    try testing.expectEqual(@as(f32, -1000), m.scores.?[7]);
    // The proto's own pieces keep their types: UNKNOWN is not "special".
    try testing.expectEqual(GGUF_TOKEN_TYPE_UNKNOWN, m.token_types[0]);
    try testing.expectEqual(@as(f32, 1.5), m.scores.?[3]);
    // The name tokenizer_config gives beats the id config.json kept from the base.
    try testing.expectEqual(@as(u32, 7), m.eos_id.?);
}

test "a BPE tune's added tokens are read from the legacy sidecars too" {
    testing.log_level = .err;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;

    var t = testing.tmpDir(.{});
    defer t.cleanup();
    // tokenizer.json lists no added tokens: this tune declares its ChatML pair
    // the older way, and a fast tokenizer reads both files. Left out, the two
    // ids stay [PAD] holes and the template's markers tokenize as text.
    try writeFile(t.dir, io, "config.json", mini_config);
    try writeFile(t.dir, io, "tokenizer.json", "{" ++ mini_pretokenizer ++
        \\  "model": { "type": "BPE",
        \\   "vocab": { "a": 0, "b": 1, "c": 2, "d": 3 },
        \\   "merges": ["a b", "ab c"] } }
    );
    try writeFile(t.dir, io, "added_tokens.json",
        \\{ "<|im_start|>": 4 }
    );
    try writeFile(t.dir, io, "tokenizer_config.json",
        \\{ "added_tokens_decoder": {
        \\    "5": { "content": "<|im_end|>", "special": true } } }
    );

    const m = (try load(io, t.dir, "Mini", a, null)).?;
    try testing.expect(!m.is_spm);
    try testing.expectEqual(@as(usize, 6), m.tokens.len);
    try testing.expectEqualStrings("<|im_start|>", m.tokens[4]);
    try testing.expectEqualStrings("<|im_end|>", m.tokens[5]);
    try testing.expectEqual(GGUF_TOKEN_TYPE_CONTROL, m.token_types[4]);
    try testing.expectEqual(GGUF_TOKEN_TYPE_CONTROL, m.token_types[5]);
}

test "a config naming no context length is refused, not written short" {
    testing.log_level = .err;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;

    // llama.cpp's loader requires <arch>.context_length, so the file this would
    // have written does not load at all.
    {
        var t = testing.tmpDir(.{});
        defer t.cleanup();
        try writeFile(t.dir, io, "config.json",
            \\{ "model_type": "qwen2", "num_hidden_layers": 2, "hidden_size": 4,
            \\  "intermediate_size": 8, "num_attention_heads": 2, "num_key_value_heads": 1,
            \\  "rms_norm_eps": 1e-5, "vocab_size": 6 }
        );
        try writeFile(t.dir, io, "tokenizer.json", mini_tokenizer);
        var why: LoadFailure = undefined;
        try testing.expectEqual(@as(?Model, null), try load(io, t.dir, "Mini", a, &why));
        try testing.expectEqual(LoadFailure.incomplete_config, why);
    }

    // The older spellings count.
    {
        var t = testing.tmpDir(.{});
        defer t.cleanup();
        try writeFile(t.dir, io, "config.json",
            \\{ "model_type": "qwen2", "num_hidden_layers": 2, "hidden_size": 4,
            \\  "intermediate_size": 8, "num_attention_heads": 2, "num_key_value_heads": 1,
            \\  "rms_norm_eps": 1e-5, "vocab_size": 6, "n_ctx": 32 }
        );
        try writeFile(t.dir, io, "tokenizer.json", mini_tokenizer);
        const m = (try load(io, t.dir, "Mini", a, null)).?;
        try testing.expectEqual(@as(u32, 32), m.context_length);
    }
}

// llama.cpp picks this tag by hashing what the real tokenizer makes of a probe
// string, and tokenizes by the tag it finds. Guessing one from model_type gets
// the llama family wrong, and a wrong tag mis-segments every prompt quietly.
test "the pre-tokenizer tag is read off tokenizer.json, not the model type" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;

    {
        var t = testing.tmpDir(.{});
        defer t.cleanup();
        try writeFile(t.dir, io, "config.json", mini_config);
        try writeFile(t.dir, io, "tokenizer.json", mini_tokenizer);
        const m = (try load(io, t.dir, "Mini", a, null)).?;
        try testing.expectEqualStrings("qwen2", m.tokenizer_pre);
    }

    // The same model_type over Llama-3's splitting. A checkpoint is free to
    // ship either, which is why the type cannot decide this.
    {
        var t = testing.tmpDir(.{});
        defer t.cleanup();
        try writeFile(t.dir, io, "config.json", mini_config);
        try writeFile(t.dir, io, "tokenizer.json", "{" ++ mini_pretokenizer_llama3 ++ mini_bpe_body);
        const m = (try load(io, t.dir, "Mini", a, null)).?;
        try testing.expectEqualStrings("llama-bpe", m.tokenizer_pre);
    }

    // Splitting no tag describes. The checkpoint loads, but with nothing to put
    // in the field: naming a family here is what would make the GGUF tokenize
    // differently from the checkpoint, so the decision goes up to the caller.
    {
        var t = testing.tmpDir(.{});
        defer t.cleanup();
        try writeFile(t.dir, io, "config.json", mini_config);
        try writeFile(t.dir, io, "tokenizer.json",
            \\{ "pre_tokenizer": { "type": "Split", "behavior": "Isolated", "pattern": { "Regex": "\\w+" } },
            \\  "model": { "type": "BPE", "vocab": { "a": 0 }, "merges": [] } }
        );
        const m = (try load(io, t.dir, "Mini", a, null)).?;
        try testing.expectEqualStrings("", m.tokenizer_pre);

        // And an empty tag never reaches a file: llama.cpp reads a GGUF with no
        // tag the same way it reads one it cannot parse.
        var meta: std.json.ObjectMap = .empty;
        try testing.expectError(error.UnknownPretokenizer, addMetadata(m, &meta, m.arch.name, a));
    }
}

test "a tag the detector writes always maps back to a regex, and default never does" {
    for (bpe_pretokenizers) |p| {
        for (p.tags) |tag| try testing.expectEqualStrings(p.regex, pretokRegex("gpt2", tag).?);
    }
    // llama.cpp's "no idea" tag, and one of the families behind it: neither
    // names splitting a single-Split tokenizer.json can carry, so neither may
    // resolve to some other family's regex.
    try testing.expectEqual(@as(?[]const u8, null), pretokRegex("gpt2", "default"));
    try testing.expectEqual(@as(?[]const u8, null), pretokRegex("gpt2", "tekken"));
    // An spm vocabulary splits on metaspace, with no regex anywhere.
    try testing.expectEqual(@as(?[]const u8, null), pretokRegex("llama", "qwen2"));
}

test "the sidecar precheck refuses what writeSidecars cannot build, before it builds any" {
    testing.log_level = .err;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var meta: std.json.ObjectMap = .empty;
    try putStr(&meta, a, "general.architecture", "llama");
    try putNum(&meta, a, "llama.block_count", @as(i64, 1));
    try putNum(&meta, a, "llama.embedding_length", @as(i64, 4));
    try putNum(&meta, a, "llama.feed_forward_length", @as(i64, 8));
    try putNum(&meta, a, "llama.attention.head_count", @as(i64, 2));
    try putStr(&meta, a, "tokenizer.ggml.model", "gpt2");
    try putStr(&meta, a, "tokenizer.ggml.pre", "default");
    try putStrArray(&meta, a, "tokenizer.ggml.tokens", &.{ "a", "b" });
    var tt: std.json.Array = .init(a);
    try tt.append(.{ .integer = 1 });
    try tt.append(.{ .integer = 1 });
    try meta.put(a, try a.dupe(u8, "tokenizer.ggml.token_type"), .{ .array = tt });
    var mg: std.json.Array = .init(a);
    try mg.append(.{ .string = "a b" });
    try meta.put(a, try a.dupe(u8, "tokenizer.ggml.merges"), .{ .array = mg });

    try testing.expectError(error.UnknownPretokenizer, sidecarPrecheck(meta));

    // intermediate_size comes off the same prefix and reads 0 the same way as
    // the rest, and a config.json with zero-width MLPs gets as far as a shape
    // mismatch against the weights.
    var no_ffn = try meta.clone(a);
    _ = no_ffn.swapRemove("llama.feed_forward_length");
    try testing.expectError(error.MissingDimensions, sidecarPrecheck(no_ffn));

    // An empty merge table is the one refusal ahead of the tag: a tokenizer.json
    // that merges nothing loads, and then segments every prompt differently.
    var empty_merges = try meta.clone(a);
    try empty_merges.put(a, try a.dupe(u8, "tokenizer.ggml.merges"), .{ .array = .init(a) });
    try testing.expectError(error.MissingMerges, sidecarPrecheck(empty_merges));
    _ = empty_merges.swapRemove("tokenizer.ggml.merges");
    try testing.expectError(error.MissingMerges, sidecarPrecheck(empty_merges));

    // Same answer from the writer, and it stops before config.json: the
    // tokenizer is the last file it writes, so finding out there would leave a
    // directory from_pretrained opens and then fails on.
    var t = testing.tmpDir(.{});
    defer t.cleanup();
    try testing.expectError(error.UnknownPretokenizer, writeSidecars(std.testing.io, t.dir, meta, &.{}, null, true, false, null, a));
    try testing.expectError(error.FileNotFound, t.dir.access(std.testing.io, "config.json", .{}));

    // -u takes the rest of the layout and leaves the vocabulary out. Writing it
    // under another family's splitting is the one thing worse than omitting it.
    try writeSidecars(std.testing.io, t.dir, meta, &.{}, null, true, true, null, a);
    try t.dir.access(std.testing.io, "config.json", .{});
    try testing.expectError(error.FileNotFound, t.dir.access(std.testing.io, "tokenizer.json", .{}));

    // A tag whose regex we hold goes through, vocabulary and all.
    try putStr(&meta, a, "tokenizer.ggml.pre", "qwen2");
    try sidecarPrecheck(meta);
    try writeSidecars(std.testing.io, t.dir, meta, &.{}, null, true, false, null, a);
    try t.dir.access(std.testing.io, "tokenizer.json", .{});

    // Each missing key names itself: the caller prints which one to go fix, and
    // a missing architecture is config.json's problem, not the vocabulary's.
    _ = meta.orderedRemove("tokenizer.ggml.tokens");
    try testing.expectError(error.MissingVocabulary, sidecarPrecheck(meta));
    _ = meta.orderedRemove("general.architecture");
    try testing.expectError(error.MissingArchitecture, sidecarPrecheck(meta));

    // A prefix nothing wrote the dimensions under is the same kind of hole:
    // config.json reads every one of them as 0 and describes no model at all.
    try putStr(&meta, a, "general.architecture", "qwen2");
    try putStrArray(&meta, a, "tokenizer.ggml.tokens", &.{ "a", "b" });
    try testing.expectError(error.MissingDimensions, sidecarPrecheck(meta));

    // A name writeConfig has no transformers shape for would become a
    // model_type and a ForCausalLM class transformers does not have.
    for ([_][]const u8{ "mymodel", "qwen3vlmoe", "qwen3moe" }) |name| {
        try putStr(&meta, a, "general.architecture", name);
        try testing.expectError(error.HfConfigUnsupported, sidecarPrecheck(meta));
    }
}

test "a merges entry in neither form is dropped, not blanked" {
    testing.log_level = .err;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var t = testing.tmpDir(.{});
    defer t.cleanup();
    _ = try makeMiniDir(&t, io);

    // A blank rule in the table is not nothing: llama.cpp's BPE loader reads it
    // as a merge over two empty pieces.
    try writeFile(t.dir, io, "tokenizer.json", "{" ++ mini_pretokenizer ++
        \\  "model": { "type": "BPE", "vocab": { "a": 0, "b": 1, "c": 2, "d": 3 },
        \\   "merges": ["a b", 7, ["ab"], ["ab", "c"]] } }
    );
    const m = (try load(io, t.dir, "Mini", a, null)).?;
    try testing.expectEqual(@as(usize, 2), m.merges.len);
    try testing.expectEqualStrings("a b", m.merges[0]);
    try testing.expectEqualStrings("ab c", m.merges[1]);
}

test "a merges array with nothing usable in it falls back to merges.txt" {
    testing.log_level = .err;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var t = testing.tmpDir(.{});
    defer t.cleanup();
    _ = try makeMiniDir(&t, io);

    // Dropping every entry leaves a gpt2 vocabulary with no merge table, which
    // llama.cpp refuses to load; the file beside it still has the rules.
    try writeFile(t.dir, io, "tokenizer.json", "{" ++ mini_pretokenizer ++
        \\  "model": { "type": "BPE", "vocab": { "a": 0, "b": 1, "c": 2, "d": 3 },
        \\   "merges": [7, ["ab"]] } }
    );
    try writeFile(t.dir, io, "merges.txt", "#version: 0.2\na b\nab c\n");
    const m = (try load(io, t.dir, "Mini", a, null)).?;
    try testing.expectEqual(@as(usize, 2), m.merges.len);
    try testing.expectEqualStrings("a b", m.merges[0]);
}

test "with no merges.txt either, the mergeless BPE checkpoint stops at the metadata" {
    testing.log_level = .err;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var t = testing.tmpDir(.{});
    defer t.cleanup();
    _ = try makeMiniDir(&t, io);

    // Nothing left to read the table off. The checkpoint still loads - its
    // vocabulary and pre-tokenizer are fine - so the refusal has to come from
    // the writer, or the GGUF goes out with no merges key at all.
    try writeFile(t.dir, io, "tokenizer.json", "{" ++ mini_pretokenizer ++ mini_bpe_body);
    const m = (try load(io, t.dir, "Mini", a, null)).?;
    try testing.expectEqual(@as(usize, 0), m.merges.len);
    try testing.expectEqualStrings("qwen2", m.tokenizer_pre);

    var meta: std.json.ObjectMap = .empty;
    try testing.expectError(error.MissingMerges, addMetadata(m, &meta, m.arch.name, a));
}

test "generateMerges survives a piece that ends mid-UTF-8" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // GGUF token strings are raw bytes, so a piece can end on a lead byte that
    // claims more than is there: "ab\xF0" promises four bytes with one left.
    const names = [_][]const u8{ "a", "b", "ab", "ab\xF0" };
    const scores = [_]f64{ -1, -2, -3, -4 };
    const merges = try generateMerges(a, &names, &scores);

    // Only "ab" splits into two in-vocab sides; the truncated piece yields none.
    try testing.expectEqual(@as(usize, 1), merges.items.len);
    try testing.expectEqualStrings("a b", merges.items[0].string);
}

test "dominantTorchDtype weighs bytes, not tensor count" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // Three F32 norms against one big BF16 weight: counting tensors would name
    // float32 and load the model at twice its size.
    var norm_dims = [_]usize{4};
    var weight_dims = [_]usize{ 64, 64 };
    const tensors = [_]types.Tensor{
        .{ .name = "n0", .type = "F32", .dims = &norm_dims, .size = 16, .offset = 0 },
        .{ .name = "n1", .type = "F32", .dims = &norm_dims, .size = 16, .offset = 0 },
        .{ .name = "n2", .type = "F32", .dims = &norm_dims, .size = 16, .offset = 0 },
        .{ .name = "w", .type = "BF16", .dims = &weight_dims, .size = 8192, .offset = 0 },
    };
    try testing.expectEqualStrings("bfloat16", (try dominantTorchDtype(&tensors, a)).?);

    // A cluster type has no torch dtype, and naming one torch cannot allocate
    // is worse than leaving the key out.
    const clustered = [_]types.Tensor{
        .{ .name = "w", .type = "INT4_CONVROT", .dims = &weight_dims, .size = 2048, .offset = 0 },
    };
    try testing.expectEqual(@as(?[]const u8, null), try dominantTorchDtype(&clustered, a));
    try testing.expectEqual(@as(?[]const u8, null), try dominantTorchDtype(&.{}, a));
}

test "writeSidecars produces loadable HF layout from GGUF metadata" {
    testing.log_level = .err;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var meta: std.json.ObjectMap = .empty;
    try putStr(&meta, a, "general.architecture", "qwen2");
    try putNum(&meta, a, "qwen2.block_count", @as(i64, 2));
    try putNum(&meta, a, "qwen2.embedding_length", @as(i64, 4));
    try putNum(&meta, a, "qwen2.feed_forward_length", @as(i64, 8));
    try putNum(&meta, a, "qwen2.attention.head_count", @as(i64, 2));
    try putNum(&meta, a, "qwen2.attention.head_count_kv", @as(i64, 1));
    try putFloat(&meta, a, "qwen2.rope.freq_base", 1000000.0);
    try putStr(&meta, a, "qwen2.rope.scaling.type", "yarn");
    try putFloat(&meta, a, "qwen2.rope.scaling.factor", 4.0);
    try putNum(&meta, a, "qwen2.rope.scaling.original_context_length", @as(i64, 32768));
    try putFloat(&meta, a, "qwen2.attention.layer_norm_rms_epsilon", 0.00001);
    try putStr(&meta, a, "tokenizer.ggml.model", "gpt2");
    try putStr(&meta, a, "tokenizer.ggml.pre", "qwen2");
    // An spm/T5 whitespace flag. The BPE layout below must ignore it.
    try meta.put(a, try a.dupe(u8, "tokenizer.ggml.remove_extra_whitespaces"), .{ .bool = true });
    try putStrArray(&meta, a, "tokenizer.ggml.tokens", &.{ "a", "b", "c", "<|im_start|>" });
    var tt: std.json.Array = .init(a);
    try tt.append(.{ .integer = 1 });
    try tt.append(.{ .integer = 1 });
    try tt.append(.{ .integer = 4 });
    try tt.append(.{ .integer = 3 });
    try meta.put(a, try a.dupe(u8, "tokenizer.ggml.token_type"), .{ .array = tt });
    var mg: std.json.Array = .init(a);
    try mg.append(.{ .string = "a b" });
    try meta.put(a, try a.dupe(u8, "tokenizer.ggml.merges"), .{ .array = mg });
    try putNum(&meta, a, "tokenizer.ggml.eos_token_id", @as(i64, 3));
    try putNum(&meta, a, "tokenizer.ggml.bos_token_id", @as(i64, 3));
    try putFloat(&meta, a, "general.sampling.temp", 0.7);

    var t = testing.tmpDir(.{});
    defer t.cleanup();
    try writeSidecars(std.testing.io, t.dir, meta, &.{}, "bfloat16", false, false, null, a);

    const cfg = try readJson(std.testing.io, t.dir, "config.json", a);
    const c = cfg.value.object;
    try testing.expectEqualStrings("qwen2", getStr(c, "model_type").?);
    // Without this from_pretrained loads the weights at float32.
    try testing.expectEqualStrings("bfloat16", getStr(c, "torch_dtype").?);
    try testing.expectEqual(@as(i64, 2), c.get("num_hidden_layers").?.integer);
    try testing.expect(c.get("tie_word_embeddings").?.bool);
    try testing.expectApproxEqRel(@as(f64, 1000000.0), getF64(c, "rope_theta").?, 0.001);
    // Scaling has to survive the trip back out, or the sidecar quietly shortens
    // the model's usable context.
    const rs = c.get("rope_scaling").?.object;
    try testing.expectEqualStrings("yarn", getStr(rs, "rope_type").?);
    try testing.expectApproxEqRel(@as(f64, 4.0), getF64(rs, "factor").?, 0.001);
    try testing.expectEqual(@as(u64, 32768), getU64(rs, "original_max_position_embeddings").?);
    // The f32 in the GGUF prints as the decimal the config said, not its f64 widening.
    try testing.expectEqual(@as(f64, 1e-5), getF64(c, "rms_norm_eps").?);
    try testing.expectEqual(@as(u64, 3), getU64(c, "eos_token_id").?);
    try testing.expectEqual(@as(u64, 3), getU64(c, "bos_token_id").?);
    // No q_proj.bias among the tensors written.
    try testing.expect(!c.get("attention_bias").?.bool);
    try testing.expectEqualStrings("4.37.0", getStr(c, "transformers_version").?);

    const gc = (try readJson(std.testing.io, t.dir, "generation_config.json", a)).value.object;
    try testing.expectEqual(@as(u64, 3), getU64(gc, "eos_token_id").?);
    try testing.expect(getBool(gc, "do_sample", false));
    try testing.expectEqual(@as(f64, 0.7), getF64(gc, "temperature").?);

    const tj = (try readJson(std.testing.io, t.dir, "tokenizer.json", a)).value.object;
    // Qwen's own tokenizer.json composes to NFC; a ByteLevel that splits again
    // on its own gpt2 regex would tokenize differently from the GGUF.
    try testing.expectEqualStrings("NFC", getStr(tj.get("normalizer").?.object, "type").?);
    try testing.expectEqualStrings("ByteLevel", getStr(tj.get("decoder").?.object, "type").?);
    const pre_toks = tj.get("pre_tokenizer").?.object.get("pretokenizers").?.array.items;
    try testing.expectEqualStrings("Split", getStr(pre_toks[0].object, "type").?);
    try testing.expect(!pre_toks[1].object.get("use_regex").?.bool);
    try testing.expect(!tj.get("post_processor").?.object.get("use_regex").?.bool);
    const model = tj.get("model").?.object;
    try testing.expectEqualStrings("BPE", getStr(model, "type").?);
    // The two added tokens are in added_tokens only.
    try testing.expectEqual(@as(usize, 2), model.get("vocab").?.object.count());
    try testing.expect(!model.get("byte_fallback").?.bool);
    try testing.expect(!model.get("ignore_merges").?.bool);
    try testing.expectEqual(@as(usize, 1), model.get("merges").?.array.items.len);
    const added = tj.get("added_tokens").?.array.items;
    try testing.expectEqual(@as(usize, 2), added.len);
    // USER_DEFINED stays special=false, CONTROL is true.
    try testing.expect(!added[0].object.get("special").?.bool);
    try testing.expect(added[1].object.get("special").?.bool);

    const tcc = (try readJson(std.testing.io, t.dir, "tokenizer_config.json", a)).value.object;
    // This GGUF carries no add_bos_token; for BPE the tokenizer class decides.
    try testing.expect(tcc.get("add_bos_token") == null);
    try testing.expect(tcc.get("unk_token").? == .null);
    try testing.expect(!tcc.get("clean_up_tokenization_spaces").?.bool);
    try testing.expect(tcc.get("model_max_length") == null);

    // Llama-3 skips the merges for whole-vocab words and does not normalize.
    try putStr(&meta, a, "tokenizer.ggml.pre", "llama-bpe");
    try putNum(&meta, a, "qwen2.context_length", @as(i64, 8192));
    var t3 = testing.tmpDir(.{});
    defer t3.cleanup();
    try writeSidecars(std.testing.io, t3.dir, meta, &.{"model.layers.0.self_attn.q_proj.bias"}, null, false, false, null, a);
    const tj3 = (try readJson(std.testing.io, t3.dir, "tokenizer.json", a)).value.object;
    try testing.expect(tj3.get("normalizer").? == .null);
    try testing.expect(tj3.get("model").?.object.get("ignore_merges").?.bool);
    try testing.expect(tj3.get("model").?.object.get("continuing_subword_prefix").? == .null);
    const tcc3 = (try readJson(std.testing.io, t3.dir, "tokenizer_config.json", a)).value.object;
    try testing.expectEqual(@as(u64, 8192), getU64(tcc3, "model_max_length").?);
    const c3 = (try readJson(std.testing.io, t3.dir, "config.json", a)).value.object;
    try testing.expect(c3.get("attention_bias").?.bool);
    const controls = tcc.get("additional_special_tokens").?.array.items;
    try testing.expectEqual(@as(usize, 1), controls.len);
    try testing.expectEqualStrings("<|im_start|>", controls[0].string);

    // The spm path rebuilds a tokenizer.model that our own reader parses.
    var smeta: std.json.ObjectMap = .empty;
    try putStr(&smeta, a, "general.architecture", "llama");
    try putNum(&smeta, a, "llama.block_count", @as(i64, 1));
    try putNum(&smeta, a, "llama.embedding_length", @as(i64, 4));
    try putNum(&smeta, a, "llama.feed_forward_length", @as(i64, 8));
    try putNum(&smeta, a, "llama.attention.head_count", @as(i64, 2));
    try putNum(&smeta, a, "llama.attention.head_count_kv", @as(i64, 1));
    try putStr(&smeta, a, "tokenizer.ggml.model", "llama");
    try putStrArray(&smeta, a, "tokenizer.ggml.tokens", &.{ "<unk>", "<s>", "</s>", "ab" });
    var stt: std.json.Array = .init(a);
    try stt.append(.{ .integer = 3 });
    try stt.append(.{ .integer = 3 });
    try stt.append(.{ .integer = 3 });
    try stt.append(.{ .integer = 1 });
    try smeta.put(a, try a.dupe(u8, "tokenizer.ggml.token_type"), .{ .array = stt });
    var sc: std.json.Array = .init(a);
    try sc.append(.{ .float = -1000 });
    try sc.append(.{ .float = -1000 });
    try sc.append(.{ .float = -1000 });
    try sc.append(.{ .float = -3.5 });
    try smeta.put(a, try a.dupe(u8, "tokenizer.ggml.scores"), .{ .array = sc });
    try putNum(&smeta, a, "tokenizer.ggml.unknown_token_id", @as(i64, 0));
    try putNum(&smeta, a, "tokenizer.ggml.bos_token_id", @as(i64, 1));
    try putNum(&smeta, a, "tokenizer.ggml.eos_token_id", @as(i64, 2));

    var t2 = testing.tmpDir(.{});
    defer t2.cleanup();
    try writeSidecars(std.testing.io, t2.dir, smeta, &.{}, null, false, false, null, a);
    // An SPM vocabulary wants BOS, so the missing key must not read as false.
    const stcc = (try readJson(std.testing.io, t2.dir, "tokenizer_config.json", a)).value.object;
    try testing.expect(stcc.get("add_bos_token").?.bool);

    const spm_bytes = t2.dir.readFileAlloc(std.testing.io, "tokenizer.model", a, .limited(1 << 20)) catch return error.TestUnexpectedResult;
    const pieces = try parseSpmModel(a, spm_bytes);
    try testing.expectEqual(@as(usize, 4), pieces.len);
    try testing.expectEqualStrings("ab", pieces[3].piece);
    try testing.expectApproxEqAbs(@as(f32, -3.5), pieces[3].score, 0.001);
    try testing.expectEqual(@as(u64, 2), pieces[0].ptype); // <unk> maps back to UNKNOWN
    try testing.expectEqual(@as(u64, 1), pieces[3].ptype);
}

test "the spm proto names an unk piece whatever the GGUF called it" {
    testing.log_level = .err;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // llama.cpp's own converter writes the unk piece as token type 2 and need
    // not call it "<unk>". sentencepiece refuses a proto with no UNKNOWN piece,
    // so this is the difference between a loadable directory and "unk is not
    // defined".
    var meta: std.json.ObjectMap = .empty;
    try putStr(&meta, a, "general.architecture", "llama");
    try putNum(&meta, a, "llama.block_count", @as(i64, 1));
    try putNum(&meta, a, "llama.embedding_length", @as(i64, 4));
    try putNum(&meta, a, "llama.feed_forward_length", @as(i64, 8));
    try putNum(&meta, a, "llama.attention.head_count", @as(i64, 2));
    try putNum(&meta, a, "llama.attention.head_count_kv", @as(i64, 1));
    try putStr(&meta, a, "tokenizer.ggml.model", "llama");
    try putStrArray(&meta, a, "tokenizer.ggml.tokens", &.{ "<pad>", "<unknown>", "</s>", "ab" });
    var tt: std.json.Array = .init(a);
    try tt.append(.{ .integer = 3 });
    try tt.append(.{ .integer = 2 });
    try tt.append(.{ .integer = 3 });
    try tt.append(.{ .integer = 1 });
    try meta.put(a, try a.dupe(u8, "tokenizer.ggml.token_type"), .{ .array = tt });
    var sc: std.json.Array = .init(a);
    for (0..4) |_| try sc.append(.{ .float = -1 });
    try meta.put(a, try a.dupe(u8, "tokenizer.ggml.scores"), .{ .array = sc });
    try putNum(&meta, a, "tokenizer.ggml.eos_token_id", @as(i64, 2));

    var t = testing.tmpDir(.{});
    defer t.cleanup();
    try writeSidecars(std.testing.io, t.dir, meta, &.{}, null, false, false, null, a);

    const bytes = try t.dir.readFileAlloc(std.testing.io, "tokenizer.model", a, .limited(1 << 20));
    const pieces = try parseSpmModel(a, bytes);
    var unknowns: usize = 0;
    for (pieces) |pc| {
        if (pc.ptype == 2) unknowns += 1;
    }
    try testing.expectEqual(@as(usize, 1), unknowns);
    try testing.expectEqual(@as(u64, 2), pieces[1].ptype);
    try testing.expectEqual(@as(u64, 3), pieces[0].ptype); // control stays control
}

test "a float too large for the integer arm reads as no number at all" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // Both sides read numbers off a file, and @intFromFloat out of range is
    // illegal behavior, so neither may hand the float to it.
    var meta: std.json.ObjectMap = .empty;
    try meta.put(a, "huge", .{ .float = 1e30 });
    try meta.put(a, "inf", .{ .float = std.math.inf(f64) });
    try meta.put(a, "neg", .{ .float = -1 });
    try meta.put(a, "ok", .{ .float = 7 });

    for ([_][]const u8{ "huge", "inf", "neg" }) |key| {
        try testing.expectEqual(@as(?u64, null), ggU64(meta, key));
        try testing.expectEqual(@as(?u64, null), getU64(meta, key));
    }
    try testing.expectEqual(@as(?u64, 7), ggU64(meta, "ok"));
    try testing.expectEqual(@as(?u64, 7), getU64(meta, "ok"));
}

/// trainer_spec.unk_id (field 2, then 40) out of a ModelProto, for the test below.
fn spmTrainerUnkId(bytes: []const u8) !?u64 {
    var pos: usize = 0;
    while (pos < bytes.len) {
        const tag = try readVarint(bytes, &pos);
        switch (@as(u3, @intCast(tag & 7))) {
            0 => _ = try readVarint(bytes, &pos),
            1 => pos += 8,
            5 => pos += 4,
            2 => {
                const len = try readVarint(bytes, &pos);
                if (len > bytes.len - pos) return error.BadProtobuf;
                const end = pos + len;
                if (tag >> 3 == 2) {
                    const tr = bytes[pos..end];
                    var tp: usize = 0;
                    while (tp < tr.len) {
                        const ttag = try readVarint(tr, &tp);
                        switch (@as(u3, @intCast(ttag & 7))) {
                            0 => {
                                const v = try readVarint(tr, &tp);
                                if (ttag >> 3 == 40) return v;
                            },
                            1 => tp += 8,
                            5 => tp += 4,
                            2 => {
                                const tlen = try readVarint(tr, &tp);
                                if (tlen > tr.len - tp) return error.BadProtobuf;
                                tp += tlen;
                            },
                            else => return error.BadProtobuf,
                        }
                    }
                }
                pos = end;
            },
            else => return error.BadProtobuf,
        }
    }
    return null;
}

test "an unk id past the end of the vocab falls back to the unk piece" {
    testing.log_level = .err;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // Writing the id through would name a piece that is not in the list, and
    // sentencepiece rejects that file after the weights are already on disk.
    var meta: std.json.ObjectMap = .empty;
    try putStr(&meta, a, "general.architecture", "llama");
    try putNum(&meta, a, "llama.block_count", @as(i64, 1));
    try putNum(&meta, a, "llama.embedding_length", @as(i64, 4));
    try putNum(&meta, a, "llama.feed_forward_length", @as(i64, 8));
    try putNum(&meta, a, "llama.attention.head_count", @as(i64, 2));
    try putNum(&meta, a, "llama.attention.head_count_kv", @as(i64, 1));
    try putStr(&meta, a, "tokenizer.ggml.model", "llama");
    try putStrArray(&meta, a, "tokenizer.ggml.tokens", &.{ "<pad>", "<unk>", "</s>", "ab" });
    var tt: std.json.Array = .init(a);
    for ([_]i64{ 3, 3, 3, 1 }) |v| try tt.append(.{ .integer = v });
    try meta.put(a, try a.dupe(u8, "tokenizer.ggml.token_type"), .{ .array = tt });
    var sc: std.json.Array = .init(a);
    for (0..4) |_| try sc.append(.{ .float = -1 });
    try meta.put(a, try a.dupe(u8, "tokenizer.ggml.scores"), .{ .array = sc });
    try putNum(&meta, a, "tokenizer.ggml.eos_token_id", @as(i64, 2));
    try putNum(&meta, a, "tokenizer.ggml.unknown_token_id", @as(i64, 99));

    var t = testing.tmpDir(.{});
    defer t.cleanup();
    try writeSidecars(std.testing.io, t.dir, meta, &.{}, null, false, false, null, a);

    const bytes = try t.dir.readFileAlloc(std.testing.io, "tokenizer.model", a, .limited(1 << 20));
    const pieces = try parseSpmModel(a, bytes);
    try testing.expectEqual(@as(u64, 2), pieces[1].ptype);
    try testing.expectEqual(@as(?u64, 1), try spmTrainerUnkId(bytes));
}

test "a card keeps the entries after one with no huggingface url" {
    testing.log_level = .err;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // A base model named without an "org/" half stored no repo_url, so index 0
    // is a gap rather than the end of the list.
    var meta: std.json.ObjectMap = .empty;
    try putNum(&meta, a, "general.base_model.count", @as(i64, 3));
    try putStr(&meta, a, "general.base_model.0.name", "my-model");
    try putStr(&meta, a, "general.base_model.1.repo_url", "https://example.com/elsewhere");
    try putStr(&meta, a, "general.base_model.2.repo_url", "https://huggingface.co/org/other");

    const card = (try buildCard(meta, a)).?;
    try testing.expect(std.mem.find(u8, card, "- org/other\n") != null);
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, card, "base_model:\n"));
}

test "a card whose list opens on the next line still loads" {
    testing.log_level = .err;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const io = std.testing.io;
    var t = testing.tmpDir(.{});
    defer t.cleanup();
    const dir = try makeMiniDir(&t, io);

    // A bare "[" opens a flow sequence this parser does not follow. Slicing it
    // as an inline list read past the end of the value and killed the run.
    try writeFile(t.dir, io, "README.md",
        \\---
        \\license: mit
        \\tags: [
        \\  text-generation,
        \\]
        \\---
        \\body
    );

    const m = (try load(io, t.dir, dir, arena.allocator(), null)).?;
    try testing.expectEqualStrings("mit", m.license.?);
}

test "writeCard names each frontmatter list once" {
    testing.log_level = .err;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var meta: std.json.ObjectMap = .empty;
    try putStrArray(&meta, a, "general.tags", &.{ "text-generation", "gguf", "qwen" });
    try putStrArray(&meta, a, "general.languages", &.{ "en", "fr" });

    const card = (try buildCard(meta, a)).?;

    // A repeated key is duplicate-key YAML: HF's parser keeps only the last one.
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, card, "tags:\n"));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, card, "language:\n"));
    for ([_][]const u8{ "- text-generation\n", "- gguf\n", "- qwen\n", "- en\n", "- fr\n" }) |item| {
        try testing.expect(std.mem.find(u8, card, item) != null);
    }
}

test "an existing sidecar stops the write until forced" {
    testing.log_level = .err;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var meta: std.json.ObjectMap = .empty;
    try putStr(&meta, a, "general.architecture", "llama");
    try putNum(&meta, a, "llama.block_count", @as(i64, 1));
    try putNum(&meta, a, "llama.embedding_length", @as(i64, 4));
    try putNum(&meta, a, "llama.feed_forward_length", @as(i64, 8));
    try putNum(&meta, a, "llama.attention.head_count", @as(i64, 2));
    try putStr(&meta, a, "tokenizer.ggml.model", "gpt2");
    try putStrArray(&meta, a, "tokenizer.ggml.tokens", &.{ "a", "b" });
    var tt: std.json.Array = .init(a);
    try tt.append(.{ .integer = 1 });
    try tt.append(.{ .integer = 1 });
    try meta.put(a, try a.dupe(u8, "tokenizer.ggml.token_type"), .{ .array = tt });
    try putStr(&meta, a, "tokenizer.ggml.pre", "llama-bpe");
    var mg: std.json.Array = .init(a);
    try mg.append(.{ .string = "a b" });
    try meta.put(a, try a.dupe(u8, "tokenizer.ggml.merges"), .{ .array = mg });

    var t = testing.tmpDir(.{});
    defer t.cleanup();
    const original = "{\"model_type\":\"mine\"}";
    try writeFile(t.dir, std.testing.io, "config.json", original);
    // An spm vocabulary from an earlier run, which this bpe write never
    // overwrites but LlamaTokenizer would read in preference to tokenizer.json.
    try writeFile(t.dir, std.testing.io, "tokenizer.model", "someone else's pieces");

    var clashed = false;
    for (try existingSidecars(std.testing.io, t.dir, meta, null, a)) |n| {
        if (std.mem.eql(u8, n, "tokenizer.model")) clashed = true;
    }
    try testing.expect(clashed);

    try testing.expectError(error.SidecarExists, writeSidecars(std.testing.io, t.dir, meta, &.{}, null, false, false, null, a));
    // The refusal is total: nothing else was created either.
    try testing.expectEqualStrings(original, try t.dir.readFileAlloc(std.testing.io, "config.json", a, .limited(1 << 16)));
    try testing.expectError(error.FileNotFound, t.dir.access(std.testing.io, "tokenizer.json", .{}));

    try writeSidecars(std.testing.io, t.dir, meta, &.{}, null, true, false, null, a);
    const forced = try t.dir.readFileAlloc(std.testing.io, "config.json", a, .limited(1 << 16));
    try testing.expect(!std.mem.eql(u8, original, forced));
    try testing.expectError(error.FileNotFound, t.dir.access(std.testing.io, "tokenizer.model", .{}));
}

test "the weights file counts as a clash before it is written" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;

    var meta: std.json.ObjectMap = .empty;
    try putStr(&meta, a, "tokenizer.ggml.model", "gpt2");

    var t = testing.tmpDir(.{});
    defer t.cleanup();
    // Another model's weights, with none of the sidecars beside them: the
    // conversion is about to truncate them under its own fixed output name.
    try writeFile(t.dir, io, "model.safetensors", "someone else's weights");

    const clashes = try existingSidecars(io, t.dir, meta, "model.safetensors", a);
    try testing.expectEqual(@as(usize, 1), clashes.len);
    try testing.expectEqualStrings("model.safetensors", clashes[0]);

    // A -n output name of its own still clashes with them: from_pretrained
    // opens model.safetensors before it reads the index that would point at
    // other.safetensors, so it would load these weights against our sidecars.
    const renamed = try existingSidecars(io, t.dir, meta, "other.safetensors", a);
    try testing.expectEqual(@as(usize, 1), renamed.len);
    try testing.expectEqualStrings("model.safetensors", renamed[0]);

    // The check writeSidecars runs once the weights are on disk does not,
    // though: by then model.safetensors may be this conversion's own output.
    try testing.expectEqual(@as(usize, 0), (try existingSidecars(io, t.dir, meta, null, a)).len);
}

test "boolean tokenizer flags survive an integer encoding" {
    var meta: std.json.ObjectMap = .empty;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try putNum(&meta, a, "tokenizer.ggml.add_bos_token", @as(i64, 1));
    try putNum(&meta, a, "tokenizer.ggml.add_space_prefix", @as(i64, 0));
    try putNum(&meta, a, "tokenizer.ggml.add_eos_token", @as(i64, 1));
    try testing.expect(boolOf(meta, "tokenizer.ggml.add_bos_token", false));
    try testing.expect(!boolOf(meta, "tokenizer.ggml.add_space_prefix", true));
    try testing.expect(boolOf(meta, "tokenizer.ggml.absent", true));
    try testing.expectEqual(@as(?bool, null), boolIf(meta, "tokenizer.ggml.absent"));

    // And the sidecar gets a JSON bool, which is the only form the way back in
    // reads: a copied 1 would drop the flag on the next pass.
    var t = testing.tmpDir(.{});
    defer t.cleanup();
    try writeTokenizerConfig(std.testing.io, t.dir, meta, a, "qwen2", &.{}, &.{}, null, null, null, null, true);
    const tc = (try readJson(std.testing.io, t.dir, "tokenizer_config.json", a)).value.object;
    try testing.expectEqual(@as(?std.json.Value, .{ .bool = true }), tc.get("add_eos_token"));
    try testing.expectEqual(@as(?std.json.Value, .{ .bool = false }), tc.get("add_space_prefix"));
}

test "idToTitle leaves a digit-led word alone" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectEqualStrings("Qwen2.5 0.5b Instruct", try idToTitle(a, "qwen2.5-0.5b-instruct"));
    try testing.expectEqualStrings("v1.5 Beta", try idToTitle(a, "v1.5-beta"));
    try testing.expectEqualStrings("Llama 3.1 8b", try idToTitle(a, "llama-3.1-8b"));
}

test "parseSpmModel rejects a length that would wrap the offset" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    // Field 1, wire type 2, length 2^64-1: pos + len wraps back under bytes.len.
    const bytes = [_]u8{0x0A} ++ @as([9]u8, @splat(0xFF)) ++ [_]u8{0x01};
    try testing.expectError(error.BadProtobuf, parseSpmModel(arena.allocator(), &bytes));
}
