//! std::regex_search (ECMAScript) for llama-quantize's --tensor-type patterns.
//! Lookaround, backreferences, collating elements and \cX are refused. A Pike
//! VM, so matching is linear in the subject.

const std = @import("std");
const Allocator = std.mem.Allocator;

const Regex = @This();

pub const Error = error{ InvalidPattern, UnsupportedPattern, PatternTooLarge, OutOfMemory };

const Set = std.bit_set.Static(256);

const Inst = union(enum) {
    set: u32,
    split: [2]u32,
    jmp: u32,
    bol,
    eol,
    word_boundary,
    not_word_boundary,
    match,
};

const max_insts = 20_000;

prog: []Inst,
sets: []Set,
clist: []u32,
nlist: []u32,
mark: []u32,
stack: []u32,
scratch: []u32,
gen: u32 = 0,

pub fn compile(gpa: Allocator, pattern: []const u8) Error!Regex {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    var p: Parser = .{ .a = arena_state.allocator(), .src = pattern };
    const root = try p.parseAlt();
    // parseAlt stops only at the end or at a ')' it has no group for.
    if (p.i < pattern.len) return error.InvalidPattern;

    var c: Compiler = .{ .gpa = gpa };
    errdefer c.prog.deinit(gpa);
    errdefer c.sets.deinit(gpa);
    try c.emit(&root);
    _ = try c.add(.match);

    const n = c.prog.items.len;
    const prog = try c.prog.toOwnedSlice(gpa);
    errdefer gpa.free(prog);
    const sets = try c.sets.toOwnedSlice(gpa);
    errdefer gpa.free(sets);
    const scratch = try gpa.alloc(u32, 4 * n);
    @memset(scratch[2 * n .. 3 * n], 0);
    return .{
        .prog = prog,
        .sets = sets,
        .clist = scratch[0..n],
        .nlist = scratch[n .. 2 * n],
        .mark = scratch[2 * n .. 3 * n],
        .stack = scratch[3 * n .. 4 * n],
        .scratch = scratch,
    };
}

pub fn deinit(self: *Regex, gpa: Allocator) void {
    gpa.free(self.prog);
    gpa.free(self.sets);
    gpa.free(self.scratch);
    self.* = undefined;
}

/// Whether the pattern matches anywhere in `s`, as std::regex_search.
pub fn search(self: *Regex, s: []const u8) bool {
    var clen = self.addThread(self.clist, 0, 0, s, self.nextGen());
    var pos: usize = 0;
    while (true) : (pos += 1) {
        const g = self.nextGen();
        var nlen: usize = 0;
        for (self.clist[0..clen]) |pc| switch (self.prog[pc]) {
            .match => return true,
            .set => |si| if (pos < s.len and self.sets[si].isSet(s[pos])) {
                nlen = self.addThreadInto(self.nlist, nlen, pc + 1, pos + 1, s, g);
            },
            else => unreachable,
        };
        if (pos == s.len) return false;
        nlen = self.addThreadInto(self.nlist, nlen, 0, pos + 1, s, g);
        std.mem.swap([]u32, &self.clist, &self.nlist);
        clen = nlen;
    }
}

fn nextGen(self: *Regex) u32 {
    self.gen +%= 1;
    if (self.gen == 0) {
        @memset(self.mark, 0);
        self.gen = 1;
    }
    return self.gen;
}

fn addThread(self: *Regex, list: []u32, pc: u32, pos: usize, s: []const u8, g: u32) usize {
    return self.addThreadInto(list, 0, pc, pos, s, g);
}

/// Follow every non-consuming instruction from `pc` at `pos`, appending the
/// consuming ones (and match) to `list[len..]`.
fn addThreadInto(self: *Regex, list: []u32, len_in: usize, pc0: u32, pos: usize, s: []const u8, g: u32) usize {
    var len = len_in;
    var sp: usize = 0;
    if (self.mark[pc0] == g) return len;
    self.mark[pc0] = g;
    self.stack[sp] = pc0;
    sp += 1;
    while (sp > 0) {
        sp -= 1;
        const pc = self.stack[sp];
        const next: [2]?u32 = switch (self.prog[pc]) {
            .set, .match => {
                list[len] = pc;
                len += 1;
                continue;
            },
            .jmp => |t| .{ t, null },
            .split => |ts| .{ ts[0], ts[1] },
            .bol => .{ if (pos == 0) pc + 1 else null, null },
            .eol => .{ if (pos == s.len) pc + 1 else null, null },
            .word_boundary => .{ if (atWordBoundary(s, pos)) pc + 1 else null, null },
            .not_word_boundary => .{ if (!atWordBoundary(s, pos)) pc + 1 else null, null },
        };
        for (next) |maybe| {
            const t = maybe orelse continue;
            if (self.mark[t] == g) continue;
            self.mark[t] = g;
            self.stack[sp] = t;
            sp += 1;
        }
    }
    return len;
}

fn isWordChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

fn atWordBoundary(s: []const u8, pos: usize) bool {
    const before = pos > 0 and isWordChar(s[pos - 1]);
    const after = pos < s.len and isWordChar(s[pos]);
    return before != after;
}

const Node = union(enum) {
    set: Set,
    concat: []Node,
    alt: []Node,
    repeat: struct { sub: *Node, min: u32, max: ?u32 },
    bol,
    eol,
    word_boundary,
    not_word_boundary,
    empty,
};

const Parser = struct {
    a: Allocator,
    src: []const u8,
    i: usize = 0,

    fn peek(p: *Parser) ?u8 {
        return if (p.i < p.src.len) p.src[p.i] else null;
    }

    fn parseAlt(p: *Parser) Error!Node {
        var branches: std.ArrayList(Node) = .empty;
        try branches.append(p.a, try p.parseConcat());
        while (p.peek() == '|') {
            p.i += 1;
            try branches.append(p.a, try p.parseConcat());
        }
        if (branches.items.len == 1) return branches.items[0];
        return .{ .alt = branches.items };
    }

    fn parseConcat(p: *Parser) Error!Node {
        var items: std.ArrayList(Node) = .empty;
        while (p.peek()) |c| {
            if (c == '|' or c == ')') break;
            try items.append(p.a, try p.parseRepeat());
        }
        return switch (items.items.len) {
            0 => .empty,
            1 => items.items[0],
            else => .{ .concat = items.items },
        };
    }

    fn parseRepeat(p: *Parser) Error!Node {
        var node = try p.parseAtom();
        // std::regex takes a quantifier on a quantifier (a**), so loop.
        while (try p.parseQuantifier()) |q| {
            switch (node) {
                .bol, .eol, .word_boundary, .not_word_boundary => return error.InvalidPattern,
                else => {},
            }
            if (p.peek() == '?') p.i += 1; // lazy: same answer for a yes/no search
            const sub = try p.a.create(Node);
            sub.* = node;
            node = .{ .repeat = .{ .sub = sub, .min = q.min, .max = q.max } };
        }
        return node;
    }

    const Quant = struct { min: u32, max: ?u32 };

    fn parseQuantifier(p: *Parser) Error!?Quant {
        const c = p.peek() orelse return null;
        switch (c) {
            '*' => {
                p.i += 1;
                return .{ .min = 0, .max = null };
            },
            '+' => {
                p.i += 1;
                return .{ .min = 1, .max = null };
            },
            '?' => {
                p.i += 1;
                return .{ .min = 0, .max = 1 };
            },
            '{' => {
                p.i += 1;
                const min = p.parseInt() orelse return error.InvalidPattern;
                var max: ?u32 = min;
                if (p.peek() == ',') {
                    p.i += 1;
                    max = p.parseInt();
                }
                if (p.peek() != '}') return error.InvalidPattern;
                p.i += 1;
                if (max) |m| if (m < min) return error.InvalidPattern;
                return .{ .min = min, .max = max };
            },
            else => return null,
        }
    }

    fn parseInt(p: *Parser) ?u32 {
        const start = p.i;
        while (p.peek()) |c| {
            if (!std.ascii.isDigit(c)) break;
            p.i += 1;
        }
        if (p.i == start) return null;
        return std.fmt.parseInt(u32, p.src[start..p.i], 10) catch std.math.maxInt(u32);
    }

    fn parseAtom(p: *Parser) Error!Node {
        const c = p.src[p.i];
        p.i += 1;
        switch (c) {
            '(' => {
                if (p.peek() == '?') {
                    if (p.i + 1 < p.src.len and p.src[p.i + 1] == ':') {
                        p.i += 2;
                    } else return error.UnsupportedPattern;
                }
                const inner = try p.parseAlt();
                if (p.peek() != ')') return error.InvalidPattern;
                p.i += 1;
                return inner;
            },
            '[' => return .{ .set = try p.parseClass() },
            '.' => {
                var s = Set.full;
                s.unset('\n');
                s.unset('\r');
                return .{ .set = s };
            },
            '^' => return .bol,
            '$' => return .eol,
            '\\' => return p.parseEscape(),
            '*', '+', '?', '{' => return error.InvalidPattern,
            else => return .{ .set = single(c) },
        }
    }

    fn parseEscape(p: *Parser) Error!Node {
        const c = p.peek() orelse return error.InvalidPattern;
        switch (c) {
            'b' => {
                p.i += 1;
                return .word_boundary;
            },
            'B' => {
                p.i += 1;
                return .not_word_boundary;
            },
            '1'...'9' => return error.UnsupportedPattern,
            else => return .{ .set = try p.parseEscapeSet(false) },
        }
    }

    /// A character escape or class escape, with the backslash consumed.
    fn parseEscapeSet(p: *Parser, in_class: bool) Error!Set {
        const c = p.peek() orelse return error.InvalidPattern;
        p.i += 1;
        return switch (c) {
            'd' => classSet(.digit, false),
            'D' => classSet(.digit, true),
            'w' => classSet(.word, false),
            'W' => classSet(.word, true),
            's' => classSet(.space, false),
            'S' => classSet(.space, true),
            't' => single('\t'),
            'n' => single('\n'),
            'r' => single('\r'),
            'f' => single(0x0c),
            'v' => single(0x0b),
            'b' => if (in_class) single(0x08) else unreachable,
            '0' => single(0),
            'x' => single(try p.parseHex(2)),
            'u' => single(try p.parseHex(4)),
            // libstdc++ misreads \ca as a, and no tensor name holds a control character.
            'c' => error.UnsupportedPattern,
            '1'...'9' => error.UnsupportedPattern,
            else => single(c),
        };
    }

    fn parseHex(p: *Parser, n: usize) Error!u8 {
        if (p.i + n > p.src.len) return error.InvalidPattern;
        const v = std.fmt.parseInt(u32, p.src[p.i .. p.i + n], 16) catch return error.InvalidPattern;
        p.i += n;
        // Names are bytes, so a wider code point could never match.
        return std.math.cast(u8, v) orelse error.UnsupportedPattern;
    }

    fn parseClass(p: *Parser) Error!Set {
        var negate = false;
        if (p.peek() == '^') {
            negate = true;
            p.i += 1;
        }
        var set = Set.empty;
        while (true) {
            const c = p.peek() orelse return error.InvalidPattern;
            if (c == ']') {
                p.i += 1;
                break;
            }
            const lo = try p.parseClassAtom();
            if (p.peek() == '-' and p.i + 1 < p.src.len and p.src[p.i + 1] != ']') {
                p.i += 1;
                const hi = try p.parseClassAtom();
                const a = lo.char orelse return error.InvalidPattern;
                const b = hi.char orelse return error.InvalidPattern;
                if (b < a) return error.InvalidPattern;
                set.setRangeValue(.{ .start = a, .end = @as(usize, b) + 1 }, true);
            } else {
                set.setUnion(lo.set);
            }
        }
        if (negate) set.toggleAll();
        return set;
    }

    const ClassAtom = struct { set: Set, char: ?u8 };

    fn parseClassAtom(p: *Parser) Error!ClassAtom {
        const c = p.src[p.i];
        p.i += 1;
        if (c == '\\') {
            const e = p.peek() orelse return error.InvalidPattern;
            const s = try p.parseEscapeSet(true);
            const is_class = switch (e) {
                'd', 'D', 'w', 'W', 's', 'S' => true,
                else => false,
            };
            return .{ .set = s, .char = if (is_class) null else firstSet(s) };
        }
        if (c == '[' and p.peek() != null) {
            switch (p.src[p.i]) {
                ':' => return .{ .set = try p.parseNamedClass(), .char = null },
                '.', '=' => return error.UnsupportedPattern,
                else => {},
            }
        }
        return .{ .set = single(c), .char = c };
    }

    fn parseNamedClass(p: *Parser) Error!Set {
        p.i += 1;
        const end = std.mem.findPos(u8, p.src, p.i, ":]") orelse return error.InvalidPattern;
        const name = p.src[p.i..end];
        p.i = end + 2;
        const pred: *const fn (u8) bool = if (std.mem.eql(u8, name, "alnum")) std.ascii.isAlphanumeric else if (std.mem.eql(u8, name, "alpha")) std.ascii.isAlphabetic else if (std.mem.eql(u8, name, "digit") or std.mem.eql(u8, name, "d")) std.ascii.isDigit else if (std.mem.eql(u8, name, "lower")) std.ascii.isLower else if (std.mem.eql(u8, name, "upper")) std.ascii.isUpper else if (std.mem.eql(u8, name, "space") or std.mem.eql(u8, name, "s")) std.ascii.isWhitespace else if (std.mem.eql(u8, name, "xdigit")) std.ascii.isHex else if (std.mem.eql(u8, name, "punct")) isPunct else if (std.mem.eql(u8, name, "w")) isWordChar else return error.UnsupportedPattern;
        var s = Set.empty;
        for (0..256) |b| if (pred(@intCast(b))) s.set(b);
        return s;
    }
};

fn isPunct(c: u8) bool {
    return std.ascii.isPrint(c) and !std.ascii.isAlphanumeric(c) and c != ' ';
}

fn single(c: u8) Set {
    var s = Set.empty;
    s.set(c);
    return s;
}

fn firstSet(s: Set) ?u8 {
    return if (s.findFirstSet()) |i| @intCast(i) else null;
}

const Class = enum { digit, word, space };

fn classSet(class: Class, negate: bool) Set {
    var s = Set.empty;
    for (0..256) |b| {
        const c: u8 = @intCast(b);
        const in = switch (class) {
            .digit => std.ascii.isDigit(c),
            .word => isWordChar(c),
            .space => c == ' ' or (c >= 0x09 and c <= 0x0d),
        };
        if (in) s.set(b);
    }
    if (negate) s.toggleAll();
    return s;
}

const Compiler = struct {
    gpa: Allocator,
    prog: std.ArrayList(Inst) = .empty,
    sets: std.ArrayList(Set) = .empty,

    fn add(c: *Compiler, inst: Inst) Error!u32 {
        if (c.prog.items.len >= max_insts) return error.PatternTooLarge;
        try c.prog.append(c.gpa, inst);
        return @intCast(c.prog.items.len - 1);
    }

    fn here(c: *Compiler) u32 {
        return @intCast(c.prog.items.len);
    }

    fn emit(c: *Compiler, n: *const Node) Error!void {
        switch (n.*) {
            .set => |s| {
                try c.sets.append(c.gpa, s);
                _ = try c.add(.{ .set = @intCast(c.sets.items.len - 1) });
            },
            .concat => |items| for (items) |*it| try c.emit(it),
            .alt => |branches| {
                var jumps: std.ArrayList(u32) = .empty;
                defer jumps.deinit(c.gpa);
                for (branches, 0..) |*b, k| {
                    if (k + 1 < branches.len) {
                        const sp = try c.add(.{ .split = .{ 0, 0 } });
                        try c.emit(b);
                        try jumps.append(c.gpa, try c.add(.{ .jmp = 0 }));
                        c.prog.items[sp].split = .{ sp + 1, c.here() };
                    } else try c.emit(b);
                }
                for (jumps.items) |j| c.prog.items[j].jmp = c.here();
            },
            .repeat => |r| {
                // Unrolled, so refuse a huge count before building it.
                if (r.min > max_insts) return error.PatternTooLarge;
                for (0..r.min) |_| try c.emit(r.sub);
                if (r.max) |max| {
                    if (max - r.min > max_insts) return error.PatternTooLarge;
                    var skips: std.ArrayList(u32) = .empty;
                    defer skips.deinit(c.gpa);
                    for (0..max - r.min) |_| {
                        try skips.append(c.gpa, try c.add(.{ .split = .{ 0, 0 } }));
                        try c.emit(r.sub);
                    }
                    for (skips.items) |sp| c.prog.items[sp].split = .{ sp + 1, c.here() };
                } else {
                    const sp = try c.add(.{ .split = .{ 0, 0 } });
                    try c.emit(r.sub);
                    _ = try c.add(.{ .jmp = sp });
                    c.prog.items[sp].split = .{ sp + 1, c.here() };
                }
            },
            .bol => _ = try c.add(.bol),
            .eol => _ = try c.add(.eol),
            .word_boundary => _ = try c.add(.word_boundary),
            .not_word_boundary => _ = try c.add(.not_word_boundary),
            .empty => {},
        }
    }
};

test "search agrees with std::regex_search on the fixture corpus" {
    const a = std.testing.allocator;
    const corpus = @embedFile("test_fixtures/regex_cases.txt");
    var lines = std.mem.splitScalar(u8, corpus, '\n');
    var cached_pattern: ?[]const u8 = null;
    var cached: ?Regex = null;
    defer if (cached) |*r| r.deinit(a);
    var failures: usize = 0;
    var checked: usize = 0;
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        var fields = std.mem.splitScalar(u8, line, '\t');
        const pattern = fields.next().?;
        const subject = fields.next().?;
        const want = fields.next().?;
        checked += 1;

        if (std.mem.eql(u8, want, "ERR")) {
            if (compile(a, pattern)) |r| {
                var re = r;
                re.deinit(a);
                std.debug.print("pattern {s}: std::regex refuses it, we compiled it\n", .{pattern});
                failures += 1;
            } else |_| {}
            continue;
        }

        if (cached_pattern == null or !std.mem.eql(u8, cached_pattern.?, pattern)) {
            if (cached) |*r| r.deinit(a);
            cached = null;
            cached = compile(a, pattern) catch |e| {
                std.debug.print("pattern {s}: std::regex compiles it, we failed with {s}\n", .{ pattern, @errorName(e) });
                failures += 1;
                cached_pattern = null;
                continue;
            };
            cached_pattern = pattern;
        }
        const got = cached.?.search(subject);
        if (got != std.mem.eql(u8, want, "1")) {
            std.debug.print("pattern {s} on {s}: std::regex says {s}, we say {}\n", .{ pattern, subject, want, got });
            failures += 1;
        }
    }
    try std.testing.expect(checked > 5000);
    try std.testing.expectEqual(@as(usize, 0), failures);
}

test "lookaround, backreferences and collating elements are refused, not misread" {
    const a = std.testing.allocator;
    for ([_][]const u8{ "(?=a)", "(?!a)b", "(a)\\1", "[[.a.]]", "[[=a=]]", "\\u0100", "\\ca" }) |p| {
        try std.testing.expectError(error.UnsupportedPattern, compile(a, p));
    }
}

test "a counted repeat too large to unroll is refused" {
    try std.testing.expectError(error.PatternTooLarge, compile(std.testing.allocator, "a{100000}"));
    try std.testing.expectError(error.PatternTooLarge, compile(std.testing.allocator, "(a{200}){200}"));
}
