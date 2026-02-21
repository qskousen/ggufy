const std = @import("std");
const st = @import("Safetensor.zig");
const types = @import("types.zig");
const nw = @import("NullWriter.zig");
const gguf = @import("Gguf.zig");
const clap = @import("clap");
const imagearch = @import("ImageArch.zig");
const conv = @import("Convert.zig");

const Command = enum {
    header,
    tree,
    metadata,
    convert,
    template,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const params = comptime clap.parseParamsComptime(
        \\-h, --help                     Display this help and exit.
        \\-d, --datatype <DATATYPE>      When converting, the target datatype (default fp16).
        \\-f, --filetype <FILETYPE>      When converting, the target filetype: gguf (default), safetensors.
        \\-t, --template <FILENAME>      When converting, specify a template to use.
        \\-o, --output-dir <DIR>         Output directory (default: same as source file).
        \\-n, --output-name <FILENAME>   Output filename without extension (default: source name + datatype).
        \\-j, --threads <INT>            Threads to use when quantizing. Defaults to number of cores - 2.
        \\-a, --aggressiveness <INT>     How aggressively to quantize layers when using sensitivity. 100 is most aggressive, 1 is least.
        \\-x, --skip-sensitivity         Pass this to not use a built-in layer sensitivity file and just blindly quantize to target type.
        \\<COMMAND>    Specify a command: header, tree, metadata, convert, template
        \\<FILENAME>   The file to use for input
    );

    const parsers = comptime .{
        .DATATYPE = clap.parsers.enumeration(types.DataType),
        .FILETYPE = clap.parsers.enumeration(types.FileType),
        .COMMAND = clap.parsers.enumeration(Command),
        .FILENAME = clap.parsers.string,
        .DIR = clap.parsers.string,
        .INT = clap.parsers.int(usize, 10),
    };

    // Initialize our diagnostics, which can be used for reporting useful errors.
    // This is optional. You can also pass `.{}` to `clap.parse` if you don't
    // care about the extra information `Diagnostic` provides.
    var diag = clap.Diagnostic{};
    var res = clap.parse(clap.Help, &params, parsers, .{
        .diagnostic = &diag,
        .allocator = allocator,
    }) catch |err| {
        // Report useful error and exit.
        try diag.reportToFile(.stderr(), err);
        return err;
    };
    defer res.deinit();

    var stderr_buffer: [256]u8 = undefined;
    var err_writer = std.fs.File.stderr().writer(&stderr_buffer);
    const stderr = &err_writer.interface;
    _ = stderr;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    if (res.args.help != 0) {
        try stdout.print("ggufy is a tool for LLM model files, particularly for converting between file types.\n\n", .{});
        try stdout.print("Usage: ggufy <COMMAND> <FILENAME> [options]\n\n", .{});
        try stdout.print("Possible commands:\n", .{});
        try stdout.print("  header         Shows header information for the specified file\n", .{});
        try stdout.print("  tree           Output tensor data in a tree format (SafeTensors only)\n", .{});
        try stdout.print("  metadata       Shows metadata information for the specified file\n", .{});
        try stdout.print("  convert        Convert the specified file into a different format or datatype\n", .{});
        try stdout.print("  template       Creates a json template from the specified file\n\n", .{});
        try stdout.print("Options:\n", .{});
        try stdout.flush();
        return clap.helpToFile(.stderr(), clap.Help, &params, .{});
    }

    const command = res.positionals[0] orelse {
        std.log.err("No command given. Use --help to get more information.", .{});
        return;
    };
    const path = res.positionals[1] orelse {
        std.log.err("No model file specified.", .{});
        return;
    };
    const filetype = res.args.filetype orelse types.FileType.gguf;
    const datatype: ?types.DataType = res.args.datatype;
    const template_path = res.args.template;
    const output_dir = res.args.@"output-dir";
    const output_name = res.args.@"output-name";
    const threads = res.args.threads orelse @max(1, try std.Thread.getCpuCount() - 2);
    const skip_sensitivity = res.args.@"skip-sensitivity" != 0;
    const quantization_aggressiveness: f32 = @floatFromInt(res.args.aggressiveness orelse 50);

    const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });

    var read_buffer: [8]u8 = undefined;
    var reader = file.reader(&read_buffer);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    const file_type = types.FileType.detect_from_file(&reader.interface, allocator) catch types.FileType.safetensors;
    file.close();
    switch (file_type) {
        .safetensors => {
            var f = try st.init(path, allocator, arena_alloc);
            defer f.deinit();

            switch (command) {
                .header => {
                    try f.printHeader(stdout);
                },
                .tree => {
                    try f.printTensorTree(stdout);
                },
                .metadata => {
                    try f.printMetadata(stdout);
                },
                .convert => {
                    try conv.convert(&f, .{
                        .path = path,
                        .filetype = filetype,
                        .datatype = datatype,
                        .template_path = template_path,
                        .output_dir = output_dir,
                        .output_name = output_name,
                        .threads = threads,
                        .skip_sensitivity = skip_sensitivity,
                        .quantization_aggressiveness = quantization_aggressiveness,
                    }, allocator, arena_alloc);
                },
                .template => {
                    return error.Unimplimented;
                },
            }
        },
        .gguf => {
            var f = try gguf.init(path, allocator, arena_alloc, false);
            defer f.deinit();

            std.log.info("GGUF format version {}", .{f.version});
            switch (command) {
                .header => {
                    try f.readGgufTensorHeader();
                },
                .tree => {
                    return error.Unimplemented;
                },
                .metadata => {
                    try f.readGgufMetadata(stdout);
                },
                .convert => {
                    return error.Unimplimented;
                },
                .template => {
                    const out_file = try std.fs.cwd().createFile("template.json", .{ .truncate = true });
                    defer out_file.close();
                    var writer_buffer: [1024]u8 = undefined;
                    var out_writer = out_file.writer(&writer_buffer);
                    var writer = &out_writer.interface;

                    try f.writeTemplate(writer);
                    try writer.flush();
                    std.log.info("Template exported to template.json", .{});
                },
            }
        },
    }
    //try stdout.print("Total bytes used in arena allocator: {}\n", .{arena.queryCapacity()});
    try stdout.flush();
}
