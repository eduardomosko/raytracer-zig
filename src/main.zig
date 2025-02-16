const std = @import("std");
const Allocator = std.mem.Allocator;

const Color = @Vector(3, u8);
const Vec3 = @Vector(3, f64);

const COLOR_BLACK: Color = @splat(0);
const COLOR_WHITE: Color = @splat(255);
const COLOR_BLUE = Color{ 0, 0, 255 };
const COLOR_RED = Color{ 255, 0, 0 };

const Image = struct {
    data: []Color,
    w: usize,
    h: usize,
    allocator: Allocator,

    fn create(allocator: Allocator, w: usize, h: usize) !@This() {
        return create_color(allocator, w, h, COLOR_BLACK);
    }

    fn create_color(allocator: Allocator, w: usize, h: usize, color: Color) !@This() {
        const data = try allocator.alloc(Color, w * h);
        @memset(data, color);
        return .{ .data = data, .w = w, .h = h, .allocator = allocator };
    }

    fn write_ppm(this: *const @This(), w: std.io.AnyWriter) !void {
        try w.writeAll("P3\n");
        try w.print("{d} {d}\n", .{ this.w, this.h });
        try w.writeAll("255\n");

        for (this.data) |color| {
            try w.print("{d} {d} {d}\n", .{ color[0], color[1], color[2] });
        }
    }

    fn write_tga(this: *const @This(), w: std.io.AnyWriter) !void {
        std.debug.assert(this.w <= std.math.maxInt(u16));
        std.debug.assert(this.h <= std.math.maxInt(u16));

        const TGAImageDescriptor = enum(u8) {
            DEFAULTS = 0,
            TOP_TO_BOTTOM = (1 << 5),
            RIGHT_TO_LEFT = (1 << 4),
        };

        const TGAHeader = extern struct {
            id_length: u8 = 0,
            color_map_type: u8 = 0,
            image_type: u8 = 0,
            color_map_spec: [5]u8 = [_]u8{0} ** 5,

            // image spec
            x_origin: u16 = 0,
            y_origin: u16 = 0,
            width: u16 = 0,
            height: u16 = 0,
            pix_depth: u8 = 0,
            img_descriptor: TGAImageDescriptor = .DEFAULTS,
        };

        try w.writeStructEndian(TGAHeader{
            .image_type = 2, // uncompressed true-color
            .width = @intCast(this.w),
            .height = @intCast(this.h),
            .pix_depth = 24,
            .img_descriptor = .TOP_TO_BOTTOM,
        }, std.builtin.Endian.little);

        try w.writeAll(std.mem.sliceAsBytes(this.data));
    }
};

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const image = try Image.create_color(alloc, 200, 200, COLOR_RED);

    std.debug.print("{d} x {d}\n", .{ image.w, image.h });

    { // write ppm
        const output = try std.fs.cwd().createFile("output.ppm", .{});
        defer output.close();
        try image.write_ppm(output.writer().any());
    }

    { // write tga
        const output = try std.fs.cwd().createFile("output.tga", .{});
        defer output.close();
        try image.write_tga(output.writer().any());
    }
}
