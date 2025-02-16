const std = @import("std");
const Allocator = std.mem.Allocator;

const Color = @Vector(3, u8);
const Vec3 = @Vector(3, f64);

const COLOR_BLACK: Color = @splat(0);
const COLOR_WHITE: Color = @splat(255);
const COLOR_BLUE = Color{ 0, 0, 255 };
const COLOR_RED = Color{ 255, 0, 0 };

const Ray = struct {
    origin: Vec3,
    direction: Vec3,

    fn color(this: *const @This()) Vec3 {
        const white = Vec3{ 1, 1, 1 };
        const blue = Vec3{ 0.5, 0.7, 1.0 };

        const a = (this.direction[1] + 1) / 2;
        return white * vec3(1 - a) + blue * vec3(a);
    }
};

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

        for (this.data) |color| {
            const data: [3]u8 = std.simd.reverseOrder(color);
            try w.writeAll(&data);
        }
    }
};

fn vec_to_color(color: Vec3) Color {
	const max = 0.9999;
    return Color{
        // clamp prevents overflow
        @intFromFloat(std.math.clamp(color[0], 0, max) * 256),
        @intFromFloat(std.math.clamp(color[1], 0, max) * 256),
        @intFromFloat(std.math.clamp(color[2], 0, max) * 256),
    };
}

fn vec3(v: anytype) Vec3 {
    if (@TypeOf(v) == usize) {
        return @splat(@floatFromInt(v));
    }
    return @splat(@as(f64, v));
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const aspect_ratio = 16.0 / 9.0;
    const width: usize = 512;
    const height: usize = @intFromFloat(@max(@as(f64, width) / aspect_ratio, 1.0));

    const camera_center = Vec3{ 0, 0, 1 };
    const focal_length = Vec3{ 0, 0, 2.0 };

    // Viewport
    const viewport_height = 2.0;
    const viewport_width = viewport_height * (@as(f64, width) / height);

    const viewport_u = Vec3{ viewport_width, 0, 0 };
    const viewport_v = Vec3{ 0, -viewport_height, 0 };

    const pix_delta_u = viewport_u / vec3(width);
    const pix_delta_v = viewport_v / vec3(height);

    const viewport_upper_left = camera_center - focal_length - (viewport_u + viewport_v) / vec3(2);
    const pix00_location = viewport_upper_left + (pix_delta_u + pix_delta_v) / vec3(2);

    const image = try Image.create_color(alloc, width, height, COLOR_RED);

    for (0..image.h) |i| {
        for (0..image.w) |j| {
            const pix_center = pix00_location + pix_delta_u * vec3(j) + pix_delta_v * vec3(i);
            const ray = Ray{
                .origin = camera_center,
                .direction = pix_center - camera_center,
            };

            //const color: Vec3 = @splat(0);
            //const samples = 8.0;
            //for (0..samples) |ry| {
            //}

            image.data[i * width + j] = vec_to_color(ray.color());
        }
    }

    //{ // write ppm
    //    const output = try std.fs.cwd().createFile("output.ppm", .{});
    //    defer output.close();
    //    try image.write_ppm(output.writer().any());
    //}

    { // write tga
        const output = try std.fs.cwd().createFile("output.tga", .{});
        defer output.close();
        var buffered = std.io.bufferedWriter(output.writer());
        try image.write_tga(buffered.writer().any());
        try buffered.flush();
    }
}

test "sanity check" {
    // To iterate over consecutive integers, use the range syntax.
    // Unbounded range is always a compile error.
    var sum3: usize = 0;
    for (0..5) |i| {
        std.debug.print("{d}", .{i});
        sum3 += i;
    }
    try std.testing.expect(sum3 == 10);
}
