const std = @import("std");
const Allocator = std.mem.Allocator;

const Color = @Vector(3, u8);
const Vec3 = @Vector(3, f64);

const COLOR_BLACK: Color = @splat(0);
const COLOR_WHITE: Color = @splat(255);
const COLOR_BLUE = Color{ 0, 0, 255 };
const COLOR_RED = Color{ 255, 0, 0 };

fn vec3_dot(a: Vec3, b: Vec3) f64 {
    return @reduce(.Add, a * b);
}

fn vec3_len2(a: Vec3) f64 {
    return vec3_dot(a, a);
}

fn vec3_len(a: Vec3) f64 {
    return @sqrt(vec3_len2(a));
}

fn to_vec(v: anytype) Vec3 {
    if (@TypeOf(v) == usize) {
        return @splat(@floatFromInt(v));
    }
    return @splat(@as(f64, v));
}

threadlocal var global_prng = std.rand.DefaultPrng.init(69);

fn vec3_rand(min: f64, max: f64) Vec3 {
    std.debug.assert(max > min);
    const d = max - min;
    return Vec3{
        std.rand.float(global_prng.random(), f64) * d + min,
        std.rand.float(global_prng.random(), f64) * d + min,
        std.rand.float(global_prng.random(), f64) * d + min,
    };
}

test "vec3_rand" {
    var min: f64 = 0;
    var max: f64 = 1;
    var r = vec3_rand(min, max);
    try std.testing.expect(min <= r[0] and r[0] < max);
    try std.testing.expect(min <= r[1] and r[1] < max);
    try std.testing.expect(min <= r[2] and r[2] < max);

    min = 10;
    max = 12;
    r = vec3_rand(min, max);
    try std.testing.expect(min <= r[0] and r[0] < max);
    try std.testing.expect(min <= r[1] and r[1] < max);
    try std.testing.expect(min <= r[2] and r[2] < max);

    // gives different numbers
    const r1 = vec3_rand(min, max);
    const r2 = vec3_rand(min, max);
    try std.testing.expect(r1[0] != r2[0]);
    try std.testing.expect(r1[1] != r2[1]);
    try std.testing.expect(r1[2] != r2[2]);
}

fn vec3_rand_unit() Vec3 {
    while (true) {
        const vec = vec3_rand(-1, 1);
        const len2 = vec3_len2(vec);
        if (-1e160 < len2 and len2 <= 1) {
            return vec / to_vec(@sqrt(len2));
        }
    }
}

test "vec3_rand_unit" {
    const r = vec3_rand_unit();
    try std.testing.expect(-1 <= r[0] and r[0] < 1);
    try std.testing.expect(-1 <= r[1] and r[1] < 1);
    try std.testing.expect(-1 <= r[2] and r[2] < 1);
    try std.testing.expect(vec3_len(r) == 1);
}

fn vec3_rand_hemisphere(normal: Vec3) Vec3 {
    const rand = vec3_rand_unit();
    const sign = vec3_dot(rand, normal);
    if (sign < 0) {
        return -rand;
    }
    return rand;
}

fn Interval(comptime T: type) type {
    return struct {
        min: T,
        max: T,

        fn new(min: T, max: T) @This() {
            return @This(){ .min = min, .max = max };
        }

        fn surrounds(this: *const @This(), n: T) bool {
            return this.min <= n and n <= this.max;
        }

        fn includes(this: *const @This(), n: T) bool {
            return this.min < n and n < this.max;
        }
    };
}

const Hit = struct {
    t: f64 = 0,
    point: Vec3 = @splat(0),
    normal: Vec3 = @splat(0),
    is_front_face: bool = false,

    pub fn format(
        this: @This(),
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: std.io.AnyWriter,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print("Hit{{ t: {d}, point: {d}, normal: {d}, is_front_face: {} }}", .{
            this.t,
            this.point,
            this.normal,
            this.is_front_face,
        });
    }
};

const Sphere = struct {
    center: Vec3,
    radius: f64,

    fn compute_hit(this: @This(), ray: Ray, t: Interval(f64)) ?Hit {
        const oc = this.center - ray.origin;

        const a = vec3_len2(ray.direction);
        // const b = -2. * vec3_dot(ray.direction, oc);
        const h = vec3_dot(ray.direction, oc);
        const c = vec3_len2(oc) - this.radius * this.radius;

        const discriminant = h * h - a * c;
        if (discriminant < 0) {
            return null; // no hit
        }

        const sqrt = @sqrt(discriminant);

        // find closest point in range
        // var root = (-b +- sqrt) / 2 * a;
        var root = (h - sqrt) / a;

        if (!t.includes(root)) {
            root = (h + sqrt) / a; // verify second solution

            if (!t.includes(root)) {
                return null; // no hit in range
            }
        }

        var hit = Hit{};
        hit.t = root;
        hit.point = ray.origin + ray.direction * to_vec(root);

        const outward_normal = (hit.point - this.center) / to_vec(this.radius);
        hit.is_front_face = vec3_dot(ray.direction, outward_normal) < 0;
        hit.normal = if (hit.is_front_face) outward_normal else -outward_normal;

        return hit;
    }

    test "Sphere.compute_hit" {
        const sphere = Sphere{
            .center = Vec3{ 0, 0, 0 },
            .radius = 1,
        };

        const no_hit = sphere.compute_hit(Ray{
            .origin = Vec3{ 0, 1, 0 },
            .direction = Vec3{ 0, 1, 0 },
        }, Interval(f64).new(0, 1));

        std.testing.expect(no_hit == null) catch |err| {
            std.log.err("{s}", .{no_hit.?});
            return err;
        };

        const hit = sphere.compute_hit(Ray{
            .origin = Vec3{ 0, 3, 0 },
            .direction = Vec3{ 0, -1, 0 },
        }, Interval(f64).new(0, 5));

        try std.testing.expect(hit != null);
    }
};

const Hittable = union(enum) {
    sphere: Sphere,

    fn compute_hit(this: @This(), ray: Ray, t: Interval(f64)) ?Hit {
        return switch (this) {
            .sphere => |sphere| sphere.compute_hit(ray, t),
        };
    }

    fn compute_hit_many(hittables: []const @This(), ray: Ray, t: Interval(f64)) ?Hit {
        var result: ?Hit = null;
        var interval = t;

        for (hittables) |hittable| {
            const this_hit = hittable.compute_hit(ray, interval);
            if (this_hit) |hit| {
                result = hit;
                interval.max = hit.t;
            }
        }

        return result;
    }

    test "Hittable.compute_hit" {
        const hittable = Hittable{
            .sphere = Sphere{
                .center = Vec3{ 0, 0, 0 },
                .radius = 1,
            },
        };

        const no_hit = hittable.compute_hit(Ray{
            .origin = Vec3{ 0, 1, 0 },
            .direction = Vec3{ 0, 1, 0 },
        }, Interval(f64).new(0, 1));

        std.testing.expect(no_hit == null) catch |err| {
            std.log.err("{s}", .{no_hit.?});
            return err;
        };

        const hit = hittable.compute_hit(Ray{
            .origin = Vec3{ 0, 3, 0 },
            .direction = Vec3{ 0, -1, 0 },
        }, Interval(f64).new(0, 5));

        try std.testing.expect(hit != null);
    }

    test "Hittable.compute_hit_many" {
        const hittables = [_]Hittable{
            .{ .sphere = .{
                .center = Vec3{ 0, 0, 0 },
                .radius = 1,
            } },
        };

        const no_hit = Hittable.compute_hit_many(&hittables, Ray{
            .origin = Vec3{ 0, 1, 0 },
            .direction = Vec3{ 0, 1, 0 },
        }, Interval(f64).new(0, 1));

        std.testing.expect(no_hit == null) catch |err| {
            std.log.err("{s}", .{no_hit.?});
            return err;
        };

        const hit = Hittable.compute_hit_many(&hittables, Ray{
            .origin = Vec3{ 0, 3, 0 },
            .direction = Vec3{ 0, -1, 0 },
        }, Interval(f64).new(0, 5));

        try std.testing.expect(hit != null);
    }
};

const Ray = struct {
    origin: Vec3,
    direction: Vec3,

    fn color(this: @This(), world: []const Hittable, max_bounces: i32) Vec3 {
        std.debug.assert(max_bounces >= -1);
        if (max_bounces < 0) {
            return to_vec(0);
        }

        const maybe_hit = Hittable.compute_hit_many(world, this, Interval(f64).new(0.00001, 10));
        if (maybe_hit) |hit| {
            const next_ray = Ray{
                .origin = hit.point,
                .direction = vec3_rand_hemisphere(hit.normal),
            };

            return next_ray.color(world, max_bounces - 1) / to_vec(2);
        }

        const white = Vec3{ 1, 1, 1 };
        const blue = Vec3{ 0.5, 0.7, 1.0 };

        const a = (this.direction[1] + 1) / 2;
        return white * to_vec(1 - a) + blue * to_vec(a);
    }
};

const Image = struct {
    data: []Color,
    w: isize,
    h: isize,
    allocator: Allocator,

    fn new(allocator: Allocator, w: isize, h: isize) !@This() {
        return new_with_color(allocator, w, h, COLOR_BLACK);
    }

    fn new_with_color(allocator: Allocator, w: isize, h: isize, color: Color) !@This() {
        std.debug.assert(w > 0);
        std.debug.assert(h > 0);

        const data = try allocator.alloc(Color, @intCast(w * h));
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

fn gamma_correction(color: Vec3) Vec3 {
    return Vec3{
        if (color[0] > 0) @sqrt(color[0]) else 0,
        if (color[1] > 0) @sqrt(color[1]) else 0,
        if (color[2] > 0) @sqrt(color[2]) else 0,
    };
}

const Camera = struct {
    // image output
    aspect_ratio: f64 = 0,
    width: isize,
    height: isize = 0,

    // camera settings
    center: Vec3 = @splat(0),
    focal_length: f64 = 1,
    pix_samples: i32 = 10,
    max_bounces: i32 = 10,
    sampler: *const fn () Vec3 = sample_square,
    gamma_correction: *const fn (color: Vec3) Vec3 = gamma_correction,

    // calculated
    pix00_location: Vec3 = @splat(0),
    pix_delta_u: Vec3 = @splat(0),
    pix_delta_v: Vec3 = @splat(0),

    fn init(this: *@This()) void {
        // either height or aspect_ratio
        std.debug.assert(this.height != 0 or this.aspect_ratio != 0);
        std.debug.assert((this.height > 0 and this.aspect_ratio == 0) or (this.height == 0 and this.aspect_ratio > 0));

        std.debug.assert(this.width > 0);
        std.debug.assert(this.focal_length >= 0);
        std.debug.assert(this.pix_samples >= 0);

        if (this.height == 0) {
            // height from aspect_ratio
            const w: f64 = @floatFromInt(this.width);
            this.height = @intFromFloat(w / this.aspect_ratio);
            this.height = @max(this.height, 1);
        }

        const w: f64 = @floatFromInt(this.width);
        const h: f64 = @floatFromInt(this.height);

        this.aspect_ratio = w / h; // recalculate aspect_ratio to compensate height to int

        // Viewport
        const viewport_height = 2;
        const viewport_width = viewport_height * this.aspect_ratio;

        const viewport_u = Vec3{ viewport_width, 0, 0 };
        const viewport_v = Vec3{ 0, -viewport_height, 0 };

        this.pix_delta_u = viewport_u / to_vec(w);
        this.pix_delta_v = viewport_v / to_vec(h);

        var viewport_upper_left = this.center;
        viewport_upper_left -= Vec3{ 0, 0, this.focal_length };
        viewport_upper_left -= (viewport_u + viewport_v) / to_vec(2);

        this.pix00_location = viewport_upper_left + (this.pix_delta_u + this.pix_delta_v) / to_vec(2);
    }

    fn sample_square() Vec3 {
        return Vec3{ std.rand.float(global_prng.random(), f64) - 0.5, std.rand.float(global_prng.random(), f64), 0 };
    }

    fn render_section(this: @This(), image: *const Image, world: []const Hittable, hrange: Interval(isize)) void {
        std.debug.assert(image.w == this.width);
        std.debug.assert(image.h == this.height);

        const h_min: usize = @intCast(hrange.min);
        const h_max: usize = @intCast(hrange.max);
        const w_usize: usize = @intCast(this.width);

        for (h_min..h_max) |i| {
            for (0..w_usize) |j| {
                const pix_center = this.pix00_location + this.pix_delta_u * to_vec(j) + this.pix_delta_v * to_vec(i);

                var color: Vec3 = @splat(0);

                const samples_f64: f64 = @floatFromInt(this.pix_samples);
                const samples: usize = @intCast(this.pix_samples);

                for (0..samples) |ry| {
                    const ry_f64: f64 = @floatFromInt(ry);

                    for (0..samples) |rx| {
                        const rx_f64: f64 = @floatFromInt(rx);

                        var center = pix_center;
                        center += this.pix_delta_u * to_vec((ry_f64 / samples_f64) - 0.5);
                        center += this.pix_delta_v * to_vec((rx_f64 / samples_f64) - 0.5);

                        const ray = Ray{
                            .origin = this.center,
                            .direction = center - this.center,
                        };

                        color += ray.color(world, 10);
                    }
                }

                color = color / to_vec(samples * samples);
                color = gamma_correction(color);

                image.data[i * w_usize + j] = vec_to_color(color);
            }
        }
    }

    test "init" {
        var camera = Camera{
            .width = 160,
            .aspect_ratio = 16.0 / 9.0,
        };
        camera.init();

        try std.testing.expect(camera.height == 90);
    }

    test "sample_square" {
        const min: f64 = -0.5;
        const max: f64 = 0.5;
        const r = sample_square();
        try std.testing.expect(min <= r[0] and r[0] < max);
        try std.testing.expect(min <= r[1] and r[1] < max);
        try std.testing.expect(r[2] == 0);
    }
};

fn join_threads(threads: []?std.Thread) void {
    for (threads) |maybe_thread| {
        if (maybe_thread) |thread| {
            thread.join();
        }
    }
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    //const aspect_ratio = 16.0 / 9.0;
    //const width: usize = 512;
    //const height: usize = @intFromFloat(@max(@as(f64, width) / aspect_ratio, 1.0));

    //const camera_center = Vec3{ 0, 0, 1 };
    //const focal_length = Vec3{ 0, 0, 2.0 };

    //// Viewport
    //const viewport_height = 2.0;
    //const viewport_width = viewport_height * (@as(f64, width) / height);

    //const viewport_u = Vec3{ viewport_width, 0, 0 };
    //const viewport_v = Vec3{ 0, -viewport_height, 0 };

    //const pix_delta_u = viewport_u / to_vec(width);
    //const pix_delta_v = viewport_v / to_vec(height);

    //const viewport_upper_left = camera_center - focal_length - (viewport_u + viewport_v) / to_vec(2);
    //const pix00_location = viewport_upper_left + (pix_delta_u + pix_delta_v) / to_vec(2);

    var camera = Camera{
        .width = 1200,
        .aspect_ratio = 16.0 / 9.0,
        .focal_length = 2,
        .center = Vec3{ 0, 0, 1 },
        .pix_samples = 30,
    };
    camera.init();

    const image = try Image.new_with_color(alloc, camera.width, camera.height, COLOR_RED);

    const world = [_]Hittable{ .{
        .sphere = .{
            .center = Vec3{ 0, 0, -1 },
            .radius = 0.5,
        },
    }, .{
        .sphere = .{
            .center = Vec3{ -1, 0, -1 },
            .radius = 0.4,
        },
    }, .{
        .sphere = .{
            .center = Vec3{ 0, -100.5, -1 },
            .radius = 100,
        },
    } };

    const thread_count = 19;
    const rows = @divTrunc(camera.height, thread_count);
    std.debug.print("threads: {d}, rows: {d}\n", .{ thread_count, rows });
    std.debug.print("h: {d}, got: {d}\n", .{ camera.height, rows * thread_count });

    var threads = [_]?std.Thread{null} ** thread_count;

    for (0..thread_count) |i| {
        errdefer join_threads(&threads);

        const i_isize: isize = @intCast(i);
        const interval = Interval(isize).new(i_isize * rows, (i_isize + 1) * rows);

        threads[i] = try std.Thread.spawn(.{}, Camera.render_section, .{ camera, &image, &world, interval });
    }
    join_threads(&threads);

    { // write tga
        const output = try std.fs.cwd().createFile("output.tga", .{});
        defer output.close();
        var buffered = std.io.bufferedWriter(output.writer());
        try image.write_tga(buffered.writer().any());
        try buffered.flush();
    }
}

test {
    std.testing.refAllDecls(Sphere);
    std.testing.refAllDecls(Hittable);
    std.testing.refAllDecls(Camera);
}
