



//SDF Primitives
// would be nice to generalise and have a single SDF struct & function with a type parameter
struct Primitive {
  // meta
  kind: u32;
  material_id: u32;

  // common spatial anchor
  center: vec3<f32>;  // world-space center / reference point
  param0: f32;        // main scalar parameter

  // extra shape parameters
  params1: vec3<f32>;
  params2: vec3<f32>;
};

// Sphere
struct Sphere {
  center: vec3<f32>,
  radius: f32,
}
fn sd_sphere(p: vec3<f32>, s: Sphere) -> f32 {
  return length(p - s.center) - s.radius;
}

// Plane
struct Plane {
  normal: vec3<f32>,
  height: f32,
}
fn sd_plane(p: vec3<f32>, pl: Plane) -> f32 {
  return dot(p, pl.normal) + pl.height;
}

// Box
struct Box {
  center: vec3<f32>,
  size: vec3<f32>,
}
fn sd_box(p: vec3<f32>, b: Box) -> f32 {
  let q = abs(p - b.center) - b.size;
  return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// Rounded Box
struct RoundedBox {
  center: vec3<f32>,
  size: vec3<f32>,
  radius: f32,
}
fn sd_rounded_box(p: vec3<f32>, rb: RoundedBox) -> f32 {
  let q = abs(p - rb.center) - rb.size;
  return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - rb.radius;
}

// Cylinder
struct Cylinder {
  center: vec3<f32>,
  height: f32,
  radius: f32,
}
fn sd_cylinder(p: vec3<f32>, cy: Cylinder) -> f32 {
  let q = abs(vec2<f32>(length(p.xz - cy.center.xz), p.y - cy.center.y)) - vec2<f32>(cy.radius, cy.height * 0.5);
  return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0)));
}

// Torus
struct Torus {
  center: vec3<f32>,
  major_radius: f32,
  minor_radius: f32,
}
fn sd_torus(p: vec3<f32>, t: Torus) -> f32 {
  let q = vec2<f32>(length(p.xz - t.center.xz) - t.major_radius, p.y - t.center.y);
  return length(q) - t.minor_radius;
}

// Pyramid
struct Pyramid {
  center: vec3<f32>,
  height: f32,
}
fn sd_pyramid(p: vec3<f32>, py: Pyramid) -> f32 {
  // Move to local space
  let p_local = p - py.center;

  // Reflect into +X+Z quadrant
  let px = abs(p_local.x);
  let pz = abs(p_local.z);

  // Compute the slanted face distance
  let q = vec2<f32>(px + pz, p_local.y);
  let d = max(q.x * 0.70710678 + q.y, -p_local.y);

  return d;
}

// Capsule
struct Capsule {
  a: vec3<f32>,
  b: vec3<f32>,
  radius: f32,
}
fn sd_capsule(p: vec3<f32>, c: Capsule) -> f32 {
  let pa = p - c.a;
  let ba = c.b - c.a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - c.radius;
}

// Scene

struct Scene {
  spheres: array<Sphere, 32>,
  boxes: array<Box, 32>,
  planes: array<Plane, 32>,
  rounded_boxes: array<RoundedBox, 32>,
  cylinders: array<Cylinder, 32>,
  tori: array<Torus, 32>,
  pyramids: array<Pyramid, 32>,
  capsules: array<Capsule, 32>,
}

fn create_scene() -> Scene {
  let spheres = array<Sphere, 32>();
  let boxes = array<Box, 32>();
  let planes = array<Plane, 32>();
  let rounded_boxes = array<RoundedBox, 32>();
  let cylinders = array<Cylinder, 32>();
  let tori = array<Torus, 32>();
  let pyramids = array<Pyramid, 32>();
  let capsules = array<Capsule, 32>();

  // Ground plane
  let ground = Plane(normal: vec3<f32>(0.0, 1.0, 0.0), height: 0.0);
  planes[0] = ground;

  return scene;
}