// Ray Marching with Reflection and Refraction
@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = (fragCoord.xy - uniforms.resolution * 0.5) / min(uniforms.resolution.x, uniforms.resolution.y);

  // Pitch Controll
  let pitch = clamp((1.0 - uniforms.mouse.y / uniforms.resolution.y), 0.05, 1.5);
  let yaw = uniforms.time * 0.5;

  // Camera Coords
  let cam_dist = 4.0; // Distance from the target
  let cam_target = vec3<f32>(0.0, 0.0, 0.0);
  let cam_pos = vec3<f32>(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch)) * cam_dist;

  // Camera Matrix
  let cam_forward = normalize(cam_target - cam_pos);
  let cam_right = normalize(cross(cam_forward, vec3<f32>(0.0, 1.0, 0.0)));
  let cam_up = cross(cam_right, cam_forward); // Re-orthogonalized up

  // Ray Direction
  // 1.5 is the "focal length" or distance to the projection plane
  let focal_length = 1.5;
  let rd = normalize(cam_right * uv.x - cam_up * uv.y + cam_forward * focal_length);

  // Render with reflections and refractions
  let color = render(cam_pos, rd, fragCoord.xy);
  return vec4<f32>(gamma_correct(color), 1.0);
}

// Gamma Correction
fn gamma_correct(color: vec3<f32>) -> vec3<f32> {
  return pow(color, vec3<f32>(1.0 / 2.2));
}

// Constants
const MAX_DIST: f32 = 100.0;
const SURF_DIST: f32 = 0.0001;
const MAX_STEPS: i32 = 256;
const MAX_BOUNCES: i32 = 16;

const IOR_AIR: f32 = 1.0;
const IOR_GLASS: f32 = 1.5;
const IOR_WATER: f32 = 1.33;

// Material types
const MAT_GROUND: f32 = 0.0;
const MAT_METAL: f32 = 1.0;
const MAT_GLASS: f32 = 2.0;
const MAT_WATER: f32 = 3.0;
const MAT_DIFFUSE: f32 = 4.0;

fn get_material_color(mat_id: f32, p: vec3<f32>) -> vec3<f32> {
  if mat_id == MAT_GROUND {
    // Checkerboard pattern
    let checker = floor(p.x) + floor(p.z);
    let col1 = vec3<f32>(0.9, 0.9, 0.9);
    let col2 = vec3<f32>(0.2, 0.2, 0.2);
    return select(col2, col1, i32(checker) % 2 == 0);
  } else if mat_id == MAT_METAL {
    return vec3<f32>(0.8, 0.85, 0.9);
  } else if mat_id == MAT_GLASS {
    return vec3<f32>(0.9, 0.9, 1.0);
  } else if mat_id == MAT_WATER {
    return vec3<f32>(0.8, 0.9, 1.0);
  } else if mat_id == MAT_DIFFUSE {
    return vec3<f32>(1.0, 0.5, 0.3);
  }
  return vec3<f32>(0.5, 0.5, 0.5);
}

// SDF Primitives
struct Primitive {
  // header: x = kind, y = material_id, z/w unused
  header: vec4<u32>,

  // center.xyz = center, center_param0.w = param0 (radius, height, etc.)
  center_param0: vec4<f32>,

  // extra shape parameters
  params1: vec4<f32>,
  params2: vec4<f32>,
}

// Sphere
// center = center_param0.xyz
// radius = center_param0.w
fn sd_sphere(p: vec3<f32>, s: Primitive) -> f32 {
  return length(p - s.center_param0.xyz) - s.center_param0.w;
}

// Plane
// normal   = params1.xyz (must be normalized)
// offset h = center_param0.w   (equation: dot(p, n) + h = 0)
fn sd_plane(p: vec3<f32>, pl: Primitive) -> f32 {
  let n = normalize(pl.params1.xyz);
  let h = pl.center_param0.w;
  return dot(p, n) + h;
}

// Box
// center = center_param0.xyz
// size   = params1.xyz (half-size)
fn sd_box(p: vec3<f32>, b: Primitive) -> f32 {
  let c = b.center_param0.xyz;
  let half_size = b.params1.xyz;
  let q = abs(p - c) - half_size;
  return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// Rounded Box
// center = center_param0.xyz
// size   = params1.xyz (half-size)
// radius = center_param0.w
fn sd_rounded_box(p: vec3<f32>, rb: Primitive) -> f32 {
  let c = rb.center_param0.xyz;
  let half_size = rb.params1.xyz;
  let r = rb.center_param0.w;
  let q = abs(p - c) - half_size;
  return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

// Cylinder
// center = center_param0.xyz
// radius = params1.x
// height = center_param0.w
fn sd_cylinder(p: vec3<f32>, cy: Primitive) -> f32 {
  let c = cy.center_param0.xyz;
  let radius = cy.params1.x;
  let height = cy.center_param0.w;
  let q = abs(vec2<f32>(length(p.xz - c.xz), p.y - c.y))
          - vec2<f32>(radius, height * 0.5);
  return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0)));
}

// Torus
// center        = center_param0.xyz
// major_radius  = center_param0.w
// minor_radius  = params1.x
fn sd_torus(p: vec3<f32>, t: Primitive) -> f32 {
  let c = t.center_param0.xyz;
  let R = t.center_param0.w;  // major radius
  let r = t.params1.x;        // minor radius
  let q = vec2<f32>(length(p.xz - c.xz) - R, p.y - c.y);
  return length(q) - r;
}

// Capsule
// a      = params1.xyz
// b      = params2.xyz
// radius = center_param0.w
fn sd_capsule(p: vec3<f32>, c: Primitive) -> f32 {
  let a = c.params1.xyz;
  let b = c.params2.xyz;
  let radius = c.center_param0.w;

  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - radius;
}

// Primitive kinds
const SPHERE      : u32 = 0u;
const PLANE       : u32 = 1u;
const BOX         : u32 = 2u;
const ROUNDED_BOX : u32 = 3u;
const CYLINDER    : u32 = 4u;
const TORUS       : u32 = 5u;
const CAPSULE     : u32 = 6u;

fn sd_primitive(p: vec3<f32>, prim: Primitive) -> f32 {
  switch (prim.header.x) {
    case SPHERE: { // SPHERE
      return sd_sphere(p, prim);
    }
    case PLANE: { // PLANE
      return sd_plane(p, prim);
    }
    case BOX: { // BOX
      return sd_box(p, prim);
    }
    case ROUNDED_BOX: { // ROUNDED_BOX
      return sd_rounded_box(p, prim);
    }
    case CYLINDER: { // CYLINDER
      return sd_cylinder(p, prim);
    }
    case TORUS: { // TORUS
      return sd_torus(p, prim);
    }
    case CAPSULE: { // CAPSULE
      return sd_capsule(p, prim);
    }
    default: {
      return 1e6; // large distance for unknown primitive
    }
  }
}

// Scene
const MAX_PRIMS: u32 = 16u;

struct Scene {
  count: u32,
  _pad: vec3<u32>,
  primitives: array<Primitive, MAX_PRIMS>,
};

@group(0) @binding(1)
var<uniform> scene: Scene;

// Scene description - returns (distance, material_id)
fn get_dist(p: vec3<f32>) -> vec2<f32> {
  var res = vec2<f32>(MAX_DIST, -1.0);

  for (var i: u32 = 0u; i < scene.count; i = i + 1u) {
    let prim = scene.primitives[i];
    let dist = sd_primitive(p, prim);
    if dist < res.x {
      res = vec2<f32>(dist, f32(prim.header.y));
    }
  }

  return res;
}

// Ray marching
fn ray_march(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
  var q = 0.0;
  var mat_id = -1.0;

  for (var i = 0; i < MAX_STEPS; i++) {
    let p = ro + rd * q;
    let dist_mat = get_dist(p);
    q += abs(dist_mat.x);
    mat_id = dist_mat.y;

    if abs(dist_mat.x) < SURF_DIST || q > MAX_DIST {
      break;
    }
  }

  return vec2<f32>(q, mat_id);
}

// Calculate normal
fn get_normal(p: vec3<f32>) -> vec3<f32> {
  let e = vec2<f32>(0.0001, 0.0);
  let n = vec3<f32>(
    get_dist(p + e.xyy).x - get_dist(p - e.xyy).x,
    get_dist(p + e.yxy).x - get_dist(p - e.yxy).x,
    get_dist(p + e.yyx).x - get_dist(p - e.yyx).x
  );
  return normalize(n);
}

// Fresnel equation
fn fresnel(cos_theta: f32, ior_ratio: f32) -> f32 {
  let r0 = pow((1.0 - ior_ratio) / (1.0 + ior_ratio), 2.0);
  return r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);
}

// Calculate refraction ray
fn refract_ray(incident: vec3<f32>, normal: vec3<f32>, ior_ratio: f32) -> vec3<f32> {
  let cos_i = -dot(incident, normal);
  let sin2_t = ior_ratio * ior_ratio * (1.0 - cos_i * cos_i);
  if sin2_t > 1.0 {
    // Total internal reflection (TIR)
    return vec3<f32>(0.0);
  }
  let cos_t = sqrt(1.0 - sin2_t);
  return ior_ratio * incident + (ior_ratio * cos_i - cos_t) * normal;
}

// Sky gradient with sun
fn get_sky(rd: vec3<f32>) -> vec3<f32> {
  let sun_dir = normalize(vec3<f32>(1.0, 0.5, -0.5));
  let sun = pow(max(dot(rd, sun_dir), 0.0), 128.0) * 2.0;
  let sky = mix(vec3<f32>(0.5, 0.7, 0.9), vec3<f32>(0.2, 0.4, 0.7), rd.y * 0.5 + 0.5);
  return sky + vec3<f32>(1.0, 0.9, 0.7) * sun;
}

// Hash function for stochastic elements
fn hash21(seed: vec2<f32>) -> f32 {
  return fract(sin(dot(seed, vec2<f32>(12.9898, 78.233))) * 43758.5453123);
}

// Main rendering function with iterative bounces
fn render(initial_ro: vec3<f32>, initial_rd: vec3<f32>, fragCoord_xy: vec2<f32>) -> vec3<f32> {
  var ro = initial_ro;
  var rd = initial_rd;
  var color = vec3<f32>(0.0);
  var mask = vec3<f32>(1.0); // Accumulates color contribution for each bounce
  var result = vec2<f32>(0.0, -1.0);

  for (var depth = 0; depth < MAX_BOUNCES; depth++) {
    result = ray_march(ro, rd);

    if result.x < MAX_DIST {
      let hit_pos = ro + rd * result.x;
      let normal = get_normal(hit_pos);
      let mat_id = result.y;
      let albedo = get_material_color(mat_id, hit_pos);

      // Lighting for current hit
      let light_pos = vec3<f32>(5.0, 8.0, -5.0);
      let light_dir = normalize(light_pos - hit_pos);
      let diffuse = max(dot(normal, light_dir), 0.0);

      // Shadow for current hit
      let shadow_origin = hit_pos + normal * 0.01; // Increased bias
      let shadow_result = ray_march(shadow_origin, light_dir);
      let shadow = select(0.3, 1.0, shadow_result.x > length(light_pos - shadow_origin));

      if mat_id == MAT_METAL {
        color += mask * albedo * diffuse * shadow * 0.2; // Add some base color even for metal
        rd = reflect(rd, normal);
        ro = hit_pos + normal * 0.01; // Increased bias
        mask *= 0.8; // Attenuate mask for next bounce, slightly less for metal to represent energy loss
      }
      else if mat_id == MAT_GLASS || mat_id == MAT_WATER {
        let entering = dot(rd, normal) < 0.0;
        let n = select(-normal, normal, entering);
        let ior = select(IOR_WATER, IOR_GLASS, mat_id == MAT_GLASS);
        let ior_ratio = select(ior / IOR_AIR, IOR_AIR / ior, entering);

        let cos_theta = min(-dot(rd, n), 1.0);
        let fresnel_val = fresnel(cos_theta, ior_ratio);

        let reflect_dir = reflect(rd, n);
        let reflect_origin = hit_pos + n * 0.01;
        let refract_dir_potential = refract_ray(rd, n, ior_ratio);
        let is_tir = dot(refract_dir_potential, refract_dir_potential) < 0.0001; // Check if it's a zero vector

        // Reflection color contribution from environment
        color += mask * fresnel_val * get_sky(reflect_dir);

        if is_tir { // Total Internal Reflection, only reflection occurs
          ro = reflect_origin;
          rd = reflect_dir;
          mask *= albedo; // Attenuate by material color (for absorption)
        } else { // Both reflection and refraction occur
          let refract_dir = refract_dir_potential;
          let refract_origin = hit_pos - n * 0.01;

          // Transmitted light, weighted by (1 - Fresnel)
          ro = refract_origin; // Continue primary ray as refracted ray
          rd = refract_dir;
          mask *= (1.0 - fresnel_val) * albedo; // Attenuate by transmitted Fresnel AND material color (for absorption)
        }
      }
      else {
        // Diffuse material (terminates bounces)
        let ambient = 0.2;
        color += mask * albedo * (ambient + diffuse * shadow * 0.8);
        mask = vec3<f32>(0.0); // Stop further bounces
        break; // Exit loop for diffuse materials
      }
    }
    else {
      // Hit nothing, add sky contribution and break
      color += mask * get_sky(rd);
      mask = vec3<f32>(0.0); // No further contribution
      break;
    }

    if (dot(mask, mask) < 0.001) { // If mask becomes too small, stop bouncing
      break;
    }
  }

  // If after all bounces, mask still has value (e.g., if a transparent object didn't hit anything in its last bounce)
  if (dot(mask, mask) > 0.001) { // Use a small epsilon to check
    color += mask * get_sky(rd);
  }

  let fog = exp(-result.x * 0.02);
  return mix(get_sky(rd), color, fog);
}
