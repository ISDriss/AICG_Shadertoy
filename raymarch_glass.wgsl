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
fn sd_sphere(p: vec3<f32>, r: f32) -> f32 {
  return length(p) - r;
}

fn sd_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sd_plane(p: vec3<f32>, n: vec3<f32>, h: f32) -> f32 {
  return dot(p, n) + h;
}

fn sd_rounded_box(p: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

// Scene description - returns (distance, material_id)
fn get_dist(p: vec3<f32>) -> vec2<f32> {
  var res = vec2<f32>(MAX_DIST, -1.0);

  // Ground
  let ground_dist = sd_plane(p, vec3<f32>(0.0, 1.0, 0.0), 1.0);
  if ground_dist < res.x {
    res = vec2<f32>(ground_dist, MAT_GROUND);
  }

  // Glass sphere
  let glass_sphere_dist = sd_sphere(p - vec3<f32>(0.0, 0.0, 0.0), 0.8);
  if glass_sphere_dist < res.x {
    res = vec2<f32>(glass_sphere_dist, MAT_GLASS);
  }

  // Metal sphere
  let metal_sphere_dist = sd_sphere(p - vec3<f32>(2.0, -0.2, 0.0), 0.8);
  if metal_sphere_dist < res.x {
    res = vec2<f32>(metal_sphere_dist, MAT_METAL);
  }

  // Water box
  let water_box_dist = sd_rounded_box(p - vec3<f32>(-2.0, -0.5, 0.0), vec3<f32>(0.7, 0.5, 0.7), 0.1);
  if water_box_dist < res.x {
    res = vec2<f32>(water_box_dist, MAT_WATER);
  }

  // Small Diffuse sphere
  let diffuse_sphere_dist = sd_sphere(p - vec3<f32>(0.0, -0.5, 2.0), 0.5);
  if diffuse_sphere_dist < res.x {
    res = vec2<f32>(diffuse_sphere_dist, MAT_DIFFUSE);
  }

  return res;
}

// Ray marching
fn ray_march(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
  var d = 0.0;
  var mat_id = -1.0;

  for (var i = 0; i < MAX_STEPS; i++) {
    let p = ro + rd * d;
    let dist_mat = get_dist(p);
    d += abs(dist_mat.x);
    mat_id = dist_mat.y;

    if abs(dist_mat.x) < SURF_DIST || d > MAX_DIST {
      break;
    }
  }

  return vec2<f32>(d, mat_id);
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
