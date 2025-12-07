# AICG_Shadertoy
Scene Uniforms & Shader Integration

This project extends the WebGPU Shadertoy starter by introducing GPU-driven scene data. Instead of hardcoding primitives in WGSL, all objects (spheres, boxes, planes, etc.) are now defined in JavaScript, serialized into a GPU buffer, and consumed by the ray-marching shader.

1. Scene Data Structures (WGSL)

We defined two new GPU-friendly structs:  
Primitive: packed into 64 bytes using four vec4 fields (header, center_param0, params1, params2), ensuring proper alignment for uniform buffers.  
Scene: contains a primitive count, padding, and a fixed-size array of primitives (array<Primitive, MAX_PRIMS>).  
A new uniform binding (@binding(1)) exposes the scene to the shader.

2. JavaScript GPUBuffer Layout

A matching ArrayBuffer layout is constructed manually in JS using Uint32Array/Float32Array.  
We allocate a sceneBuffer of size:

16 bytes (header) + MAX_PRIMS × 64 bytes

This buffer is uploaded to the GPU and bound alongside the existing uniforms.

3. Shader Integration

The ray marcher now reads all primitive data dynamically:  
Replaced all hardcoded objects with scene.primitives[i].  
get_dist() iterates over the scene using scene.count.  
All SDFs pull parameters from the unified Primitive struct (no helper structs needed).  
This completes the low-level scene system: the shader now renders whatever the JavaScript scene buffer provides, enabling future UI-driven scene editing.  


Github pages at : **https://isdriss.github.io/AICG_Shadertoy/**