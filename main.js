let fallbackShader = `// Fragment shader - runs once per pixel
@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    // Simple gradient as fallback
    let uv = fragCoord.xy / uniforms.resolution;
    return vec4<f32>(uv, 0.5, 1.0);
}`;

CodeMirror.defineSimpleMode("wgsl", {
        start: [
          { regex: /\b(fn|let|var|const|if|else|for|while|loop|return|break|continue|discard|switch|case|default|struct|type|alias)\b/, token: "keyword" },
          { regex: /\b(bool|i32|u32|f32|f16|vec2|vec3|vec4|mat2x2|mat3x3|mat4x4|array|sampler|texture_2d|texture_3d)\b/, token: "type" },
          { regex: /\b(vec2|vec3|vec4|mat2x2|mat3x3|mat4x4|array)<[^>]+>/, token: "type" },
          { regex: /\b(abs|acos|all|any|asin|atan|atan2|ceil|clamp|cos|cosh|cross|degrees|determinant|distance|dot|exp|exp2|faceforward|floor|fma|fract|frexp|inversesqrt|ldexp|length|log|log2|max|min|mix|modf|normalize|pow|radians|reflect|refract|round|sign|sin|sinh|smoothstep|sqrt|step|tan|tanh|transpose|trunc)\b/, token: "builtin" },
          { regex: /@(vertex|fragment|compute|builtin|location|binding|group|stage|workgroup_size|interpolate|invariant)/, token: "attribute" },
          { regex: /\b\d+\.?\d*[fu]?\b|0x[0-9a-fA-F]+[ul]?/, token: "number" },
          { regex: /\/\/.*/, token: "comment" },
          { regex: /\/\*/, token: "comment", next: "comment" },
          { regex: /[+\-*/%=<>!&|^~?:]/, token: "operator" },
          { regex: /[{}()\[\];,\.]/, token: "punctuation" },
        ],
        comment: [
          { regex: /.*?\*\//, token: "comment", next: "start" },
          { regex: /.*/, token: "comment" },
        ],
      }); // prettier-ignore

const editor = CodeMirror.fromTextArea(document.getElementById("code-editor"), {
  mode: "wgsl",
  theme: "gruvbox-dark-hard",
  lineNumbers: true,
  lineWrapping: true,
  value: fallbackShader,
  tabSize: 2,
  indentUnit: 2,
  viewportMargin: Infinity,
  scrollbarStyle: "native",
});
editor.setValue(fallbackShader);

let device;
let context;
let pipeline;
let uniformBuffer;
let bindGroup;
let startTime = performance.now();
let lastFrameTime = startTime;
let frameCount = 0;
let lastFpsUpdate = startTime;
let mouseX = 0;
let mouseY = 0;
let mouseDown = false;
let CodePanelOpen = true;
let isFullscreen = false;

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const errorMsg = $("error-message");
const compileBtn = $("compile-btn");
const fullscreenBtn = $("fullscreen-btn");
const fullscreenEnterIcon = $("fullscreen-enter-icon");
const fullscreenExitIcon = $("fullscreen-exit-icon");
const canvasContainer = $("canvas-container");
const editorContainer = $("editor-container");

function setText(id, text) {
  const el = $(id);
  if (el) el.textContent = text;
}

const uniforms = {
  resolution: {
    label: "resolution",
    initial: "0 × 0",
    update: (w, h) => `${w} × ${h}`,
  },
  time: {
    label: "time",
    initial: "0.00s",
    update: (t) => `${t.toFixed(2)}s`,
  },
  deltaTime: {
    label: "deltaTime",
    initial: "0.00ms",
    update: (dt) => `${(dt * 1000).toFixed(2)}ms`,
  },
  mousexy: {
    label: "mouse.xy",
    initial: "0, 0",
    update: (x, y) => `${Math.round(x)}, ${Math.round(y)}`,
  },
  mousez: {
    label: "mouse.z",
    initial:
      '<span class="inline-block w-2 h-2 rounded-full" id="mouse-ind" style="background:#928374"></span>',
    update: (down) => {
      $("mouse-ind").style.background = down ? "#b8bb26" : "#928374";
      return null;
    },
  },
  frame: {
    label: "frame",
    initial: "0",
    update: (f) => f.toString(),
  },
};

canvas.addEventListener("mousemove", (e) => {
  const rect = canvas.getBoundingClientRect();
  const dpr = devicePixelRatio || 1;
  [mouseX, mouseY] = [
    (e.clientX - rect.left) * dpr,
    (e.clientY - rect.top) * dpr,
  ];
});
canvas.addEventListener("mousedown", () => (mouseDown = true));
canvas.addEventListener("mouseup", () => (mouseDown = false));
canvas.addEventListener("mouseleave", () => (mouseDown = false));

// Toggle the WGSL code editor panel
$("code-toggle").onclick = () => {
  CodePanelOpen = !CodePanelOpen;
  $("code-content").style.display = CodePanelOpen? "block" : "none";
  $("code-panel").style.flex = CodePanelOpen? "1 1 30%": "0 0 24px";
  $("code-arrow").textContent = CodePanelOpen? "▼ Shader Code": "▶ Shader Code";
  // CodeMirror needs a refresh when its container size changes
  setTimeout(() => editor.refresh(), 0);
};


const vertexShader = `@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(pos[vertexIndex], 0.0, 1.0);
}`;

const uniformsStruct = `struct Uniforms {
  resolution: vec2<f32>, time: f32, deltaTime: f32, mouse: vec4<f32>, frame: u32,
  _padding: u32, _padding2: u32, _padding3: u32,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;`;

// Scene buffer
let sceneBuffer;

const MAX_PRIMS = 16;             // must match WGSL MAX_PRIMS
const PRIMITIVE_SIZE = 64;        // bytes (Primitive = 4 * vec4 = 64)
const SCENE_HEADER_SIZE = 32;     // bytes (count + padding + vec3<u32>)
const SCENE_SIZE = SCENE_HEADER_SIZE + MAX_PRIMS * PRIMITIVE_SIZE;

// Match WGSL constants
const SPHERE      = 0;
const PLANE       = 1;
const BOX         = 2;
const ROUNDED_BOX = 3;
const CYLINDER    = 4;
const TORUS       = 5;
const CAPSULE     = 6;

const MAT_GROUND  = 0;
const MAT_METAL   = 1;
const MAT_GLASS   = 2;
const MAT_WATER   = 3;
const MAT_DIFFUSE = 4;

// This array is what your future UI will modify
let scenePrimitives = [
  // Ground plane
  {
    kind: PLANE,
    materialId: MAT_GROUND,
    center: [0.0, 0.0, 0.0],         // unused for plane
    param0: 1.0,                     // offset h
    params1: [0.0, 1.0, 0.0, 0.0],   // normal
    params2: [0.0, 0.0, 0.0, 0.0],
  },
  // Glass sphere
  {
    kind: SPHERE,
    materialId: MAT_GLASS,
    center: [0.0, 0.0, 0.0],
    param0: 0.8,
    params1: [0.0, 0.0, 0.0, 0.0],
    params2: [0.0, 0.0, 0.0, 0.0],
  },
  // Metal sphere
  {
    kind: SPHERE,
    materialId: MAT_METAL,
    center: [2.0, -0.2, 0.0],
    param0: 0.8,
    params1: [0.0, 0.0, 0.0, 0.0],
    params2: [0.0, 0.0, 0.0, 0.0],
  },
  // Water rounded box
  {
    kind: ROUNDED_BOX,
    materialId: MAT_WATER,
    center: [-2.0, -0.5, 0.0],
    param0: 0.1,                           // corner radius
    params1: [0.7, 0.5, 0.7, 0.0],         // size
    params2: [0.0, 0.0, 0.0, 0.0],
  },
  // Diffuse sphere
  {
    kind: SPHERE,
    materialId: MAT_DIFFUSE,
    center: [0.0, -0.5, 2.0],
    param0: 0.5,
    params1: [0.0, 0.0, 0.0, 0.0],
    params2: [0.0, 0.0, 0.0, 0.0],
  },
];

function buildSceneData(primitives) {
  const buffer = new ArrayBuffer(SCENE_SIZE);
  const u32 = new Uint32Array(buffer);
  const f32 = new Float32Array(buffer);

  const count = Math.min(primitives.length, MAX_PRIMS);

  // Scene header = 32 bytes = 8 * 4 bytes
  // Each Primitive = 64 bytes = 16 * 4 bytes

  u32[0] = count; // count
  u32[1] = 0;     // _pad.x
  u32[2] = 0;     // _pad.y
  u32[3] = 0;     // _pad.z

  function writePrimitive(index, spec) {
    const headerWords = SCENE_HEADER_SIZE / 4;      // 32 / 4 = 8
    const wordsPerPrimitive = PRIMITIVE_SIZE / 4;   // 64 / 4 = 16

    const baseIndex = headerWords + index * wordsPerPrimitive;

    // header: x = kind, y = materialId
    u32[baseIndex + 0] = spec.kind;
    u32[baseIndex + 1] = spec.materialId;
    u32[baseIndex + 2] = 0; // unused
    u32[baseIndex + 3] = 0; // unused

    // center_param0
    f32[baseIndex + 4] = spec.center[0];
    f32[baseIndex + 5] = spec.center[1];
    f32[baseIndex + 6] = spec.center[2];
    f32[baseIndex + 7] = spec.param0;

    // params1
    f32[baseIndex + 8]  = spec.params1[0];
    f32[baseIndex + 9]  = spec.params1[1];
    f32[baseIndex + 10] = spec.params1[2];
    f32[baseIndex + 11] = spec.params1[3];

    // params2
    f32[baseIndex + 12] = spec.params2[0];
    f32[baseIndex + 13] = spec.params2[1];
    f32[baseIndex + 14] = spec.params2[2];
    f32[baseIndex + 15] = spec.params2[3];
  }

  for (let i = 0; i < count; i++) {
    writePrimitive(i, primitives[i]);
  }

  // Unused slots (if any) can stay zeroed

  return buffer;
}

async function initWebGPU() {
  if (!navigator.gpu) return ((errorMsg.textContent = "WebGPU not supported"), false);
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return ((errorMsg.textContent = "No GPU adapter"), false);
  device = await adapter.requestDevice();
  context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format });

  uniformBuffer = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // create and upload initial scene buffer
  const sceneData = buildSceneData(scenePrimitives);
  sceneBuffer = device.createBuffer({
    size: SCENE_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(sceneBuffer, 0, new Uint8Array(sceneData));

  await compileShader(fallbackShader);
  return true;
}

async function compileShader(fragmentCode) {
  const start = performance.now();
  try {
    errorMsg.classList.add("hidden");
    const code = vertexShader + "\n" + uniformsStruct + "\n" + fragmentCode; // prettier-ignore
    const shaderModule = device.createShaderModule({ code });
    const info = await shaderModule.getCompilationInfo();
    const lineOffset = (vertexShader + "\n" + uniformsStruct).split("\n").length; // prettier-ignore
    const errors = info.messages
      .filter((m) => m.type === "error")
      .map((m) => {
        const fragmentLine = m.lineNum - lineOffset;
        return fragmentLine > 0 ? `Line ${fragmentLine}: ${m.message}` : `Line ${m.lineNum}: ${m.message}`;
      })
      .join("\n");
    if (errors)
      return ((errorMsg.textContent = "Shader error:\n" + errors), errorMsg.classList.remove("hidden"));

    const format = navigator.gpu.getPreferredCanvasFormat();
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
      ],
    });
    pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      vertex: { module: shaderModule, entryPoint: "vs_main" },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [{ format }],
      },
      primitive: { topology: "triangle-list" },
    });
    bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: sceneBuffer } },
      ],
    });
    $("compile-time").textContent = `${(performance.now() - start).toFixed(2)}ms`; // prettier-ignore
  } catch (e) {
    errorMsg.textContent = "Compile error: " + e.message;
    errorMsg.classList.remove("hidden");
  }
}

function render() {
  // const sceneData = buildSceneData(scenePrimitives);
  // device.queue.writeBuffer(sceneBuffer, 0, new Uint8Array(sceneData));

  if (!pipeline) return;
  const currentTime = performance.now();
  const deltaTime = (currentTime - lastFrameTime) / 1000;
  const elapsedTime = (currentTime - startTime) / 1000;
  const data = [canvas.width, canvas.height, elapsedTime, deltaTime, mouseX, mouseY, mouseDown ? 1 : 0, 0, frameCount, 0, 0, 0]; // prettier-ignore
  device.queue.writeBuffer(uniformBuffer, 0, new Float32Array(data));

  const val = uniforms.resolution.update(canvas.width, canvas.height);
  if (val) setText("u-resolution", val);

  setText("u-time", uniforms.time.update(elapsedTime));
  setText("u-deltaTime", uniforms.deltaTime.update(deltaTime));
  setText("u-mousexy", uniforms.mousexy.update(mouseX, mouseY));
  setText("u-frame", uniforms.frame.update(frameCount));

  const mouseInd = $("mouse-ind");
  if (mouseInd) {
    uniforms.mousez.update(mouseDown);
  }

  lastFrameTime = currentTime;

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        loadOp: "clear",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        storeOp: "store",
      },
    ],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(3);
  pass.end();
  device.queue.submit([encoder.finish()]);

  if (++frameCount && currentTime - lastFpsUpdate > 100) {
    const fps = Math.round(frameCount / ((currentTime - lastFpsUpdate) / 1_000)); // prettier-ignore
    $("fps").textContent = fps;
    $("frame-time").textContent = `${((currentTime - lastFpsUpdate) / frameCount).toFixed(1)}ms`; // prettier-ignore
    frameCount = 0;
    lastFpsUpdate = currentTime;
  }
  requestAnimationFrame(render);
}

function resizeCanvas() {
  const container = $("canvas-container");
  const dpr = devicePixelRatio || 1;
  canvas.width = container.clientWidth * dpr;
  canvas.height = container.clientHeight * dpr;
  canvas.style.width = container.clientWidth + "px";
  canvas.style.height = container.clientHeight + "px";
}

compileBtn.onclick = () => compileShader(editor.getValue());

function toggleFullscreen() {
  if (
    !document.fullscreenElement &&
    !document.webkitFullscreenElement &&
    !document.mozFullScreenElement &&
    !document.msFullscreenElement
  ) {
    const elem = canvasContainer;
    if (elem.requestFullscreen) {
      elem.requestFullscreen();
    } else if (elem.webkitRequestFullscreen) {
      elem.webkitRequestFullscreen();
    } else if (elem.mozRequestFullScreen) {
      elem.mozRequestFullScreen();
    } else if (elem.msRequestFullscreen) {
      elem.msRequestFullscreen();
    }
  } else {
    if (document.exitFullscreen) {
      document.exitFullscreen();
    } else if (document.webkitExitFullscreen) {
      document.webkitExitFullscreen();
    } else if (document.mozCancelFullScreen) {
      document.mozCancelFullScreen();
    } else if (document.msExitFullscreen) {
      document.msExitFullscreen();
    }
  }
}

function updateFullscreenUI() {
  const fullscreenElement =
    document.fullscreenElement ||
    document.webkitFullscreenElement ||
    document.mozFullScreenElement ||
    document.msFullscreenElement;

  isFullscreen = !!fullscreenElement;
  if (isFullscreen) {
    fullscreenEnterIcon.classList.add("hidden");
    fullscreenExitIcon.classList.remove("hidden");
    editorContainer.style.display = "none";
    canvasContainer.classList.remove("landscape:w-1/2", "portrait:h-1/2");
    canvasContainer.classList.add("w-full", "h-full");
  } else {
    fullscreenEnterIcon.classList.remove("hidden");
    fullscreenExitIcon.classList.add("hidden");
    editorContainer.style.display = "";
    canvasContainer.classList.remove("w-full", "h-full");
    canvasContainer.classList.add("landscape:w-1/2", "portrait:h-1/2");
  }

  setTimeout(resizeCanvas, 50);
}

fullscreenBtn.onclick = toggleFullscreen;
document.addEventListener("fullscreenchange", updateFullscreenUI);
document.addEventListener("webkitfullscreenchange", updateFullscreenUI);
document.addEventListener("mozfullscreenchange", updateFullscreenUI);
document.addEventListener("MSFullscreenChange", updateFullscreenUI);

document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    compileShader(editor.getValue());
  }
  if (e.key === "f" && !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
    if (document.activeElement !== editor.getInputField()) {
      e.preventDefault();
      toggleFullscreen();
    }
  }
});
window.addEventListener("resize", resizeCanvas);

async function loadDefaultShader() {
  try {
    const response = await fetch("./shader.wgsl");
    if (response.ok) {
      fallbackShader = await response.text();
    } else {
      console.warn("shader.wgsl not found, using fallback shader");
    }
  } catch (err) {
    console.warn("Failed to load shader.wgsl, using fallback");
  }
  editor.setValue(fallbackShader);
}

const main = async () => {
  await loadDefaultShader();
  resizeCanvas();
  if (await initWebGPU()) render();
};
main();
