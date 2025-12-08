let fallbackShader = `// Fragment shader - runs once per pixel
@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    // Simple gradient as fallback
    let uv = fragCoord.xy / uniforms.resolution;
    return vec4<f32>(uv, 0.5, 1.0);
}`;

// --- CodeMirror setup ------------------------------------------------------

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

// --- Globals ----------------------------------------------------------------

let device;
let context;
let pipeline;
let uniformBuffer;
let bindGroup;
let sceneBuffer;

let startTime = performance.now();
let lastFrameTime = startTime;
let frameCount = 0;
let lastFpsUpdate = startTime;
let mouseX = 0;
let mouseY = 0;
let mouseDown = false;
let codePanelOpen = true;
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

// --- Scene / primitive data model ------------------------------------------

const MAX_PRIMS = 16;             // must match WGSL MAX_PRIMS
const PRIMITIVE_SIZE = 64;        // bytes (Primitive = 4 * vec4 = 64)
const SCENE_HEADER_SIZE = 32;     // bytes (count + padding + vec3<u32>)
const SCENE_SIZE = SCENE_HEADER_SIZE + MAX_PRIMS * PRIMITIVE_SIZE;

// WGSL kind IDs
const SPHERE      = 0;
const PLANE       = 1;
const BOX         = 2;
const ROUNDED_BOX = 3;
const CYLINDER    = 4;
const TORUS       = 5;
const CAPSULE     = 6;

// Material IDs
const MAT_GROUND  = 0;
const MAT_METAL   = 1;
const MAT_GLASS   = 2;
const MAT_WATER   = 3;
const MAT_DIFFUSE = 4;

const PRIM_TYPE_LABELS = {
  [PLANE]: "Plane",
  [SPHERE]: "Sphere",
  [BOX]: "Box",
  [ROUNDED_BOX]: "Rounded Box",
  [CYLINDER]: "Cylinder",
  [TORUS]: "Torus",
  [CAPSULE]: "Capsule",
};

const MATERIAL_LABELS = {
  [MAT_GROUND]:  "Ground",
  [MAT_METAL]:   "Metal",
  [MAT_GLASS]:   "Glass",
  [MAT_WATER]:   "Water",
  [MAT_DIFFUSE]: "Diffuse",
};

// This array is what the UI edits + what we upload to the GPU
let scenePrimitives = [
  // Ground plane
  {
    kind: PLANE,
    materialId: MAT_GROUND,
    center: [0.0, 0.0, 0.0],         // unused for plane, kept for consistency
    param0: 1.0,                     // offset h
    params1: [0.0, 1.0, 0.0, 0.0],   // normal
    params2: [0.0, 0.0, 0.0, 0.0],
  },
  // Metal sphere
  {
    kind: SPHERE,
    materialId: MAT_METAL,
    center: [0.0, 0.0, 0.0],
    param0: 0.8,
    params1: [0.0, 0.0, 0.0, 0.0],
    params2: [0.0, 0.0, 0.0, 0.0],
  },
];

function buildSceneData(primitives) {
  const buffer = new ArrayBuffer(SCENE_SIZE);
  const u32 = new Uint32Array(buffer);
  const f32 = new Float32Array(buffer);

  const count = Math.min(primitives.length, MAX_PRIMS);

  // Scene header (32 bytes, 8 * 4 bytes)
  u32[0] = count; // count
  u32[1] = 0;     // _pad.x
  u32[2] = 0;     // _pad.y
  u32[3] = 0;     // _pad.z
  // u32[4..7] unused, kept zero

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

  return buffer;
}

function updateSceneGPU() {
  if (!device || !sceneBuffer) return;
  const sceneData = buildSceneData(scenePrimitives);
  device.queue.writeBuffer(sceneBuffer, 0, new Uint8Array(sceneData));
}

// --- Scene editor UI -------------------------------------------------------

function createLabeledNumber(parent, label, value, min, max, step, onChange) {
  const row = document.createElement("div");
  row.className = "flex items-center justify-between gap-2 mb-1";

  const labelEl = document.createElement("span");
  labelEl.textContent = label;
  labelEl.className = "text-xs";
  row.appendChild(labelEl);

  const input = document.createElement("input");
  input.type = "number";
  input.value = value;
  input.min = min;
  input.max = max;
  input.step = step;
  input.className = "w-20 bg-gray-900 border border-gray-700 text-xs px-1 py-0.5 rounded";
  input.oninput = () => {
    const v = parseFloat(input.value);
    if (!Number.isNaN(v)) onChange(v);
  };
  row.appendChild(input);

  parent.appendChild(row);
}

function createVec3Controls(parent, label, vec, range, step, onChange) {
  const container = document.createElement("div");
  container.className = "mb-2";

  const title = document.createElement("div");
  title.textContent = label;
  title.className = "text-xs mb-1";
  container.appendChild(title);

  const row = document.createElement("div");
  row.className = "flex gap-2";

  ["X", "Y", "Z"].forEach((axis, idx) => {
    const wrap = document.createElement("div");
    wrap.className = "flex flex-col flex-1";

    const lbl = document.createElement("span");
    lbl.textContent = axis;
    lbl.className = "text-[10px] mb-0.5 opacity-70";
    wrap.appendChild(lbl);

    const input = document.createElement("input");
    input.type = "number";
    input.value = vec[idx];
    input.step = step;
    input.min = range[0];
    input.max = range[1];
    input.className = "w-full bg-gray-900 border border-gray-700 text-xs px-1 py-0.5 rounded";
    input.oninput = () => {
      const copy = [...vec];
      const v = parseFloat(input.value);
      if (!Number.isNaN(v)) {
        copy[idx] = v;
        onChange(copy);
      }
    };
    wrap.appendChild(input);

    row.appendChild(wrap);
  });

  container.appendChild(row);
  parent.appendChild(container);
}

function createMaterialSelect(parent, current, onChange) {
  const row = document.createElement("div");
  row.className = "flex items-center justify-between gap-2 mb-1";

  const labelEl = document.createElement("span");
  labelEl.textContent = "Material";
  labelEl.className = "text-xs";
  row.appendChild(labelEl);

  const select = document.createElement("select");
  select.className = "flex-1 bg-gray-900 border border-gray-700 text-xs px-1 py-0.5 rounded";
  Object.entries(MATERIAL_LABELS).forEach(([id, label]) => {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = label;
    if (parseInt(id, 10) === current) opt.selected = true;
    select.appendChild(opt);
  });
  select.onchange = () => onChange(parseInt(select.value, 10));

  row.appendChild(select);
  parent.appendChild(row);
}

function buildPrimitiveControls(body, prim, index) {
  // Position (center)
  createVec3Controls(
    body,
    "Position",
    prim.center,
    [-5, 5],
    0.1,
    (v) => {
      scenePrimitives[index].center = v;
      updateSceneGPU();
    },
  );

  // Shape-specific controls
  switch (prim.kind) {
    case SPHERE: {
      createLabeledNumber(
        body,
        "Radius",
        prim.param0,
        0.05,
        5.0,
        0.05,
        (v) => {
          scenePrimitives[index].param0 = v;
          updateSceneGPU();
        },
      );
      break;
    }
    case PLANE: {
      createVec3Controls(
        body,
        "Normal",
        prim.params1.slice(0, 3),
        [-1, 1],
        0.1,
        (v) => {
          scenePrimitives[index].params1 = [...v, prim.params1[3]];
          updateSceneGPU();
        },
      );
      createLabeledNumber(
        body,
        "Offset",
        prim.param0,
        -5.0,
        5.0,
        0.1,
        (v) => {
          scenePrimitives[index].param0 = v;
          updateSceneGPU();
        },
      );
      break;
    }
    case BOX: {
      createVec3Controls(
        body,
        "Half-size",
        prim.params1.slice(0, 3),
        [0.05, 5],
        0.05,
        (v) => {
          scenePrimitives[index].params1 = [...v, prim.params1[3]];
          updateSceneGPU();
        },
      );
      break;
    }
    case ROUNDED_BOX: {
      createVec3Controls(
        body,
        "Half-size",
        prim.params1.slice(0, 3),
        [0.05, 5],
        0.05,
        (v) => {
          scenePrimitives[index].params1 = [...v, prim.params1[3]];
          updateSceneGPU();
        },
      );
      createLabeledNumber(
        body,
        "Corner radius",
        prim.param0,
        0.0,
        2.0,
        0.02,
        (v) => {
          scenePrimitives[index].param0 = v;
          updateSceneGPU();
        },
      );
      break;
    }
    case CYLINDER: {
      createLabeledNumber(
        body,
        "Radius",
        prim.params1[0],
        0.05,
        5.0,
        0.05,
        (v) => {
          scenePrimitives[index].params1[0] = v;
          updateSceneGPU();
        },
      );
      createLabeledNumber(
        body,
        "Height",
        prim.param0,
        0.05,
        10.0,
        0.05,
        (v) => {
          scenePrimitives[index].param0 = v;
          updateSceneGPU();
        },
      );
      break;
    }
    case TORUS: {
      createLabeledNumber(
        body,
        "Major radius",
        prim.param0,
        0.1,
        5.0,
        0.05,
        (v) => {
          scenePrimitives[index].param0 = v;
          updateSceneGPU();
        },
      );
      createLabeledNumber(
        body,
        "Minor radius",
        prim.params1[0],
        0.05,
        2.0,
        0.02,
        (v) => {
          scenePrimitives[index].params1[0] = v;
          updateSceneGPU();
        },
      );
      break;
    }
    case CAPSULE: {
      createVec3Controls(
        body,
        "Point A",
        prim.params1.slice(0, 3),
        [-5, 5],
        0.1,
        (v) => {
          scenePrimitives[index].params1 = [...v, prim.params1[3]];
          updateSceneGPU();
        },
      );
      createVec3Controls(
        body,
        "Point B",
        prim.params2.slice(0, 3),
        [-5, 5],
        0.1,
        (v) => {
          scenePrimitives[index].params2 = [...v, prim.params2[3]];
          updateSceneGPU();
        },
      );
      createLabeledNumber(
        body,
        "Radius",
        prim.param0,
        0.05,
        2.0,
        0.02,
        (v) => {
          scenePrimitives[index].param0 = v;
          updateSceneGPU();
        },
      );
      break;
    }
    default:
      // Fallback: just expose param0
      createLabeledNumber(
        body,
        "Param0",
        prim.param0,
        -10.0,
        10.0,
        0.1,
        (v) => {
          scenePrimitives[index].param0 = v;
          updateSceneGPU();
        },
      );
  }

  // Material select (common)
  createMaterialSelect(body, prim.materialId, (matId) => {
    scenePrimitives[index].materialId = matId;
    updateSceneGPU();
  });
}

function buildSceneEditorUI() {
  const list = $("scene-list");
  if (!list) return;
  list.innerHTML = "";

  scenePrimitives.forEach((prim, index) => {
    const card = document.createElement("div");
    card.className = "border border-gray-700 rounded-lg bg-gray-900/60 overflow-hidden";

    // Header
    const header = document.createElement("div");
    header.className = "flex items-center justify-between px-3 py-1.5 text-xs cursor-pointer select-none";
    header.style.borderBottom = "1px solid #3c3836";

    const left = document.createElement("div");
    left.className = "flex flex-col";
    const title = document.createElement("span");
    title.textContent = `${PRIM_TYPE_LABELS[prim.kind] ?? "Primitive"} #${index}`;
    const subtitle = document.createElement("span");
    subtitle.className = "opacity-60 text-[10px]";
    subtitle.textContent = MATERIAL_LABELS[prim.materialId] ?? "Unknown material";
    left.appendChild(title);
    left.appendChild(subtitle);

    const right = document.createElement("div");
    right.className = "flex items-center gap-2";
    const arrow = document.createElement("span");
    arrow.textContent = "▼";
    arrow.className = "text-xs";
    const removeBtn = document.createElement("button");
    removeBtn.textContent = "×";
    removeBtn.className = "w-5 h-5 text-xs rounded-full flex items-center justify-center";
    removeBtn.style.background = "#cc241d";
    removeBtn.style.color = "#fbf1c7";
    removeBtn.title = "Remove object";

    right.appendChild(arrow);
    right.appendChild(removeBtn);

    header.appendChild(left);
    header.appendChild(right);

    // Body
    const body = document.createElement("div");
    body.className = "px-3 py-2 text-xs space-y-2";

    buildPrimitiveControls(body, prim, index);

    // Collapsing
    let open = true;
    header.onclick = (e) => {
      // avoid toggling when clicking the remove button
      if (e.target === removeBtn) return;
      open = !open;
      body.style.display = open ? "block" : "none";
      arrow.textContent = open ? "▼" : "▶";
    };

    // Remove primitive
    removeBtn.onclick = (e) => {
      e.stopPropagation();
      scenePrimitives.splice(index, 1);
      updateSceneGPU();
      buildSceneEditorUI();
    };

    card.appendChild(header);
    card.appendChild(body);
    list.appendChild(card);
  });
}

function addDefaultSphere() {
  if (scenePrimitives.length >= MAX_PRIMS) return;
  scenePrimitives.push({
    kind: SPHERE,
    materialId: MAT_DIFFUSE,
    center: [0.0, 0.0, 0.0],
    param0: 0.6,
    params1: [0.0, 0.0, 0.0, 0.0],
    params2: [0.0, 0.0, 0.0, 0.0],
  });
  updateSceneGPU();
  buildSceneEditorUI();
}

// --- Canvas --------------------------------------------------

// Toggle shader code panel
$("code-toggle").onclick = () => {
  codePanelOpen = !codePanelOpen;
  const codePanel = $("code-panel");
  const codeContent = $("code-content");
  const codeArrow = $("code-arrow");

  if (codePanelOpen) {
    codePanel.style.height = "260px";
    codeContent.style.display = "block";
    codeArrow.textContent = "▼ Shader Code";
  } else {
    codePanel.style.height = "24px";
    codeContent.style.display = "none";
    codeArrow.textContent = "▶ Shader Code";
  }
  setTimeout(() => editor.refresh(), 0);
};

$("add-primitive-btn").onclick = addDefaultSphere;

// --- WebGPU setup -----------------------------------------------------------

const vertexShader = `@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  return vec4<f32>(pos[vertexIndex], 0.0, 1.0);
}`;

const uniformsStruct = `struct Uniforms {
  resolution: vec2<f32>,
  frame: u32,
  _padding: u32,
  camPos: vec4<f32>,
  camDir: vec4<f32>,
  camUp: vec4<f32>,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;`;

async function initWebGPU() {
  if (!navigator.gpu) {
    errorMsg.textContent = "WebGPU not supported";
    errorMsg.classList.remove("hidden");
    return false;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    errorMsg.textContent = "No GPU adapter";
    errorMsg.classList.remove("hidden");
    return false;
  }
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
    if (errors) {
      errorMsg.textContent = "Shader error:\n" + errors;
      errorMsg.classList.remove("hidden");
      return;
    }

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
    $("compile-time").textContent = `${(performance.now() - start).toFixed(2)}ms`;
  } catch (e) {
    errorMsg.textContent = "Compile error: " + e.message;
    errorMsg.classList.remove("hidden");
  }
}

// --- Camera setup -----------------------------------------------------------

let camPos = { x: 0, y: 2, z: 5 };
let camDir = { x: 0, y: -0.3, z: -1 };
let camUp = { x: 0, y: 1, z: 0 };

let camTarget = { x: 0, y: 0, z: 0 };
let camDist   = 4.0;
let camYaw    = 0.0;
let camPitch  = 0.5; // radians

function updateCamera() {
  const cp = computeCamPosFromOrbit();
  camPos = cp; // {x,y,z}
  const dir = {
    x: camTarget.x - camPos.x,
    y: camTarget.y - camPos.y,
    z: camTarget.z - camPos.z,
  };
  const len = Math.hypot(dir.x, dir.y, dir.z);
  camDir = { x: dir.x / len, y: dir.y / len, z: dir.z / len };

  camUp = { x: 0, y: 1, z: 0 };
}

function computeCamPosFromOrbit() {
  const cp = {};
  const cosPitch = Math.cos(camPitch);
  cp.x = camTarget.x + camDist * Math.sin(camYaw) * cosPitch;
  cp.y = camTarget.y + camDist * Math.sin(camPitch);
  cp.z = camTarget.z + camDist * Math.cos(camYaw) * cosPitch;
  return cp;
}

const PITCH_MIN = -Math.PI / 2 + 0.01;
const PITCH_MAX =  Math.PI / 2 - 0.01;

let isDragging = false;
let dragMode = null; // "orbit" | "pan" | "zoom"
let lastX = 0;
let lastY = 0;

canvas.addEventListener("mousedown", (e) => {
  if (e.button === 1 || (e.button === 0 && e.altKey)) { // middle, or alt+left fallback
    isDragging = true;
    lastX = e.clientX;
    lastY = e.clientY;

    if (e.shiftKey) {
      dragMode = "pan";
    } else if (e.ctrlKey) {
      dragMode = "zoom";
    } else {
      dragMode = "orbit"; // plain MMB drag
    }

    e.preventDefault();
  }
});

window.addEventListener("mouseup", () => {
  isDragging = false;
  dragMode = null;
});

window.addEventListener("mousemove", (e) => {
  if (!isDragging || !dragMode) return;

  const dx = e.clientX - lastX;
  const dy = e.clientY - lastY;
  lastX = e.clientX;
  lastY = e.clientY;

  const ROT_SPEED = 0.005;
  const PAN_SPEED = 0.0015 * camDist;
  const ZOOM_SPEED = 0.01 * camDist;

  if (dragMode === "orbit") {
    camYaw   -= dx * ROT_SPEED;
    camPitch += dy * ROT_SPEED;
    camPitch = Math.min(Math.max(camPitch, PITCH_MIN), PITCH_MAX);
  } else if (dragMode === "pan") {
    // move target in camera's right/up plane
    const right = {
      x: camDir.z,
      y: 0,
      z: -camDir.x,
    };
    const up = { x: 0, y: 1, z: 0 };

    camTarget.x += (dx * PAN_SPEED) * right.x + (dy * PAN_SPEED) * up.x;
    camTarget.y += (dx * PAN_SPEED) * right.y + (dy * PAN_SPEED) * up.y;
    camTarget.z += (dx * PAN_SPEED) * right.z + (dy * PAN_SPEED) * up.z;
  } else if (dragMode === "zoom") {
    camDist *= 1.0 + (dy * ZOOM_SPEED * 0.1); // drag up/down to zoom
    camDist = Math.max(0.5, camDist);
  }

  updateCamera();
});

canvas.addEventListener("wheel", (e) => {
  const ZOOM_WHEEL_SPEED = 0.001;
  camDist *= 1.0 + e.deltaY * ZOOM_WHEEL_SPEED;
  camDist = Math.max(0.5, camDist);
  updateCamera();
  e.preventDefault();
}, { passive: false });

// --- Render loop ------------------------------------------------------------

function render() {
  if (!pipeline) {
    requestAnimationFrame(render);
    return;
  }

  const currentTime = performance.now();

  updateCamera();

  const data = [
    canvas.width, canvas.height,
    frameCount, 0,
    camPos.x, camPos.y, camPos.z, mouseDown ? 1 : 0,
    camDir.x, camDir.y, camDir.z, 0,
    camUp.x, camUp.y, camUp.z, 0,
  ];
  device.queue.writeBuffer(uniformBuffer, 0, new Float32Array(data));

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
    const fps = Math.round(frameCount / ((currentTime - lastFpsUpdate) / 1_000));
    $("fps").textContent = fps;
    $("frame-time").textContent = `${((currentTime - lastFpsUpdate) / frameCount).toFixed(1)}ms`;
    frameCount = 0;
    lastFpsUpdate = currentTime;
  }
  requestAnimationFrame(render);
}

// --- Misc UI / window handling ---------------------------------------------

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

// --- Shader loading + main --------------------------------------------------

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
  buildSceneEditorUI();
  if (await initWebGPU()) render();
};
main();
