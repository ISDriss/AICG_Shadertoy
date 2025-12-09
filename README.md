# AICG_SHADERTOY {#top}

*Unleash Creativity with Real-Time Visual Mastery*

![last-commit](https://img.shields.io/github/last-commit/ISDriss/AICG_Shadertoy?style=flat&logo=git&logoColor=white&color=0080ff)
![repo-top-language](https://img.shields.io/github/languages/top/ISDriss/AICG_Shadertoy?style=flat&color=0080ff)
![repo-language-count](https://img.shields.io/github/languages/count/ISDriss/AICG_Shadertoy?style=flat&color=0080ff)

Built with:
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E.svg?style=flat&logo=JavaScript&logoColor=black)
![HTML5](https://img.shields.io/badge/HTML5-E34F26.svg?style=flat&logo=html5&logoColor=white)
![WGSL](https://img.shields.io/badge/WGSL-663399.svg?style=flat&logo=webgpu&logoColor=white)

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Live Demo](#live-demo)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running Locally](#running-locally)
- [Tech Stack](#tech-stack)
- [Screenshots](#screenshots)

---

## Overview

**AICG_Shadertoy** is a full WebGPU-powered 3D shader playground and scene editor.  
It merges real-time ray marching, an interactive object editor, and a built-in WGSL IDE into a single lightweight browser application.

This project is built as a complete assignment solution for an advanced GPU programming course. It demonstrates deep integration between a WebGPU rendering pipeline, WGSL shader architecture, and dynamic UI-driven scene manipulation.

---

## Features

### Interactive Scene Editor
- Add/remove primitives dynamically  
- Edit parameters: position, radius, size, normals, height, endpoints, etc.  
- Material selector with predefined shading models  
- Auto-updating GPU buffer — no need to recompile  

### Supported Primitives
- Sphere  
- Plane  
- Box  
- Rounded Box  
- Cylinder  
- Torus  
- Capsule  

### Orbit Camera (Blender-like)
- Alt + LMB or MMB — Orbit  
- Shift + MMB — Pan  
- Ctrl + MMB — Zoom  
- Scroll wheel — Zoom  

### Fully GPU-Driven Scene
- Scene is serialized into a tightly packed uniform buffer  
- WGSL shader reads a Scene struct with a fixed-size array of primitives  
- No hardcoded scene logic inside the shader  

### Advanced Ray Marching Pipeline
- Soft shadows  
- Fresnel reflections  
- Glass & water refraction  
- Sky environment lighting  
- Normal calculation from SDF gradient  
- Multi-bounce path approximation  

---

## Live Demo

**Experience it here:**  
### 🔗 https://isdriss.github.io/AICG_Shadertoy/

---

## Getting Started

### Prerequisites
Before running locally, ensure you have:

- **Python 3** (for a local HTTP server)
- A **WebGPU-compatible browser**  
  (Chrome 113+, Edge 113+, Safari Tech Preview)

### Installation

```sh
git clone https://github.com/ISDriss/AICG_Shadertoy
cd AICG_Shadertoy
```

### Running Locally

Because WebGPU requires an HTTPS or localhost context, you must use a local server:

```sh
python -m http.server
```

Then open:
```sh
http://localhost:8000
```

---

### Tech Stack

- WebGPU — next-gen graphics API for the web  
- WGSL — shader language powering ray marching  
- JavaScript — UI, buffer management, compilation orchestration  
- HTML/CSS (Tailwind-inspired) — layout & styling  

---

[[⬆ Return](#top)]
