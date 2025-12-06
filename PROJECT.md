# WebGPU Project: Interactive Ray Marching Scene Editor 🚀

## Project Goal

The objective of this project is to extend the provided "WebGPU Shadertoy" starter code into a powerful, interactive 3D scene editor. You will modify the application so that instead of just writing shader code, you can also define and edit a 3D scene (composed of primitives like spheres, boxes, etc.) using a UI panel. These scene parameters will be passed as uniforms to your ray marching shader, allowing for real-time manipulation of the 3D world.

**To Pass the Project, a final score of** $\ge 10$ **is expected for each section.**

## 1. Scene Uniforms & Shader Integration

| Weight: 35% (Difficulty: High) |
| :--- |

This section focuses on defining a custom data structure for your scene in both JavaScript and WGSL and linking them via a new uniform buffer.

### 1.1. Defining the Scene

1.  **WGSL:** Create new `struct` definitions in your shader for primitives (e.g., `Sphere`, `Box`) and a main `Scene` struct that contains them.
2.  **JavaScript:** Create a new `GPUBuffer` to hold your `Scene` data. You will need to carefully manage the data layout (size and padding) to match the WGSL struct alignment rules (e.g., `vec3<f32>` is often padded to 16 bytes).
3.  **Bind Group:** Create a new `GPUBindGroup` (or modify the existing one) to bind this new uniform buffer to a new binding slot (e.g., `@binding(1)`).

### 1.2. Shader Modification

1.  **Adapt Shader:** Modify one of the existing ray marching shaders (e.g., `raymarch_basic.wgsl`) to use your new `Scene` uniform.
2.  **Remove Hardcoded Values:** Replace hardcoded scene definitions (like `sphere_pos = vec3<f32>(sin(time) * 1.5, 0.0, 0.0);`) with data from your uniform (e.g., `scene.spheres[0].pos`).
3.  **Render Loop:** In your JavaScript `render()` function, update your `Scene` GPU buffer *every frame* (or only when values change) using `device.queue.writeBuffer`.

| Grading Scale (0-20) | Description |
| :--- | :--- |
| **0** | No custom uniform buffer is created. Shader still uses hardcoded values. |
| **10 (Minimum Goal)** | A new uniform buffer is created. The shader can successfully render **one** primitive (e.g., a single sphere) whose properties (position, radius, color) are fully controlled by the uniform. |
| **15 (Good Goal)** | A clean `Scene` struct is defined in WGSL and JS that holds **multiple, mixed primitives** (e.g., `struct Scene { sphere1: Sphere, box1: Box }`). The shader correctly renders all objects from this struct. |
| **20 (Excellent Goal)** | A robust, extensible system is built using **arrays of primitives** (e.g., `struct Scene { spheres: array<Sphere, 5>, num_spheres: u32 }`). The shader loops over these arrays, demonstrating a fully dynamic scene structure. |

## 2. Interactive Scene Editor UI

| Weight: 30% (Difficulty: Medium) |
| :--- |

This section covers building the HTML/JS front-end that allows the user to edit the `Scene` uniforms in real-time.

### 2.1. UI Panel Creation

1.  **Modify `index.html`:** Repurpose or replace the existing `uniforms-panel` to create a new "Scene Editor" panel with interactive HTML controls.
2.  **Add Inputs:** Add HTML inputs to control your scene.
    * Use `<input type="range">` (sliders) for positions, radius, etc.
    * Use `<input type="color">` for material colors.
3.  **Labels:** Ensure all inputs are clearly labeled (e.g., "Sphere 1: X-Position", "Box 1: Color").

### 2.2. Real-Time Updates

1.  **Event Listeners:** Add JavaScript event listeners (e.g., `oninput`) to your new HTML controls.
2.  **Update Buffer:** When an input changes, update the corresponding value in your JavaScript-side `Scene` data object and write the updated data to the `Scene` GPU buffer using `device.queue.writeBuffer`.
3.  **No Re-compile:** The scene must update in real-time *without* clicking the "Compile" button.

| Grading Scale (0-20) | Description |
| :--- | :--- |
| **0** | No UI is created for editing scene parameters. |
| **10 (Minimum Goal)** | The UI panel has interactive controls for the **single primitive** from Section 1's minimum goal. Changes to sliders update the object in the viewport in real-time. |
| **15 (Good Goal)** | The UI panel is well-organized and provides controls for the **multiple, mixed primitives** from Section 1's "Good Goal". The UI feels stable and is styled to match the editor. |
| **20 (Excellent Goal)** | The UI is **dynamically generated** from the scene definition. It supports the **array of primitives** from Section 1's "Excellent Goal", perhaps with buttons to "Add Sphere" or "Select Object", which dynamically redraws the UI panel for the selected object. |

## 3. Deployment & Professional Documentation

| Weight: 20% (Difficulty: Low) |
| :--- |

This section covers the professional presentation and deployment of your project.

### 3.1. GitHub Pages Deployment (Mandatory)

* The final application must be hosted on your **GitHub Pages** URL and be fully functional and accessible to everyone. (Note: This may require a `.nojekyll` file if your assets are in folders starting with `_`).

### 3.2. Professional Repository `README.md`

Your `README.md` is the front page of your project. It should be professional, clear, and serve as a demonstration of your work.

* **Project Title & High-Quality Visual:** A clear title and a high-quality screenshot or, even better, an **animated GIF** of your editor in action.
* **Live Demo Link:** **CRUCIAL:** A prominent link to your live, functional GitHub Pages deployment.
* **Elevator Pitch:** A concise one or two-sentence overview explaining what the project is.
* **Features:** A bulleted list of what the user can do (e.g., "Real-time shader editing," "Interactive scene panel," "Click-to-select objects," "3D gizmo for translation").
* **Tech Stack:** A brief list of the core technologies (e.g., `WebGPU`, `WGSL`, `JavaScript`).
* **Local Development:** Clear, simple instructions for cloning the repo and running it locally (mentioning the `python -m http.server` requirement).

| Grading Scale (0-20) | Description |
| :--- | :--- |
| **0** | No `README.md` file. App is not deployed on GitHub Pages. |
| **10 (Minimum)** | App is deployed. A basic `README.md` exists with a title, a (possibly broken) demo link, and setup instructions. |
| **15 (Good)** | App is deployed and functional. The `README.md` includes all required sections (Title, Live Link, Features, Local Setup) and is well-formatted. |
| **20 (Very Good)** | Meets "Good" criteria. The `README.md` is exceptionally professional, featuring a high-quality screenshot or **animated GIF** and polished, clear writing. The deployed app is responsive and bug-free. |

## 4. Bonus: Viewport Gizmo Interaction

| Weight: 15% (Difficulty: High) |
| :--- |

This section is for the "bonus" feature: creating a professional, intuitive viewport interaction system.

### 4.1. Click-to-Select & Gizmo Interaction

* **Implement Ray-Object Intersection:** To select an object, you must determine *which* object the user clicked on. This requires "picking".
    * **Method 1 (Easy):** On mouse click, re-render the scene to an offscreen texture using "ID rendering" (each object has a unique flat color ID). Read the pixel color under the mouse to see which object ID was clicked.
    * **Method 2 (Hard):** Perform a ray march *in JavaScript* (or a separate compute shader) from the camera to the mouse position to find the first object hit.
* **Implement Selection State:**
    1.  When an object is "picked," set it as the `activeObject` in your JavaScript.
    2.  The "Scene Editor" UI panel should automatically scroll to and highlight the controls for this `activeObject`.
* **Implement 3D Gizmo:**
    1.  Once an object is selected, render a 3D translation gizmo (e.g., 3 colored axes) at the object's position *within your ray marching shader*.
    2.  When the user clicks and drags near the gizmo, determine which axis (X, Y, or Z) they are trying to drag.
    3.  Update the `activeObject`'s position based on the mouse drag, write the data to the uniform buffer, and see the object (and its gizmo) move in real-time.
    4.  The sliders in the UI panel must update in sync with the viewport drag.

| Grading Scale (0-20) | Description |
| :--- | :--- |
| **0** | No bonus interaction implemented. |
| **10 (Minimum)** | Implements a **Click-to-Select** system. Clicking an object in the viewport correctly highlights its properties in the "Scene Editor" UI panel. |
| **15 (Good)** | Meets "Minimum" criteria. Also implements a basic **3D Gizmo**. The gizmo is drawn at the selected object's position and can be used to drag the object, but the UI sliders might not update in sync. |
| **20 (Very Good)** | Fully implements the **Click-to-Select & Gizmo Interaction**. Clicking selects an object, the UI updates, a 3D gizmo appears, and dragging the gizmo moves the object *while also* updating the UI sliders in real-time. The interaction is smooth and intuitive. |

## Summary of Weighting and Passing Criteria

| Section | Description | Weight | Passing Score (Minimum) |
| :--- | :--- | :--- | :--- |
| **1. Scene Uniforms & Shader Integration** | | 35% | $\ge 10/20$ |
| **2. Interactive Scene Editor UI** | | 30% | $\ge 10/20$ |
| **3. Deployment & Professional Documentation** | | 20% | $\ge 10/20$ |
| **4. Bonus: Viewport Gizmo Interaction** | | 15% | $\ge 10/20$ |
| **TOTAL** | | **100%** | **Expected Pass:** $\ge 10/20$ **on all sections** |

### Appendix: Technical Setup & Resources

#### A. Local Server for Testing 💻

WebGPU security protocols (`COOP`/`COEP`) and file loading (`fetch`) require you to serve your `index.html` from a local web server.

1.  Navigate to your project's root folder in a terminal.
2.  Run the command:
    ```
    python -m http.server
    ```
3.  Open your browser to `http://localhost:8000`.

#### B. Bibliography & Resources 📚

| Resource | Description | Link |
| :--- | :--- | :--- |
| **Inigo Quilez's Articles** | The definitive resource for Signed Distance Functions (SDFs) and ray marching. | <https://iquilezles.org/articles/> |
| **WGSL Specification** | Official spec, useful for understanding struct memory layout and padding. | <https://www.w.org/TR/WGSL/> |
| **WebGPU Fundamentals** | A great guide on WebGPU concepts, including buffers and bind groups. | <https://webgpufundamentals.org/> |
