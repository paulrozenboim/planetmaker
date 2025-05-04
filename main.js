// Import necessary modules from Three.js and addons (via importmap)
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import GUI from 'lil-gui';

// --- Module-level variables ---
let scene, camera, renderer;
let planetMesh;
let controls;
let rdMaterial, displayMaterial;
let rt1, rt2; // Render targets for ping-ponging simulation
let quadScene, quadCamera; // Helper scene for rendering simulation steps

// --- Constants ---
const TEXTURE_WIDTH = 1024; // Width of the simulation texture
const TEXTURE_HEIGHT = 512; // Height (2:1 aspect ratio good for spheres)
const SIMULATION_STEPS_PER_FRAME = 5; // Number of simulation steps per display frame
const SPHERE_SEGMENTS_W = 360; // Sphere horizontal segments (for detail)
const SPHERE_SEGMENTS_H = 360;  // Sphere vertical segments (for detail)

// --- Parameters Object (for GUI controls) ---
const params = {
    // Simulation parameters
    feed: 0.03, // Feed rate (f) for Gray-Scott model
    kill: 0.06, // Kill rate (k) for Gray-Scott model
    diffA: 1.0,   // Diffusion rate for chemical A
    diffB: 0.5,   // Diffusion rate for chemical B
    timeStep: 1.0,// Timestep multiplier for simulation speed
    preset: 'Mitosis', // Default preset name

    // Display parameters
    color1: '#1a3b80', // First color for mapping RD values
    color2: '#e6cc33', // Second color for mapping RD values
    color3: '#1a1a1a', // Base/third color for mapping RD values
    displacementScale: 0.1, // Controls magnitude of vertex displacement

    // Actions (linked to functions)
    reset: resetSimulation,
    savePNG: savePNG,
    saveGLTF: saveGLTF
};

// --- Simulation Presets ---
const presets = {
     'Mitosis': { feed: 0.03, kill: 0.06 },
     'Coral Growth': { feed: 0.0545, kill: 0.062 },
     'Worms': { feed: 0.026, kill: 0.051 },
     'Waves': { feed: 0.014, kill: 0.054 },
     'Solitons': { feed: 0.025, kill: 0.06 },
     'Chaos': { feed: 0.039, kill: 0.058 },
     'Zebra': { feed: 0.029, kill: 0.057 },
};


// --- Shader Definitions ---

// Vertex shader for the simulation quad (simple pass-through)
const rdVertexShader = `
    varying vec2 vUv; // Pass UV coordinates to fragment shader
    void main() {
        vUv = uv;
        // Project vertex position
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`;

// Fragment shader for the Reaction-Diffusion step (Gray-Scott model)
const rdFragmentShader = `
    varying vec2 vUv; // Received UV coordinates
    uniform sampler2D tPrev; // Texture holding the previous state (A in .r, B in .g)
    uniform vec2 pixelSize; // Size of one pixel: (1/width, 1/height)

    // Simulation parameters passed from JavaScript
    uniform float feed;
    uniform float kill;
    uniform float diffA; // Diffusion rate for A
    uniform float diffB; // Diffusion rate for B
    uniform float timeStep; // Simulation time step (dt)

    // Calculate Laplacian using 9-point stencil (center, adjacent, diagonal)
    // Provides better stability/isotropy than 5-point
    vec2 laplacian(vec2 uv) {
        vec2 L = vec2(0.0);
        float wCenter = -1.0;   // Weight for the center pixel
        float wAdjacent = 0.2;  // Weight for N, S, E, W neighbors
        float wDiagonal = 0.05; // Weight for NE, NW, SE, SW neighbors

        // Sample adjacent neighbors (with texture wrapping using mod)
        L += texture2D(tPrev, mod(uv + vec2(-pixelSize.x, 0.0), 1.0)).rg; // Left
        L += texture2D(tPrev, mod(uv + vec2( pixelSize.x, 0.0), 1.0)).rg; // Right
        L += texture2D(tPrev, mod(uv + vec2(0.0, -pixelSize.y), 1.0)).rg; // Bottom
        L += texture2D(tPrev, mod(uv + vec2(0.0,  pixelSize.y), 1.0)).rg; // Top
        L *= wAdjacent; // Apply adjacent weight

        // Sample diagonal neighbors
        L += texture2D(tPrev, mod(uv + vec2(-pixelSize.x, -pixelSize.y), 1.0)).rg * wDiagonal;
        L += texture2D(tPrev, mod(uv + vec2( pixelSize.x, -pixelSize.y), 1.0)).rg * wDiagonal;
        L += texture2D(tPrev, mod(uv + vec2(-pixelSize.x,  pixelSize.y), 1.0)).rg * wDiagonal;
        L += texture2D(tPrev, mod(uv + vec2( pixelSize.x,  pixelSize.y), 1.0)).rg * wDiagonal;

        // Add weighted center pixel value
        L += texture2D(tPrev, uv).rg * wCenter;
        return L; // Return the calculated Laplacian (change in A and B due to diffusion)
    }

    void main() {
        // Get current chemical concentrations (A, B) at this pixel
        vec2 current = texture2D(tPrev, vUv).rg; // A = current.r, B = current.g

        // Calculate the diffusion term (Laplacian)
        vec2 L = laplacian(vUv);

        // Calculate the reaction term (A * B^2)
        float reaction = current.r * current.g * current.g;

        // Gray-Scott equations: dA/dt = D_a * laplacian(A) - A*B^2 + feed*(1-A)
        float deltaA = (diffA * L.r) - reaction + (feed * (1.0 - current.r));
        // dB/dt = D_b * laplacian(B) + A*B^2 - (kill+feed)*B
        float deltaB = (diffB * L.g) + reaction - ((kill + feed) * current.g);

        // Update A and B using simple Euler integration: next = current + delta * dt
        vec2 next = current + vec2(deltaA, deltaB) * timeStep;

        // Clamp values between 0 and 1 to prevent instability
        next = clamp(next, 0.0, 1.0);

        // Output the new state (A in red channel, B in green channel)
        // Blue and Alpha channels are not used by the simulation itself
        gl_FragColor = vec4(next.r, next.g, 0.0, 1.0);
    }
`;

// Vertex shader for the planet sphere (includes displacement)
const displayVertexShader = `
    varying vec2 vUv; // Pass UV to fragment shader
    varying vec3 vNormal; // Pass normal vector for lighting

    uniform sampler2D tDiffuse; // Reaction-diffusion texture (A=r, B=g)
    uniform float u_displacementScale; // How much to displace vertices

    void main() {
        vUv = uv; // Pass the vertex's original UV coordinates

        // Sample the reaction-diffusion texture at this vertex's UV
        vec2 state = texture2D(tDiffuse, uv).rg; // A = state.r, B = state.g

        // --- Calculate Displacement ---
        // Formula: (A - B) * scale. This enhances edges where A is high and B is low.
        // Other formulas are possible (e.g., just state.r, or state.g * scale)
        float displacement = (state.r - state.g) * u_displacementScale;

        // --- Apply Displacement ---
        // Move the vertex position outwards/inwards along its normal vector
        vec3 displacedPosition = position + normal * displacement;

        // --- Calculate Normal for Lighting ---
        // We pass the *original* normal. Recalculating the displaced normal
        // is complex and often not necessary for visual appeal.
        vNormal = normalize(normalMatrix * normal); // Transform normal to view space

        // --- Final Vertex Position ---
        // Project the *displaced* position to screen space
        gl_Position = projectionMatrix * modelViewMatrix * vec4(displacedPosition, 1.0);
    }
`;

// Fragment shader to display the RD texture colorfully on the sphere
const displayFragmentShader = `
    varying vec2 vUv; // Received UV coordinates
    varying vec3 vNormal; // Received normal vector (view space)

    uniform sampler2D tDiffuse; // Reaction-diffusion texture
    uniform float time; // Time uniform (can be used for subtle animations)

    // Controllable colors passed from JavaScript
    uniform vec3 u_color1;
    uniform vec3 u_color2;
    uniform vec3 u_color3;

    void main() {
        // Sample the RD state (A, B) at this fragment's UV
        vec2 state = texture2D(tDiffuse, vUv).rg; // A = state.r, B = state.g

        // --- Color Mapping ---
        // Map the A and B values to colors using the uniforms.
        // This example mixes between color1 and color2 based on (A - B/2),
        // then mixes with a base color3 based on B.
        float mixVal = smoothstep(0.3, 0.7, state.r - state.g * 0.5); // Create a mix factor
        vec3 color = mix(u_color1, u_color2, mixVal); // Interpolate between color1 and color2
        color = mix(color, u_color3, smoothstep(0.1, 0.3, state.g)); // Interpolate with base color

        // --- Lighting (Simple Lambertian Diffuse) ---
        vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0)); // Define light direction (view space)
        // Calculate diffuse lighting intensity based on surface normal and light direction
        float diffuse = max(0.0, dot(vNormal, lightDir)) * 0.7 + 0.3; // Add ambient term (0.3)

        // Final color is the mapped color modulated by lighting
        gl_FragColor = vec4(color * diffuse, 1.0); // Set fragment color (alpha = 1.0)
    }
`;

// --- Initialization Function ---
function init() {
    // --- Scene ---
    scene = new THREE.Scene();

    // --- Camera ---
    camera = new THREE.PerspectiveCamera(
        75, // Field of view
        window.innerWidth / window.innerHeight, // Aspect ratio
        0.1, // Near clipping plane
        1000 // Far clipping plane
    );
    camera.position.z = 2.5; // Move camera back slightly

    // --- Renderer ---
    renderer = new THREE.WebGLRenderer({
        antialias: true, // Enable anti-aliasing
        preserveDrawingBuffer: true // Needed for savePNG functionality
    });
    renderer.setSize(window.innerWidth, window.innerHeight); // Set size to full window
    renderer.setPixelRatio(window.devicePixelRatio); // Adjust for high-DPI displays
    renderer.outputColorSpace = THREE.SRGBColorSpace; // Use correct color space for display
    document.getElementById('container').appendChild(renderer.domElement); // Add canvas to DOM

    // --- Lighting ---
    const ambientLight = new THREE.AmbientLight(0xcccccc, 0.4); // Soft white ambient light
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0); // White directional light
    directionalLight.position.set(1, 1.5, 1).normalize(); // Set light direction
    scene.add(directionalLight);

    // --- Controls ---
    controls = new OrbitControls(camera, renderer.domElement); // Allow mouse interaction
    controls.enableDamping = true; // Smooth camera movement
    controls.dampingFactor = 0.05;
    controls.minDistance = 1.5; // Prevent zooming too close
    controls.maxDistance = 10;  // Prevent zooming too far

    // --- Render Targets for Simulation ---
    // We need two render targets to ping-pong between simulation steps
    const rtOptions = {
        minFilter: THREE.LinearFilter, // How to sample when texture is smaller
        magFilter: THREE.LinearFilter, // How to sample when texture is larger
        format: THREE.RGBAFormat,      // Store Red, Green, Blue, Alpha channels
        type: THREE.FloatType          // Use floating point for precision needed in simulation
    };
    rt1 = new THREE.WebGLRenderTarget(TEXTURE_WIDTH, TEXTURE_HEIGHT, rtOptions);
    rt2 = new THREE.WebGLRenderTarget(TEXTURE_WIDTH, TEXTURE_HEIGHT, rtOptions);

    // --- Reaction-Diffusion Material (for simulation step) ---
    rdMaterial = new THREE.ShaderMaterial({
        uniforms: {
            tPrev: { value: null }, // Previous state texture (set in animation loop)
            pixelSize: { value: new THREE.Vector2(1.0 / TEXTURE_WIDTH, 1.0 / TEXTURE_HEIGHT) },
            feed: { value: params.feed },
            kill: { value: params.kill },
            diffA: { value: params.diffA },
            diffB: { value: params.diffB },
            timeStep: { value: params.timeStep }
        },
        vertexShader: rdVertexShader,
        fragmentShader: rdFragmentShader
    });

    // --- Scene for Simulation Quad ---
    // We render the simulation step onto a simple quad covering the screen
    quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1); // Orthographic camera for 2D rendering
    quadScene = new THREE.Scene();
    const quadGeometry = new THREE.PlaneGeometry(2, 2); // A plane that fills the view
    const quadMesh = new THREE.Mesh(quadGeometry, rdMaterial); // Use the RD material on the quad
    quadScene.add(quadMesh);

    // --- Planet Display Material (for rendering the final sphere) ---
    displayMaterial = new THREE.ShaderMaterial({
        uniforms: {
            tDiffuse: { value: rt1.texture }, // The current RD texture state
            time: { value: 0.0 }, // Time uniform (can be used in shader)
            // Initialize color uniforms from params object
            u_color1: { value: new THREE.Color(params.color1) },
            u_color2: { value: new THREE.Color(params.color2) },
            u_color3: { value: new THREE.Color(params.color3) },
            // Initialize displacement uniform
            u_displacementScale: { value: params.displacementScale }
        },
        vertexShader: displayVertexShader,   // Use the vertex shader with displacement
        fragmentShader: displayFragmentShader, // Use the fragment shader for coloring/lighting
        side: THREE.FrontSide // Render only the front faces
    });

    // --- Planet Geometry ---
    // Use the defined segment constants for high detail
    const planetGeometry = new THREE.SphereGeometry(1, SPHERE_SEGMENTS_W, SPHERE_SEGMENTS_H);
    planetMesh = new THREE.Mesh(planetGeometry, displayMaterial); // Combine geometry and material
    scene.add(planetMesh); // Add the planet to the main scene

    // --- Initialize Simulation Texture ---
    resetSimulation(); // Set the initial state of the RD texture

    // --- GUI Setup ---
    setupGUI(); // Create the control panel

    // --- Event Listeners ---
    window.addEventListener('resize', onWindowResize); // Handle window resizing
}

// --- Initialize/Reset Simulation Texture ---
function resetSimulation() {
    const size = TEXTURE_WIDTH * TEXTURE_HEIGHT;
    // Create a Float32Array to hold RGBA data for each pixel
    const data = new Float32Array(size * 4);

    // Initialize with A=1, B=0 everywhere (base state)
    for (let i = 0; i < size; i++) {
        data[i * 4 + 0] = 1.0; // A (Red channel)
        data[i * 4 + 1] = 0.0; // B (Green channel)
        data[i * 4 + 2] = 0.0; // Blue channel (unused)
        data[i * 4 + 3] = 1.0; // Alpha channel
    }

    // Add a small "seed" area with B > 0 to kickstart pattern formation
    const centerX = Math.floor(TEXTURE_WIDTH / 2);
    const centerY = Math.floor(TEXTURE_HEIGHT / 2);
    const seedSize = 15; // Radius of the seed area

    // Iterate through pixels to create the seed
    for (let y = 0; y < TEXTURE_HEIGHT; y++) {
        for (let x = 0; x < TEXTURE_WIDTH; x++) {
             const dx = x - centerX;
             const dy = y - centerY;
             // If pixel is within the circular seed area
             if (dx*dx + dy*dy < seedSize*seedSize) {
                 const index = (y * TEXTURE_WIDTH + x) * 4; // Calculate buffer index
                 // Set A slightly lower and B slightly higher, with some randomness
                 data[index + 0] = 0.5 + Math.random() * 0.1; // A
                 data[index + 1] = 0.25 + Math.random() * 0.1; // B
             }
        }
    }

    // Create a Three.js DataTexture from the raw data
    const initialTexture = new THREE.DataTexture(data, TEXTURE_WIDTH, TEXTURE_HEIGHT, THREE.RGBAFormat, THREE.FloatType);
    initialTexture.needsUpdate = true; // Tell Three.js the texture data has changed

    // --- Render this initial state to BOTH render targets ---
    // This ensures both buffers start identically, avoiding issues on the first frame swap.
    const initialMaterial = new THREE.MeshBasicMaterial({ map: initialTexture });
    const tempQuadMesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), initialMaterial);
    quadScene.add(tempQuadMesh); // Temporarily add to the simulation scene

    // Render to rt1
    renderer.setRenderTarget(rt1);
    renderer.clear(); // Clear the target first
    renderer.render(quadScene, quadCamera);

    // Render to rt2
    renderer.setRenderTarget(rt2);
    renderer.clear(); // Clear the target first
    renderer.render(quadScene, quadCamera);

    renderer.setRenderTarget(null); // Reset renderer target back to the screen
    quadScene.remove(tempQuadMesh); // Remove the temporary mesh
    initialMaterial.dispose(); // Clean up temporary material
    initialTexture.dispose(); // Clean up temporary texture

    // Ensure the display material starts by reading from the correct texture (rt1 after swaps)
    displayMaterial.uniforms.tDiffuse.value = rt1.texture;
    console.log("Simulation Reset"); // Log reset for debugging
}


// --- GUI Setup Function ---
function setupGUI() {
    const gui = new GUI(); // Create the main GUI panel
    gui.title("Planet Controls"); // Set panel title

    // --- Simulation Folder ---
    const simFolder = gui.addFolder('Simulation Parameters'); // Create a folder
    simFolder.add(params, 'preset', Object.keys(presets)).name('Preset').onChange(value => {
        // When preset changes, update params and reset simulation
        const presetParams = presets[value];
        if (presetParams) {
            params.feed = presetParams.feed;
            params.kill = presetParams.kill;
            // Update the GUI sliders visually
            simFolder.controllers.forEach(c => {
                if (c.property === 'feed' || c.property === 'kill') {
                     c.updateDisplay(); // Reflect change in slider position
                }
            });
            // Update the actual shader uniforms
            rdMaterial.uniforms.feed.value = params.feed;
            rdMaterial.uniforms.kill.value = params.kill;
            resetSimulation(); // Reset the texture state
        }
    });
    // Add sliders for simulation parameters, linking them to shader uniforms via onChange
    simFolder.add(params, 'feed', 0.01, 0.1, 0.0001).name('Feed Rate (f)').onChange(v => rdMaterial.uniforms.feed.value = v);
    simFolder.add(params, 'kill', 0.01, 0.1, 0.0001).name('Kill Rate (k)').onChange(v => rdMaterial.uniforms.kill.value = v);
    simFolder.add(params, 'diffA', 0.1, 2.0, 0.01).name('Diffusion A (Da)').onChange(v => rdMaterial.uniforms.diffA.value = v);
    simFolder.add(params, 'diffB', 0.1, 1.0, 0.01).name('Diffusion B (Db)').onChange(v => rdMaterial.uniforms.diffB.value = v);
    simFolder.add(params, 'timeStep', 0.5, 1.5, 0.01).name('Time Step (dt)').onChange(v => rdMaterial.uniforms.timeStep.value = v);
    simFolder.add(params, 'reset').name('Reset Simulation'); // Button to call resetSimulation
    // simFolder.close(); // Optional: Start folder closed

    // --- Display Folder ---
    const displayFolder = gui.addFolder('Display Settings');
    // Add color pickers, linking them to display material uniforms
    displayFolder.addColor(params, 'color1').name('Color 1').onChange(v => displayMaterial.uniforms.u_color1.value.set(v));
    displayFolder.addColor(params, 'color2').name('Color 2').onChange(v => displayMaterial.uniforms.u_color2.value.set(v));
    displayFolder.addColor(params, 'color3').name('Base Color 3').onChange(v => displayMaterial.uniforms.u_color3.value.set(v));
    // Add slider for displacement scale
    displayFolder.add(params, 'displacementScale', 0.0, 0.5, 0.005).name('Displacement Scale').onChange(v => displayMaterial.uniforms.u_displacementScale.value = v);
    // displayFolder.close(); // Optional: Start folder closed

    // --- Export Folder ---
    const exportFolder = gui.addFolder('Export');
    // Add buttons linked to export functions, add CSS class for styling
    exportFolder.add(params, 'savePNG').name('Save PNG Snapshot').domElement.parentElement.classList.add('export-button');
    exportFolder.add(params, 'saveGLTF').name('Save Static GLTF Model').domElement.parentElement.classList.add('export-button');
    // exportFolder.open(); // Optional: Start folder open
}

// --- Utility function for triggering file downloads ---
function triggerDownload(filename, data) {
    const link = document.createElement('a'); // Create temporary link element
    link.style.display = 'none'; // Hide it
    document.body.appendChild(link); // Add to DOM to make it clickable

    // Set link properties based on data type (Blob or Data URL)
    if (data instanceof Blob) {
        link.href = URL.createObjectURL(data); // Create object URL for Blob
    } else {
        link.href = data; // Assume it's a Data URL
    }

    link.download = filename; // Set the filename for download
    link.click(); // Simulate a click to trigger download

    // Clean up the temporary link and object URL
    if (data instanceof Blob) {
        URL.revokeObjectURL(link.href); // Release object URL memory
    }
    document.body.removeChild(link); // Remove link from DOM
}

// --- PNG Export Function ---
function savePNG() {
    try {
        // Ensure the scene is fully rendered before capturing
        // (May not be strictly necessary with preserveDrawingBuffer, but good practice)
        renderer.render(scene, camera);
        // Get canvas content as PNG data URL
        const dataURL = renderer.domElement.toDataURL('image/png');
        // Trigger download
        triggerDownload('planet_snapshot.png', dataURL);
    } catch (e) {
        console.error("Error saving PNG:", e);
        alert("Could not save PNG. Check console for details.");
    }
}

// --- Static GLTF Export Function ---
function saveGLTF() {
    console.log("Starting GLTF Export...");
    try {
        // 1. Read texture data from GPU render target (rt1 contains the latest state)
        // Allocate buffer to hold RGBA float data for every pixel
        const buffer = new Float32Array(TEXTURE_WIDTH * TEXTURE_HEIGHT * 4);
        // Read pixel data from the render target into the buffer
        renderer.readRenderTargetPixels(rt1, 0, 0, TEXTURE_WIDTH, TEXTURE_HEIGHT, buffer);
        console.log("Texture data read from GPU.");

        // 2. Create a new geometry instance to modify (don't modify the one being rendered)
        const exportGeometry = new THREE.SphereGeometry(1, SPHERE_SEGMENTS_W, SPHERE_SEGMENTS_H);
        // Get buffers for position, normal, and UV attributes
        const positionAttribute = exportGeometry.attributes.position;
        const normalAttribute = exportGeometry.attributes.normal;
        const uvAttribute = exportGeometry.attributes.uv;
        // Helper vectors for calculations
        const vertex = new THREE.Vector3();
        const normal = new THREE.Vector3();

        // 3. Apply displacement to vertices based on texture data (in JavaScript)
        console.log("Applying displacement to vertices...");
        const currentDisplacementScale = params.displacementScale; // Use current GUI value

        // Iterate through each vertex in the geometry
        for (let i = 0; i < positionAttribute.count; i++) {
            // Get original vertex position, normal, and UV
            vertex.fromBufferAttribute(positionAttribute, i);
            normal.fromBufferAttribute(normalAttribute, i);
            const u = uvAttribute.getX(i);
            const v = 1.0 - uvAttribute.getY(i); // Invert v-coordinate for texture lookup

            // Map UV coordinates to texture pixel indices (integer coords with wrapping)
            let tx = Math.floor(u * TEXTURE_WIDTH) % TEXTURE_WIDTH;
            let ty = Math.floor(v * TEXTURE_HEIGHT) % TEXTURE_HEIGHT;
            // Handle potential negative results from modulo
            if (tx < 0) tx += TEXTURE_WIDTH;
            if (ty < 0) ty += TEXTURE_HEIGHT;

            // Calculate the index in the 1D buffer array (4 floats per pixel: R,G,B,A)
            const bufferIndex = (ty * TEXTURE_WIDTH + tx) * 4;

            // Sample A (Red channel) and B (Green channel) values from the buffer
            const stateA = buffer[bufferIndex];
            const stateB = buffer[bufferIndex + 1];

            // Calculate displacement amount using the same formula as the vertex shader
            const displacement = (stateA - stateB) * currentDisplacementScale;

            // Apply displacement: Move vertex along its normal vector
            vertex.addScaledVector(normal, displacement);

            // Write the new displaced position back into the geometry's position buffer
            positionAttribute.setXYZ(i, vertex.x, vertex.y, vertex.z);
        }
        positionAttribute.needsUpdate = true; // Tell Three.js the position data changed
        // Recalculate vertex normals based on the new displaced positions
        // This improves lighting appearance in the exported model
        exportGeometry.computeVertexNormals();
        console.log("Displacement applied and normals recalculated.");

        // 4. Create a mesh with the displaced geometry and a simple material
        // Complex shader materials don't export well to standard formats like GLTF.
         const exportMaterial = new THREE.MeshStandardMaterial({
             color: 0xcccccc, // Use a basic gray color
             roughness: 0.7,
             metalness: 0.1
             // We don't apply the RD texture here, as GLTF expects standard material properties
             // or baked textures, not live procedural textures.
         });
        const exportMesh = new THREE.Mesh(exportGeometry, exportMaterial);

        // 5. Use GLTFExporter to generate the file content
        console.log("Parsing scene for GLTF export...");
        const exporter = new GLTFExporter();
        exporter.parse(
            exportMesh, // Export only the modified planet mesh
            function (gltf) { // Success callback (gltf is an ArrayBuffer for binary)
                console.log("GLTF parsing successful.");
                // Create a Blob from the ArrayBuffer
                const blob = new Blob([gltf], { type: 'model/gltf-binary' });
                // Trigger the download of the .glb file
                triggerDownload('planet_model.glb', blob);
            },
            function (error) { // Error callback
                console.error('Error exporting GLTF:', error);
                alert('Could not export GLTF model. Check console for errors.');
            },
            { binary: true } // Export as binary GLTF (.glb)
        );
    } catch(e) {
         console.error("Error during GLTF export process:", e);
         alert("An error occurred during GLTF export. Check console.");
    }
}


// --- Resize Handler ---
function onWindowResize() {
    // Update camera aspect ratio and projection matrix
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    // Update renderer size
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// --- Animation Loop ---
function animate(time) {
    // Request the next frame
    requestAnimationFrame(animate);

    // Convert time to seconds (or use for other effects)
    const dt = time * 0.0001;

    // --- Run Simulation Steps ---
    renderer.autoClear = false; // Disable automatic clearing for ping-pong rendering
    // Perform multiple simulation steps per display frame for faster evolution
    for (let i = 0; i < SIMULATION_STEPS_PER_FRAME; i++) {
        // Set the input texture for the simulation shader (current state is in rt1)
        rdMaterial.uniforms.tPrev.value = rt1.texture;

        // Render the simulation step to the *other* render target (rt2)
        renderer.setRenderTarget(rt2);
        renderer.clear(); // Clear the target before rendering
        renderer.render(quadScene, quadCamera); // Render the quad using the RD material

        // Swap render targets: rt2 now holds the new state, rt1 holds the previous
        const temp = rt1;
        rt1 = rt2;
        rt2 = temp;
        // After the swap, rt1 holds the latest state, ready for display or the next step
    }
     renderer.autoClear = true; // Re-enable automatic clearing for the main render pass


    // --- Render Main Scene ---
    // Update the display material's texture uniform to the latest simulation result (in rt1)
    displayMaterial.uniforms.tDiffuse.value = rt1.texture;
    displayMaterial.uniforms.time.value = dt; // Update time uniform if needed by shader

    // Update camera controls (for damping)
    controls.update();

    // Render the main scene (planet) to the screen (canvas)
    renderer.setRenderTarget(null); // Ensure rendering goes to the canvas
    renderer.clear(); // Clear the canvas
    renderer.render(scene, camera); // Render the scene with the planet
}

// --- Start the application ---
// Use window.onload to ensure the DOM is ready and scripts are loaded
window.onload = () => {
    try {
        init(); // Initialize scene, objects, GUI
        animate(0); // Start the animation loop
    } catch (error) {
        // Basic error handling if initialization fails
        console.error("Initialization failed:", error);
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = 'position:absolute;top:0;left:0;width:100%;padding:20px;background-color:rgba(255,0,0,0.8);color:white;z-index:1000;font-family:monospace;';
        errorDiv.innerHTML = `<h2>Error Initializing Simulation</h2><p>Could not start the WebGL application. Please ensure your browser supports WebGL and check the developer console for more details.</p><pre>${error.message}\n${error.stack}</pre>`;
        document.body.appendChild(errorDiv); // Display error message on screen
    }
};
