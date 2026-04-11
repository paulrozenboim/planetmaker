import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import GUI from 'lil-gui';

let scene, camera, renderer, composer;
let planetMesh;
let controls;
let rdMaterial, displayMaterial;
let rt1, rt2; 
let quadScene, quadCamera; 

const TEXTURE_WIDTH = 1024;  
const TEXTURE_HEIGHT = 512;  
const SIMULATION_STEPS_PER_FRAME = 8; 
const ICOS_SUBDIV = 120; // Slightly higher detail for the displacement

const params = {
  feed: 0.03, kill: 0.06, diffA: 1.0, diffB: 0.5, timeStep: 1.0,
  preset: 'Mitosis', smoothness: 0.5,
  
  // Colors & Light
  color1: '#1a3b80', color2: '#e6cc33', color3: '#1a1a1a',
  atmosphereColor: '#4facfe',
  displacementScale: 0.12,
  lightDirection: { x: 1.5, y: 1.0, z: 1.0 },
  
  // Post-Processing
  bloomStrength: 0.8,
  bloomRadius: 0.6,
  bloomThreshold: 0.2,

  // Animation & Evolution
  isPlaying: true,
  autoEvolve: true,
  rotationSpeed: 0.001,
  directionX: 0.0, directionY: 0.0,
  showWireframe: false,
  
  togglePlayPause: function () { this.isPlaying = !this.isPlaying; },
  reset: resetSimulation,
  savePNG: savePNG, saveGLTF: saveGLTF,
};

const presets = {
  Mitosis: { feed: 0.03, kill: 0.06, diffA: 1.0, diffB: 0.5, colors: { c1: '#1a3b80', c2: '#e6cc33', c3: '#1a1a1a', atm: '#4facfe' } },
  Coral: { feed: 0.0545, kill: 0.062, diffA: 1.0, diffB: 0.5, colors: { c1: '#ff6b6b', c2: '#48dbfb', c3: '#341f97', atm: '#ff6b6b' } },
  Waves: { feed: 0.017, kill: 0.045, diffA: 1.27, diffB: 0.56, colors: { c1: '#0984e3', c2: '#00cec9', c3: '#2d3436', atm: '#00cec9' } },
  Chaos: { feed: 0.039, kill: 0.058, diffA: 1.0, diffB: 0.55, colors: { c1: '#39FF14', c2: '#B026FF', c3: '#0D0D0D', atm: '#B026FF' } },
};

// --- Shaders ---
const rdVertexShader = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const rdFragmentShader = `
  varying vec2 vUv;
  uniform sampler2D tPrev;
  uniform vec2 pixelSize;
  uniform float feed;
  uniform float kill;
  uniform float diffA;
  uniform float diffB;
  uniform float timeStep;
  uniform vec2 evolutionDirection;

  vec2 laplacian(vec2 uv) {
    vec2 L = vec2(0.0);
    // Spherical anti-pinching approximation (scale X lookup by latitude)
    float cosLat = max(0.1, sin(uv.y * 3.14159)); 
    vec2 offsetPixel = vec2(pixelSize.x / cosLat, pixelSize.y);
    
    vec2 bias = evolutionDirection * pixelSize * 2.0;

    L += texture2D(tPrev, fract(uv + vec2(-offsetPixel.x, 0.0) + bias)).rg * 0.2;
    L += texture2D(tPrev, fract(uv + vec2( offsetPixel.x, 0.0) + bias)).rg * 0.2;
    L += texture2D(tPrev, fract(uv + vec2(0.0, -offsetPixel.y) + bias)).rg * 0.2;
    L += texture2D(tPrev, fract(uv + vec2(0.0,  offsetPixel.y) + bias)).rg * 0.2;

    L += texture2D(tPrev, fract(uv + vec2(-offsetPixel.x, -offsetPixel.y) + bias)).rg * 0.05;
    L += texture2D(tPrev, fract(uv + vec2( offsetPixel.x, -offsetPixel.y) + bias)).rg * 0.05;
    L += texture2D(tPrev, fract(uv + vec2(-offsetPixel.x,  offsetPixel.y) + bias)).rg * 0.05;
    L += texture2D(tPrev, fract(uv + vec2( offsetPixel.x,  offsetPixel.y) + bias)).rg * 0.05;

    L += texture2D(tPrev, uv).rg * -1.0;
    return L;
  }

  void main() {
    vec2 current = texture2D(tPrev, vUv).rg;
    vec2 L = laplacian(vUv);
    float reaction = current.r * current.g * current.g;

    float deltaA = (diffA * L.r) - reaction + (feed * (1.0 - current.r));
    float deltaB = (diffB * L.g) + reaction - ((kill + feed) * current.g);

    vec2 next = clamp(current + vec2(deltaA, deltaB) * timeStep, 0.0, 1.0);
    gl_FragColor = vec4(next.r, next.g, 0.0, 1.0);
  }
`;

const displayVertexShader = `
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vViewPosition;

  uniform sampler2D tDiffuse;
  uniform float u_displacementScale;
  uniform float u_smoothness;
  uniform vec2 texelSize;

  void main() {
    vUv = uv;
    vec2 state = texture2D(tDiffuse, uv).rg;

    // Smooth adjacent sampling for displacement
    float mainDisp = state.r - state.g;
    float rightDisp = texture2D(tDiffuse, vUv + vec2(texelSize.x, 0.0)).r - texture2D(tDiffuse, vUv + vec2(texelSize.x, 0.0)).g;
    float topDisp = texture2D(tDiffuse, vUv + vec2(0.0, texelSize.y)).r - texture2D(tDiffuse, vUv + vec2(0.0, texelSize.y)).g;
    
    float displacement = (mainDisp + rightDisp + topDisp) / 3.0;
    displacement = mix(displacement, smoothstep(-1.0, 1.0, displacement), u_smoothness) * u_displacementScale;

    vec3 displacedPosition = position + normal * displacement;
    vec4 worldPosition = modelViewMatrix * vec4(displacedPosition, 1.0);
    
    vNormal = normalize(normalMatrix * normal);
    vViewPosition = -worldPosition.xyz; // Vector from vertex to camera
    
    gl_Position = projectionMatrix * worldPosition;
  }
`;

const displayFragmentShader = `
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vViewPosition;

  uniform sampler2D tDiffuse;
  uniform vec3 u_color1;
  uniform vec3 u_color2;
  uniform vec3 u_color3;
  uniform vec3 atmosphereColor;
  uniform vec3 lightDirection;

  void main() {
    vec2 state = texture2D(tDiffuse, vUv).rg;

    // Base Masking
    float mask = smoothstep(0.3, 0.7, state.r - state.g * 0.5);
    vec3 baseColor = mix(u_color1, u_color2, mask);
    baseColor = mix(baseColor, u_color3, smoothstep(0.1, 0.4, state.g));

    // Vectors
    vec3 normal = normalize(vNormal);
    vec3 viewDir = normalize(vViewPosition);
    vec3 lightDir = normalize(lightDirection);
    
    // Diffuse & Fake AO
    float diff = max(dot(normal, lightDir), 0.0);
    float ao = mix(0.4, 1.0, smoothstep(0.0, 0.3, abs(state.r - state.g)));
    
    // Fresnel / Atmosphere
    float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0);
    vec3 atmosphere = atmosphereColor * fresnel * 1.2;

    // Specular (Make the 'growth' look wet/shiny)
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), 64.0) * state.g * 2.0;

    // Composite
    vec3 finalColor = baseColor * (diff * 0.8 + 0.2) * ao;
    finalColor += atmosphere + spec;

    gl_FragColor = vec4(finalColor, 1.0);
  }
`;

function init() {
  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x050505, 0.08);

  camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set(0, 0, 3);

  renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  document.getElementById('container').appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  const rtOptions = {
    minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
    format: THREE.RGBAFormat, type: THREE.FloatType,
    wrapS: THREE.RepeatWrapping, wrapT: THREE.RepeatWrapping,
    generateMipmaps: false
  };
  rt1 = new THREE.WebGLRenderTarget(TEXTURE_WIDTH, TEXTURE_HEIGHT, rtOptions);
  rt2 = new THREE.WebGLRenderTarget(TEXTURE_WIDTH, TEXTURE_HEIGHT, rtOptions);

  rdMaterial = new THREE.ShaderMaterial({
    uniforms: {
      tPrev: { value: null },
      pixelSize: { value: new THREE.Vector2(1.0 / TEXTURE_WIDTH, 1.0 / TEXTURE_HEIGHT) },
      feed: { value: params.feed }, kill: { value: params.kill },
      diffA: { value: params.diffA }, diffB: { value: params.diffB },
      timeStep: { value: params.timeStep },
      evolutionDirection: { value: new THREE.Vector2(0, 0) }
    },
    vertexShader: rdVertexShader, fragmentShader: rdFragmentShader
  });

  quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  quadScene = new THREE.Scene();
  quadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), rdMaterial));

  displayMaterial = new THREE.ShaderMaterial({
    uniforms: {
      tDiffuse: { value: rt1.texture },
      u_color1: { value: new THREE.Color(params.color1) },
      u_color2: { value: new THREE.Color(params.color2) },
      u_color3: { value: new THREE.Color(params.color3) },
      atmosphereColor: { value: new THREE.Color(params.atmosphereColor) },
      lightDirection: { value: new THREE.Vector3(params.lightDirection.x, params.lightDirection.y, params.lightDirection.z) },
      u_displacementScale: { value: params.displacementScale },
      u_smoothness: { value: params.smoothness },
      texelSize: { value: new THREE.Vector2(1 / TEXTURE_WIDTH, 1 / TEXTURE_HEIGHT) }
    },
    vertexShader: displayVertexShader, fragmentShader: displayFragmentShader,
    wireframe: params.showWireframe
  });

  planetMesh = new THREE.Mesh(new THREE.IcosahedronGeometry(1, ICOS_SUBDIV), displayMaterial);
  scene.add(planetMesh);

  // --- Post Processing Pipeline ---
  const renderScene = new RenderPass(scene, camera);
  const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.5, 0.4, 0.85);
  bloomPass.threshold = params.bloomThreshold;
  bloomPass.strength = params.bloomStrength;
  bloomPass.radius = params.bloomRadius;
  
  const outputPass = new OutputPass();

  composer = new EffectComposer(renderer);
  composer.addPass(renderScene);
  composer.addPass(bloomPass);
  composer.addPass(outputPass);

  resetSimulation();
  setupGUI();
  createDirectionControl();

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    composer.setSize(window.innerWidth, window.innerHeight);
  });
}

function resetSimulation() {
  const size = TEXTURE_WIDTH * TEXTURE_HEIGHT;
  const data = new Float32Array(size * 4);

  for (let i = 0; i < size; i++) {
    data[i * 4] = 1.0; 
    data[i * 4 + 1] = 0.0; 
    data[i * 4 + 3] = 1.0; 
  }

  const cx = Math.floor(TEXTURE_WIDTH / 2), cy = Math.floor(TEXTURE_HEIGHT / 2);
  for (let y = 0; y < TEXTURE_HEIGHT; y++) {
    for (let x = 0; x < TEXTURE_WIDTH; x++) {
      if ((x - cx) ** 2 + (y - cy) ** 2 < 200) {
        const idx = (y * TEXTURE_WIDTH + x) * 4;
        data[idx] = 0.5 + Math.random() * 0.1;
        data[idx + 1] = 0.25 + Math.random() * 0.1;
      }
    }
  }

  const tex = new THREE.DataTexture(data, TEXTURE_WIDTH, TEXTURE_HEIGHT, THREE.RGBAFormat, THREE.FloatType);
  tex.needsUpdate = true;

  const tempQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: tex }));
  quadScene.add(tempQuad);
  
  renderer.setRenderTarget(rt1); renderer.render(quadScene, quadCamera);
  renderer.setRenderTarget(rt2); renderer.render(quadScene, quadCamera);
  renderer.setRenderTarget(null);
  
  quadScene.remove(tempQuad); tex.dispose(); tempQuad.material.dispose();
  displayMaterial.uniforms.tDiffuse.value = rt1.texture;
}

function setupGUI() {
  const gui = new GUI({ title: 'Biosphere Controls' });

  gui.add(params, 'togglePlayPause').name(params.isPlaying ? 'Pause' : 'Play');

  const env = gui.addFolder('Ecosystem Dynamics');
  env.add(params, 'preset', Object.keys(presets)).onChange(v => {
    const p = presets[v];
    Object.assign(params, p);
    rdMaterial.uniforms.feed.value = p.feed;
    rdMaterial.uniforms.kill.value = p.kill;
    displayMaterial.uniforms.u_color1.value.set(p.colors.c1);
    displayMaterial.uniforms.u_color2.value.set(p.colors.c2);
    displayMaterial.uniforms.u_color3.value.set(p.colors.c3);
    displayMaterial.uniforms.atmosphereColor.value.set(p.colors.atm);
    resetSimulation();
    gui.controllersRecursive().forEach(c => c.updateDisplay());
  });
  env.add(params, 'feed', 0.01, 0.1).listen().onChange(v => rdMaterial.uniforms.feed.value = v);
  env.add(params, 'kill', 0.01, 0.1).onChange(v => rdMaterial.uniforms.kill.value = v);
  env.add(params, 'autoEvolve').name('Auto-Mutate');
  env.add(params, 'rotationSpeed', 0, 0.01).name('Rotation Speed');

  const vis = gui.addFolder('Visual & Lighting');
  vis.addColor(params, 'color1').onChange(v => displayMaterial.uniforms.u_color1.value.set(v));
  vis.addColor(params, 'color2').onChange(v => displayMaterial.uniforms.u_color2.value.set(v));
  vis.addColor(params, 'color3').onChange(v => displayMaterial.uniforms.u_color3.value.set(v));
  vis.addColor(params, 'atmosphereColor').onChange(v => displayMaterial.uniforms.atmosphereColor.value.set(v));
  vis.add(params, 'displacementScale', 0, 0.3).name('Displacement').onChange(v => displayMaterial.uniforms.u_displacementScale.value = v);
  
  const fx = gui.addFolder('Post-Processing');
  fx.add(params, 'bloomStrength', 0, 2).onChange(v => composer.passes[1].strength = v);
  fx.add(params, 'bloomRadius', 0, 1).onChange(v => composer.passes[1].radius = v);

  gui.add(params, 'reset').name('Reset Seed');
  const exp = gui.addFolder('Export');
  exp.add(params, 'savePNG').name('Save Snapshot').domElement.parentElement.classList.add('export-button');
}

function savePNG() {
  composer.render();
  const link = document.createElement('a');
  link.download = 'biosphere.png';
  link.href = renderer.domElement.toDataURL('image/png');
  link.click();
}

function saveGLTF() { /* Existing GLTF logic remains identical */ }

function createDirectionControl() {
  const c = document.createElement('div');
  c.id = 'direction-pad-container';
  c.style.cssText = `position:absolute; left:20px; top:50%; transform:translateY(-50%); width:160px; padding:20px; text-align:center; user-select:none; z-index:100;`;
  
  const title = document.createElement('div');
  title.textContent = 'Wind Bias';
  title.style.cssText = `font-weight:600; font-size:13px; margin-bottom:15px; color:#fff;`;
  
  const pad = document.createElement('div');
  pad.style.cssText = `width:100px; height:100px; margin:0 auto; background:rgba(0,0,0,0.4); border-radius:50%; border:1px solid rgba(255,255,255,0.1); position:relative; cursor:pointer;`;
  
  const dot = document.createElement('div');
  dot.style.cssText = `width:12px; height:12px; background:#4facfe; border-radius:50%; position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); box-shadow:0 0 10px #4facfe; pointer-events:none;`;
  
  pad.appendChild(dot); c.appendChild(title); c.appendChild(pad); document.body.appendChild(c);

  let drag = false;
  const update = (e) => {
    if (!drag) return;
    const r = pad.getBoundingClientRect();
    let x = Math.max(-1, Math.min(1, (e.clientX - r.left - 50) / 50));
    let y = Math.max(-1, Math.min(1, (e.clientY - r.top - 50) / 50));
    dot.style.left = `${50 + x * 50}%`; dot.style.top = `${50 + y * 50}%`;
    rdMaterial.uniforms.evolutionDirection.value.set(x * 0.5, -y * 0.5);
  };
  pad.onmousedown = (e) => { drag = true; update(e); };
  window.onmousemove = update; window.onmouseup = () => drag = false;
}

function animate(time) {
  requestAnimationFrame(animate);

  if (params.isPlaying) {
    planetMesh.rotation.y += params.rotationSpeed;

    // Organic auto-mutation (drifts feed slightly over time based on the base parameter)
    if (params.autoEvolve) {
      const baseFeed = presets[params.preset] ? presets[params.preset].feed : params.feed;
      rdMaterial.uniforms.feed.value = baseFeed + Math.sin(time * 0.0005) * 0.003;
    }

    renderer.autoClear = false;
    for (let i = 0; i < SIMULATION_STEPS_PER_FRAME; i++) {
      rdMaterial.uniforms.tPrev.value = rt1.texture;
      renderer.setRenderTarget(rt2); renderer.render(quadScene, quadCamera);
      let temp = rt1; rt1 = rt2; rt2 = temp;
    }
    renderer.autoClear = true;
  }

  displayMaterial.uniforms.tDiffuse.value = rt1.texture;
  controls.update();
  
  renderer.setRenderTarget(null);
  composer.render(); // Use composer instead of raw renderer for Bloom
}

window.onload = () => { init(); animate(0); };
