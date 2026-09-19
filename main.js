import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

/**
 * Planet Maker — a Gray-Scott reaction-diffusion field wrapped onto a sphere.
 *
 * Two fields, A and B, diffuse at different rates and react; B eats A. The
 * result is displaced and shaded as terrain. Everything you can change moves
 * one of four numbers: feed, kill, and the two diffusion rates.
 *
 * The limits below are measured rather than guessed. Sweeping the parameter
 * space on 19 September 2026 found:
 *   - diffusion A above 1.1 saturates the field at this timestep: B floods
 *     everything and the planet goes flat. The old Waves preset shipped at
 *     1.27 and did exactly that — though it never showed, because the preset's
 *     diffusion values were assigned to an object and never reached the shader.
 *   - the living band is narrow, and narrower still than it first appears:
     many settings are patterned for a few hundred steps and then homogenise.
 *   - feed and kill are coupled; the living region is a diagonal band, which
 *     is why they are one map here instead of two independent sliders.
 */

/* ---- constants --------------------------------------------------------- */
const TEXTURE_WIDTH = 1024;
const TEXTURE_HEIGHT = 512;
const STEPS_PER_FRAME = 8;
/* Establishing a pattern takes a few thousand steps. Running them before the
   first paint is the difference between opening on a planet and opening on a
   plain sphere you have to sit and wait out. */
const WARMUP_STEPS = 2600;

/** Hard stability ceiling, from the sweep. Above this the field saturates. */
const DIFF_A_MAX = 1.10;
const FEED_MIN = 0.018, FEED_MAX = 0.100;
const KILL_MIN = 0.042, KILL_MAX = 0.066;

/* The measured viability map: rows are feed, columns kill, each character the
   variation left in the B field after 2,600 steps from an even seed, at the
   diffusion rates below.
       .  collapsed to one value   -  faint
       o  moderate structure       @  strong structure

   The horizon matters more than it looks. An earlier sweep stopped at 720
   steps and found a band roughly twice this wide — because many settings are
   only transiently patterned and homogenise if you leave them running. Three
   presets picked off that map looked right for a few seconds and then went
   featureless. Measure the steady state, not the first thing you see. */
const VIABILITY = {
  feeds: [0.018, 0.026, 0.034, 0.042, 0.050, 0.058, 0.066, 0.074, 0.082, 0.090, 0.100],
  kills: [0.042, 0.046, 0.050, 0.054, 0.057, 0.060, 0.062, 0.064, 0.066],
  rows: [
    '.@@oo....',
    '...ooo...',
    '....oo@o.',
    '.....o@@.',
    '.....-@@-',
    '......@@-',
    '.....-@@.',
    '.....@o-.',
    '.....@--.',
    '....@o-..',
    '...@-....',
  ],
};

/* Every preset sits on a cell the steady-state sweep marked strong, and every
   one holds the diffusion rates the sweep was run at — so the map behind the
   control is telling the truth about where these sit. Changing diffusion moves
   the band, which is why it is a separate control rather than part of a
   preset. */
const PRESETS = {
  Mitosis: { feed: 0.034, kill: 0.062, diffA: 1.00, diffB: 0.50,
             c1: '#0d1f0a', c2: '#c9ff36', c3: '#090909', atm: '#c9ff36' },
  Coral:   { feed: 0.058, kill: 0.062, diffA: 1.00, diffB: 0.50,
             c1: '#7a1f3d', c2: '#ffb38a', c3: '#2b0d1a', atm: '#ff7a59' },
  Waves:   { feed: 0.018, kill: 0.046, diffA: 1.00, diffB: 0.50,
             c1: '#06304a', c2: '#63e3d4', c3: '#04141f', atm: '#63e3d4' },
  Chaos:   { feed: 0.042, kill: 0.064, diffA: 1.00, diffB: 0.50,
             c1: '#2b0b4a', c2: '#fbea4b', c3: '#0d0416', atm: '#b026ff' },
  Lace:    { feed: 0.066, kill: 0.062, diffA: 1.00, diffB: 0.50,
             c1: '#1a1a1a', c2: '#f7f7ef', c3: '#050505', atm: '#aaa9a4' },
  Fissure: { feed: 0.090, kill: 0.057, diffA: 1.00, diffB: 0.50,
             c1: '#3d1a06', c2: '#ff8c1a', c3: '#140702', atm: '#ff5a1a' },
};

const params = {
  feed: PRESETS.Mitosis.feed, kill: PRESETS.Mitosis.kill,
  diffA: PRESETS.Mitosis.diffA, diffB: PRESETS.Mitosis.diffB,
  preset: 'Mitosis',
  smoothness: 0.5, displacementScale: 0.12,
  color1: PRESETS.Mitosis.c1, color2: PRESETS.Mitosis.c2,
  color3: PRESETS.Mitosis.c3, atmosphereColor: PRESETS.Mitosis.atm,
  bloomStrength: 0.45, bloomRadius: 0.55, bloomThreshold: 0.35,
  isPlaying: true, drift: true, rotationSpeed: 0.0012, wireframe: false,
};

let scene, camera, renderer, composer, controls, bloomPass;
let planetMesh, rdMaterial, displayMaterial;
let rt1, rt2, quadScene, quadCamera;
let frameCount = 0, fpsAt = 0;

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

/* ---- shaders ----------------------------------------------------------- */
const rdVertexShader = /* glsl */`
  varying vec2 vUv;
  void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
`;

const rdFragmentShader = /* glsl */`
  varying vec2 vUv;
  uniform sampler2D tPrev;
  uniform vec2 pixelSize;
  uniform float feed, kill, diffA, diffB, timeStep;
  uniform vec2 evolutionDirection;

  vec2 laplacian(vec2 uv){
    vec2 L = vec2(0.0);
    /* An equirectangular map crowds its columns together towards the poles, so
       one pixel of x is a much shorter step on the sphere up there. Scaling the
       x lookup by latitude keeps the neighbourhood roughly round; the clamp
       stops the scale running away in the last few rows. */
    float cosLat = max(0.12, sin(uv.y * 3.14159265));
    vec2 o = vec2(pixelSize.x / cosLat, pixelSize.y);
    vec2 bias = evolutionDirection * pixelSize * 2.0;

    L += texture2D(tPrev, fract(uv + vec2(-o.x, 0.0) + bias)).rg * 0.2;
    L += texture2D(tPrev, fract(uv + vec2( o.x, 0.0) + bias)).rg * 0.2;
    L += texture2D(tPrev, fract(uv + vec2(0.0, -o.y) + bias)).rg * 0.2;
    L += texture2D(tPrev, fract(uv + vec2(0.0,  o.y) + bias)).rg * 0.2;
    L += texture2D(tPrev, fract(uv + vec2(-o.x, -o.y) + bias)).rg * 0.05;
    L += texture2D(tPrev, fract(uv + vec2( o.x, -o.y) + bias)).rg * 0.05;
    L += texture2D(tPrev, fract(uv + vec2(-o.x,  o.y) + bias)).rg * 0.05;
    L += texture2D(tPrev, fract(uv + vec2( o.x,  o.y) + bias)).rg * 0.05;
    L += texture2D(tPrev, uv).rg * -1.0;
    return L;
  }

  void main(){
    vec2 c = texture2D(tPrev, vUv).rg;
    vec2 L = laplacian(vUv);
    float reaction = c.r * c.g * c.g;
    float dA = (diffA * L.r) - reaction + (feed * (1.0 - c.r));
    float dB = (diffB * L.g) + reaction - ((kill + feed) * c.g);
    vec2 next = clamp(c + vec2(dA, dB) * timeStep, 0.0, 1.0);
    gl_FragColor = vec4(next.r, next.g, 0.0, 1.0);
  }
`;

const displayVertexShader = /* glsl */`
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vViewPosition;
  uniform sampler2D tDiffuse;
  uniform float u_displacementScale, u_smoothness;
  uniform vec2 texelSize;

  float height(vec2 uv){
    vec2 s = texture2D(tDiffuse, fract(uv)).rg;
    return s.r - s.g;
  }

  void main(){
    vUv = uv;
    /* Five taps rather than three, and centred rather than leaning up-right,
       so the relief does not drift off the pattern it is supposed to follow.
       fract() on every lookup keeps the seam from tearing. */
    float h = (height(uv) * 2.0
             + height(uv + vec2(texelSize.x, 0.0))
             + height(uv - vec2(texelSize.x, 0.0))
             + height(uv + vec2(0.0, texelSize.y))
             + height(uv - vec2(0.0, texelSize.y))) / 6.0;
    float d = mix(h, smoothstep(-1.0, 1.0, h), u_smoothness) * u_displacementScale;

    vec3 displaced = position + normal * d;
    vec4 worldPosition = modelViewMatrix * vec4(displaced, 1.0);
    vNormal = normalize(normalMatrix * normal);
    vViewPosition = -worldPosition.xyz;
    gl_Position = projectionMatrix * worldPosition;
  }
`;

const displayFragmentShader = /* glsl */`
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vViewPosition;
  uniform sampler2D tDiffuse;
  uniform vec3 u_color1, u_color2, u_color3, atmosphereColor, lightDirection;

  void main(){
    vec2 s = texture2D(tDiffuse, vUv).rg;

    /* A narrower window than the original, which mapped an untouched field
       straight onto colour 2 — so a planet that had not grown yet was a flat
       saturated ball, and with bloom on top, a blown-out one. */
    float mask = smoothstep(0.42, 0.78, s.r - s.g * 0.5);
    vec3 base = mix(u_color1, u_color2, mask);
    base = mix(base, u_color3, smoothstep(0.12, 0.42, s.g));

    vec3 n = normalize(vNormal);
    vec3 viewDir = normalize(vViewPosition);
    vec3 lightDir = normalize(lightDirection);

    float diff = max(dot(n, lightDir), 0.0);
    float ao = mix(0.45, 1.0, smoothstep(0.0, 0.3, abs(s.r - s.g)));
    float fresnel = pow(1.0 - max(dot(n, viewDir), 0.0), 3.0);
    vec3 atmosphere = atmosphereColor * fresnel * 1.1;
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(n, halfDir), 0.0), 64.0) * s.g * 1.6;

    vec3 col = base * (diff * 0.82 + 0.18) * ao + atmosphere + spec;
    gl_FragColor = vec4(col, 1.0);
  }
`;

/* ---- setup ------------------------------------------------------------- */
function init(){
  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x090909, 0.07);

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
  controls.minDistance = 1.4;
  controls.maxDistance = 8;

  /* Float targets with linear filtering are not universal. Half float is, near
     enough, and the simulation does not need the range — the field is clamped
     to 0..1 on every step. */
  const linearFloat = renderer.extensions.has('OES_texture_float_linear');
  const rtOptions = {
    minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
    format: THREE.RGBAFormat, type: linearFloat ? THREE.FloatType : THREE.HalfFloatType,
    wrapS: THREE.RepeatWrapping, wrapT: THREE.RepeatWrapping,
    generateMipmaps: false, depthBuffer: false, stencilBuffer: false,
  };
  rt1 = new THREE.WebGLRenderTarget(TEXTURE_WIDTH, TEXTURE_HEIGHT, rtOptions);
  rt2 = new THREE.WebGLRenderTarget(TEXTURE_WIDTH, TEXTURE_HEIGHT, rtOptions);

  rdMaterial = new THREE.ShaderMaterial({
    uniforms: {
      tPrev: { value: null },
      pixelSize: { value: new THREE.Vector2(1 / TEXTURE_WIDTH, 1 / TEXTURE_HEIGHT) },
      feed: { value: params.feed }, kill: { value: params.kill },
      diffA: { value: params.diffA }, diffB: { value: params.diffB },
      timeStep: { value: 1.0 },
      evolutionDirection: { value: new THREE.Vector2(0, 0) },
    },
    vertexShader: rdVertexShader, fragmentShader: rdFragmentShader,
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
      lightDirection: { value: new THREE.Vector3(1.5, 1.0, 1.0) },
      u_displacementScale: { value: params.displacementScale },
      u_smoothness: { value: params.smoothness },
      texelSize: { value: new THREE.Vector2(1 / TEXTURE_WIDTH, 1 / TEXTURE_HEIGHT) },
    },
    vertexShader: displayVertexShader, fragmentShader: displayFragmentShader,
    wireframe: params.wireframe,
  });

  /* 120 subdivisions is about 290,000 triangles, which a phone will not carry.
     The mesh only has to out-resolve a 1024x512 texture. */
  const subdivision = window.innerWidth < 720 ? 48 : 96;
  planetMesh = new THREE.Mesh(new THREE.IcosahedronGeometry(1, subdivision), displayMaterial);
  scene.add(planetMesh);

  bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    params.bloomStrength, params.bloomRadius, params.bloomThreshold);
  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  composer.addPass(bloomPass);
  composer.addPass(new OutputPass());

  seed();
  buildUI();
  window.addEventListener('resize', onResize);
}

function onResize(){
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
}

/**
 * Fill the field and let it establish.
 *
 * The original seeded one small disc at the equator, which took thousands of
 * steps to reach the poles — so the tool opened on a blank sphere and stayed
 * that way long enough to look broken.
 */
function seed(){
  const size = TEXTURE_WIDTH * TEXTURE_HEIGHT;
  const data = new Float32Array(size * 4);
  for (let i = 0; i < size; i++){ data[i * 4] = 1.0; data[i * 4 + 3] = 1.0; }

  const blobs = 120, r = 10;
  for (let b = 0; b < blobs; b++){
    const cx = Math.random() * TEXTURE_WIDTH;
    /* Keep seeds off the poles, where the columns converge and a blob smears
       into a ring. */
    const cy = TEXTURE_HEIGHT * 0.12 + Math.random() * TEXTURE_HEIGHT * 0.76;
    for (let y = Math.max(0, (cy - r) | 0); y < Math.min(TEXTURE_HEIGHT, cy + r); y++){
      for (let x = (cx - r) | 0; x < cx + r; x++){
        if ((x - cx) ** 2 + (y - cy) ** 2 > r * r) continue;
        const xi = ((x % TEXTURE_WIDTH) + TEXTURE_WIDTH) % TEXTURE_WIDTH;
        const idx = (y * TEXTURE_WIDTH + xi) * 4;
        data[idx] = 0.5 + Math.random() * 0.1;
        data[idx + 1] = 0.25 + Math.random() * 0.1;
      }
    }
  }

  const tex = new THREE.DataTexture(data, TEXTURE_WIDTH, TEXTURE_HEIGHT, THREE.RGBAFormat, THREE.FloatType);
  tex.needsUpdate = true;
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: tex }));
  quadScene.add(quad);
  renderer.setRenderTarget(rt1); renderer.render(quadScene, quadCamera);
  renderer.setRenderTarget(rt2); renderer.render(quadScene, quadCamera);
  renderer.setRenderTarget(null);
  quadScene.remove(quad); tex.dispose(); quad.material.dispose();

  step(WARMUP_STEPS);
  displayMaterial.uniforms.tDiffuse.value = rt1.texture;
}

function step(n){
  const prevAutoClear = renderer.autoClear;
  renderer.autoClear = false;
  for (let i = 0; i < n; i++){
    rdMaterial.uniforms.tPrev.value = rt1.texture;
    renderer.setRenderTarget(rt2);
    renderer.render(quadScene, quadCamera);
    const t = rt1; rt1 = rt2; rt2 = t;
  }
  renderer.setRenderTarget(null);
  renderer.autoClear = prevAutoClear;
}

/* ---- applying values --------------------------------------------------- */
function setFeedKill(feed, kill){
  params.feed = clamp(feed, FEED_MIN, FEED_MAX);
  params.kill = clamp(kill, KILL_MIN, KILL_MAX);
  rdMaterial.uniforms.feed.value = params.feed;
  rdMaterial.uniforms.kill.value = params.kill;
}

function setColors(c1, c2, c3, atm){
  params.color1 = c1; params.color2 = c2; params.color3 = c3; params.atmosphereColor = atm;
  displayMaterial.uniforms.u_color1.value.set(c1);
  displayMaterial.uniforms.u_color2.value.set(c2);
  displayMaterial.uniforms.u_color3.value.set(c3);
  displayMaterial.uniforms.atmosphereColor.value.set(atm);
}

function applyPreset(name){
  const p = PRESETS[name];
  if (!p) return;
  params.preset = name;
  /* The original assigned these onto a plain object and never pushed them into
     the shader, so a preset's diffusion values did nothing at all. */
  params.diffA = Math.min(p.diffA, DIFF_A_MAX);
  params.diffB = p.diffB;
  rdMaterial.uniforms.diffA.value = params.diffA;
  rdMaterial.uniforms.diffB.value = params.diffB;
  setFeedKill(p.feed, p.kill);
  setColors(p.c1, p.c2, p.c3, p.atm);
  seed();
  syncUI();
}

/* ---- interface --------------------------------------------------------- */
const ui = {};

function el(tag, attrs = {}, text){
  const n = document.createElement(tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (text != null) n.textContent = text;
  return n;
}

function buildUI(){
  const panel = el('aside', { class: 'panel' });
  panel.innerHTML = `
    <div class="panel-head">
      <h1>Planet Maker</h1><span class="by">unapaulogetic</span>
    </div>

    <div class="group">
      <h2>Presets</h2>
      <div class="presets" id="presets"></div>
    </div>

    <div class="group">
      <h2>The living region</h2>
      <div class="map-wrap">
        <canvas id="map" width="480" height="440" aria-label="Feed and kill map. Drag to choose a point."></canvas>
        <div class="map-axis"><span>kill ${KILL_MIN}</span><span>kill ${KILL_MAX}</span></div>
      </div>
      <div class="map-read">
        <span>feed <b id="r-feed"></b></span>
        <span>kill <b id="r-kill"></b></span>
        <span class="life" id="r-life"></span>
      </div>
    </div>

    <div class="group">
      <h2>Diffusion</h2>
      <div class="row">
        <label for="s-diffA">A — spread</label><span class="val" id="v-diffA"></span>
        <input type="range" id="s-diffA" min="0.60" max="${DIFF_A_MAX}" step="0.01">
      </div>
      <div class="row">
        <label for="s-diffB">B — spread</label><span class="val" id="v-diffB"></span>
        <input type="range" id="s-diffB" min="0.20" max="0.80" step="0.01">
      </div>
    </div>

    <div class="group">
      <h2>Surface</h2>
      <div class="row">
        <label for="s-disp">Relief</label><span class="val" id="v-disp"></span>
        <input type="range" id="s-disp" min="0" max="0.30" step="0.005">
      </div>
      <div class="row">
        <label for="s-smooth">Smoothing</label><span class="val" id="v-smooth"></span>
        <input type="range" id="s-smooth" min="0" max="1" step="0.01">
      </div>
      <div class="swatches">
        <label class="sw"><span>Low</span><input type="color" id="c1"></label>
        <label class="sw"><span>High</span><input type="color" id="c2"></label>
        <label class="sw"><span>Growth</span><input type="color" id="c3"></label>
        <label class="sw"><span>Air</span><input type="color" id="atm"></label>
      </div>
    </div>

    <div class="group">
      <h2>Light</h2>
      <div class="row">
        <label for="s-bloom">Glow</label><span class="val" id="v-bloom"></span>
        <input type="range" id="s-bloom" min="0" max="1.4" step="0.01">
      </div>
      <div class="row">
        <label for="s-thresh">Glow threshold</label><span class="val" id="v-thresh"></span>
        <input type="range" id="s-thresh" min="0" max="1" step="0.01">
      </div>
    </div>

    <div class="group">
      <h2>Wind</h2>
      <div class="pad-wrap">
        <div id="pad"><div id="pad-dot"></div></div>
        <button class="btn" id="b-windreset" style="width:100%">Still</button>
      </div>
    </div>

    <div class="group">
      <h2>Motion</h2>
      <div class="row">
        <label for="s-rot">Rotation</label><span class="val" id="v-rot"></span>
        <input type="range" id="s-rot" min="0" max="0.006" step="0.0001">
      </div>
      <div class="btn-row">
        <button class="btn" id="b-play" aria-pressed="true">Pause</button>
        <button class="btn" id="b-drift" aria-pressed="true">Drift</button>
      </div>
    </div>

    <div class="group">
      <div class="btn-row">
        <button class="btn" id="b-reseed">Reseed</button>
        <button class="btn" id="b-wire" aria-pressed="false">Wire</button>
      </div>
      <div class="btn-row one" style="margin-top:6px">
        <button class="btn is-primary" id="b-png">Save image</button>
      </div>
    </div>
  `;
  document.body.appendChild(panel);

  const toggle = el('button', { id: 'toggle' }, 'Hide');
  toggle.addEventListener('click', () => {
    const hidden = document.body.classList.toggle('hidden-ui');
    toggle.textContent = hidden ? 'Show' : 'Hide';
  });
  document.body.appendChild(toggle);

  const hud = el('div', { id: 'hud' });
  hud.innerHTML = `<span>Drag to orbit · scroll to zoom</span><span>fps <b id="h-fps">–</b></span>`;
  document.body.appendChild(hud);

  const host = document.getElementById('presets');
  Object.keys(PRESETS).forEach(name => {
    const b = el('button', { class: 'btn', 'data-preset': name }, name);
    b.addEventListener('click', () => applyPreset(name));
    host.appendChild(b);
  });

  bindRange('s-diffA', 'v-diffA', v => { params.diffA = v; rdMaterial.uniforms.diffA.value = v; }, v => v.toFixed(2));
  bindRange('s-diffB', 'v-diffB', v => { params.diffB = v; rdMaterial.uniforms.diffB.value = v; }, v => v.toFixed(2));
  bindRange('s-disp', 'v-disp', v => { params.displacementScale = v; displayMaterial.uniforms.u_displacementScale.value = v; }, v => v.toFixed(3));
  bindRange('s-smooth', 'v-smooth', v => { params.smoothness = v; displayMaterial.uniforms.u_smoothness.value = v; }, v => v.toFixed(2));
  bindRange('s-bloom', 'v-bloom', v => { params.bloomStrength = v; bloomPass.strength = v; }, v => v.toFixed(2));
  bindRange('s-thresh', 'v-thresh', v => { params.bloomThreshold = v; bloomPass.threshold = v; }, v => v.toFixed(2));
  bindRange('s-rot', 'v-rot', v => { params.rotationSpeed = v; }, v => v.toFixed(4));

  bindColor('c1', v => setColors(v, params.color2, params.color3, params.atmosphereColor));
  bindColor('c2', v => setColors(params.color1, v, params.color3, params.atmosphereColor));
  bindColor('c3', v => setColors(params.color1, params.color2, v, params.atmosphereColor));
  bindColor('atm', v => setColors(params.color1, params.color2, params.color3, v));

  ui.play = document.getElementById('b-play');
  ui.play.addEventListener('click', () => {
    params.isPlaying = !params.isPlaying;
    ui.play.textContent = params.isPlaying ? 'Pause' : 'Play';
    ui.play.setAttribute('aria-pressed', String(params.isPlaying));
  });

  ui.drift = document.getElementById('b-drift');
  ui.drift.addEventListener('click', () => {
    params.drift = !params.drift;
    ui.drift.setAttribute('aria-pressed', String(params.drift));
  });

  ui.wire = document.getElementById('b-wire');
  ui.wire.addEventListener('click', () => {
    params.wireframe = !params.wireframe;
    displayMaterial.wireframe = params.wireframe;
    ui.wire.setAttribute('aria-pressed', String(params.wireframe));
  });

  document.getElementById('b-reseed').addEventListener('click', () => seed());
  document.getElementById('b-png').addEventListener('click', savePNG);
  document.getElementById('b-windreset').addEventListener('click', () => setWind(0, 0));

  buildMap();
  buildWindPad();
  syncUI();
}

function bindRange(id, valId, apply, fmt){
  const input = document.getElementById(id);
  const out = document.getElementById(valId);
  ui[id] = { input, out, apply, fmt };
  input.addEventListener('input', () => {
    const v = parseFloat(input.value);
    apply(v);
    out.textContent = fmt(v);
  });
}

function bindColor(id, apply){
  const input = document.getElementById(id);
  ui[id] = input;
  input.addEventListener('input', () => apply(input.value));
}

function syncUI(){
  const set = (id, v) => { const c = ui[id]; if (c){ c.input.value = v; c.out.textContent = c.fmt(v); } };
  set('s-diffA', params.diffA);
  set('s-diffB', params.diffB);
  set('s-disp', params.displacementScale);
  set('s-smooth', params.smoothness);
  set('s-bloom', params.bloomStrength);
  set('s-thresh', params.bloomThreshold);
  set('s-rot', params.rotationSpeed);
  if (ui.c1) ui.c1.value = params.color1;
  if (ui.c2) ui.c2.value = params.color2;
  if (ui.c3) ui.c3.value = params.color3;
  if (ui.atm) ui.atm.value = params.atmosphereColor;
  document.querySelectorAll('[data-preset]').forEach(b =>
    b.setAttribute('aria-pressed', String(b.dataset.preset === params.preset)));
  drawMap();
}

/* ---- the map ----------------------------------------------------------- */
const SCORE = { '.': 0, '-': 1, 'o': 2, '@': 3 };
let mapCanvas, mapCtx;

function buildMap(){
  mapCanvas = document.getElementById('map');
  mapCtx = mapCanvas.getContext('2d');

  const pick = e => {
    const r = mapCanvas.getBoundingClientRect();
    const fx = clamp((e.clientX - r.left) / r.width, 0, 1);
    const fy = clamp((e.clientY - r.top) / r.height, 0, 1);
    setFeedKill(FEED_MAX - fy * (FEED_MAX - FEED_MIN), KILL_MIN + fx * (KILL_MAX - KILL_MIN));
    params.preset = '';
    syncUI();
  };

  let dragging = false;
  mapCanvas.addEventListener('pointerdown', e => { dragging = true; mapCanvas.setPointerCapture(e.pointerId); pick(e); });
  mapCanvas.addEventListener('pointermove', e => { if (dragging) pick(e); });
  mapCanvas.addEventListener('pointerup', () => { dragging = false; });
  mapCanvas.addEventListener('pointercancel', () => { dragging = false; });
}

/**
 * Where a value sits on an axis, as a fractional index.
 *
 * The kill axis is not evenly spaced — it is sampled finely through the
 * interesting middle (0.055, 0.058, 0.060, 0.062) and coarsely at the ends —
 * so treating it as linear puts every reading in the wrong column.
 */
function axisIndex(arr, v){
  if (v <= arr[0]) return 0;
  const last = arr.length - 1;
  if (v >= arr[last]) return last;
  for (let i = 0; i < last; i++){
    if (v <= arr[i + 1]) return i + (v - arr[i]) / (arr[i + 1] - arr[i]);
  }
  return last;
}

/** Bilinear read of the measured grid, so the readout is not stepped. */
function viabilityAt(feed, kill){
  const { feeds, kills, rows } = VIABILITY;
  const fi = axisIndex(feeds, feed);
  const ki = axisIndex(kills, kill);
  const f0 = Math.floor(fi), k0 = Math.floor(ki);
  const f1 = Math.min(f0 + 1, feeds.length - 1), k1 = Math.min(k0 + 1, kills.length - 1);
  const tf = fi - f0, tk = ki - k0;
  const at = (fr, kc) => SCORE[rows[fr][kc]] ?? 0;
  return at(f0, k0) * (1 - tf) * (1 - tk) + at(f1, k0) * tf * (1 - tk)
       + at(f0, k1) * (1 - tf) * tk + at(f1, k1) * tf * tk;
}

function drawMap(){
  if (!mapCtx) return;
  const w = mapCanvas.width, h = mapCanvas.height;
  const img = mapCtx.createImageData(w, h);
  for (let y = 0; y < h; y++){
    const feed = FEED_MAX - (y / (h - 1)) * (FEED_MAX - FEED_MIN);
    for (let x = 0; x < w; x++){
      const kill = KILL_MIN + (x / (w - 1)) * (KILL_MAX - KILL_MIN);
      const v = viabilityAt(feed, kill) / 3;
      const i = (y * w + x) * 4;
      /* Towards the signal green as the pattern gets stronger. */
      img.data[i]     = 17 + v * (201 - 17);
      img.data[i + 1] = 17 + v * (255 - 17);
      img.data[i + 2] = 17 + v * (54 - 17);
      img.data[i + 3] = 255;
    }
  }
  mapCtx.putImageData(img, 0, 0);

  const px = (params.kill - KILL_MIN) / (KILL_MAX - KILL_MIN) * w;
  const py = (FEED_MAX - params.feed) / (FEED_MAX - FEED_MIN) * h;

  mapCtx.strokeStyle = 'rgba(9,9,9,0.6)';
  mapCtx.lineWidth = 2;
  mapCtx.beginPath();
  mapCtx.moveTo(px, 0); mapCtx.lineTo(px, h);
  mapCtx.moveTo(0, py); mapCtx.lineTo(w, py);
  mapCtx.stroke();

  mapCtx.beginPath(); mapCtx.arc(px, py, 7, 0, Math.PI * 2);
  mapCtx.strokeStyle = '#090909'; mapCtx.lineWidth = 3; mapCtx.stroke();
  mapCtx.fillStyle = '#f7f7ef'; mapCtx.fill();

  const v = viabilityAt(params.feed, params.kill);
  document.getElementById('r-feed').textContent = params.feed.toFixed(3);
  document.getElementById('r-kill').textContent = params.kill.toFixed(3);
  const life = document.getElementById('r-life');
  life.textContent = v < 0.6 ? 'barren' : v < 1.4 ? 'sparse' : v < 2.3 ? 'living' : 'teeming';
  life.dataset.state = v < 0.6 ? 'dead' : 'alive';
}

/* ---- wind pad ---------------------------------------------------------- */
function setWind(x, y){
  rdMaterial.uniforms.evolutionDirection.value.set(x * 0.5, -y * 0.5);
  const dot = document.getElementById('pad-dot');
  if (dot){ dot.style.left = `${50 + x * 42}%`; dot.style.top = `${50 + y * 42}%`; }
}

function buildWindPad(){
  const pad = document.getElementById('pad');
  let dragging = false;
  const move = e => {
    if (!dragging) return;
    const r = pad.getBoundingClientRect();
    setWind(
      clamp((e.clientX - r.left - r.width / 2) / (r.width / 2), -1, 1),
      clamp((e.clientY - r.top - r.height / 2) / (r.height / 2), -1, 1));
  };
  /* Pointer events on the pad rather than mouse events on window: the original
     bound window.onmousemove, which clobbered any other handler and left the
     pad completely dead on a touchscreen. */
  pad.addEventListener('pointerdown', e => { dragging = true; pad.setPointerCapture(e.pointerId); move(e); });
  pad.addEventListener('pointermove', move);
  pad.addEventListener('pointerup', () => { dragging = false; });
  pad.addEventListener('pointercancel', () => { dragging = false; });
}

/* ---- export ------------------------------------------------------------ */
function savePNG(){
  composer.render();
  const link = document.createElement('a');
  link.download = `planet-${params.preset || 'custom'}-${Date.now()}.png`;
  link.href = renderer.domElement.toDataURL('image/png');
  link.click();
}

/* ---- loop -------------------------------------------------------------- */
function animate(time){
  requestAnimationFrame(animate);

  if (params.isPlaying){
    planetMesh.rotation.y += params.rotationSpeed;

    /* Drift wanders around whatever feed is set now, not around the preset's.
       The original read the preset every frame, which quietly overwrote the
       feed slider and made it look broken whenever drift was on — which was
       its default. */
    rdMaterial.uniforms.feed.value = params.drift
      ? clamp(params.feed + Math.sin(time * 0.0004) * 0.0022, FEED_MIN, FEED_MAX)
      : params.feed;

    step(STEPS_PER_FRAME);
  }

  displayMaterial.uniforms.tDiffuse.value = rt1.texture;
  controls.update();
  renderer.setRenderTarget(null);
  composer.render();

  frameCount++;
  if (time - fpsAt > 500){
    const out = document.getElementById('h-fps');
    if (out) out.textContent = Math.round(frameCount * 1000 / (time - fpsAt));
    frameCount = 0; fpsAt = time;
  }
}

/* A small surface so the Lab page, or anything embedding this, can drive it
   without reaching into the module. */
/**
 * Describe the live field: how much B there is and how varied it is.
 * `flat` means the pattern has collapsed to one value, which is the failure
 * mode worth catching — it looks like a plain sphere however pretty the
 * colours are.
 */
function stats(){
  const w = 256, h = 128;
  const buf = new Float32Array(w * h * 4);
  renderer.readRenderTargetPixels(rt1, (TEXTURE_WIDTH - w) >> 1, (TEXTURE_HEIGHT - h) >> 1, w, h, buf);
  let n = 0, sum = 0, sumsq = 0, bad = 0;
  for (let i = 0; i < w * h; i++){
    const b = buf[i * 4 + 1];
    if (!Number.isFinite(b)){ bad++; continue; }
    n++; sum += b; sumsq += b * b;
  }
  const mean = sum / n;
  const sd = Math.sqrt(Math.max(0, sumsq / n - mean * mean));
  return { mean: +mean.toFixed(4), sd: +sd.toFixed(4), nonFinite: bad, flat: sd < 0.04 };
}

window.planetMaker = {
  presets: () => Object.keys(PRESETS),
  stats,
  /** Run the simulation forward without waiting on frames. */
  settle: n => step(n),
  diffusion: (a, b) => {
    params.diffA = clamp(a, 0.6, DIFF_A_MAX); params.diffB = clamp(b, 0.2, 0.8);
    rdMaterial.uniforms.diffA.value = params.diffA;
    rdMaterial.uniforms.diffB.value = params.diffB;
    syncUI();
  },
  pause: () => { params.isPlaying = false; },
  apply: applyPreset,
  set: (feed, kill) => { setFeedKill(feed, kill); params.preset = ''; syncUI(); },
  reseed: () => seed(),
  snapshot: savePNG,
  limits: { FEED_MIN, FEED_MAX, KILL_MIN, KILL_MAX, DIFF_A_MAX },
};

init();
animate(0);
