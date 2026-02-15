<template>
  <div class="canvas-container" ref="container">
    <canvas ref="canvasEl" 
      @mousedown="startPan" 
      @mousemove="onMouseMove" 
      @wheel.prevent="onWheel"
    ></canvas>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue';
import { StyleManager, GraphRenderer, GraphLayoutEngine } from '../composables/useGraphEngine';

const props = defineProps({
  data: { type: Object, required: true },
  isDark: Boolean,
  showLabels: Boolean
});

const container = ref(null);
const canvasEl = ref(null);
let renderer = null;
let layoutEngine = null;
let styleManager = null;
let isPanning = false;
let lastMousePos = { x: 0, y: 0 };

onMounted(() => {
  styleManager = new StyleManager();
  renderer = new GraphRenderer(canvasEl.value, styleManager);
  layoutEngine = new GraphLayoutEngine();

  styleManager.discoverTypes(props.data.nodes, props.data.edges);
  layoutEngine.applyForceDirectedLayout(props.data.nodes, props.data.edges);
  
  window.addEventListener('resize', handleResize);
  handleResize();
  animate();
});

onUnmounted(() => {
  window.removeEventListener('resize', handleResize);
});

const handleResize = () => {
  renderer.resize();
  draw();
};

const draw = () => {
  if (renderer) {
    renderer.setTheme(props.isDark);
    renderer.showLabels = props.showLabels;
    renderer.render(props.data.nodes, props.data.edges);
  }
};

const animate = () => {
  draw();
  requestAnimationFrame(animate);
};

// Event Handlers
const startPan = (e) => {
  isPanning = true;
  lastMousePos = { x: e.clientX, y: e.clientY };
  window.addEventListener('mouseup', stopPan);
};

const stopPan = () => {
  isPanning = false;
  window.removeEventListener('mouseup', stopPan);
};

const onMouseMove = (e) => {
  if (isPanning) {
    const dx = e.clientX - lastMousePos.x;
    const dy = e.clientY - lastMousePos.y;
    renderer.transform.x += dx / renderer.transform.k;
    renderer.transform.y += dy / renderer.transform.k;
    lastMousePos = { x: e.clientX, y: e.clientY };
  }
};

const onWheel = (e) => {
  const delta = e.deltaY;
  const factor = delta > 0 ? 0.9 : 1.1;
  renderer.transform.k *= factor;
};

// External Controls
defineExpose({
  zoom: (factor) => renderer.transform.k *= factor,
  recenter: () => {
    renderer.transform = { x: 0, y: 0, k: 1 };
  }
});
</script>

<style scoped>
.canvas-container {
  width: 100%;
  height: 100%;
  background: var(--bg-main);
  overflow: hidden;
}

canvas {
  display: block;
  cursor: grab;
}

canvas:active {
  cursor: grabbing;
}
</style>
