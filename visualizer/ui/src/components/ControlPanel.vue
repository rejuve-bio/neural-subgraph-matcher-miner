<template>
  <div class="control-panel">
    <!-- Top Stats Bar -->
    <div class="stats-bar ui-card">
      <div class="title">{{ title }}</div>
      <div class="subtitle">{{ stats }}</div>
    </div>

    <!-- Floating Zoom Controls -->
    <div class="zoom-controls ui-card">
      <button @click="$emit('zoom-in')" title="Zoom In">+</button>
      <button @click="$emit('zoom-out')" title="Zoom Out">−</button>
      <button @click="$emit('recenter')" title="Recenter">⌂</button>
      <button @click="toggleTheme" title="Toggle Theme">
        {{ isDark ? '🌙' : '☀️' }}
      </button>
    </div>

    <!-- Bottom Settings Bar -->
    <div class="settings-card ui-card">
      <div class="toggle-group">
        <label>Show Labels</label>
        <input type="checkbox" :checked="showLabels" @change="$emit('toggle-labels')">
      </div>
      <div class="export-group">
        <button @click="$emit('export-pdf')">PDF</button>
        <button @click="$emit('export-png')">PNG</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue';

const props = defineProps({
  title: String,
  metadata: Object,
  showLabels: Boolean,
  isDark: Boolean
});

const emit = defineEmits(['zoom-in', 'zoom-out', 'recenter', 'toggle-theme', 'toggle-labels', 'export-pdf', 'export-png']);

const stats = computed(() => {
  if (!props.metadata) return 'Loading...';
  return `${props.metadata.nodeCount} nodes • ${props.metadata.edgeCount} edges • ${props.metadata.isDirected ? 'Directed' : 'Undirected'}`;
});

const toggleTheme = () => {
  emit('toggle-theme');
};
</script>

<style scoped>
.control-panel {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 100;
}

.stats-bar {
  position: absolute;
  top: 20px;
  left: 50%;
  transform: translateX(-50%);
  text-align: center;
  pointer-events: auto;
  min-width: 250px;
}

.zoom-controls {
  position: absolute;
  top: 20px;
  right: 20px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  pointer-events: auto;
}

.settings-card {
  position: absolute;
  bottom: 20px;
  right: 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  pointer-events: auto;
  min-width: 180px;
}

.title { font-weight: 600; font-size: 15px; }
.subtitle { font-size: 11px; color: var(--text-secondary); margin-top: 2px; }

button {
  width: 36px;
  height: 36px;
  border: 1px solid var(--border-light);
  border-radius: 8px;
  background: var(--button-bg);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  pointer-events: auto;
}

button:hover { background: var(--button-hover); }

.toggle-group {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
}

.export-group {
  display: flex;
  gap: 8px;
}

.export-group button {
  flex: 1;
  font-size: 12px;
  height: 32px;
}
</style>
