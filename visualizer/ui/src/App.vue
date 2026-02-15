<template>
  <div :class="['app-wrapper', isDark ? 'dark' : 'light']">
    <!-- Main rendering surface -->
    <GraphCanvas 
      v-if="state.isLoaded"
      ref="graphRef"
      :data="state.graphData" 
      :is-dark="isDark"
      :show-labels="showLabels"
    />

    <!-- UI Overlay Components -->
    <div class="ui-overlay">
      <LegendPanel 
        v-if="state.isLoaded"
        :legend="state.graphData.legend" 
      />
      
      <ControlPanel 
        :title="state.graphData.metadata?.title"
        :metadata="state.graphData.metadata"
        :is-dark="isDark"
        :show-labels="showLabels"
        @zoom-in="graphRef?.zoom(1.2)"
        @zoom-out="graphRef?.zoom(0.8)"
        @recenter="graphRef?.recenter()"
        @toggle-theme="isDark = !isDark"
        @toggle-labels="showLabels = !showLabels"
      />
    </div>

    <!-- Loading State -->
    <div v-if="!state.isLoaded" class="loading-screen">
      <div class="loader"></div>
      <p>Loading Graph Visualizer...</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { useGlobalData } from './composables/useGlobalData';
import GraphCanvas from './components/GraphCanvas.vue';
import LegendPanel from './components/LegendPanel.vue';
import ControlPanel from './components/ControlPanel.vue';

const { state } = useGlobalData();
const graphRef = ref(null);
const isDark = ref(true);
const showLabels = ref(true);
</script>

<style>
/* Global CSS Variables for Themes */
:root {
  --bg-main: #ffffff;
  --text-primary: #1f2937;
  --text-secondary: #6b7280;
  --card-bg: rgba(255, 255, 255, 0.9);
  --border-light: rgba(0, 0, 0, 0.1);
  --button-bg: #ffffff;
  --button-hover: #f3f4f6;
}

.dark {
  --bg-main: #0f172a;
  --text-primary: #f8fafc;
  --text-secondary: #94a3b8;
  --card-bg: rgba(30, 41, 59, 0.8);
  --border-light: rgba(255, 255, 255, 0.1);
  --button-bg: #1e293b;
  --button-hover: #334155;
}

.app-wrapper {
  width: 100vw;
  height: 100vh;
  margin: 0;
  padding: 0;
  overflow: hidden;
  font-family: 'Inter', -apple-system, sans-serif;
  background: var(--bg-main);
  color: var(--text-primary);
}

.ui-overlay {
  position: absolute;
  top: 20px;
  left: 20px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  pointer-events: none;
}

.ui-overlay > * {
  pointer-events: auto;
}

.ui-card {
  background: var(--card-bg);
  backdrop-filter: blur(8px);
  border: 1px solid var(--border-light);
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.loading-screen {
  position: fixed;
  inset: 0;
  background: #0f172a;
  color: white;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.loader {
  width: 40px;
  height: 40px;
  border: 3px solid rgba(255,255,255,0.1);
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 16px;
}

@keyframes spin { to { transform: rotate(360deg); } }
</style>
