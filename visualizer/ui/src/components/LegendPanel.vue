<template>
  <div class="legend-panel ui-card">
    <h3>Legend</h3>
    <div v-if="nodeTypes.length" class="legend-section">
      <h4>Node Types</h4>
      <div v-for="type in nodeTypes" :key="type.label" class="legend-item">
        <div class="legend-color" :style="{ backgroundColor: type.color }"></div>
        <span>{{ type.label }}</span>
      </div>
    </div>

    <div v-if="edgeTypes.length" class="legend-section">
      <h4>Edge Types</h4>
      <div v-for="type in edgeTypes" :key="type.label" class="legend-item">
        <div class="legend-color" :style="{ backgroundColor: type.color }"></div>
        <span>{{ type.label }}</span>
      </div>
    </div>

    <div class="legend-section">
      <h4>Special Nodes</h4>
      <div class="legend-item">
        <div class="legend-shape anchor-box"></div>
        <span>Anchor Nodes</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue';

const props = defineProps({
  legend: {
    type: Object,
    required: true
  }
});

const nodeTypes = computed(() => props.legend?.nodeTypes || []);
const edgeTypes = computed(() => props.legend?.edgeTypes || []);
</script>

<style scoped>
.legend-panel {
  min-width: 200px;
  max-width: 280px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

h3 {
  font-size: 14px;
  font-weight: 600;
  border-bottom: 1px solid var(--border-light);
  padding-bottom: 8px;
  margin: 0;
}

.legend-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

h4 {
  font-size: 11px;
  text-transform: uppercase;
  color: var(--text-secondary);
  letter-spacing: 0.5px;
  margin: 0;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 13px;
}

.legend-color {
  width: 14px;
  height: 14px;
  border-radius: 3px;
  border: 1px solid rgba(0,0,0,0.1);
}

.anchor-box {
  width: 14px;
  height: 14px;
  background-color: rgba(239, 68, 68, 0.8);
  border-radius: 2px;
  border: 1px solid rgba(0,0,0,0.2);
}
</style>
