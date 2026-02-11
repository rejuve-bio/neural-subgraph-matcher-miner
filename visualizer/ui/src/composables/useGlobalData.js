/**
 * Global Data Bridge Composable
 * Connects the Python-injected global data to the Vue reactive state.
 */
import { reactive, onMounted } from 'vue';

const sampleData = {
  metadata: { title: "Visualization Loading", nodeCount: 0, edgeCount: 0, isDirected: true, density: 0 },
  nodes: [],
  edges: [],
  legend: { nodeTypes: [], edgeTypes: [] }
};

export function useGlobalData() {
  const state = reactive({
    graphData: sampleData,
    isLoaded: false
  });

  onMounted(() => {
    // Check if GRAPH_DATA exists in the global scope 
    if (window.GRAPH_DATA) {
      state.graphData = window.GRAPH_DATA;
      state.isLoaded = true;
      console.log("Graph data loaded from global scope", state.graphData);
    } else {
      console.warn("GRAPH_DATA not found in window. Are you running in development mode?");
    }
  });

  return { state };
}
