/**
 * Graph Engine Composable
 * Handles the core logic for style management, layout, and rendering.
 */

// Style Manager handles dynamic color generation for node and edge types
export class StyleManager {
  constructor() {
    this.nodeTypeColors = new Map();
    this.edgeTypeColors = new Map();
    this.discoveredNodeTypes = new Set();
    this.discoveredEdgeTypes = new Set();
    this.colorSeed = 0;
  }

  discoverTypes(nodes, edges) {
    this.discoveredNodeTypes.clear();
    this.discoveredEdgeTypes.clear();

    nodes.forEach(node => {
      if (node.label && !node.anchor) {
        this.discoveredNodeTypes.add(node.label);
      }
    });

    edges.forEach(edge => {
      if (edge.label) {
        this.discoveredEdgeTypes.add(edge.label);
      }
    });

    this.generateColorsForTypes();
  }

  generateColorsForTypes() {
    this.colorSeed = 0;
    const sortedNodeTypes = Array.from(this.discoveredNodeTypes).sort();
    sortedNodeTypes.forEach(type => {
      if (!this.nodeTypeColors.has(type)) {
        this.nodeTypeColors.set(type, this.generateLightTransparentColor(0.7));
      }
    });

    const sortedEdgeTypes = Array.from(this.discoveredEdgeTypes).sort();
    sortedEdgeTypes.forEach(type => {
      if (!this.edgeTypeColors.has(type)) {
        this.edgeTypeColors.set(type, this.generateLightTransparentColor(0.6));
      }
    });
  }

  generateLightTransparentColor(alpha = 0.7) {
    const hue = (this.colorSeed * 137.508) % 360;
    const saturation = 50 + (this.colorSeed * 19) % 25;
    const lightness = 65 + (this.colorSeed * 13) % 20;
    this.colorSeed++;
    const rgb = this.hslToRgb(hue / 360, saturation / 100, lightness / 100);
    return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
  }

  hslToRgb(h, s, l) {
    let r, g, b;
    if (s === 0) {
      r = g = b = l;
    } else {
      const hue2rgb = (p, q, t) => {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1 / 6) return p + (q - p) * 6 * t;
        if (t < 1 / 2) return q;
        if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
        return p;
      };
      const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      const p = 2 * l - q;
      r = hue2rgb(p, q, h + 1 / 3);
      g = hue2rgb(p, q, h);
      b = hue2rgb(p, q, h - 1 / 3);
    }
    return { r: Math.round(r * 255), g: Math.round(g * 255), b: Math.round(b * 255) };
  }

  getNodeColor(type) {
    if (!type) type = 'default';
    if (!this.nodeTypeColors.has(type)) {
      this.nodeTypeColors.set(type, this.generateLightTransparentColor(0.7));
      this.discoveredNodeTypes.add(type);
    }
    return this.nodeTypeColors.get(type);
  }

  getEdgeColor(type) {
    if (!type) type = 'default';
    if (!this.edgeTypeColors.has(type)) {
      this.edgeTypeColors.set(type, this.generateLightTransparentColor(0.6));
      this.discoveredEdgeTypes.add(type);
    }
    return this.edgeTypeColors.get(type);
  }
}

// Layout Engine handles the positioning of nodes
export class GraphLayoutEngine {
  constructor() {
    this.minDistance = 460;
    this.iterations = 100;
    this.repulsionStrength = 800;
    this.attractionStrength = 0.05;
    this.damping = 0.9;
  }

  applyForceDirectedLayout(nodes, edges) {
    const nodeMap = new Map();
    nodes.forEach(node => nodeMap.set(node.id, node));

    // Initial setup
    nodes.forEach(node => {
      if (!node.x) node.x = (Math.random() - 0.5) * 500;
      if (!node.y) node.y = (Math.random() - 0.5) * 500;
      node.vx = 0;
      node.vy = 0;
    });

    for (let i = 0; i < this.iterations; i++) {
      // Repulsion
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const u = nodes[i];
          const v = nodes[j];
          const dx = v.x - u.x;
          const dy = v.y - u.y;
          const distSq = dx * dx + dy * dy || 1;
          const dist = Math.sqrt(distSq);

          if (dist < this.minDistance) {
            const force = (this.minDistance - dist) / dist;
            const fx = dx * force * 0.1;
            const fy = dy * force * 0.1;
            u.vx -= fx;
            u.vy -= fy;
            v.vx += fx;
            v.vy += fy;
          }
        }
      }

      // Attraction
      edges.forEach(edge => {
        const u = nodeMap.get(edge.source);
        const v = nodeMap.get(edge.target);
        if (!u || !v) return;

        const dx = v.x - u.x;
        const dy = v.y - u.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = dist * this.attractionStrength;

        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        u.vx += fx;
        u.vy += fy;
        v.vx -= fx;
        v.vy -= fy;
      });

      // Update positions
      nodes.forEach(node => {
        node.x += node.vx * this.damping;
        node.y += node.vy * this.damping;
        node.vx *= 0.5;
        node.vy *= 0.5;
      });
    }
  }
}

// Renderer handles drawing to the canvas
export class GraphRenderer {
  constructor(canvas, styleManager) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.styleManager = styleManager;
    this.pixelRatio = window.devicePixelRatio || 1;
    this.transform = { x: 0, y: 0, k: 1 };
    this.hoveredNode = null;
    this.selectedNode = null;
    this.showLabels = true;
    this.isDarkMode = true;
  }

  setTheme(isDark) {
    this.isDarkMode = isDark;
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = rect.width * this.pixelRatio;
    this.canvas.height = rect.height * this.pixelRatio;
    this.canvas.style.width = `${rect.width}px`;
    this.canvas.style.height = `${rect.height}px`;
    this.ctx.scale(this.pixelRatio, this.pixelRatio);
  }

  render(nodes, edges) {
    if (!this.ctx) return;

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.save();
    this.ctx.translate(this.canvas.width / (2 * this.pixelRatio) + this.transform.x,
      this.canvas.height / (2 * this.pixelRatio) + this.transform.y);
    this.ctx.scale(this.transform.k, this.transform.k);

    // Draw edges
    edges.forEach(edge => this.drawEdge(edge, nodes));

    // Draw nodes
    nodes.forEach(node => this.drawNode(node));

    this.ctx.restore();
  }

  drawNode(node) {
    const isHovered = this.hoveredNode === node.id;
    const isSelected = this.selectedNode === node.id;
    const radius = node.anchor ? 18 : 12;
    const color = node.anchor ? 'rgba(239, 68, 68, 0.8)' : this.styleManager.getNodeColor(node.label);

    this.ctx.beginPath();
    if (node.anchor) {
      this.ctx.rect(node.x - radius, node.y - radius, radius * 2, radius * 2);
    } else {
      this.ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
    }

    this.ctx.fillStyle = color;
    this.ctx.fill();

    this.ctx.strokeStyle = this.isDarkMode ? 'rgba(255, 255, 255, 0.4)' : 'rgba(0, 0, 0, 0.3)';
    this.ctx.lineWidth = (isHovered || isSelected) ? 3 : 1.5;
    this.ctx.stroke();

    if (this.showLabels) {
      this.ctx.fillStyle = this.isDarkMode ? '#e5e7eb' : '#374151';
      this.ctx.font = node.anchor ? 'bold 12px sans-serif' : '10px sans-serif';
      this.ctx.textAlign = 'center';
      this.ctx.fillText(node.label || '', node.x, node.y + radius + 15);
    }
  }

  drawEdge(edge, nodes) {
    const source = nodes.find(n => n.id === edge.source);
    const target = nodes.find(n => n.id === edge.target);
    if (!source || !target) return;

    this.ctx.beginPath();
    this.ctx.moveTo(source.x, source.y);
    this.ctx.lineTo(target.x, target.y);

    this.ctx.strokeStyle = this.styleManager.getEdgeColor(edge.label);
    this.ctx.lineWidth = 2 / this.transform.k;
    this.ctx.stroke();

    // Draw arrow if directed
    if (edge.directed) {
      this.drawArrow(source, target);
    }
  }

  drawArrow(source, target) {
    const headlen = 10;
    const angle = Math.atan2(target.y - source.y, target.x - source.x);

    // Move to target minus radius
    const radius = target.anchor ? 22 : 15;
    const tx = target.x - radius * Math.cos(angle);
    const ty = target.y - radius * Math.sin(angle);

    this.ctx.beginPath();
    this.ctx.moveTo(tx, ty);
    this.ctx.lineTo(tx - headlen * Math.cos(angle - Math.PI / 6), ty - headlen * Math.sin(angle - Math.PI / 6));
    this.ctx.lineTo(tx - headlen * Math.cos(angle + Math.PI / 6), ty - headlen * Math.sin(angle + Math.PI / 6));
    this.ctx.closePath();
    this.ctx.fillStyle = this.ctx.strokeStyle;
    this.ctx.fill();
  }
}
