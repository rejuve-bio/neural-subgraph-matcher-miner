"""
Index HTML generation for pattern visualization browsing.
"""
import os
from typing import Optional


class IndexHTMLGenerator:
    """Generates index.html files for browsing pattern instances."""
    
    def create_pattern_index(self, pattern_key: str, count: int, pattern_dir: str,
                            has_representative: bool = False, has_instances: bool = False,
                            representative_idx: int = -1) -> None:
        """Create an index.html to browse all instances of a pattern with tabs."""
        
        html_content = self._build_html_structure(
            pattern_key, count, has_representative, has_instances, representative_idx
        )
        
        index_path = os.path.join(pattern_dir, "index.html")
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _build_html_structure(self, pattern_key: str, count: int,
                             has_representative: bool, has_instances: bool,
                             representative_idx: int) -> str:
        """Build complete HTML structure."""
        parts = [
            self._build_header(pattern_key, has_instances),
            self._build_tabs(pattern_key, count, has_instances),
            self._build_representative_tab(has_representative),
            self._build_instances_tab(pattern_key, count, has_instances, representative_idx),
            self._build_scripts(has_instances),
        ]
        return ''.join(parts)
    
    def _build_header(self, pattern_key: str, has_instances: bool) -> str:
        """Build HTML header section."""
        return f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{pattern_key} - Pattern Overview</title>
    {self._get_styles(has_instances)}
</head>
<body>
    <div class="header">
        <div class="header-content">
            <h1>{pattern_key}</h1>
            <div class="subtitle">Pattern Discovery and Analysis{' - Representative Only' if not has_instances else ''}</div>
        </div>
        <button id="theme-toggle" title="Toggle Dark/Light Mode">🌙</button>
    </div>
"""
    
    def _build_tabs(self, pattern_key: str, count: int, has_instances: bool) -> str:
        """Build tabs navigation."""
        if not has_instances:
            return ""
        
        return f"""
    <div class="tabs">
        <button class="tab active" onclick="openTab(event, 'representative-tab')">
            📊 Representative Pattern
        </button>
        <button class="tab" onclick="openTab(event, 'instances-tab')">
            📁 All Instances ({count})
        </button>
    </div>
"""
    
    def _build_representative_tab(self, has_representative: bool) -> str:
        """Build representative pattern tab content."""
        content = """
    <div id="representative-tab" class="tab-content active">
        <div class="representative-section">
"""
        
        if has_representative:
            content += """
            <iframe src="representative.html" class="representative-frame"></iframe>
"""
        else:
            content += """
            <div class="no-representative">
                <strong>⚠️ Representative pattern not available</strong>
                <p>No representative visualization was created for this pattern.</p>
            </div>
"""
        
        content += """
        </div>
    </div>
"""
        return content
    
    def _build_instances_tab(self, pattern_key: str, count: int,
                            has_instances: bool, representative_idx: int) -> str:
        """Build instances tab content."""
        if not has_instances:
            return ""
        
        content = f"""
    <div id="instances-tab" class="tab-content">
        <div class="instances-section">
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-label">Total Instances</div>
                    <div class="stat-value">{count}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Pattern Type</div>
                    <div class="stat-value">{pattern_key.split('_')[1] if '_' in pattern_key else 'N/A'}</div>
                </div>
            </div>

            <div class="grid">
"""
        
        for i in range(1, count + 1):
            href = f"instance_{i:04d}.html"
            is_rep = (i - 1) == representative_idx
            
            content += f"""
                <div class="instance-card{' representative' if is_rep else ''}">
                    <a href="{href}" target="_blank">Instance {i}{' (Rep)' if is_rep else ''}</a>
                    <div class="instance-number">#{i:04d}</div>
                </div>
"""
        
        content += """
            </div>
        </div>
    </div>
"""
        return content
    
    def _build_scripts(self, has_instances: bool) -> str:
        """Build JavaScript section."""
        script = """
    <script>
        // Theme Management
        document.addEventListener('DOMContentLoaded', function() {
            const themeToggle = document.getElementById('theme-toggle');
            const html = document.documentElement;

            // Load saved theme from localStorage or default to dark
            const savedTheme = localStorage.getItem('neural-miner-theme') || 'dark';
            if (savedTheme === 'dark') {
                html.classList.add('dark');
                themeToggle.textContent = '🌙';
            } else {
                html.classList.remove('dark');
                themeToggle.textContent = '☀️';
            }

            // Theme toggle event listener
            themeToggle.addEventListener('click', function() {
                const isDark = html.classList.contains('dark');
                const newTheme = isDark ? 'light' : 'dark';

                if (isDark) {
                    html.classList.remove('dark');
                    themeToggle.textContent = '☀️';
                    localStorage.setItem('neural-miner-theme', 'light');
                } else {
                    html.classList.add('dark');
                    themeToggle.textContent = '🌙';
                    localStorage.setItem('neural-miner-theme', 'dark');
                }

                // Notify iframe to update theme
                const iframe = document.querySelector('.representative-frame');
                if (iframe && iframe.contentWindow) {
                    iframe.contentWindow.postMessage({
                        type: 'theme-change',
                        theme: newTheme
                    }, '*');
                }
            });

            // Listen for theme changes from other windows/tabs/iframes
            window.addEventListener('storage', function(e) {
                if (e.key === 'neural-miner-theme' && e.newValue) {
                    const newTheme = e.newValue;
                    if (newTheme === 'dark') {
                        html.classList.add('dark');
                        themeToggle.textContent = '🌙';
                    } else {
                        html.classList.remove('dark');
                        themeToggle.textContent = '☀️';
                    }
                }
            });
"""
        
        if has_instances:
            script += """
            // Tab Switching
            window.openTab = function(evt, tabName) {
                // Hide all tab contents
                const tabContents = document.getElementsByClassName('tab-content');
                for (let i = 0; i < tabContents.length; i++) {
                    tabContents[i].classList.remove('active');
                }

                // Remove active class from all tabs
                const tabs = document.getElementsByClassName('tab');
                for (let i = 0; i < tabs.length; i++) {
                    tabs[i].classList.remove('active');
                }

                // Show the selected tab content and mark tab as active
                document.getElementById(tabName).classList.add('active');
                evt.currentTarget.classList.add('active');
            };
"""
        
        script += """
        });
    </script>
</body>
</html>
"""
        return script
    
    def _get_styles(self, has_instances: bool) -> str:
        """Get CSS styles for the index page."""
        return f"""
    <style>
        :root {{
            /* Light Theme */
            --bg-primary: oklch(0.96 0 0);
            --bg-secondary: oklch(0.97 0 0);
            --border-light: oklch(0.86 0 0);
            --text-primary: oklch(0.32 0 0);
            --text-secondary: oklch(0.51 0 0);
            --card-bg: oklch(0.97 0 0);
            --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --card-border: oklch(0.86 0 0);
            --primary: oklch(0.21 0.006 285.885);
            --primary-foreground: oklch(0.985 0 0);
            --accent: oklch(0.81 0 0);
            --hover-bg: oklch(0.98 0 0);
            --button-bg: white;
            --button-hover: oklch(0.96 0 0);
        }}

        /* Dark Theme */
        .dark {{
            --bg-primary: oklch(0.22 0 0);
            --bg-secondary: oklch(0.24 0 0);
            --border-light: oklch(0.33 0 0);
            --text-primary: oklch(0.89 0 0);
            --text-secondary: oklch(0.6 0 0);
            --card-bg: oklch(0.24 0 0);
            --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
            --card-border: oklch(0.33 0 0);
            --primary: oklch(0.92 0.004 286.32);
            --primary-foreground: oklch(0.21 0.006 285.885);
            --accent: oklch(0.37 0 0);
            --hover-bg: oklch(0.29 0 0);
            --button-bg: oklch(0.31 0 0);
            --button-hover: oklch(0.37 0 0);
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            transition: background-color 0.3s, color 0.3s;
        }}

        .header {{
            background: var(--card-bg);
            padding: 20px;
            border-bottom: 2px solid var(--border-light);
            box-shadow: var(--card-shadow);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .header-content {{
            flex: 1;
        }}

        h1 {{
            color: var(--text-primary);
            font-size: 24px;
            margin-bottom: 8px;
        }}

        .subtitle {{
            color: var(--text-secondary);
            font-size: 14px;
        }}

        #theme-toggle {{
            width: 40px;
            height: 40px;
            border: 1px solid var(--border-light);
            border-radius: 6px;
            background: var(--button-bg);
            color: var(--text-primary);
            cursor: pointer;
            font-size: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
        }}

        #theme-toggle:hover {{
            background: var(--button-hover);
            border-color: var(--text-secondary);
        }}

        /* Tabs */
        .tabs {{
            display: {'flex' if has_instances else 'none'};
            background: var(--card-bg);
            border-bottom: 1px solid var(--border-light);
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: var(--card-shadow);
        }}

        .tab {{
            padding: 16px 24px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
            font-weight: 500;
            color: var(--text-secondary);
            background: none;
            border: none;
            font-size: 15px;
        }}

        .tab:hover {{
            background: var(--hover-bg);
            color: var(--text-primary);
        }}

        .tab.active {{
            color: var(--primary);
            border-bottom-color: var(--primary);
        }}

        /* Tab content */
        .tab-content {{
            display: {'none' if has_instances else 'block'};
            padding: 24px;
        }}

        .tab-content.active {{
            display: block;
        }}

        /* Representative pattern section */
        .representative-section {{
            max-width: 100%;
            margin: 0 auto;
            height: calc(100vh - 140px);
        }}

        .representative-frame {{
            width: 100%;
            height: 100%;
            border: none;
            background: var(--card-bg);
        }}

        /* Instances grid */
        .instances-section {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        .stats {{
            background: var(--card-bg);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 24px;
            box-shadow: var(--card-shadow);
            display: flex;
            gap: 32px;
            align-items: center;
            border: 1px solid var(--border-light);
        }}

        .stat-item {{
            display: flex;
            flex-direction: column;
        }}

        .stat-label {{
            color: var(--text-secondary);
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}

        .stat-value {{
            color: var(--text-primary);
            font-size: 24px;
            font-weight: 700;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 16px;
        }}

        .instance-card {{
            background: var(--card-bg);
            border: 2px solid var(--border-light);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            transition: all 0.2s;
            cursor: pointer;
            box-shadow: var(--card-shadow);
        }}

        .instance-card:hover {{
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            transform: translateY(-4px);
            border-color: var(--primary);
        }}

        .instance-card a {{
            text-decoration: none;
            color: var(--primary);
            font-weight: 600;
            font-size: 16px;
            display: block;
        }}

        .instance-number {{
            color: var(--text-secondary);
            font-size: 13px;
            margin-top: 8px;
            font-family: 'Courier New', monospace;
        }}

        .instance-card.representative {{
            border-color: var(--primary);
            background: var(--accent);
        }}

        /* No representative message */
        .no-representative {{
            background: var(--accent);
            border: 1px solid var(--border-light);
            color: var(--text-primary);
            padding: 16px;
            border-radius: 8px;
            margin: 20px auto;
            max-width: 600px;
            text-align: center;
        }}
    </style>
"""
