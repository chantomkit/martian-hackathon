#!/usr/bin/env python3
"""
Create a static HTML report with all MI visualizations
Professional report for Mechanistically Interpretable Micro Judges
"""

import json
from pathlib import Path
from datetime import datetime
import base64
import re


def create_static_report(viz_dir: Path = Path("outputs/checkpoints/training_visualizations"),
                        output_file: Path = Path("mi_report.html")):
    """Create a single HTML file with all visualizations embedded"""
    
    print("Creating static MI report...")
    
    # Find all visualization files
    viz_files = {
        'Token Attribution': sorted(viz_dir.glob("*tokens*.html")),
        'Layer Activations': sorted(viz_dir.glob("*layers*.html")),
        'Circuit Comparisons': sorted(viz_dir.glob("*comparison*.html")),
        'Neuron Maps': sorted(viz_dir.glob("*neuron_map*.html")),
        'Circuit Diagrams': sorted(viz_dir.glob("*circuits*.html")),
        'Training Metrics': sorted(viz_dir.glob("*progress*.html"))
    }
    
    # Load evaluation results if available
    eval_results = {}
    eval_file = Path("outputs/evaluation/evaluation_results.json")
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            eval_results = json.load(f)
    
    # Create HTML report with professional styling
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MI Micro Judges Report</title>
    <style>
        /* Professional Nord theme styling */
        :root {{
            --brand-primary: #2E3440;
            --brand-secondary: #4C566A;
            --brand-accent: #5E81AC;
            --brand-highlight: #88C0D0;
            --brand-success: #A3BE8C;
            --brand-warning: #EBCB8B;
            --brand-danger: #BF616A;
            --bg-primary: #ECEFF4;
            --bg-secondary: #E5E9F0;
            --text-primary: #2E3440;
            --text-secondary: #4C566A;
            --border: #D8DEE9;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        .header {{
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(135deg, #2E3440 0%, #3B4252 100%);
            color: white;
            margin: -2rem -2rem 3rem -2rem;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            font-weight: 300;
            letter-spacing: 2px;
            margin-bottom: 0.5rem;
        }}
        
        .header h2 {{
            font-size: 2rem;
            font-weight: 600;
            color: var(--brand-highlight);
            margin-bottom: 1rem;
        }}
        
        .header p {{
            font-size: 1rem;
            color: #D8DEE9;
            margin-bottom: 0.5rem;
        }}
        
        .header .timestamp {{
            font-size: 0.875rem;
            opacity: 0.8;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }}
        
        .metric-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid var(--border);
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--brand-accent);
            margin: 0.5rem 0;
        }}
        
        .metric-label {{
            color: var(--text-secondary);
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 500;
        }}
        
        .viz-section {{
            background: white;
            padding: 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        .viz-section h2 {{
            font-size: 1.75rem;
            margin-bottom: 1rem;
            color: var(--text-primary);
            font-weight: 600;
        }}
        
        .info-box {{
            background-color: var(--bg-secondary);
            border-left: 4px solid var(--brand-accent);
            padding: 1rem;
            margin-bottom: 1.5rem;
            border-radius: 4px;
            font-size: 0.95rem;
            color: var(--text-secondary);
        }}
        
        .viz-tabs {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
            border-bottom: 2px solid var(--border);
            padding-bottom: 0;
        }}
        
        .tab-btn {{
            padding: 0.75rem 1.5rem;
            background: transparent;
            border: none;
            border-radius: 4px 4px 0 0;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.9rem;
            font-weight: 500;
            margin-bottom: -2px;
        }}
        
        .tab-btn:hover {{
            color: var(--text-primary);
            background: var(--bg-secondary);
        }}
        
        .tab-btn.active {{
            background: var(--brand-accent);
            color: white;
            border-bottom: 2px solid var(--brand-accent);
        }}
        
        .viz-container {{
            background: var(--bg-secondary);
            border-radius: 4px;
            padding: 1rem;
            min-height: 600px;
            overflow: auto;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .footer {{
            text-align: center;
            padding: 3rem 0 2rem;
            color: var(--text-secondary);
            border-top: 1px solid var(--border);
            margin-top: 4rem;
        }}
        
        .no-viz {{
            text-align: center;
            color: var(--text-secondary);
            padding: 3rem;
        }}
        
        .footer p {{
            margin: 0.25rem 0;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 1rem;
            }}
            
            .header h1 {{
                font-size: 1.75rem;
            }}
            
            .header h2 {{
                font-size: 1.5rem;
            }}
            
            .metrics-grid {{
                grid-template-columns: repeat(2, 1fr);
                gap: 1rem;
            }}
        }}
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>MECHANISTICALLY INTERPRETABLE</h1>
            <h2>MICRO JUDGES</h2>
            <p>AI Safety Analysis Report</p>
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
"""
    
    # Add metrics if available
    if eval_results:
        html_content += """
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{:.1%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">F1 Score</div>
                <div class="metric-value">{:.1%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">AUC-ROC</div>
                <div class="metric-value">{:.1%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">False Positive Rate</div>
                <div class="metric-value">{:.1%}</div>
            </div>
        </div>
        """.format(
            eval_results.get('accuracy', 0),
            eval_results.get('f1', 0),
            eval_results.get('auc', 0),
            eval_results.get('false_positive_rate', 0)
        )
    
    # Add visualization sections
    section_id = 0
    info_texts = {
        'Token Attribution': 'Shows which tokens (words/subwords) contribute most to safety classification. Red indicates "unsafe" signals, blue indicates "safe" signals.',
        'Layer Activations': 'Visualizes activation magnitude across transformer layers. Higher activations indicate layers more involved in safety detection.',
        'Circuit Comparisons': 'Compares how the model processes safe versus unsafe prompts, showing divergence patterns across layers.',
        'Neuron Maps': 'Interactive plots showing specific neurons most active in detecting unsafe content. Larger points indicate stronger safety signals.',
        'Circuit Diagrams': 'Network diagrams showing computational pathways involved in safety detection. Nodes represent neurons, edges show information flow.',
        'Training Metrics': 'Model performance metrics over training epochs including accuracy, F1 score, and loss curves.'
    }
    
    for section_name, files in viz_files.items():
        if not files:
            continue
            
        section_id += 1
        html_content += f"""
        <div class="viz-section">
            <h2>{section_name}</h2>
            <div class="info-box">
                {info_texts.get(section_name, '')}
            </div>
            <div class="viz-tabs">
        """
        
        # Create tabs for each file
        for i, file in enumerate(files):
            active_class = "active" if i == 0 else ""
            display_name = file.stem.replace('_', ' ').title()
            html_content += f"""
                <button class="tab-btn {active_class}" onclick="showTab('section{section_id}', 'tab{section_id}_{i}', this)">
                    {display_name}
                </button>
            """
        
        html_content += """
            </div>
            <div class="viz-container">
        """
        
        # Add tab contents
        for i, file in enumerate(files):
            active_class = "active" if i == 0 else ""
            
            # Load and embed visualization
            try:
                with open(file, 'r') as f:
                    viz_html = f.read()
                
                # Extract just the Plotly content
                plot_match = re.search(
                    r'<div id="([^"]+)".*?</div>.*?<script type="text/javascript">(.*?)</script>',
                    viz_html,
                    re.DOTALL
                )
                
                if plot_match:
                    plot_id = f"plot_{section_id}_{i}"
                    plot_script = plot_match.group(2).replace(plot_match.group(1), plot_id)
                    
                    html_content += f"""
                    <div id="tab{section_id}_{i}" class="tab-content {active_class}">
                        <div id="{plot_id}" style="width:100%;height:600px;"></div>
                        <script type="text/javascript">
                            {plot_script}
                        </script>
                    </div>
                    """
                else:
                    html_content += f"""
                    <div id="tab{section_id}_{i}" class="tab-content {active_class}">
                        {viz_html}
                    </div>
                    """
            except Exception as e:
                html_content += f"""
                <div id="tab{section_id}_{i}" class="tab-content {active_class}">
                    <div class="no-viz">Error loading visualization: {str(e)}</div>
                </div>
                """
        
        html_content += """
            </div>
        </div>
        """
    
    # Add footer and scripts
    html_content += """
        <div class="footer">
            <p><strong>Mechanistically Interpretable Micro Judges</strong></p>
            <p>Built for AI Safety Research | Guardian-Loop Project</p>
            <p style="font-size: 0.875rem; margin-top: 1rem;">&copy; 2024 All rights reserved</p>
        </div>
    </div>
    
    <script>
        function showTab(sectionId, tabId, btn) {
            // Hide all tabs in this section
            const section = btn.closest('.viz-section');
            section.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            section.querySelectorAll('.tab-btn').forEach(button => {
                button.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabId).classList.add('active');
            btn.classList.add('active');
        }
    </script>
</body>
</html>
"""
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Report created successfully: {output_file}")
    print(f"Total visualizations embedded: {sum(len(files) for files in viz_files.values())}")
    print(f"\nTo view the report:")
    print(f"  1. Open {output_file} in your browser")
    print(f"  2. Or run: python -m http.server 8000")
    print(f"     Then visit: http://localhost:8000/{output_file.name}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create static MI report")
    parser.add_argument('--viz-dir', type=str, 
                       default='outputs/checkpoints/training_visualizations',
                       help='Directory containing visualizations')
    parser.add_argument('--output', type=str, default='mi_report.html',
                       help='Output HTML file')
    
    args = parser.parse_args()
    
    create_static_report(Path(args.viz_dir), Path(args.output)) 