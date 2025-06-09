import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def create_robustness_analysis():
    """Create detailed robustness analysis charts"""
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Scenario Performance Heatmap
    scenarios = ['Standard', 'Obfuscated', 'Edge Cases', 'Benign Heavy', 'Mixed Complex', 'Realistic']
    metrics = ['Accuracy', 'Attack Recall', 'False Positive Rate', 'Avg Confidence']
    
    data = np.array([
        [100.0, 19.3, 84.3, 100.0, 100.0, 97.6],  # Accuracy
        [100.0, 57.8, 51.9, 100.0, 100.0, 100.0], # Attack Recall
        [0.0, 0.0, 0.0, 0.0, 0.0, 3.4],          # False Positive Rate (inverted for display)
        [70.7, 41.8, 10.8, 68.3, 74.1, 42.7]     # Avg Confidence
    ])
    
    sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn', 
                xticklabels=scenarios, yticklabels=metrics, ax=ax1)
    ax1.set_title('Performance Heatmap Across Scenarios', fontsize=14, fontweight='bold')
    
    # Attack Recall vs Accuracy Scatter
    accuracy_vals = [100.0, 19.3, 84.3, 100.0, 100.0, 97.6]
    recall_vals = [100.0, 57.8, 51.9, 100.0, 100.0, 100.0]
    
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#2ecc71', '#2ecc71', '#3498db']
    
    for i, (acc, rec, scenario) in enumerate(zip(accuracy_vals, recall_vals, scenarios)):
        ax2.scatter(acc, rec, s=200, c=colors[i], alpha=0.7, edgecolors='black', linewidth=1)
        ax2.annotate(scenario, (acc, rec), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Accuracy (%)')
    ax2.set_ylabel('Attack Recall (%)')
    ax2.set_title('Security vs Accuracy Trade-off', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 105)
    ax2.set_ylim(40, 105)
    
    # Response Time Distribution - Fixed deprecation warning
    response_times = [26.6, 32.7, 31.8, 11.5, 33.2, 24.4]
    ax3.boxplot([response_times], tick_labels=['All Scenarios'])  # Fixed: changed 'labels' to 'tick_labels'
    ax3.scatter(np.ones(len(response_times)), response_times, alpha=0.6, s=100)
    
    for i, (scenario, time) in enumerate(zip(scenarios, response_times)):
        ax3.annotate(f'{scenario}: {time}ms', (1, time), 
                    xytext=(10, 0), textcoords='offset points', fontsize=8)
    
    ax3.set_ylabel('Response Time (ms)')
    ax3.set_title('Response Time Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Confidence Score Analysis
    confidence_scores = [70.7, 41.8, 10.8, 68.3, 74.1, 42.7]
    bars = ax4.bar(scenarios, confidence_scores, 
                   color=['#2ecc71' if c > 60 else '#f39c12' if c > 30 else '#e74c3c' for c in confidence_scores],
                   alpha=0.8)
    
    ax4.set_ylabel('Average Confidence (%)')
    ax4.set_title('Confidence Scores by Scenario', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add threshold line
    ax4.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% Threshold')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save with proper path handling
    output_path = os.path.join(output_dir, 'robustness_analysis.png')
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Robustness analysis saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving analysis: {e}")
        # Fallback to current directory
        plt.savefig('robustness_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Saved to current directory: robustness_analysis.png")
    
    plt.show()

if __name__ == "__main__":
    create_robustness_analysis() 