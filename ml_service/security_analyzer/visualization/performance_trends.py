import matplotlib.pyplot as plt
import numpy as np
import os

def create_performance_trends():
    """Create performance trends and improvement analysis"""
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # Simulated improvement over time (based on your described improvements)
    iterations = ['Initial', 'After Pattern Fix', 'After ML Tuning', 'Final Optimization']
    overall_accuracy = [25.0, 54.3, 83.4, 83.5]
    false_positive_rate = [46.5, 36.1, 0.0, 0.56]
    robustness_score = [13.1, 54.6, 80.8, 82.6]
    test_success_rate = [66.7, 71.4, 100.0, 100.0]
    
    # Overall Accuracy Improvement
    ax1.plot(iterations, overall_accuracy, marker='o', linewidth=3, markersize=8, color='#2ecc71')
    ax1.fill_between(iterations, overall_accuracy, alpha=0.3, color='#2ecc71')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Overall Accuracy Improvement', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # False Positive Rate Reduction
    ax2.plot(iterations, false_positive_rate, marker='s', linewidth=3, markersize=8, color='#e74c3c')
    ax2.fill_between(iterations, false_positive_rate, alpha=0.3, color='#e74c3c')
    ax2.set_ylabel('False Positive Rate (%)')
    ax2.set_title('False Positive Rate Reduction', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Multi-metric Progress
    x = np.arange(len(iterations))
    width = 0.2
    
    bars1 = ax3.bar(x - width, overall_accuracy, width, label='Accuracy', color='#3498db', alpha=0.8)
    bars2 = ax3.bar(x, robustness_score, width, label='Robustness', color='#9b59b6', alpha=0.8)
    bars3 = ax3.bar(x + width, test_success_rate, width, label='Test Success', color='#2ecc71', alpha=0.8)
    
    ax3.set_ylabel('Score (%)')
    ax3.set_title('Multi-Metric Progress', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(iterations, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Final Grade Comparison
    grade_categories = ['Security\n(Attack Recall)', 'Accuracy\n(Overall)', 'Performance\n(Speed)', 'Reliability\n(Tests)']
    scores = [84.9, 83.5, 95.0, 100.0]  # Performance score based on speed
    
    colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']
    bars = ax4.bar(grade_categories, scores, color=colors, alpha=0.8)
    
    ax4.set_ylabel('Score (%)')
    ax4.set_title('Final Grade Breakdown', fontsize=14, fontweight='bold')
    ax4.axhline(y=90, color='green', linestyle='--', alpha=0.7, label='A Grade (90%)')
    ax4.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='B Grade (80%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{score}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save with proper path handling
    output_path = os.path.join(output_dir, 'performance_trends.png')
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Performance trends saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving trends: {e}")
        # Fallback to current directory
        plt.savefig('performance_trends.png', dpi=300, bbox_inches='tight')
        print("✅ Saved to current directory: performance_trends.png")
    
    plt.show()

if __name__ == "__main__":
    create_performance_trends() 