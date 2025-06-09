#!/usr/bin/env python3
"""
Master script to generate all visualizations
"""
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def main():
    """Generate all visualizations"""
    print("🎨 Generating Security Analyzer Visualizations...")
    print("=" * 50)
    
    try:
        # Import and run performance dashboard
        print("1️⃣ Creating Performance Dashboard...")
        from performance_dashboard import create_performance_dashboard
        create_performance_dashboard()
        
        print("\n2️⃣ Creating Performance Trends...")
        from performance_trends import create_performance_trends
        create_performance_trends()
        
        print("\n3️⃣ Creating Robustness Analysis...")
        from robustness_analysis import create_robustness_analysis
        create_robustness_analysis()
        
        print("\n🎉 All visualizations created successfully!")
        print(f"📁 Check the outputs folder: {os.path.join(os.path.dirname(__file__), 'outputs')}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages are installed:")
        print("pip install matplotlib seaborn numpy pandas")
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")

if __name__ == "__main__":
    main() 