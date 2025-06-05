"""
Security Analyzer Package

A comprehensive, production-ready HTTP request security analysis system 
that uses multi-layered machine learning techniques to detect and classify 
malicious requests in real-time.

Modules:
- core: Main SecurityAnalyzer implementation
- tests: Comprehensive test suite  
- evaluation: Performance evaluation and analysis tools
- visualizations: Chart and plot generation
- models: Saved model files and persistence
- data: Test data and analysis results
"""

from .core.security_analyzer import SecurityAnalyzer

__version__ = "1.0.0"
__author__ = "Security Analysis Team"

__all__ = ["SecurityAnalyzer"] 