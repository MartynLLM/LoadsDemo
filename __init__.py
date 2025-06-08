"""
Flow Data Analysis Package

A modular package for generating correlated time series with specific statistical properties.

Main Features:
- Generate time series correlated with original data
- Ensure non-negative values
- Precise control over mean, variance, and correlation
- Comprehensive visualization tools
- Batch processing capabilities

Quick Start:
-----------
from flow_analysis import generate_correlated_time_series, create_visualization

# Generate a correlated series
results = generate_correlated_time_series('your_data.csv', target_correlation=0.75)

# Create visualizations
if results:
    create_visualization(results)
"""

# Import main interface functions
from .main_interface import (
    generate_correlated_time_series,
    batch_generate_correlations,
    analyze_existing_series,
    create_visualization,
    main,  # Backward compatibility
    main_enhanced_autocorr,  # Backward compatibility
    print_usage_examples
)

# Import key classes for advanced usage
from .generation_criteria import GenerationCriteria, create_default_criteria
from .statistics import calculate_series_statistics, calculate_autocorrelation
from .visualization import create_visualizer
from .data_io import save_generated_series, create_output_filename

# Version information
__version__ = "1.0.0"
__author__ = "Flow Analysis Team"

# Define what gets imported with "from flow_analysis import *"
__all__ = [
    # Main interface functions
    'generate_correlated_time_series',
    'batch_generate_correlations', 
    'analyze_existing_series',
    'create_visualization',
    'main',
    'main_enhanced_autocorr',
    'print_usage_examples',
    
    # Key classes
    'GenerationCriteria',
    'create_default_criteria',
    
    # Utility functions
    'calculate_series_statistics',
    'calculate_autocorrelation',
    'create_visualizer',
    'save_generated_series',
    'create_output_filename',
]

# Package metadata
DESCRIPTION = "A modular package for generating correlated time series"
LONG_DESCRIPTION = """
Flow Data Analysis Package
=========================

This package provides tools for generating time series that are correlated with 
existing data while maintaining specific statistical properties such as mean, 
variance, and autocorrelation.

Key Features:
- **Correlation Control**: Generate series with precise correlation to original data
- **Statistical Precision**: Achieve 1% precision on mean, variance, and correlation
- **Non-negative Constraint**: Ensure generated values are always >= 0
- **Visualization**: Comprehensive plotting and analysis tools
- **Batch Processing**: Generate multiple series with different parameters
- **Modular Design**: Clean separation of concerns for easy maintenance

Usage Patterns:
--------------

**Basic Usage:**
```python
from flow_analysis import generate_correlated_time_series, create_visualization

# Generate a correlated series
results = generate_correlated_time_series('data.csv', target_correlation=0.75)

# Visualize results
if results:
    create_visualization(results)
```

**Batch Processing:**
```python
from flow_analysis import batch_generate_correlations

# Generate multiple series
correlations = [0.9, 0.75, 0.5, 0.0, -0.3]
results_list = batch_generate_correlations('data.csv', correlations)
```

**Advanced Configuration:**
```python
from flow_analysis import generate_correlated_time_series, GenerationCriteria

# Custom criteria
criteria = GenerationCriteria(
    target_precision=0.005,  # 0.5% precision
    ensure_nonnegative=True,
    max_iterations=300
)

results = generate_correlated_time_series(
    'data.csv', 
    target_correlation=0.8,
    require_1pct_precision=True
)
```

Module Structure:
----------------
- `main_interface`: High-level user functions
- `optimization_engine`: Core optimization algorithms  
- `series_generators`: Time series generation methods
- `statistics`: Statistical calculations and analysis
- `generation_criteria`: Validation and criteria management
- `visualization`: Plotting and visualization tools
- `data_io`: File input/output operations
"""

# Configuration defaults
DEFAULT_CONFIG = {
    'target_precision': 0.01,  # 1% precision
    'max_iterations': 200,
    'max_attempts': 5,
    'ensure_nonnegative': True,
    'correlation_focused': True,
}

def get_version():
    """Get package version."""
    return __version__

def get_config():
    """Get default configuration."""
    return DEFAULT_CONFIG.copy()

def print_package_info():
    """Print package information and quick start guide."""
    print("="*70)
    print(f"Flow Data Analysis Package v{__version__}")
    print("="*70)
    print(DESCRIPTION)
    print()
    print("Quick Start:")
    print("-" * 20)
    print("from flow_analysis import generate_correlated_time_series")
    print("results = generate_correlated_time_series('your_data.csv', 0.75)")
    print()
    print("For more examples, run:")
    print("from flow_analysis import print_usage_examples")
    print("print_usage_examples()")
    print("="*70)