# Flow Data Analysis Package

A modular Python package for generating time series that are correlated with existing data while maintaining specific statistical properties.

## Features

- **Correlation Control**: Generate series with precise correlation to original data
- **Statistical Precision**: Achieve 1% precision on mean, variance, and correlation
- **Non-negative Constraint**: Ensure generated values are always ≥ 0
- **Visualization**: Comprehensive plotting and analysis tools
- **Batch Processing**: Generate multiple series with different parameters
- **Modular Design**: Clean separation of concerns for easy maintenance

## Package Structure

```
flow_analysis/
├── __init__.py                 # Package initialization and main exports
├── main_interface.py          # High-level user interface functions
├── optimization_engine.py     # Core optimization algorithms
├── series_generators.py       # Time series generation methods
├── statistics.py              # Statistical calculations and analysis
├── generation_criteria.py     # Validation and criteria management
├── visualization.py           # Plotting and visualization tools
├── data_io.py                 # File input/output operations
├── example_usage.py           # Example usage script
└── README.md                  # This documentation
```

## Quick Start

### Basic Usage

```python
from flow_analysis import generate_correlated_time_series, create_visualization

# Generate a correlated series
results = generate_correlated_time_series(
    csv_file_path='your_data.csv',
    target_correlation=0.75,
    random_seed=42
)

# Create visualizations
if results:
    create_visualization(results)
    print(f"Achieved correlation: {results['generated_stats']['correlation_with_original']:.4f}")
```

### Batch Processing

```python
from flow_analysis import batch_generate_correlations

# Generate multiple series with different correlations
correlations = [0.9, 0.75, 0.5, 0.0, -0.3]
results_list = batch_generate_correlations('your_data.csv', correlations)

# Results include error analysis plots
```

### Analysis Only

```python
from flow_analysis import analyze_existing_series

# Analyze existing time series
analysis = analyze_existing_series('your_data.csv')
if analysis:
    print(f"Mean: {analysis['mean']:.4f}")
    print(f"Autocorrelation: {analysis['first_order_autocorrelation']:.4f}")
```

## Key Functions

### `generate_correlated_time_series()`

Main function for generating correlated time series.

**Parameters:**
- `csv_file_path` (str): Path to CSV file with 'Date' and 'Flow' columns
- `target_correlation` (float): Desired correlation with original series (-1 to 1)
- `random_seed` (int): Random seed for reproducibility
- `require_1pct_precision` (bool): Retry until 1% precision achieved on required metrics
- `correlation_focused` (bool): Prioritize correlation over autocorrelation
- `ensure_nonnegative` (bool): Ensure generated values are ≥ 0
- `max_attempts` (int): Maximum attempts to achieve precision

**Returns:**
- Dictionary containing original data, generated series, and statistics

### `batch_generate_correlations()`

Generate multiple series with different correlation targets.

### `analyze_existing_series()`

Analyze statistical properties of existing time series.

### `create_visualization()`

Create comprehensive visualizations comparing original and generated series.

## Input Data Format

The package expects CSV files with the following columns:

- **Date** (optional): Date in formats like DD/MM/YYYY, DD-MM-YYYY, etc.
- **Flow** (required): Numeric flow values

Example CSV:
```csv
Date,Flow
01/01/2020,45.2
02/01/2020,48.7
03/01/2020,43.1
...
```

## Output Files

The package automatically generates:

1. **CSV files**: Generated series with original data for comparison
   - Format: `generated_corr_focused_corr_0p75_seed_42_1pct.csv`

2. **PNG plots**: Comprehensive visualization plots
   - Time series comparison
   - Scatter plots with correlation analysis
   - Statistical summary panels

## Precision Guarantees

The package focuses on achieving **1% precision** on:

- **Mean**: Target vs achieved mean
- **Variance**: Target vs achieved variance  
- **Correlation**: Target vs achieved correlation with original series

**Autocorrelation** is reported for information but not required to meet 1% precision in correlation-focused mode.

## Advanced Configuration

```python
from flow_analysis import generate_correlated_time_series, GenerationCriteria

# Custom generation criteria
results = generate_correlated_time_series(
    'data.csv',
    target_correlation=0.8,
    correlation_focused=False,  # Prioritize autocorrelation
    require_1pct_precision=True,
    max_attempts=3
)
```

## Module Responsibilities

### Core Logic Separation

1. **Application Logic** (`optimization_engine.py`, `series_generators.py`)
   - Time series generation algorithms
   - Optimization and blending strategies
   - Core mathematical operations

2. **Calculation Methods** (`statistics.py`)
   - Statistical property calculations
   - Error percentage computations
   - Autocorrelation and correlation analysis

3. **Criteria and Validation** (`generation_criteria.py`)
   - Acceptance criteria for simulations
   - Precision requirements and validation
   - Error tolerance specifications

4. **User Interface** (`main_interface.py`)
   - High-level function interfaces
   - Parameter validation and defaults
   - Result formatting and output

5. **Generated Outputs** (`visualization.py`, `data_io.py`)
   - Plot generation and customization
   - File I/O operations
   - Output filename conventions

## Dependencies

### Required:
- `numpy`: Numerical computations
- `csv`: CSV file handling (standard library)
- `datetime`: Date parsing (standard library)

### Optional:
- `matplotlib`: Plotting and visualization
- `scipy`: Advanced statistical functions (fallbacks provided)

## Installation

1. **Copy the package files** to your working directory
2. **Install dependencies**:
   ```bash
   pip install numpy matplotlib scipy
   ```
3. **Test the installation**:
   ```python
   python example_usage.py
   ```

## Examples

Run the example script to see all functionality:

```bash
python example_usage.py
```

This will demonstrate:
1. Basic usage with single correlation
2. Batch processing multiple correlations
3. Analysis of existing time series
4. Custom configuration options
5. Autocorrelation-focused generation

## Error Handling

The package includes comprehensive error handling:

- **File not found**: Clear error messages with suggestions
- **Invalid data**: Automatic skipping of NaN values and invalid rows
- **Precision failures**: Automatic retries with different random seeds
- **Missing dependencies**: Graceful fallbacks when optional packages unavailable

## Performance Considerations

- **Memory usage**: Efficient numpy array operations
- **Computation time**: Configurable iteration limits
- **File I/O**: Optimized CSV reading with progress indicators
- **Plotting**: Optional matplotlib dependency for headless environments

## Contributing

The modular structure makes it easy to:

- Add new generation algorithms in `series_generators.py`
- Implement additional statistical measures in `statistics.py`
- Extend validation criteria in `generation_criteria.py`
- Create new visualization types in `visualization.py`

## License

This package is provided as-is for research and analysis purposes.

## Support

For issues or questions:

1. Check the example usage script
2. Review the function docstrings
3. Run the built-in help: `print_usage_examples()`

---

**Note**: This refactored version separates the original monolithic code into focused modules while maintaining all functionality and improving maintainability.