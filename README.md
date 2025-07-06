# Loads Demo Repository Documentation

## Repository Overview

The **Loads Demo** repository is a Python-based system for assessing the performance of loads statistics by simulating streamflow and chemistry time series with arbitrary correlations and variance relationships. It's designed for environmental monitoring and water quality assessment research.

## Repository Structure

### Core Applications

#### 1. `loads.py` - Main Time Series Generator
**Primary Purpose**: Combined Flow and Chemistry Time Series Generator
- **Class**: `CombinedFlowChemistryGenerator`
- **Key Features**:
  - Generates synthetic flow and chemistry data with configurable correlations
  - Supports positive and negative correlations (-1.0 to 1.0)
  - Ensures non-negative time series generation
  - Creates flow:chemistry tuples and calculates load statistics
  - Uses vectorized operations for performance optimization
  - Stores results in normalized SQLite database

#### 2. `cq_dual_generator.py` - Enhanced Generator
**Enhanced Version** of `loads.py` with additional features:
- **Correlation Refinement**: Iterative improvement for chemistry correlations
- **Post-generation Adjustment**: Fine-tuning correlation accuracy
- **Enhanced Views**: Additional database views including `scenario_summary`
- **Better Chemistry Handling**: Specialized correlation refinement for chemistry data

#### 3. `scenario_viewer.py` - Interactive Visualization Tool
**GUI Application** for exploring generated scenarios:
- **Framework**: Tkinter with matplotlib integration
- **Purpose**: Interactive visualization of load correlation vs chemistry correlation
- **Features**:
  - Factor selection interface (4 of 5 factors)
  - Dynamic plotting with series grouping
  - Database connectivity to scenario_summary view
  - Plot export capabilities

### Configuration Files

#### `CQ_generator_parameters.json` - Main Configuration
```json
{
    "flow": {
        "input_csv_file": "Flows.csv",
        "mean_percentages": [60, 80, 100, 120, 140],
        "variance_percentages": [60, 80, 100, 120, 140], 
        "correlations": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    },
    "chemistry": {
        "input_csv_file": "Chemistry.csv",
        "correlations": [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]
    },
    "output_database_file": "LoadsDemo.db"
}
```

#### `generator_parameters.json` - Single Dataset Configuration
- Simplified configuration for single time series generation
- Uses `SavjaForClaude.csv` as input
- Limited to positive correlations only

### Sample Data

#### `sampleData/shortFlows.csv`
- **Period**: January 1990 - December 1999 (3,653 daily records)
- **Format**: Date (dd/mm/yyyy), Flow (m³/s)
- **Range**: 0.06 to 30.96 m³/s
- **Characteristics**: Seasonal patterns with winter peaks, summer lows

#### `sampleData/shortChemistry.csv`
- **Period**: January 1990 - December 1999 (120 monthly records)
- **Format**: Date (dd/mm/yyyy), Chemistry (concentration units)
- **Range**: 81 to 3,762 units
- **Characteristics**: Seasonal chemistry variations, higher winter concentrations

## Technical Architecture

### Database Schema Design

**Normalized SQLite Database** with the following structure:

#### Core Tables
- **input_flow** / **input_chem**: Original input time series data
- **flow_combinations** / **chem_combinations**: Parameter combination lookup tables
- **Flows** / **Chem**: Generated time series data (normalized design)
- **flow_metadata** / **chem_metadata**: Generation statistics and validation metrics
- **Loads**: Flow:chemistry tuples with calculated loads
- **loads_statistics**: Comprehensive scenario statistics

#### Database Views
- **Flows_view** / **Chem_view**: Denormalized views combining input and generated data
- **comprehensive_metadata**: Complete scenario information with parameters and statistics
- **scenario_summary**: Streamlined view for analysis and visualization

### Statistical Methods

#### Time Series Generation
1. **Perfect Correlation (r=±1.0)**: Linear/inverse transformation of original data
2. **Zero Correlation (r=0.0)**: Independent random series generation
3. **Partial Correlation**: Correlated normal distribution mixing
4. **Non-negative Constraints**: Automatic shifting and minimum value enforcement

#### Load Calculation
```
Load = 0.0864 × Chemistry × Flow
```
(Conversion factor for daily loads in appropriate units)

#### Performance Metrics
- **Time Series Accuracy**: Mean, variance, correlation error percentages
- **Load Statistics**: Mean, standard deviation, min/max, correlation
- **Regression Analysis**: Linear slope, intercept, R-squared
- **Error Metrics**: RMSE, Mean Absolute Error

### Performance Optimizations

#### Computational Efficiency
- **Vectorized Operations**: NumPy array operations for load calculations
- **Batch Processing**: Chunked time series generation
- **Memory Management**: Progressive cleanup and efficient data structures
- **Database Optimization**: Batch inserts, transaction management, indexing

#### Data Processing
- **Flexible Date Parsing**: Multiple format support with fallback
- **Robust Data Cleaning**: Missing value handling and validation
- **Efficient Grouping**: Pandas groupby operations for statistics

## Usage Workflows

### Basic Generation Workflow
```bash
# Generate using main configuration
python loads.py --config CQ_generator_parameters.json

# Enhanced generation with correlation refinement  
python cq_dual_generator.py --config CQ_generator_parameters.json

# Test mode with verification
python loads.py --config CQ_generator_parameters.json --test
```

### Analysis Workflow
```bash
# Launch interactive viewer
python scenario_viewer.py

# Or with pre-loaded database
python scenario_viewer.py LoadsDemo.db
```

### Configuration Parameters

#### Required Parameters
- **input_csv_file**: Path to input CSV data
- **mean_percentages**: Target means as percentage of original (e.g., [60, 80, 100, 120, 140])
- **variance_percentages**: Target variances as percentage of original
- **correlations**: Target correlations with original series
- **output_database_file**: SQLite database output path

#### Optional Parameters
- **min_value**: Minimum allowed value (default: 0.001)
- **random_seed**: Reproducibility seed (default: 42)

## Key Features

### Data Generation Capabilities
- **Arbitrary Correlations**: Full range from -1.0 to +1.0
- **Flexible Statistics**: Configurable mean and variance as percentages of original
- **Non-negative Assurance**: Automatic constraint enforcement
- **Quality Validation**: Comprehensive error metrics and reporting

### Analysis Features
- **Comprehensive Statistics**: Multi-level statistical analysis
- **Interactive Visualization**: GUI-based scenario exploration
- **Database Integration**: Normalized storage with efficient querying
- **Export Capabilities**: Plot and data export functionality

### Quality Assurance
- **Validation Metrics**: Correlation accuracy, statistical fidelity
- **Error Reporting**: Detailed error tracking and reporting
- **Test Mode**: Built-in verification and validation
- **Robustness**: Comprehensive error handling and edge case management

## Dependencies

### Core Requirements
- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations and array operations
- **sqlite3**: Database storage and management
- **matplotlib**: Plotting and visualization (for scenario_viewer.py)
- **tkinter**: GUI framework (for scenario_viewer.py)

### Standard Library
- **json**: Configuration file parsing
- **argparse**: Command-line interface
- **os**: File system operations

## License and Usage

**License**: GNU Lesser General Public License (LGPL) Version 2.1
- **Permissions**: Copy, distribute, and modify the library
- **Requirements**: Derivative works must be under same license
- **Commercial Use**: Allows linking with non-free programs

## Research Applications

### Scientific Use Cases
- **Environmental Monitoring**: Streamflow and water quality assessment
- **Statistical Validation**: Performance testing of load calculation methods
- **Scenario Analysis**: Impact assessment under varying conditions
- **Method Development**: Algorithm testing and validation

### Data Analysis Capabilities
- **Correlation Studies**: Relationship analysis between flow and chemistry
- **Uncertainty Quantification**: Statistical performance under different scenarios
- **Sensitivity Analysis**: Parameter impact assessment
- **Comparative Studies**: Method performance comparison

This repository provides a comprehensive framework for generating and analyzing synthetic environmental time series data, particularly focused on the relationship between streamflow, water chemistry, and calculated loads.
