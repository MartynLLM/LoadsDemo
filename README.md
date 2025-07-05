# Loads Demo Repository Documentation

## Overview
The **Loads Demo** repository contains Python code for assessing the performance of loads statistics by simulating streamflow and chemistry time series with arbitrary correlations and variance relationships. The system generates synthetic datasets to evaluate environmental monitoring and water quality assessment methods.

## Repository Structure

### Core Files

#### 1. `loads.py` - Main Application
**Purpose**: Combined Flow and Chemistry Time Series Generator
- **Primary Function**: Generates both flow and chemistry data with separate database tables
- **Key Features**:
  - Creates flow:chemistry tuples and calculates load statistics
  - Supports arbitrary correlations (including negative correlations)
  - Ensures non-negative time series generation
  - Vectorized operations for performance optimization
  - Comprehensive statistical analysis and validation

**Main Class**: `CombinedFlowChemistryGenerator`
- Handles configuration loading from JSON files
- Manages data reading with flexible date format support
- Generates correlated time series using various methods
- Creates normalized SQLite database with comprehensive schema
- Calculates load statistics using vectorized operations

#### 2. Configuration Files

##### `CQ_generator_parameters.json` - Main Configuration
```json
{
    "flow": {
        "input_csv_file": "Flows.csv",
        "mean_percentages": [60, 80, 100, 120, 140],
        "variance_percentages": [60, 80, 100, 120, 140],
        "correlations": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
        "min_value": 0.001,
        "random_seed": 42
    },
    "chemistry": {
        "input_csv_file": "Chemistry.csv",
        "mean_percentages": [60, 80, 100, 120, 140],
        "variance_percentages": [60, 80, 100, 120, 140],
        "correlations": [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0],
        "min_value": 0.001,
        "random_seed": 42
    },
    "output_database_file": "LoadsDemo.db"
}
```

##### `generator_parameters.json` - Alternative Configuration
- Simplified configuration for single dataset generation
- Uses `SavjaForClaude.csv` as input
- Limited correlation range: [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

### Sample Data

#### `sampleData/shortFlows.csv`
- **Period**: January 1990 - December 1999 (daily data)
- **Format**: Date (dd/mm/yyyy), Flow (cubic meters/second)
- **Characteristics**: 
  - 3,653 daily flow measurements
  - Range: 0.06 to 30.96 m³/s
  - Seasonal patterns with winter peaks and summer lows
  - Represents typical streamflow variability

#### `sampleData/shortChemistry.csv`
- **Period**: January 1990 - December 1999 (monthly data)
- **Format**: Date (dd/mm/yyyy), Chemistry (concentration units)
- **Characteristics**:
  - 120 monthly chemistry measurements
  - Range: 81 to 3,762 units
  - Shows seasonal chemistry variation patterns
  - Higher concentrations typically in winter months

### Documentation

#### `README.md`
Basic repository description explaining:
- Purpose: Performance assessment of loads statistics
- Input: JSON configuration file specifying parameters
- Output: SQLite database with simulated time series
- Sample data location: `sampleData/` folder

#### `LICENSE`
- **License Type**: GNU Lesser General Public License (LGPL) Version 2.1
- **Permissions**: Copy, distribute, and modify the library
- **Requirements**: Derivative works must be under same license
- **Use Case**: Allows linking with non-free programs

## Technical Architecture

### Database Schema

The application creates a normalized SQLite database with the following structure:

#### Core Tables
1. **input_flow** / **input_chem**: Store original input data
2. **flow_combinations** / **chem_combinations**: Lookup tables for parameter combinations
3. **Flows** / **Chem**: Generated time series data (normalized)
4. **flow_metadata** / **chem_metadata**: Statistics and validation metrics
5. **Loads**: Flow:chemistry tuples with calculated loads
6. **loads_statistics**: Comprehensive scenario statistics

#### Views
- **Flows_view** / **Chem_view**: Denormalized views combining input and generated data
- **comprehensive_metadata**: Complete scenario information with all parameters and statistics

### Generation Methods

#### Correlation Handling
- **Perfect Positive (r=1.0)**: Linear transformation of original data
- **Perfect Negative (r=-1.0)**: Inverse linear transformation
- **Zero Correlation (r=0.0)**: Independent random series
- **Partial Correlation**: Correlated normal distribution mixing

#### Non-negative Constraints
- Automatic shifting of negative values
- Minimum value enforcement (default: 0.001)
- Mean adjustment after constraint application

### Statistical Calculations

#### Load Calculation
```
Load = 0.0864 × Chemistry × Flow
```
(Conversion factor for daily loads in appropriate units)

#### Comprehensive Statistics
- **Time Series**: Mean, variance, correlation, generation method accuracy
- **Load Statistics**: Mean, standard deviation, min/max, correlation
- **Regression Analysis**: Linear slope, intercept, R-squared
- **Error Metrics**: RMSE, Mean Absolute Error

## Usage Examples

### Basic Usage
```bash
python loads.py --config CQ_generator_parameters.json
```

### Test Mode
```bash
python loads.py --config CQ_generator_parameters.json --test
```

### Configuration Parameters

#### Required Parameters
- **input_csv_file**: Path to input data CSV
- **mean_percentages**: Target means as percentage of original (e.g., [60, 80, 100, 120, 140])
- **variance_percentages**: Target variances as percentage of original
- **correlations**: Target correlations with original series
- **output_database_file**: SQLite database output path

#### Optional Parameters
- **min_value**: Minimum allowed value (default: 0.001)
- **random_seed**: Reproducibility seed (default: 42)

## Performance Optimizations

### Vectorized Operations
- Batch processing for time series generation
- NumPy array operations for load calculations
- Pandas groupby for statistical aggregations

### Database Optimizations
- Batch inserts with configurable batch sizes
- Transaction management for bulk operations
- Pragma settings optimization for write performance
- Comprehensive indexing strategy

### Memory Management
- Chunked processing for large combination matrices
- Progressive memory cleanup
- Efficient data structure usage

## Output Analysis

### Database Queries
The generated database supports complex analytical queries:

```sql
-- Scenario performance comparison
SELECT scenario_id, correlation, rmse, r_squared 
FROM loads_statistics 
WHERE flow_target_correlation > 0.8 
ORDER BY rmse;

-- Method effectiveness analysis
SELECT generation_method, AVG(correlation_error_pct)
FROM comprehensive_metadata
GROUP BY generation_method;
```

### Key Metrics
- **Correlation Accuracy**: How well target correlations are achieved
- **Load Relationship Quality**: R-squared values for input vs generated loads
- **Statistical Fidelity**: Mean and variance preservation
- **Non-negative Compliance**: Absence of negative values

## Dependencies
- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations and array operations
- **sqlite3**: Database storage and management
- **json**: Configuration file parsing
- **argparse**: Command-line interface

## Error Handling
- Comprehensive input validation
- Date format flexibility with multiple parsing attempts
- Missing data handling with informative logging
- Database transaction rollback on errors
- Detailed error reporting and stack traces

## Future Enhancements
- Support for additional correlation generation methods
- Extended statistical distribution options
- Parallel processing capabilities
- Enhanced visualization and reporting features
- Integration with external data sources
