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

## Dependencies and Installation

### System Requirements
- **Python**: Version 3.6 or higher
- **Operating System**: Cross-platform (Windows, macOS, Linux)

### Required Non-Standard Libraries

#### Core Dependencies (Required for all applications)
```bash
# Install via pip
pip install pandas numpy

# Or via conda
conda install pandas numpy
```

**pandas** (Data manipulation and analysis)
- **Version**: 1.0.0 or higher recommended
- **Purpose**: CSV reading, data cleaning, DataFrame operations, statistical grouping
- **Key Functions**: `read_csv()`, `DataFrame`, `groupby()`, date parsing

**numpy** (Numerical computing)
- **Version**: 1.18.0 or higher recommended  
- **Purpose**: Array operations, statistical calculations, random number generation
- **Key Functions**: Vectorized operations, correlation calculations, mathematical functions

#### Visualization Dependencies (Required for scenario_viewer.py only)
```bash
# Install via pip
pip install matplotlib

# Or via conda
conda install matplotlib
```

**matplotlib** (Plotting and visualization)
- **Version**: 3.1.0 or higher recommended
- **Purpose**: Interactive plotting, figure export, GUI integration
- **Key Components**: `pyplot`, `backends.backend_tkagg`, figure/axis management

### Standard Library Dependencies
The following are included with Python and require no additional installation:
- **sqlite3**: Database operations
- **tkinter**: GUI framework (may need separate installation on some Linux distributions)
- **json**: Configuration file parsing
- **argparse**: Command-line argument parsing
- **os**: File system operations

### Installation Commands

#### Complete Installation (All Features)
```bash
# Using pip
pip install pandas numpy matplotlib

# Using conda
conda install pandas numpy matplotlib

# Alternative: Install from requirements file
pip install -r requirements.txt
```

#### Minimal Installation (Core Generation Only)
```bash
# For loads.py and cq_dual_generator.py only
pip install pandas numpy
```

#### Verification
```python
# Test installation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # If using scenario_viewer.py
print("All dependencies successfully installed!")
```

## SQLite Database Viewing and Analysis

The system generates comprehensive SQLite databases containing all generated time series data, statistics, and metadata. Here are multiple approaches for viewing and analyzing the database contents.

### Database Structure Overview

#### Core Tables
- **input_flow**, **input_chem**: Original input data
- **Flows**, **Chem**: Generated time series data
- **flow_combinations**, **chem_combinations**: Parameter combinations
- **flow_metadata**, **chem_metadata**: Generation statistics
- **Loads**: Flow:chemistry tuples with calculated loads
- **loads_statistics**: Comprehensive scenario statistics

#### Database Views
- **Flows_view**, **Chem_view**: Denormalized data views
- **comprehensive_metadata**: Complete scenario information
- **scenario_summary**: Streamlined analysis view

### Viewing Methods

#### 1. Command-Line SQLite Tools

**Install SQLite Command-Line Tool:**
```bash
# Ubuntu/Debian
sudo apt-get install sqlite3

# macOS (via Homebrew)
brew install sqlite

# Windows: Download from https://sqlite.org/download.html
```

**Basic Database Exploration:**
```bash
# Open database
sqlite3 LoadsDemo.db

# List all tables and views
.tables

# View table schema
.schema Loads

# Quick data overview
SELECT COUNT(*) FROM loads_statistics;
SELECT * FROM loads_statistics LIMIT 5;

# Export data to CSV
.headers on
.mode csv
.output scenarios.csv
SELECT * FROM scenario_summary;
.quit
```

#### 2. GUI Database Browsers

**Recommended Tools:**

**DB Browser for SQLite** (Free, Cross-platform)
```bash
# Installation
# Ubuntu/Debian
sudo apt-get install sqlitebrowser

# macOS
brew install --cask db-browser-for-sqlite

# Windows: Download from https://sqlitebrowser.org/
```

**DBeaver** (Free, Professional)
- Download from: https://dbeaver.io/
- Supports SQLite and many other databases
- Advanced query tools and data visualization

**SQLiteStudio** (Free, Lightweight)
- Download from: https://sqlitestudio.pl/
- Portable application, no installation required

#### 3. Python-Based Viewing

**Quick Data Exploration Script:**
```python
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('LoadsDemo.db')

# List all tables and views
tables_query = """
    SELECT name, type FROM sqlite_master 
    WHERE type IN ('table', 'view') 
    ORDER BY type, name
"""
tables = pd.read_sql_query(tables_query, conn)
print("Available tables and views:")
print(tables)

# Load key summary data
summary = pd.read_sql_query("SELECT * FROM scenario_summary LIMIT 10", conn)
print("\nSample scenario data:")
print(summary.head())

# Basic statistics
stats_query = """
    SELECT 
        COUNT(*) as total_scenarios,
        MIN(load_correlation) as min_load_corr,
        MAX(load_correlation) as max_load_corr,
        AVG(load_correlation) as avg_load_corr,
        AVG(rmse) as avg_rmse
    FROM loads_statistics
"""
stats = pd.read_sql_query(stats_query, conn)
print("\nDatabase statistics:")
print(stats)

conn.close()
```

**Data Export Script:**
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('LoadsDemo.db')

# Export scenario summary to Excel
scenario_summary = pd.read_sql_query("SELECT * FROM scenario_summary", conn)
scenario_summary.to_excel('scenario_analysis.xlsx', index=False)

# Export time series data for specific scenario
time_series_query = """
    SELECT l.date, l.flow_input_value, l.flow_generated_value,
           l.chem_input_value, l.chem_generated_value,
           l.input_load, l.generated_load
    FROM Loads l
    WHERE l.scenario_id = 'F_M100_V100_C0.50_C_M100_V100_C-0.50'
    ORDER BY l.date
"""
time_series = pd.read_sql_query(time_series_query, conn)
time_series.to_csv('sample_time_series.csv', index=False)

conn.close()
print("Data exported successfully!")
```

### Key Analysis Queries

#### Scenario Performance Analysis
```sql
-- Top performing scenarios by correlation
SELECT scenario_id, flow_correlation, chem_correlation, 
       load_correlation, rmse, r_squared
FROM scenario_summary
ORDER BY load_correlation DESC
LIMIT 10;

-- Chemistry correlation impact
SELECT chem_correlation, 
       AVG(load_correlation) as avg_load_corr,
       COUNT(*) as scenario_count
FROM scenario_summary
GROUP BY chem_correlation
ORDER BY chem_correlation;
```

#### Time Series Data Analysis
```sql
-- Load statistics by flow correlation
SELECT f.correlation as flow_corr,
       AVG(l.generated_load) as avg_load,
       STDDEV(l.generated_load) as std_load
FROM Loads l
JOIN flow_combinations f ON l.flow_combination_id = f.combination_id
GROUP BY f.correlation
ORDER BY f.correlation;

-- Monthly load patterns
SELECT strftime('%m', date) as month,
       AVG(generated_load) as avg_monthly_load
FROM Loads
WHERE scenario_id = 'F_M100_V100_C1.00_C_M100_V100_C1.00'
GROUP BY month
ORDER BY month;
```

#### Data Quality Verification
```sql
-- Check for negative values (should be zero)
SELECT 
    'Flows' as table_name,
    COUNT(*) as negative_count
FROM Flows 
WHERE generated_value < 0
UNION ALL
SELECT 
    'Chem' as table_name,
    COUNT(*) as negative_count
FROM Chem 
WHERE generated_value < 0;

-- Correlation accuracy summary
SELECT 
    AVG(ABS(flow_achieved_correlation - flow_target_correlation)) as avg_flow_corr_error,
    AVG(ABS(chem_achieved_correlation - chem_target_correlation)) as avg_chem_corr_error,
    MAX(ABS(flow_achieved_correlation - flow_target_correlation)) as max_flow_corr_error,
    MAX(ABS(chem_achieved_correlation - chem_target_correlation)) as max_chem_corr_error
FROM comprehensive_metadata;
```

### Database Performance Tips

#### Indexing for Large Datasets
```sql
-- Additional indexes for performance (if needed)
CREATE INDEX IF NOT EXISTS idx_loads_date_scenario ON Loads(date, scenario_id);
CREATE INDEX IF NOT EXISTS idx_loads_values ON Loads(flow_generated_value, chem_generated_value);
```

#### Memory Optimization for Large Queries
```python
# For large datasets, use chunked reading
def read_large_table(conn, table_name, chunk_size=10000):
    offset = 0
    while True:
        query = f"SELECT * FROM {table_name} LIMIT {chunk_size} OFFSET {offset}"
        chunk = pd.read_sql_query(query, conn)
        if chunk.empty:
            break
        yield chunk
        offset += chunk_size

# Example usage
conn = sqlite3.connect('LoadsDemo.db')
for chunk in read_large_table(conn, 'Loads', 5000):
    # Process chunk
    print(f"Processing {len(chunk)} records...")
conn.close()
```

## Technical Architecture

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
