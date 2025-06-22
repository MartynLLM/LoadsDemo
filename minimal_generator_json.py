#!/usr/bin/env python3
"""
Minimal Non-Negative Time Series Generator
JSON configuration version with flexible parameters.
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import json
import argparse

class MinimalNonNegativeGenerator:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.min_value = 0.001
        np.random.seed(random_seed)
        
        # Default parameters (will be overridden by JSON config)
        self.mean_percentages = [100]
        self.variance_percentages = [100]
        self.correlations = [1.0]
    
    def load_config(self, config_file):
        """Load configuration from JSON file."""
        print(f"Loading configuration from: {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"Configuration file not found: {config_file}")
            print("Creating default configuration file...")
            config = self.create_default_config(config_file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        
        # Validate required fields
        required_fields = ['input_csv_file', 'output_database_file']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")
        
        # Set parameters from config
        if 'mean_percentages' in config:
            self.mean_percentages = config['mean_percentages']
        if 'variance_percentages' in config:
            self.variance_percentages = config['variance_percentages']
        if 'correlations' in config:
            self.correlations = config['correlations']
        if 'min_value' in config:
            self.min_value = config['min_value']
        if 'random_seed' in config:
            self.random_seed = config['random_seed']
            np.random.seed(self.random_seed)
        
        print(f"Configuration loaded:")
        print(f"  Input CSV: {config['input_csv_file']}")
        print(f"  Output DB: {config['output_database_file']}")
        print(f"  Mean percentages: {self.mean_percentages}")
        print(f"  Variance percentages: {self.variance_percentages}")
        print(f"  Correlations: {self.correlations}")
        print(f"  Min value: {self.min_value}")
        print(f"  Random seed: {self.random_seed}")
        
        return config
    
    def create_default_config(self, config_file):
        """Create a default configuration file."""
        default_config = {
            "input_csv_file": "input_data.csv",
            "output_database_file": "generated_timeseries.db",
            "mean_percentages": [100],
            "variance_percentages": [100],
            "correlations": [1.0],
            "min_value": 0.001,
            "random_seed": 42,
            "description": "Configuration file for non-negative time series generator",
            "notes": [
                "mean_percentages: Target mean as percentage of original mean",
                "variance_percentages: Target variance as percentage of original variance", 
                "correlations: Target correlations with original series",
                "min_value: Minimum allowed value in generated series",
                "All generated series will be non-negative"
            ]
        }
        
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"Default configuration created: {config_file}")
        print("Please edit the configuration file with your desired parameters and run again.")
        
        return default_config
    
    def read_data(self, csv_file):
        """Read CSV data."""
        print(f"Reading data from: {csv_file}")
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Input CSV file not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
        
        # Use first two columns as date and flow
        date_col, flow_col = df.columns[0], df.columns[1]
        print(f"Using columns: {date_col} (date), {flow_col} (flow)")
        
        # Clean data
        df = df.dropna(subset=[flow_col])
        df[flow_col] = pd.to_numeric(df[flow_col], errors='coerce')
        df = df.dropna()
        
        # Parse dates
        df['parsed_date'] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')
        if df['parsed_date'].isna().all():
            df['parsed_date'] = pd.to_datetime(df[date_col], errors='coerce')
        
        df = df.dropna(subset=['parsed_date']).sort_values('parsed_date')
        
        flow_data = df[flow_col].values
        date_strings = df['parsed_date'].dt.strftime('%Y-%m-%d').values
        
        stats = {
            'data': flow_data,
            'date_strings': date_strings,
            'mean': np.mean(flow_data),
            'variance': np.var(flow_data, ddof=1),
            'std': np.std(flow_data, ddof=1),
            'min': np.min(flow_data),
            'max': np.max(flow_data),
            'n_samples': len(flow_data),
            'source_file': csv_file
        }
        
        print(f"Loaded {len(flow_data)} points")
        print(f"Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        print(f"Range: {stats['min']:.4f} to {stats['max']:.4f}")
        
        return stats
    
    def generate_nonnegative_series(self, original_data, target_correlation, target_mean, target_variance, seed):
        """Generate non-negative correlated series."""
        np.random.seed(seed)
        n = len(original_data)
        
        # Standardize original data
        orig_mean = np.mean(original_data)
        orig_std = np.std(original_data, ddof=1)
        
        if orig_std > 0:
            standardized_orig = (original_data - orig_mean) / orig_std
        else:
            standardized_orig = np.zeros(n)
        
        # Generate correlated series
        if abs(target_correlation - 1.0) < 1e-10:
            # Perfect correlation
            method = "perfect_correlation"
            target_std = np.sqrt(target_variance)
            a = target_std / orig_std if orig_std > 0 else 1.0
            b = target_mean - a * orig_mean
            series = a * original_data + b
        else:
            # Regular correlation
            method = "truncated_normal"
            noise = np.random.normal(0, 1, n)
            corr_factor = np.sqrt(1 - target_correlation**2)
            standardized_new = target_correlation * standardized_orig + corr_factor * noise
            
            target_std = np.sqrt(target_variance)
            series = standardized_new * target_std + target_mean
        
        # Ensure non-negative
        if np.any(series < 0):
            min_val = np.min(series)
            shift = -min_val + self.min_value
            series = series + shift
            current_mean = np.mean(series)
            series = series + (target_mean - current_mean)
        
        series = np.maximum(series, self.min_value)
        
        return series, method
    
    def generate_all_combinations(self, original_stats):
        """Generate all combinations."""
        print("\nGenerating all combinations...")
        
        original_data = original_stats['data']
        original_mean = original_stats['mean']
        original_variance = original_stats['variance']
        
        results = []
        combo_count = 0
        
        total_combos = len(self.mean_percentages) * len(self.variance_percentages) * len(self.correlations)
        
        for mean_pct in self.mean_percentages:
            for var_pct in self.variance_percentages:
                for correlation in self.correlations:
                    combo_count += 1
                    
                    target_mean = original_mean * (mean_pct / 100.0)
                    target_variance = original_variance * (var_pct / 100.0)
                    combo_id = f"M{mean_pct}_V{var_pct}_C{correlation:.2f}"
                    
                    print(f"  {combo_count}/{total_combos}: {combo_id}")
                    
                    generated_series, method = self.generate_nonnegative_series(
                        original_data, correlation, target_mean, target_variance,
                        self.random_seed + combo_count * 100
                    )
                    
                    # Calculate statistics
                    achieved_mean = np.mean(generated_series)
                    achieved_var = np.var(generated_series, ddof=1)
                    achieved_corr = np.corrcoef(original_data, generated_series)[0, 1]
                    min_value = np.min(generated_series)
                    negative_count = np.sum(generated_series < 0)
                    
                    result = {
                        'combination_id': combo_id,
                        'mean_percentage': mean_pct,
                        'variance_percentage': var_pct,
                        'target_correlation': correlation,
                        'target_mean': target_mean,
                        'target_variance': target_variance,
                        'achieved_mean': achieved_mean,
                        'achieved_variance': achieved_var,
                        'achieved_correlation': achieved_corr,
                        'min_value': min_value,
                        'negative_values_count': negative_count,
                        'generation_method': method,
                        'generated_series': generated_series
                    }
                    
                    results.append(result)
                    
                    print(f"    Min: {min_value:.6f}, Negatives: {negative_count}")
        
        print(f"Generated {len(results)} combinations successfully")
        return results
    

    
    def create_database(self, db_path):
        """Create database schema."""
        # Always remove existing database to avoid conflicts
        if os.path.exists(db_path):
            print(f"Removing existing database: {db_path}")
            os.remove(db_path)
        
        print(f"Creating new database: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE timeseries_data (
                id INTEGER PRIMARY KEY,
                combination_id TEXT,
                date TEXT,
                input_value REAL,
                generated_value REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE metadata (
                combination_id TEXT PRIMARY KEY,
                mean_percentage INTEGER,
                variance_percentage INTEGER,
                target_correlation REAL,
                achieved_correlation REAL,
                generation_method TEXT,
                min_value REAL,
                negative_count INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
        return True
    
    def save_to_database(self, original_stats, results, db_path):
        """Save results to database."""
        print(f"\nSaving to database: {db_path}")
        
        # Create database
        self.create_database(db_path)
        
        original_data = original_stats['data']
        date_strings = original_stats['date_strings']
        n_samples = original_stats['n_samples']
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Save metadata
        metadata_rows = []
        for result in results:
            metadata_row = (
                result['combination_id'],
                result['mean_percentage'],
                result['variance_percentage'],
                result['target_correlation'],
                result['achieved_correlation'],
                result['generation_method'],
                result['min_value'],
                result['negative_values_count']
            )
            metadata_rows.append(metadata_row)
        
        cursor.executemany(
            'INSERT INTO metadata VALUES (?,?,?,?,?,?,?,?)',
            metadata_rows
        )
        
        print(f"Metadata saved: {len(metadata_rows)} rows")
        
        # Save time series data
        total_rows = 0
        for combo_idx, result in enumerate(results):
            combo_id = result['combination_id']
            generated_series = result['generated_series']
            
            batch_data = []
            for i in range(n_samples):
                row = (
                    combo_id,
                    date_strings[i],
                    float(original_data[i]),
                    float(generated_series[i])
                )
                batch_data.append(row)
            
            cursor.executemany(
                'INSERT INTO timeseries_data (combination_id, date, input_value, generated_value) VALUES (?,?,?,?)',
                batch_data
            )
            
            total_rows += len(batch_data)
            
            if (combo_idx + 1) % 10 == 0:
                conn.commit()
                print(f"  Progress: {combo_idx + 1}/{len(results)} combinations, {total_rows} rows")
        
        conn.commit()
        
        # Verify
        cursor.execute("SELECT COUNT(*) FROM timeseries_data")
        db_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM timeseries_data WHERE generated_value < 0")
        negative_count = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"Database saved successfully!")
        print(f"Total rows: {db_count}")
        print(f"Negative values: {negative_count}")
        
        return True


def main(config_file='generator_parameters.json', test_mode=False):
    """Main function."""
    print("NON-NEGATIVE TIME SERIES GENERATOR")
    print("=" * 50)
    print(f"Configuration file: {config_file}")
    
    generator = MinimalNonNegativeGenerator()
    
    # Load configuration
    config = generator.load_config(config_file)
    
    csv_file = config['input_csv_file']
    db_path = config['output_database_file']
    
    print(f"Input: {csv_file}")
    print(f"Output: {db_path}")
    
    # Read data
    original_stats = generator.read_data(csv_file)
    
    # Generate combinations
    results = generator.generate_all_combinations(original_stats)
    
    # Save to database
    generator.save_to_database(original_stats, results, db_path)
    
    if test_mode:
        # Test database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM timeseries_data")
        count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM timeseries_data WHERE generated_value < 0")
        negatives = cursor.fetchone()[0]
        conn.close()
        
        print(f"\nDatabase verification:")
        print(f"  Total rows: {count}")
        print(f"  Negative values: {negatives}")
        print(f"  Status: {'PASSED' if negatives == 0 else 'FAILED'}")
    
    return "Generation completed successfully"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate non-negative time series using JSON configuration')
    parser.add_argument('--config', '-c', default='generator_parameters.json', 
                       help='JSON configuration file (default: generator_parameters.json)')
    parser.add_argument('--test', '-t', action='store_true', help='Run verification test')
    
    args = parser.parse_args()
    
    try:
        result = main(args.config, args.test)
        print(f"\nResult: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
