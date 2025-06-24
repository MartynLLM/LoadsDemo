#!/usr/bin/env python3
"""
Combined Flow and Chemistry Time Series Generator
Generates both flow and chemistry data with separate database tables.
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import json
import argparse

class CombinedFlowChemistryGenerator:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.min_value = 0.001
        np.random.seed(random_seed)
        
        # Default parameters (will be overridden by JSON config)
        self.flow_params = {}
        self.chemistry_params = {}
        self.output_db = None
    
    def load_config(self, config_file):
        """Load configuration from JSON file."""
        print(f"Loading configuration from: {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"Configuration file not found: {config_file}")
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        
        # Validate required fields
        required_fields = ['flow', 'chemistry', 'output_database_file']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")
        
        # Extract flow and chemistry parameters
        self.flow_params = config['flow']
        self.chemistry_params = config['chemistry']
        self.output_db = config['output_database_file']
        
        # Validate that each parameter set has required fields
        for param_name, params in [('flow', self.flow_params), ('chemistry', self.chemistry_params)]:
            required_param_fields = ['input_csv_file', 'mean_percentages', 'variance_percentages', 'correlations']
            for field in required_param_fields:
                if field not in params:
                    raise ValueError(f"Missing required field '{field}' in {param_name} parameters")
        
        # Set global parameters if present
        if 'random_seed' in config:
            self.random_seed = config['random_seed']
            np.random.seed(self.random_seed)
        
        print(f"Configuration loaded:")
        print(f"  Flow input: {self.flow_params['input_csv_file']}")
        print(f"  Chemistry input: {self.chemistry_params['input_csv_file']}")
        print(f"  Output DB: {self.output_db}")
        print(f"  Flow correlations: {self.flow_params['correlations']}")
        print(f"  Chemistry correlations: {self.chemistry_params['correlations']}")
        
        return config
    
    def read_data(self, csv_file, data_type="flow"):
        """Read CSV data."""
        print(f"Reading {data_type} data from: {csv_file}")
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Input CSV file not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
        
        # Use first two columns as date and value
        date_col, value_col = df.columns[0], df.columns[1]
        print(f"Using columns: {date_col} (date), {value_col} ({data_type})")
        
        # Clean data
        df = df.dropna(subset=[value_col])
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna()
        
        # Parse dates
        df['parsed_date'] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')
        if df['parsed_date'].isna().all():
            df['parsed_date'] = pd.to_datetime(df[date_col], errors='coerce')
        
        df = df.dropna(subset=['parsed_date']).sort_values('parsed_date')
        
        value_data = df[value_col].values
        date_strings = df['parsed_date'].dt.strftime('%Y-%m-%d').values
        
        stats = {
            'data': value_data,
            'date_strings': date_strings,
            'mean': np.mean(value_data),
            'variance': np.var(value_data, ddof=1),
            'std': np.std(value_data, ddof=1),
            'min': np.min(value_data),
            'max': np.max(value_data),
            'n_samples': len(value_data),
            'source_file': csv_file,
            'data_type': data_type
        }
        
        print(f"Loaded {len(value_data)} {data_type} points")
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
            # Perfect positive correlation
            method = "perfect_positive_correlation"
            target_std = np.sqrt(target_variance)
            a = target_std / orig_std if orig_std > 0 else 1.0
            b = target_mean - a * orig_mean
            series = a * original_data + b
        elif abs(target_correlation + 1.0) < 1e-10:
            # Perfect negative correlation
            method = "perfect_negative_correlation"
            target_std = np.sqrt(target_variance)
            a = -target_std / orig_std if orig_std > 0 else -1.0
            b = target_mean - a * orig_mean
            series = a * original_data + b
        else:
            # Regular correlation (positive or negative)
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
    
    def generate_combinations_for_dataset(self, original_stats, params, data_type):
        """Generate all combinations for a specific dataset (flow or chemistry)."""
        print(f"\nGenerating {data_type} combinations...")
        
        original_data = original_stats['data']
        original_mean = original_stats['mean']
        original_variance = original_stats['variance']
        
        mean_percentages = params['mean_percentages']
        variance_percentages = params['variance_percentages']
        correlations = params['correlations']
        min_value = params.get('min_value', self.min_value)
        
        results = []
        combo_count = 0
        
        total_combos = len(mean_percentages) * len(variance_percentages) * len(correlations)
        
        for mean_pct in mean_percentages:
            for var_pct in variance_percentages:
                for correlation in correlations:
                    combo_count += 1
                    
                    target_mean = original_mean * (mean_pct / 100.0)
                    target_variance = original_variance * (var_pct / 100.0)
                    combo_id = f"M{mean_pct}_V{var_pct}_C{correlation:.2f}"
                    
                    print(f"  {combo_count}/{total_combos}: {combo_id}")
                    
                    generated_series, method = self.generate_nonnegative_series(
                        original_data, correlation, target_mean, target_variance,
                        self.random_seed + combo_count * 100 + hash(data_type) % 10000
                    )
                    
                    # Calculate statistics
                    achieved_mean = np.mean(generated_series)
                    achieved_var = np.var(generated_series, ddof=1)
                    achieved_corr = np.corrcoef(original_data, generated_series)[0, 1]
                    min_value_achieved = np.min(generated_series)
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
                        'min_value': min_value_achieved,
                        'negative_values_count': negative_count,
                        'generation_method': method,
                        'generated_series': generated_series,
                        'data_type': data_type
                    }
                    
                    results.append(result)
                    
                    print(f"    Min: {min_value_achieved:.6f}, Negatives: {negative_count}, Corr: {achieved_corr:.3f}")
        
        print(f"Generated {len(results)} {data_type} combinations successfully")
        return results
    
    def create_database(self, db_path):
        """Create database schema with separate tables for flow and chemistry."""
        # Always remove existing database to avoid conflicts
        if os.path.exists(db_path):
            print(f"Removing existing database: {db_path}")
            os.remove(db_path)
        
        print(f"Creating new database: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create Flows table
        cursor.execute('''
            CREATE TABLE Flows (
                id INTEGER PRIMARY KEY,
                combination_id TEXT,
                date TEXT,
                input_value REAL,
                generated_value REAL
            )
        ''')
        
        # Create Chem table
        cursor.execute('''
            CREATE TABLE Chem (
                id INTEGER PRIMARY KEY,
                combination_id TEXT,
                date TEXT,
                input_value REAL,
                generated_value REAL
            )
        ''')
        
        # Create metadata tables
        cursor.execute('''
            CREATE TABLE flow_metadata (
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
        
        cursor.execute('''
            CREATE TABLE chem_metadata (
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
    
    def save_to_database(self, flow_stats, flow_results, chem_stats, chem_results, db_path):
        """Save results to database with separate tables."""
        print(f"\nSaving to database: {db_path}")
        
        # Create database
        self.create_database(db_path)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Save flow data
        self._save_dataset_to_db(cursor, flow_stats, flow_results, 'flow', 'Flows', 'flow_metadata')
        
        # Save chemistry data
        self._save_dataset_to_db(cursor, chem_stats, chem_results, 'chemistry', 'Chem', 'chem_metadata')
        
        conn.commit()
        
        # Verify
        cursor.execute("SELECT COUNT(*) FROM Flows")
        flow_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM Chem")
        chem_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM Flows WHERE generated_value < 0")
        flow_negatives = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM Chem WHERE generated_value < 0")
        chem_negatives = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"Database saved successfully!")
        print(f"Flow rows: {flow_count}, negative values: {flow_negatives}")
        print(f"Chemistry rows: {chem_count}, negative values: {chem_negatives}")
        
        return True
    
    def _save_dataset_to_db(self, cursor, original_stats, results, data_type, data_table, metadata_table):
        """Save a single dataset (flow or chemistry) to database."""
        print(f"\nSaving {data_type} data...")
        
        original_data = original_stats['data']
        date_strings = original_stats['date_strings']
        n_samples = original_stats['n_samples']
        
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
            f'INSERT INTO {metadata_table} VALUES (?,?,?,?,?,?,?,?)',
            metadata_rows
        )
        
        print(f"{data_type} metadata saved: {len(metadata_rows)} rows")
        
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
                f'INSERT INTO {data_table} (combination_id, date, input_value, generated_value) VALUES (?,?,?,?)',
                batch_data
            )
            
            total_rows += len(batch_data)
            
            if (combo_idx + 1) % 10 == 0:
                print(f"  {data_type} progress: {combo_idx + 1}/{len(results)} combinations, {total_rows} rows")
        
        print(f"{data_type} time series saved: {total_rows} rows")


def main(config_file='CQ_generator_parameters.json', test_mode=False):
    """Main function."""
    print("COMBINED FLOW AND CHEMISTRY TIME SERIES GENERATOR")
    print("=" * 60)
    print(f"Configuration file: {config_file}")
    
    generator = CombinedFlowChemistryGenerator()
    
    # Load configuration
    config = generator.load_config(config_file)
    
    print(f"Output database: {generator.output_db}")
    
    # Read flow data
    flow_stats = generator.read_data(generator.flow_params['input_csv_file'], 'flow')
    
    # Read chemistry data
    chem_stats = generator.read_data(generator.chemistry_params['input_csv_file'], 'chemistry')
    
    # Generate flow combinations
    flow_results = generator.generate_combinations_for_dataset(flow_stats, generator.flow_params, 'flow')
    
    # Generate chemistry combinations
    chem_results = generator.generate_combinations_for_dataset(chem_stats, generator.chemistry_params, 'chemistry')
    
    # Save to database
    generator.save_to_database(flow_stats, flow_results, chem_stats, chem_results, generator.output_db)
    
    if test_mode:
        # Test database
        conn = sqlite3.connect(generator.output_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM Flows")
        flow_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM Flows WHERE generated_value < 0")
        flow_negatives = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM Chem")
        chem_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM Chem WHERE generated_value < 0")
        chem_negatives = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"\nDatabase verification:")
        print(f"  Flow rows: {flow_count}, negative values: {flow_negatives}")
        print(f"  Chemistry rows: {chem_count}, negative values: {chem_negatives}")
        print(f"  Status: {'PASSED' if (flow_negatives + chem_negatives) == 0 else 'FAILED'}")
    
    return f"Generation completed successfully - Flow: {len(flow_results)} combinations, Chemistry: {len(chem_results)} combinations"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate combined flow and chemistry time series')
    parser.add_argument('--config', '-c', default='CQ_generator_parameters.json', 
                       help='JSON configuration file (default: CQ_generator_parameters.json)')
    parser.add_argument('--test', '-t', action='store_true', help='Run verification test')
    
    args = parser.parse_args()
    
    try:
        result = main(args.config, args.test)
        print(f"\nResult: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()