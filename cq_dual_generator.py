#!/usr/bin/env python3
"""
Combined Flow and Chemistry Time Series Generator
Generates both flow and chemistry data with separate database tables.
Creates flow:chemistry tuples and calculates load statistics.
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
        
        # Will be populated from config
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
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        
        # Validate required sections
        required_sections = ['flow', 'chemistry', 'output_database_file']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section in config: {section}")
        
        # Extract parameters
        self.flow_params = config['flow']
        self.chemistry_params = config['chemistry']
        self.output_db = config['output_database_file']
        
        # Validate parameter sections
        for param_name, params in [('flow', self.flow_params), ('chemistry', self.chemistry_params)]:
            required_fields = ['input_csv_file', 'mean_percentages', 'variance_percentages', 'correlations']
            for field in required_fields:
                if field not in params:
                    raise ValueError(f"Missing required field '{field}' in {param_name} parameters")
        
        # Set min_value and random_seed from parameters
        self.min_value = self.flow_params.get('min_value', 0.001)
        if 'random_seed' in self.flow_params:
            self.random_seed = self.flow_params['random_seed']
            np.random.seed(self.random_seed)
        
        print(f"Configuration loaded successfully:")
        print(f"  Flow input: {self.flow_params['input_csv_file']}")
        print(f"  Chemistry input: {self.chemistry_params['input_csv_file']}")
        print(f"  Output database: {self.output_db}")
        print(f"  Flow correlations: {self.flow_params['correlations']}")
        print(f"  Chemistry correlations: {self.chemistry_params['correlations']}")
        
        return config
    
    def read_data(self, csv_file, data_type="data"):
        """Read CSV data with flexible column detection."""
        print(f"Reading {data_type} data from: {csv_file}")
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Input CSV file not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
        
        # Use first two columns as date and value
        date_col, value_col = df.columns[0], df.columns[1]
        print(f"Using columns: {date_col} (date), {value_col} (value)")
        
        # Clean data
        original_count = len(df)
        df = df.dropna(subset=[value_col])
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna()
        
        # Parse dates with multiple format attempts
        date_formats = ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d', '%d-%m-%Y']
        parsed_dates = None
        
        for fmt in date_formats:
            try:
                parsed_dates = pd.to_datetime(df[date_col], format=fmt)
                print(f"  Parsed dates using format: {fmt}")
                break
            except:
                continue
        
        if parsed_dates is None:
            parsed_dates = pd.to_datetime(df[date_col], errors='coerce')
        
        df = df[parsed_dates.notna()].copy()
        df['parsed_date'] = parsed_dates[parsed_dates.notna()]
        df = df.sort_values('parsed_date').drop_duplicates(subset=['parsed_date'])
        
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
        
        print(f"Loaded {len(value_data)} {data_type} points (removed {original_count - len(value_data)})")
        print(f"Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        print(f"Range: {stats['min']:.4f} to {stats['max']:.4f}")
        
        return stats
    
    def generate_nonnegative_series(self, original_data, target_correlation, target_mean, target_variance, seed):
        """Generate non-negative correlated series with support for negative correlations."""
        np.random.seed(seed)
        n = len(original_data)
        
        # Standardize original data
        orig_mean = np.mean(original_data)
        orig_std = np.std(original_data, ddof=1)
        
        if orig_std > 0:
            standardized_orig = (original_data - orig_mean) / orig_std
        else:
            standardized_orig = np.zeros(n)
        
        # Handle different correlation cases
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
        elif target_correlation == 0.0:
            # Zero correlation - independent series
            method = "independent"
            target_std = np.sqrt(target_variance)
            series = np.random.normal(target_mean, target_std, n)
        else:
            # Regular correlation (positive or negative)
            method = "correlated_normal"
            noise = np.random.normal(0, 1, n)
            corr_factor = np.sqrt(1 - target_correlation**2)
            standardized_new = target_correlation * standardized_orig + corr_factor * noise
            
            target_std = np.sqrt(target_variance)
            series = standardized_new * target_std + target_mean
        
        # Ensure non-negative values
        if np.any(series < 0):
            min_val = np.min(series)
            shift = -min_val + self.min_value
            series = series + shift
            # Adjust mean back to target after shifting
            current_mean = np.mean(series)
            series = series + (target_mean - current_mean)
        
        series = np.maximum(series, self.min_value)
        
        return series, method
    
    def generate_chemistry_with_correlation_refinement(self, original_data, target_correlation, target_mean, target_variance, seed, max_attempts=15):
        """Enhanced chemistry generation with iterative correlation refinement."""
        np.random.seed(seed)
        best_series = None
        best_correlation_error = float('inf')
        best_method = "none"
        
        tolerance = 0.01  # Within 1% of target correlation
        
        for attempt in range(max_attempts):
            # Generate initial series with different seed each attempt
            attempt_seed = seed + attempt * 1000
            series, method = self.generate_nonnegative_series(
                original_data, target_correlation, target_mean, target_variance, attempt_seed
            )
            
            # Calculate achieved correlation
            achieved_corr = np.corrcoef(original_data, series)[0, 1]
            correlation_error = abs(achieved_corr - target_correlation)
            
            # Keep best result
            if correlation_error < best_correlation_error:
                best_series = series.copy()
                best_correlation_error = correlation_error
                best_method = f"{method}_refined_attempt_{attempt + 1}"
            
            # Early stopping if good enough
            if correlation_error <= tolerance:
                print(f"        Chemistry correlation refined in {attempt + 1} attempts: "
                      f"target={target_correlation:.3f}, achieved={achieved_corr:.3f}, "
                      f"error={correlation_error:.4f}")
                break
        else:
            # If we didn't break early, report final result
            final_corr = np.corrcoef(original_data, best_series)[0, 1]
            print(f"        Chemistry correlation after {max_attempts} attempts: "
                  f"target={target_correlation:.3f}, achieved={final_corr:.3f}, "
                  f"error={best_correlation_error:.4f}")
        
        # Apply post-processing correlation adjustment for chemistry
        best_series = self.adjust_correlation_post_generation(
            best_series, original_data, target_correlation, target_mean
        )
        
        return best_series, best_method
    
    def adjust_correlation_post_generation(self, series, original_data, target_correlation, target_mean):
        """Fine-tune correlation after initial generation."""
        current_corr = np.corrcoef(original_data, series)[0, 1]
        
        if abs(current_corr - target_correlation) > 0.02:  # If error > 2%
            # Calculate adjustment weights
            orig_standardized = (original_data - np.mean(original_data)) / np.std(original_data, ddof=1)
            series_mean = np.mean(series)
            series_std = np.std(series, ddof=1)
            
            # Blend with correlation-adjusted component
            alpha = 0.15  # Adjustment factor
            correlation_component = target_correlation * orig_standardized * series_std
            adjusted_series = (1 - alpha) * series + alpha * (correlation_component + series_mean)
            
            # Ensure non-negativity and mean preservation
            adjusted_series = np.maximum(adjusted_series, self.min_value)
            adjusted_series = adjusted_series + (target_mean - np.mean(adjusted_series))
            
            # Check if adjustment improved correlation
            new_corr = np.corrcoef(original_data, adjusted_series)[0, 1]
            if abs(new_corr - target_correlation) < abs(current_corr - target_correlation):
                return adjusted_series
        
        return series
    
    def generate_combinations_for_dataset(self, original_stats, params, data_type):
        """Generate all combinations for a specific dataset."""
        print(f"\nGenerating {data_type} combinations...")
        
        original_data = original_stats['data']
        original_mean = original_stats['mean']
        original_variance = original_stats['variance']
        
        mean_percentages = params['mean_percentages']
        variance_percentages = params['variance_percentages']
        correlations = params['correlations']
        
        results = []
        combo_count = 0
        
        total_combos = len(mean_percentages) * len(variance_percentages) * len(correlations)
        print(f"Total {data_type} combinations: {total_combos}")
        
        for mean_pct in mean_percentages:
            for var_pct in variance_percentages:
                for correlation in correlations:
                    combo_count += 1
                    
                    target_mean = original_mean * (mean_pct / 100.0)
                    target_variance = original_variance * (var_pct / 100.0)
                    combo_id = f"M{mean_pct}_V{var_pct}_C{correlation:.2f}"
                    
                    # Use different seed offsets for flow vs chemistry to ensure independence
                    seed_offset = hash(data_type) % 10000
                    
                    # Use enhanced correlation refinement for chemistry
                    if data_type == 'chemistry':
                        generated_series, method = self.generate_chemistry_with_correlation_refinement(
                            original_data, correlation, target_mean, target_variance,
                            self.random_seed + combo_count * 100 + seed_offset
                        )
                    else:
                        generated_series, method = self.generate_nonnegative_series(
                            original_data, correlation, target_mean, target_variance,
                            self.random_seed + combo_count * 100 + seed_offset
                        )
                    
                    # Calculate achieved statistics
                    achieved_mean = np.mean(generated_series)
                    achieved_var = np.var(generated_series, ddof=1)
                    achieved_corr = np.corrcoef(original_data, generated_series)[0, 1]
                    min_value_achieved = np.min(generated_series)
                    negative_count = np.sum(generated_series < 0)
                    
                    # Calculate errors
                    mean_error = abs(achieved_mean - target_mean) / target_mean * 100 if target_mean > 0 else 0
                    var_error = abs(achieved_var - target_variance) / target_variance * 100 if target_variance > 0 else 0
                    corr_error = abs(achieved_corr - correlation) / abs(correlation) * 100 if correlation != 0 else abs(achieved_corr) * 100
                    
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
                        'mean_error_pct': mean_error,
                        'variance_error_pct': var_error,
                        'correlation_error_pct': corr_error,
                        'generated_series': generated_series,
                        'data_type': data_type
                    }
                    
                    results.append(result)
                    
                    # Progress reporting
                    if combo_count % 25 == 0 or combo_count == total_combos:
                        print(f"  Progress: {combo_count}/{total_combos} "
                              f"({combo_count/total_combos*100:.1f}%)")
        
        # Summary statistics
        total_negatives = sum(r['negative_values_count'] for r in results)
        avg_corr_error = np.mean([r['correlation_error_pct'] for r in results])
        
        print(f"Generated {len(results)} {data_type} combinations successfully")
        print(f"  Total negative values: {total_negatives}")
        print(f"  Average correlation error: {avg_corr_error:.2f}%")
        
        return results
    
    def create_database(self, db_path):
        """Create database schema with normalized design and views."""
        if os.path.exists(db_path):
            print(f"Removing existing database: {db_path}")
            os.remove(db_path)
        
        print(f"Creating new database: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create input data tables (normalized)
        cursor.execute('''
            CREATE TABLE input_flow (
                date TEXT PRIMARY KEY,
                input_value REAL NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE input_chem (
                date TEXT PRIMARY KEY,
                input_value REAL NOT NULL
            )
        ''')
        
        # Create lookup tables for combinations
        cursor.execute('''
            CREATE TABLE flow_combinations (
                combination_id TEXT PRIMARY KEY,
                mean_percentage INTEGER NOT NULL,
                variance_percentage INTEGER NOT NULL,
                correlation REAL NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE chem_combinations (
                combination_id TEXT PRIMARY KEY,
                mean_percentage INTEGER NOT NULL,
                variance_percentage INTEGER NOT NULL,
                correlation REAL NOT NULL
            )
        ''')
        
        # Create normalized generated data tables (no input values)
        cursor.execute('''
            CREATE TABLE Flows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                combination_id TEXT NOT NULL,
                date TEXT NOT NULL,
                generated_value REAL NOT NULL,
                FOREIGN KEY (combination_id) REFERENCES flow_combinations(combination_id),
                FOREIGN KEY (date) REFERENCES input_flow(date)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE Chem (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                combination_id TEXT NOT NULL,
                date TEXT NOT NULL,
                generated_value REAL NOT NULL,
                FOREIGN KEY (combination_id) REFERENCES chem_combinations(combination_id),
                FOREIGN KEY (date) REFERENCES input_chem(date)
            )
        ''')
        
        # Create metadata tables
        cursor.execute('''
            CREATE TABLE flow_metadata (
                combination_id TEXT PRIMARY KEY,
                target_mean REAL,
                target_variance REAL,
                achieved_mean REAL,
                achieved_variance REAL,
                achieved_correlation REAL,
                generation_method TEXT,
                min_value REAL,
                negative_count INTEGER,
                mean_error_pct REAL,
                variance_error_pct REAL,
                correlation_error_pct REAL,
                FOREIGN KEY (combination_id) REFERENCES flow_combinations(combination_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE chem_metadata (
                combination_id TEXT PRIMARY KEY,
                target_mean REAL,
                target_variance REAL,
                achieved_mean REAL,
                achieved_variance REAL,
                achieved_correlation REAL,
                generation_method TEXT,
                min_value REAL,
                negative_count INTEGER,
                mean_error_pct REAL,
                variance_error_pct REAL,
                correlation_error_pct REAL,
                FOREIGN KEY (combination_id) REFERENCES chem_combinations(combination_id)
            )
        ''')
        
        # Create loads table for flow:chemistry tuples
        cursor.execute('''
            CREATE TABLE Loads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flow_combination_id TEXT NOT NULL,
                chem_combination_id TEXT NOT NULL,
                scenario_id TEXT NOT NULL,
                date TEXT NOT NULL,
                flow_input_value REAL NOT NULL,
                flow_generated_value REAL NOT NULL,
                chem_input_value REAL NOT NULL,
                chem_generated_value REAL NOT NULL,
                input_load REAL NOT NULL,
                generated_load REAL NOT NULL,
                FOREIGN KEY (flow_combination_id) REFERENCES flow_combinations(combination_id),
                FOREIGN KEY (chem_combination_id) REFERENCES chem_combinations(combination_id)
            )
        ''')
        
        # Create loads statistics table
        cursor.execute('''
            CREATE TABLE loads_statistics (
                scenario_id TEXT PRIMARY KEY,
                flow_combination_id TEXT NOT NULL,
                chem_combination_id TEXT NOT NULL,
                n_points INTEGER,
                
                -- Flow time series parameters and statistics
                flow_mean_percentage INTEGER,
                flow_variance_percentage INTEGER,
                flow_target_correlation REAL,
                flow_target_mean REAL,
                flow_target_variance REAL,
                flow_achieved_mean REAL,
                flow_achieved_variance REAL,
                flow_achieved_correlation REAL,
                
                -- Chemistry time series parameters and statistics  
                chem_mean_percentage INTEGER,
                chem_variance_percentage INTEGER,
                chem_target_correlation REAL,
                chem_target_mean REAL,
                chem_target_variance REAL,
                chem_achieved_mean REAL,
                chem_achieved_variance REAL,
                chem_achieved_correlation REAL,
                
                -- Load statistics
                input_load_mean REAL,
                input_load_std REAL,
                input_load_min REAL,
                input_load_max REAL,
                generated_load_mean REAL,
                generated_load_std REAL,
                generated_load_min REAL,
                generated_load_max REAL,
                correlation REAL,
                linear_slope REAL,
                linear_intercept REAL,
                r_squared REAL,
                rmse REAL,
                mean_absolute_error REAL,
                FOREIGN KEY (flow_combination_id) REFERENCES flow_combinations(combination_id),
                FOREIGN KEY (chem_combination_id) REFERENCES chem_combinations(combination_id)
            )
        ''')
        
        # Create views that replicate original structure
        cursor.execute('''
            CREATE VIEW Flows_view AS
            SELECT 
                f.id,
                f.combination_id,
                f.date,
                i.input_value,
                f.generated_value
            FROM Flows f
            JOIN input_flow i ON f.date = i.date
        ''')
        
        cursor.execute('''
            CREATE VIEW Chem_view AS
            SELECT 
                c.id,
                c.combination_id,
                c.date,
                i.input_value,
                c.generated_value
            FROM Chem c
            JOIN input_chem i ON c.date = i.date
        ''')
        
        # Create comprehensive metadata view
        cursor.execute('''
            CREATE VIEW comprehensive_metadata AS
            SELECT 
                ls.scenario_id,
                ls.flow_combination_id,
                ls.chem_combination_id,
                
                -- Flow combination details
                fc.mean_percentage as flow_mean_pct,
                fc.variance_percentage as flow_variance_pct,
                fc.correlation as flow_target_correlation,
                
                -- Chemistry combination details  
                cc.mean_percentage as chem_mean_pct,
                cc.variance_percentage as chem_variance_pct,
                cc.correlation as chem_target_correlation,
                
                -- Flow metadata
                fm.achieved_mean as flow_achieved_mean,
                fm.achieved_variance as flow_achieved_variance,
                fm.achieved_correlation as flow_achieved_correlation,
                fm.generation_method as flow_generation_method,
                fm.mean_error_pct as flow_mean_error_pct,
                fm.variance_error_pct as flow_variance_error_pct,
                fm.correlation_error_pct as flow_correlation_error_pct,
                
                -- Chemistry metadata
                cm.achieved_mean as chem_achieved_mean,
                cm.achieved_variance as chem_achieved_variance,
                cm.achieved_correlation as chem_achieved_correlation,
                cm.generation_method as chem_generation_method,
                cm.mean_error_pct as chem_mean_error_pct,
                cm.variance_error_pct as chem_variance_error_pct,
                cm.correlation_error_pct as chem_correlation_error_pct,
                
                -- Load statistics
                ls.n_points,
                ls.input_load_mean,
                ls.input_load_std,
                ls.input_load_min,
                ls.input_load_max,
                ls.generated_load_mean,
                ls.generated_load_std,
                ls.generated_load_min,
                ls.generated_load_max,
                ls.correlation as load_correlation,
                ls.linear_slope,
                ls.linear_intercept,
                ls.r_squared,
                ls.rmse,
                ls.mean_absolute_error
                
            FROM loads_statistics ls
            JOIN flow_combinations fc ON ls.flow_combination_id = fc.combination_id
            JOIN chem_combinations cc ON ls.chem_combination_id = cc.combination_id
            JOIN flow_metadata fm ON ls.flow_combination_id = fm.combination_id
            JOIN chem_metadata cm ON ls.chem_combination_id = cm.combination_id
        ''')
        
        # Create scenario summary view (loads_statistics + combinations only)
        cursor.execute('''
            CREATE VIEW scenario_summary AS
            SELECT 
                ls.scenario_id,
                ls.flow_combination_id,
                ls.chem_combination_id,
                ls.n_points,
                
                -- Flow combination parameters
                fc.mean_percentage as flow_mean_percentage,
                fc.variance_percentage as flow_variance_percentage,
                fc.correlation as flow_correlation,
                
                -- Chemistry combination parameters
                cc.mean_percentage as chem_mean_percentage,
                cc.variance_percentage as chem_variance_percentage,
                cc.correlation as chem_correlation,
                
                -- Flow time series statistics from loads_statistics
                ls.flow_target_mean,
                ls.flow_target_variance,
                ls.flow_target_correlation,
                ls.flow_achieved_mean,
                ls.flow_achieved_variance,
                ls.flow_achieved_correlation,
                
                -- Chemistry time series statistics from loads_statistics
                ls.chem_target_mean,
                ls.chem_target_variance,
                ls.chem_target_correlation,
                ls.chem_achieved_mean,
                ls.chem_achieved_variance,
                ls.chem_achieved_correlation,
                
                -- Load statistics
                ls.input_load_mean,
                ls.input_load_std,
                ls.input_load_min,
                ls.input_load_max,
                ls.generated_load_mean,
                ls.generated_load_std,
                ls.generated_load_min,
                ls.generated_load_max,
                ls.correlation as load_correlation,
                ls.linear_slope,
                ls.linear_intercept,
                ls.r_squared,
                ls.rmse,
                ls.mean_absolute_error
                
            FROM loads_statistics ls
            JOIN flow_combinations fc ON ls.flow_combination_id = fc.combination_id
            JOIN chem_combinations cc ON ls.chem_combination_id = cc.combination_id
        ''')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX idx_flows_combo ON Flows(combination_id)')
        cursor.execute('CREATE INDEX idx_chem_combo ON Chem(combination_id)')
        cursor.execute('CREATE INDEX idx_flows_date ON Flows(date)')
        cursor.execute('CREATE INDEX idx_chem_date ON Chem(date)')
        cursor.execute('CREATE INDEX idx_loads_scenario ON Loads(scenario_id)')
        cursor.execute('CREATE INDEX idx_loads_date ON Loads(date)')
        cursor.execute('CREATE INDEX idx_loads_flow_combo ON Loads(flow_combination_id)')
        cursor.execute('CREATE INDEX idx_loads_chem_combo ON Loads(chem_combination_id)')
        
        conn.commit()
        conn.close()
        return True
    
    def save_dataset_to_database(self, cursor, original_stats, results, table_name, metadata_table):
        """Save a dataset (flow or chemistry) to the normalized database."""
        data_type = results[0]['data_type'] if results else 'unknown'
        print(f"\nSaving {data_type} data to normalized tables...")
        
        original_data = original_stats['data']
        date_strings = original_stats['date_strings']
        n_samples = original_stats['n_samples']
        
        # Map data types to actual table names
        input_table_map = {
            'flow': 'input_flow',
            'chemistry': 'input_chem'
        }
        
        combination_table_map = {
            'flow': 'flow_combinations',
            'chemistry': 'chem_combinations'
        }
        
        # Save input data to input tables (only once per data type)
        input_table = input_table_map.get(data_type, f"input_{data_type}")
        print(f"  Saving input data to {input_table}...")
        
        input_rows = [(date, float(value)) for date, value in zip(date_strings, original_data)]
        
        cursor.executemany(
            f'INSERT INTO {input_table} (date, input_value) VALUES (?,?)',
            input_rows
        )
        print(f"  Saved {len(input_rows)} input records to {input_table}")
        
        # Save combination lookup data
        combination_table = combination_table_map.get(data_type, f"{data_type}_combinations")
        print(f"  Saving combinations to {combination_table}...")
        
        combination_rows = []
        for result in results:
            combo_row = (
                result['combination_id'],
                result['mean_percentage'],
                result['variance_percentage'],
                result['target_correlation']
            )
            combination_rows.append(combo_row)
        
        cursor.executemany(
            f'INSERT INTO {combination_table} (combination_id, mean_percentage, variance_percentage, correlation) VALUES (?,?,?,?)',
            combination_rows
        )
        print(f"  Saved {len(combination_rows)} combinations to {combination_table}")
        
        # Save metadata
        metadata_rows = []
        for result in results:
            metadata_row = (
                result['combination_id'],
                result['target_mean'],
                result['target_variance'],
                result['achieved_mean'],
                result['achieved_variance'],
                result['achieved_correlation'],
                result['generation_method'],
                result['min_value'],
                result['negative_values_count'],
                result['mean_error_pct'],
                result['variance_error_pct'],
                result['correlation_error_pct']
            )
            metadata_rows.append(metadata_row)
        
        cursor.executemany(
            f'INSERT INTO {metadata_table} VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
            metadata_rows
        )
        print(f"  Saved metadata: {len(metadata_rows)} combinations")
        
        # Save generated time series data (normalized - no input values)
        batch_size = 1000
        total_rows = 0
        
        print(f"  Saving generated data to {table_name}...")
        
        for combo_idx, result in enumerate(results):
            combo_id = result['combination_id']
            generated_series = result['generated_series']
            
            # Prepare batch data (only combination_id, date, generated_value)
            batch_data = []
            for i in range(n_samples):
                row = (
                    combo_id,
                    date_strings[i],
                    float(generated_series[i])
                )
                batch_data.append(row)
            
            # Insert batch
            cursor.executemany(
                f'INSERT INTO {table_name} (combination_id, date, generated_value) VALUES (?,?,?)',
                batch_data
            )
            
            total_rows += len(batch_data)
            
            # Progress reporting
            if (combo_idx + 1) % 10 == 0 or (combo_idx + 1) == len(results):
                print(f"    Progress: {combo_idx + 1}/{len(results)} combinations, {total_rows:,} rows")
        
        print(f"  {data_type} generated data saved: {total_rows:,} rows")
        return total_rows
    
    def create_flow_chemistry_tuples(self, flow_stats, flow_results, chem_stats, chem_results):
        """Create flow:chemistry tuples and calculate loads using vectorized operations."""
        print(f"\nCreating flow:chemistry tuples and calculating loads (vectorized)...")
        
        # Ensure dates match between flow and chemistry data
        flow_dates = set(flow_stats['date_strings'])
        chem_dates = set(chem_stats['date_strings'])
        common_dates = sorted(flow_dates.intersection(chem_dates))
        
        if len(common_dates) == 0:
            raise ValueError("No common dates found between flow and chemistry data")
        
        print(f"  Found {len(common_dates)} common dates")
        print(f"  Date range: {common_dates[0]} to {common_dates[-1]}")
        
        # Create date mapping for quick lookup
        flow_date_map = {date: idx for idx, date in enumerate(flow_stats['date_strings'])}
        chem_date_map = {date: idx for idx, date in enumerate(chem_stats['date_strings'])}
        
        # Filter data to common dates and create aligned arrays
        common_indices_flow = [flow_date_map[date] for date in common_dates]
        common_indices_chem = [chem_date_map[date] for date in common_dates]
        
        # Extract input data for common dates
        flow_input_aligned = flow_stats['data'][common_indices_flow]
        chem_input_aligned = chem_stats['data'][common_indices_chem]
        
        # Create arrays for all generated series (aligned to common dates)
        flow_generated_matrix = np.array([
            result['generated_series'][common_indices_flow] 
            for result in flow_results
        ])  # Shape: (n_flow_combinations, n_common_dates)
        
        chem_generated_matrix = np.array([
            result['generated_series'][common_indices_chem] 
            for result in chem_results
        ])  # Shape: (n_chem_combinations, n_common_dates)
        
        print(f"  Flow matrix shape: {flow_generated_matrix.shape}")
        print(f"  Chemistry matrix shape: {chem_generated_matrix.shape}")
        
        # Process in chunks to manage memory
        chunk_size = 50  # Process 50 flow combinations at a time
        total_scenarios = len(flow_results) * len(chem_results)
        processed_scenarios = 0
        
        print(f"  Processing {total_scenarios} scenarios in chunks of {chunk_size * len(chem_results)}")
        
        tuple_data = []
        
        for flow_chunk_start in range(0, len(flow_results), chunk_size):
            flow_chunk_end = min(flow_chunk_start + chunk_size, len(flow_results))
            flow_chunk_results = flow_results[flow_chunk_start:flow_chunk_end]
            flow_chunk_data = flow_generated_matrix[flow_chunk_start:flow_chunk_end]
            
            # Vectorized operations for this chunk
            # flow_chunk_data shape: (chunk_size, n_dates)
            # chem_generated_matrix shape: (n_chem_combinations, n_dates)
            
            # Broadcast for all combinations in this chunk
            flow_broadcast = flow_chunk_data[:, np.newaxis, :]  # (chunk_size, 1, n_dates)
            chem_broadcast = chem_generated_matrix[np.newaxis, :, :]  # (1, n_chem_combinations, n_dates)
            
            # Calculate all generated loads for this chunk at once
            generated_loads_chunk = 0.0864 * chem_broadcast * flow_broadcast  # (chunk_size, n_chem_combinations, n_dates)
            
            # Also calculate input loads (same for all scenarios, but broadcast for consistency)
            input_loads_base = 0.0864 * chem_input_aligned * flow_input_aligned  # (n_dates,)
            
            # Convert to tuples for this chunk
            for i, flow_result in enumerate(flow_chunk_results):
                for j, chem_result in enumerate(chem_results):
                    processed_scenarios += 1
                    
                    flow_combo_id = flow_result['combination_id']
                    chem_combo_id = chem_result['combination_id']
                    scenario_id = f"F_{flow_combo_id}_C_{chem_combo_id}"
                    
                    # Extract the generated loads for this specific scenario
                    scenario_generated_loads = generated_loads_chunk[i, j, :]  # (n_dates,)
                    scenario_flow_generated = flow_chunk_data[i, :]  # (n_dates,)
                    scenario_chem_generated = chem_generated_matrix[j, :]  # (n_dates,)
                    
                    # Create tuples for all dates in this scenario
                    for date_idx, date in enumerate(common_dates):
                        tuple_row = {
                            'flow_combination_id': flow_combo_id,
                            'chem_combination_id': chem_combo_id,
                            'scenario_id': scenario_id,
                            'date': date,
                            'flow_input_value': flow_input_aligned[date_idx],
                            'flow_generated_value': scenario_flow_generated[date_idx],
                            'chem_input_value': chem_input_aligned[date_idx],
                            'chem_generated_value': scenario_chem_generated[date_idx],
                            'input_load': input_loads_base[date_idx],
                            'generated_load': scenario_generated_loads[date_idx]
                        }
                        tuple_data.append(tuple_row)
            
            # Progress reporting
            print(f"    Progress: {processed_scenarios}/{total_scenarios} scenarios "
                  f"({processed_scenarios/total_scenarios*100:.1f}%) - "
                  f"Memory: {len(tuple_data):,} tuples")
        
        print(f"  Created {len(tuple_data):,} flow:chemistry tuples using vectorized operations")
        return tuple_data
    
    def calculate_load_statistics(self, tuple_data, flow_results, chem_results):
        """Calculate comprehensive statistics for each scenario using vectorized operations."""
        print(f"\nCalculating load statistics (vectorized)...")
        
        # Create lookup dictionaries for flow and chemistry results
        flow_lookup = {result['combination_id']: result for result in flow_results}
        chem_lookup = {result['combination_id']: result for result in chem_results}
        
        # Convert to DataFrame for more efficient grouping
        df = pd.DataFrame(tuple_data)
        
        print(f"  Processing {df['scenario_id'].nunique()} unique scenarios...")
        
        statistics = []
        
        # Group by scenario and calculate statistics vectorized
        grouped = df.groupby('scenario_id')
        
        scenario_count = 0
        total_scenarios = len(grouped)
        
        for scenario_id, group_data in grouped:
            scenario_count += 1
            
            # Extract load arrays
            input_loads = group_data['input_load'].values
            generated_loads = group_data['generated_load'].values
            
            # Get combination IDs from first row of group
            flow_combo_id = group_data.iloc[0]['flow_combination_id']
            chem_combo_id = group_data.iloc[0]['chem_combination_id']
            
            # Get flow and chemistry statistics from lookup
            flow_result = flow_lookup[flow_combo_id]
            chem_result = chem_lookup[chem_combo_id]
            
            # Basic load statistics (vectorized)
            n_points = len(input_loads)
            input_stats = {
                'mean': np.mean(input_loads),
                'std': np.std(input_loads, ddof=1),
                'min': np.min(input_loads),
                'max': np.max(input_loads)
            }
            
            generated_stats = {
                'mean': np.mean(generated_loads),
                'std': np.std(generated_loads, ddof=1),
                'min': np.min(generated_loads),
                'max': np.max(generated_loads)
            }
            
            # Correlation (vectorized)
            if n_points > 1 and np.var(input_loads) > 0 and np.var(generated_loads) > 0:
                correlation = np.corrcoef(input_loads, generated_loads)[0, 1]
            else:
                correlation = 0.0
            
            # Linear regression using vectorized operations
            if n_points > 1 and np.var(input_loads) > 0:
                # Vectorized calculation of slope and intercept
                x_mean = input_stats['mean']
                y_mean = generated_stats['mean']
                
                # Use numpy's efficient operations
                numerator = np.sum((input_loads - x_mean) * (generated_loads - y_mean))
                denominator = np.sum((input_loads - x_mean) ** 2)
                
                linear_slope = numerator / denominator if denominator > 0 else 0.0
                linear_intercept = y_mean - linear_slope * x_mean
                
                # Vectorized R-squared calculation
                y_pred = linear_slope * input_loads + linear_intercept
                ss_res = np.sum((generated_loads - y_pred) ** 2)
                ss_tot = np.sum((generated_loads - y_mean) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            else:
                linear_slope = 0.0
                linear_intercept = generated_stats['mean']
                r_squared = 0.0
            
            # Error metrics (vectorized)
            rmse = np.sqrt(np.mean((input_loads - generated_loads) ** 2))
            mae = np.mean(np.abs(input_loads - generated_loads))
            
            # Store comprehensive statistics including flow and chemistry data
            stats = {
                'scenario_id': scenario_id,
                'flow_combination_id': flow_combo_id,
                'chem_combination_id': chem_combo_id,
                'n_points': n_points,
                
                # Flow time series parameters and statistics
                'flow_mean_percentage': flow_result['mean_percentage'],
                'flow_variance_percentage': flow_result['variance_percentage'],
                'flow_target_correlation': flow_result['target_correlation'],
                'flow_target_mean': flow_result['target_mean'],
                'flow_target_variance': flow_result['target_variance'],
                'flow_achieved_mean': flow_result['achieved_mean'],
                'flow_achieved_variance': flow_result['achieved_variance'],
                'flow_achieved_correlation': flow_result['achieved_correlation'],
                
                # Chemistry time series parameters and statistics
                'chem_mean_percentage': chem_result['mean_percentage'],
                'chem_variance_percentage': chem_result['variance_percentage'],
                'chem_target_correlation': chem_result['target_correlation'],
                'chem_target_mean': chem_result['target_mean'],
                'chem_target_variance': chem_result['target_variance'],
                'chem_achieved_mean': chem_result['achieved_mean'],
                'chem_achieved_variance': chem_result['achieved_variance'],
                'chem_achieved_correlation': chem_result['achieved_correlation'],
                
                # Load statistics
                'input_load_mean': input_stats['mean'],
                'input_load_std': input_stats['std'],
                'input_load_min': input_stats['min'],
                'input_load_max': input_stats['max'],
                'generated_load_mean': generated_stats['mean'],
                'generated_load_std': generated_stats['std'],
                'generated_load_min': generated_stats['min'],
                'generated_load_max': generated_stats['max'],
                'correlation': correlation,
                'linear_slope': linear_slope,
                'linear_intercept': linear_intercept,
                'r_squared': r_squared,
                'rmse': rmse,
                'mean_absolute_error': mae
            }
            
            statistics.append(stats)
            
            # Progress reporting (less frequent for performance)
            if scenario_count % 500 == 0 or scenario_count == total_scenarios:
                print(f"    Progress: {scenario_count}/{total_scenarios} scenarios "
                      f"({scenario_count/total_scenarios*100:.1f}%)")
        
        print(f"  Calculated statistics for {len(statistics)} scenarios using vectorized operations")
        return statistics
    
    def save_loads_to_database(self, cursor, tuple_data, statistics):
        """Save loads data and statistics to database with optimized batch processing."""
        print(f"\nSaving loads data to database (optimized)...")
        
        # Optimize database settings for bulk inserts
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL") 
        cursor.execute("PRAGMA cache_size=100000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        
        # Save tuple data to Loads table with larger batches
        batch_size = 5000  # Increased batch size for better performance
        total_tuples = len(tuple_data)
        
        print(f"  Saving {total_tuples:,} tuples in batches of {batch_size:,}")
        
        # Prepare all data first (more efficient than building in loop)
        all_rows = [
            (
                row['flow_combination_id'],
                row['chem_combination_id'], 
                row['scenario_id'],
                row['date'],
                row['flow_input_value'],
                row['flow_generated_value'],
                row['chem_input_value'],
                row['chem_generated_value'],
                row['input_load'],
                row['generated_load']
            )
            for row in tuple_data
        ]
        
        # Insert in optimized batches
        cursor.execute("BEGIN TRANSACTION")
        
        for i in range(0, total_tuples, batch_size):
            batch = all_rows[i:i + batch_size]
            
            cursor.executemany('''
                INSERT INTO Loads (flow_combination_id, chem_combination_id, scenario_id, 
                                 date, flow_input_value, flow_generated_value, 
                                 chem_input_value, chem_generated_value, 
                                 input_load, generated_load) 
                VALUES (?,?,?,?,?,?,?,?,?,?)
            ''', batch)
            
            if (i + batch_size) % 50000 == 0 or (i + batch_size) >= total_tuples:
                print(f"    Loads progress: {min(i + batch_size, total_tuples):,}/{total_tuples:,} "
                      f"({min(i + batch_size, total_tuples)/total_tuples*100:.1f}%)")
        
        cursor.execute("COMMIT")
        print(f"  Saved {total_tuples:,} load tuples")
        
        # Save statistics with single transaction
        print(f"  Saving {len(statistics)} statistics records...")
        
        stats_rows = [
            (
                stats['scenario_id'],
                stats['flow_combination_id'],
                stats['chem_combination_id'],
                stats['n_points'],
                
                # Flow time series parameters and statistics
                stats['flow_mean_percentage'],
                stats['flow_variance_percentage'],
                stats['flow_target_correlation'],
                stats['flow_target_mean'],
                stats['flow_target_variance'],
                stats['flow_achieved_mean'],
                stats['flow_achieved_variance'],
                stats['flow_achieved_correlation'],
                
                # Chemistry time series parameters and statistics
                stats['chem_mean_percentage'],
                stats['chem_variance_percentage'],
                stats['chem_target_correlation'],
                stats['chem_target_mean'],
                stats['chem_target_variance'],
                stats['chem_achieved_mean'],
                stats['chem_achieved_variance'],
                stats['chem_achieved_correlation'],
                
                # Load statistics
                stats['input_load_mean'],
                stats['input_load_std'],
                stats['input_load_min'],
                stats['input_load_max'],
                stats['generated_load_mean'],
                stats['generated_load_std'],
                stats['generated_load_min'],
                stats['generated_load_max'],
                stats['correlation'],
                stats['linear_slope'],
                stats['linear_intercept'],
                stats['r_squared'],
                stats['rmse'],
                stats['mean_absolute_error']
            )
            for stats in statistics
        ]
        
        cursor.execute("BEGIN TRANSACTION")
        cursor.executemany('''
            INSERT INTO loads_statistics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', stats_rows)
        cursor.execute("COMMIT")
        
        print(f"  Saved statistics for {len(stats_rows)} scenarios")
        
        # Reset pragma settings
        cursor.execute("PRAGMA journal_mode=DELETE")
        cursor.execute("PRAGMA synchronous=FULL")
        
        return True
    
    def save_to_database(self, flow_stats, flow_results, chem_stats, chem_results, db_path):
        """Save all results to database."""
        print(f"\nSaving results to database: {db_path}")
        
        # Create database
        self.create_database(db_path)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # Save flow data
            flow_rows = self.save_dataset_to_database(
                cursor, flow_stats, flow_results, 'Flows', 'flow_metadata'
            )
            
            # Save chemistry data
            chem_rows = self.save_dataset_to_database(
                cursor, chem_stats, chem_results, 'Chem', 'chem_metadata'
            )
            
            conn.commit()
            
            # Create and save loads data
            tuple_data = self.create_flow_chemistry_tuples(
                flow_stats, flow_results, chem_stats, chem_results
            )
            
            statistics = self.calculate_load_statistics(tuple_data, flow_results, chem_results)
            
            self.save_loads_to_database(cursor, tuple_data, statistics)
            
            conn.commit()
            
            # Verification
            cursor.execute("SELECT COUNT(*) FROM Flows")
            db_flow_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM Chem")
            db_chem_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM Loads")
            db_loads_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM loads_statistics")
            db_stats_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM Flows WHERE generated_value < 0")
            flow_negatives = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM Chem WHERE generated_value < 0")
            chem_negatives = cursor.fetchone()[0]
            
            print(f"\nDatabase verification:")
            print(f"  Flows table: {db_flow_count:,} rows, {flow_negatives} negative values")
            print(f"  Chem table: {db_chem_count:,} rows, {chem_negatives} negative values")
            print(f"  Loads table: {db_loads_count:,} tuples")
            print(f"  Statistics table: {db_stats_count:,} scenarios")
            print(f"  Total negative values: {flow_negatives + chem_negatives}")
            
            return True
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()


def main(config_file='CQ_generator_parameters.json', test_mode=False):
    """Main function."""
    print("COMBINED FLOW AND CHEMISTRY TIME SERIES GENERATOR")
    print("=" * 60)
    print(f"Configuration file: {config_file}")
    
    try:
        generator = CombinedFlowChemistryGenerator()
        
        # Load configuration
        config = generator.load_config(config_file)
        
        # Read input data
        flow_stats = generator.read_data(generator.flow_params['input_csv_file'], 'flow')
        chem_stats = generator.read_data(generator.chemistry_params['input_csv_file'], 'chemistry')
        
        # Generate combinations
        flow_results = generator.generate_combinations_for_dataset(
            flow_stats, generator.flow_params, 'flow'
        )
        chem_results = generator.generate_combinations_for_dataset(
            chem_stats, generator.chemistry_params, 'chemistry'
        )
        
        # Save to database
        generator.save_to_database(
            flow_stats, flow_results, chem_stats, chem_results, generator.output_db
        )
        
        # Test mode verification
        if test_mode:
            conn = sqlite3.connect(generator.output_db)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM Flows WHERE generated_value < 0")
            flow_negatives = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM Chem WHERE generated_value < 0")
            chem_negatives = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT combination_id) FROM flow_metadata")
            flow_combos = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT combination_id) FROM chem_metadata")
            chem_combos = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM loads_statistics")
            db_stats_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(correlation), MIN(correlation), MAX(correlation) FROM loads_statistics")
            corr_stats = cursor.fetchone()
            
            cursor.execute("SELECT AVG(rmse), MIN(rmse), MAX(rmse) FROM loads_statistics") 
            rmse_stats = cursor.fetchone()
            
            conn.close()
            
            print(f"\nTest verification:")
            print(f"  Flow combinations: {flow_combos}")
            print(f"  Chemistry combinations: {chem_combos}")
            print(f"  Load scenarios: {db_stats_count}")
            print(f"  Total negative values: {flow_negatives + chem_negatives}")
            print(f"  Correlation stats: avg={corr_stats[0]:.3f}, min={corr_stats[1]:.3f}, max={corr_stats[2]:.3f}")
            print(f"  RMSE stats: avg={rmse_stats[0]:.3f}, min={rmse_stats[1]:.3f}, max={rmse_stats[2]:.3f}")
            print(f"  Test status: {'PASSED' if (flow_negatives + chem_negatives) == 0 else 'FAILED'}")
        
        return f"Generation completed - Flow: {len(flow_results)} combinations, Chemistry: {len(chem_results)} combinations"
        
    except Exception as e:
        print(f"Error during generation: {e}")
        raise


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