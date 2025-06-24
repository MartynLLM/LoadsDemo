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
        """Create database schema with separate tables for Flows and Chem."""
        if os.path.exists(db_path):
            print(f"Removing existing database: {db_path}")
            os.remove(db_path)
        
        print(f"Creating new database: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create Flows table
        cursor.execute('''
            CREATE TABLE Flows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                combination_id TEXT NOT NULL,
                date TEXT NOT NULL,
                input_value REAL NOT NULL,
                generated_value REAL NOT NULL
            )
        ''')
        
        # Create Chem table
        cursor.execute('''
            CREATE TABLE Chem (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                combination_id TEXT NOT NULL,
                date TEXT NOT NULL,
                input_value REAL NOT NULL,
                generated_value REAL NOT NULL
            )
        ''')
        
        # Create metadata tables
        cursor.execute('''
            CREATE TABLE flow_metadata (
                combination_id TEXT PRIMARY KEY,
                mean_percentage INTEGER,
                variance_percentage INTEGER,
                target_correlation REAL,
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
                correlation_error_pct REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE chem_metadata (
                combination_id TEXT PRIMARY KEY,
                mean_percentage INTEGER,
                variance_percentage INTEGER,
                target_correlation REAL,
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
                correlation_error_pct REAL
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
                generated_load REAL NOT NULL
            )
        ''')
        
        # Create loads statistics table
        cursor.execute('''
            CREATE TABLE loads_statistics (
                scenario_id TEXT PRIMARY KEY,
                flow_combination_id TEXT NOT NULL,
                chem_combination_id TEXT NOT NULL,
                n_points INTEGER,
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
                mean_absolute_error REAL
            )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX idx_flows_combo ON Flows(combination_id)')
        cursor.execute('CREATE INDEX idx_chem_combo ON Chem(combination_id)')
        cursor.execute('CREATE INDEX idx_flows_date ON Flows(date)')
        cursor.execute('CREATE INDEX idx_chem_date ON Chem(date)')
        cursor.execute('CREATE INDEX idx_loads_scenario ON Loads(scenario_id)')
        cursor.execute('CREATE INDEX idx_loads_date ON Loads(date)')
        
        conn.commit()
        conn.close()
        return True
    
    def save_dataset_to_database(self, cursor, original_stats, results, table_name, metadata_table):
        """Save a dataset (flow or chemistry) to the database."""
        data_type = results[0]['data_type'] if results else 'unknown'
        print(f"\nSaving {data_type} data to {table_name} table...")
        
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
            f'INSERT INTO {metadata_table} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            metadata_rows
        )
        
        print(f"  Metadata saved: {len(metadata_rows)} combinations")
        
        # Save time series data in batches
        batch_size = 1000
        total_rows = 0
        
        for combo_idx, result in enumerate(results):
            combo_id = result['combination_id']
            generated_series = result['generated_series']
            
            # Prepare batch data
            batch_data = []
            for i in range(n_samples):
                row = (
                    combo_id,
                    date_strings[i],
                    float(original_data[i]),
                    float(generated_series[i])
                )
                batch_data.append(row)
            
            # Insert batch
            cursor.executemany(
                f'INSERT INTO {table_name} (combination_id, date, input_value, generated_value) VALUES (?,?,?,?)',
                batch_data
            )
            
            total_rows += len(batch_data)
            
            # Progress reporting
            if (combo_idx + 1) % 10 == 0 or (combo_idx + 1) == len(results):
                print(f"  Progress: {combo_idx + 1}/{len(results)} combinations, {total_rows:,} rows")
        
        print(f"  {data_type} time series saved: {total_rows:,} rows")
        return total_rows
    
    def create_flow_chemistry_tuples(self, flow_stats, flow_results, chem_stats, chem_results):
        """Create flow:chemistry tuples and calculate loads."""
        print(f"\nCreating flow:chemistry tuples and calculating loads...")
        
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
        
        tuple_data = []
        scenario_count = 0
        total_scenarios = len(flow_results) * len(chem_results)
        
        print(f"  Generating {total_scenarios} scenarios...")
        
        for flow_result in flow_results:
            for chem_result in chem_results:
                scenario_count += 1
                
                flow_combo_id = flow_result['combination_id']
                chem_combo_id = chem_result['combination_id']
                scenario_id = f"F_{flow_combo_id}_C_{chem_combo_id}"
                
                flow_series = flow_result['generated_series']
                chem_series = chem_result['generated_series']
                flow_input = flow_stats['data']
                chem_input = chem_stats['data']
                
                scenario_tuples = []
                
                for date in common_dates:
                    flow_idx = flow_date_map[date]
                    chem_idx = chem_date_map[date]
                    
                    flow_input_val = flow_input[flow_idx]
                    flow_generated_val = flow_series[flow_idx]
                    chem_input_val = chem_input[chem_idx]
                    chem_generated_val = chem_series[chem_idx]
                    
                    # Calculate loads using the formula: 0.0864 * chemistry * flow
                    input_load = 0.0864 * chem_input_val * flow_input_val
                    generated_load = 0.0864 * chem_generated_val * flow_generated_val
                    
                    tuple_row = {
                        'flow_combination_id': flow_combo_id,
                        'chem_combination_id': chem_combo_id,
                        'scenario_id': scenario_id,
                        'date': date,
                        'flow_input_value': flow_input_val,
                        'flow_generated_value': flow_generated_val,
                        'chem_input_value': chem_input_val,
                        'chem_generated_value': chem_generated_val,
                        'input_load': input_load,
                        'generated_load': generated_load
                    }
                    
                    scenario_tuples.append(tuple_row)
                
                tuple_data.extend(scenario_tuples)
                
                # Progress reporting
                if scenario_count % 50 == 0 or scenario_count == total_scenarios:
                    print(f"    Progress: {scenario_count}/{total_scenarios} scenarios "
                          f"({scenario_count/total_scenarios*100:.1f}%)")
        
        print(f"  Created {len(tuple_data):,} flow:chemistry tuples")
        return tuple_data
    
    def calculate_load_statistics(self, tuple_data):
        """Calculate comprehensive statistics for each scenario."""
        print(f"\nCalculating load statistics...")
        
        # Group tuples by scenario
        scenarios = {}
        for row in tuple_data:
            scenario_id = row['scenario_id']
            if scenario_id not in scenarios:
                scenarios[scenario_id] = []
            scenarios[scenario_id].append(row)
        
        statistics = []
        scenario_count = 0
        total_scenarios = len(scenarios)
        
        for scenario_id, scenario_data in scenarios.items():
            scenario_count += 1
            
            # Extract load arrays
            input_loads = np.array([row['input_load'] for row in scenario_data])
            generated_loads = np.array([row['generated_load'] for row in scenario_data])
            
            # Basic statistics
            n_points = len(input_loads)
            input_mean = np.mean(input_loads)
            input_std = np.std(input_loads, ddof=1)
            input_min = np.min(input_loads)
            input_max = np.max(input_loads)
            
            generated_mean = np.mean(generated_loads)
            generated_std = np.std(generated_loads, ddof=1)
            generated_min = np.min(generated_loads)
            generated_max = np.max(generated_loads)
            
            # Correlation
            correlation = np.corrcoef(input_loads, generated_loads)[0, 1]
            
            # Linear regression using basic numpy operations
            if len(input_loads) > 1 and np.var(input_loads) > 0:
                # Calculate slope and intercept manually
                x_mean = np.mean(input_loads)
                y_mean = np.mean(generated_loads)
                
                numerator = np.sum((input_loads - x_mean) * (generated_loads - y_mean))
                denominator = np.sum((input_loads - x_mean) ** 2)
                
                linear_slope = numerator / denominator if denominator > 0 else 0.0
                linear_intercept = y_mean - linear_slope * x_mean
                
                # Calculate R-squared
                y_pred = linear_slope * input_loads + linear_intercept
                ss_res = np.sum((generated_loads - y_pred) ** 2)
                ss_tot = np.sum((generated_loads - y_mean) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            else:
                linear_slope = 0.0
                linear_intercept = generated_mean
                r_squared = 0.0
            
            # Error metrics
            rmse = np.sqrt(np.mean((input_loads - generated_loads) ** 2))
            mae = np.mean(np.abs(input_loads - generated_loads))
            
            # Store statistics
            flow_combo_id = scenario_data[0]['flow_combination_id']
            chem_combo_id = scenario_data[0]['chem_combination_id']
            
            stats = {
                'scenario_id': scenario_id,
                'flow_combination_id': flow_combo_id,
                'chem_combination_id': chem_combo_id,
                'n_points': n_points,
                'input_load_mean': input_mean,
                'input_load_std': input_std,
                'input_load_min': input_min,
                'input_load_max': input_max,
                'generated_load_mean': generated_mean,
                'generated_load_std': generated_std,
                'generated_load_min': generated_min,
                'generated_load_max': generated_max,
                'correlation': correlation,
                'linear_slope': linear_slope,
                'linear_intercept': linear_intercept,
                'r_squared': r_squared,
                'rmse': rmse,
                'mean_absolute_error': mae
            }
            
            statistics.append(stats)
            
            # Progress reporting
            if scenario_count % 100 == 0 or scenario_count == total_scenarios:
                print(f"    Progress: {scenario_count}/{total_scenarios} scenarios "
                      f"({scenario_count/total_scenarios*100:.1f}%)")
        
        print(f"  Calculated statistics for {len(statistics)} scenarios")
        return statistics
    
    def save_loads_to_database(self, cursor, tuple_data, statistics):
        """Save loads data and statistics to database."""
        print(f"\nSaving loads data to database...")
        
        # Save tuple data to Loads table
        batch_size = 1000
        total_tuples = len(tuple_data)
        
        for i in range(0, total_tuples, batch_size):
            batch = tuple_data[i:i + batch_size]
            batch_rows = []
            
            for row in batch:
                db_row = (
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
                batch_rows.append(db_row)
            
            cursor.executemany('''
                INSERT INTO Loads (flow_combination_id, chem_combination_id, scenario_id, 
                                 date, flow_input_value, flow_generated_value, 
                                 chem_input_value, chem_generated_value, 
                                 input_load, generated_load) 
                VALUES (?,?,?,?,?,?,?,?,?,?)
            ''', batch_rows)
            
            if (i + batch_size) % 10000 == 0 or (i + batch_size) >= total_tuples:
                print(f"    Loads data progress: {min(i + batch_size, total_tuples):,}/{total_tuples:,}")
        
        print(f"  Saved {total_tuples:,} load tuples")
        
        # Save statistics
        stats_rows = []
        for stats in statistics:
            stats_row = (
                stats['scenario_id'],
                stats['flow_combination_id'],
                stats['chem_combination_id'],
                stats['n_points'],
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
            stats_rows.append(stats_row)
        
        cursor.executemany('''
            INSERT INTO loads_statistics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', stats_rows)
        
        print(f"  Saved statistics for {len(stats_rows)} scenarios")
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
            
            statistics = self.calculate_load_statistics(tuple_data)
            
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