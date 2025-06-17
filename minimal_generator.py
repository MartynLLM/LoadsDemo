#!/usr/bin/env python3
"""
Minimal Non-Negative Time Series Generator
Syntax-safe version with core functionality.
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import time
import argparse

class MinimalNonNegativeGenerator:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.min_value = 0.001
        np.random.seed(random_seed)
        
        # Parameters
        self.mean_percentages = [80, 90, 100, 110, 120]
        self.variance_percentages = [70, 80, 90, 100, 110]
        self.correlations = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    def read_data(self, csv_file):
        """Read CSV data."""
        print(f"Reading data from: {csv_file}")
        
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
                    combo_id = f"M{mean_pct}_V{var_pct}_C{correlation:.1f}"
                    
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
    
    def save_to_csv(self, original_stats, results, csv_path=None):
        """Save results to CSV."""
        if csv_path is None:
            base_name = os.path.splitext(os.path.basename(original_stats['source_file']))[0]
            csv_path = f"nonnegative_{base_name}_results.csv"
        
        print(f"\nSaving to CSV: {csv_path}")
        
        original_data = original_stats['data']
        date_strings = original_stats['date_strings']
        n_samples = original_stats['n_samples']
        
        csv_data = []
        
        for result in results:
            combo_id = result['combination_id']
            generated_series = result['generated_series']
            
            for i in range(n_samples):
                row = {
                    'combination_id': combo_id,
                    'date': date_strings[i],
                    'original_value': original_data[i],
                    'generated_value': generated_series[i],
                    'mean_percentage': result['mean_percentage'],
                    'variance_percentage': result['variance_percentage'],
                    'target_correlation': result['target_correlation'],
                    'achieved_correlation': result['achieved_correlation'],
                    'generation_method': result['generation_method']
                }
                csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        
        print(f"CSV saved: {len(df)} rows")
        print(f"Min generated value: {df['generated_value'].min():.6f}")
        print(f"Negative values: {(df['generated_value'] < 0).sum()}")
        
        return True
    
    def create_database(self, db_path):
        """Create database schema."""
        if os.path.exists(db_path):
            os.remove(db_path)
        
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


def main(csv_file, db_path=None, test_mode=False):
    """Main function."""
    if db_path is None:
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        db_path = f"nonnegative_{base_name}_timeseries.db"
    
    print("NON-NEGATIVE TIME SERIES GENERATOR")
    print("=" * 50)
    print(f"Input: {csv_file}")
    print(f"Output: {db_path}")
    
    generator = MinimalNonNegativeGenerator()
    
    # Read data
    original_stats = generator.read_data(csv_file)
    
    # Generate combinations
    results = generator.generate_all_combinations(original_stats)
    
    # Save to CSV
    generator.save_to_csv(original_stats, results)
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Input CSV file')
    parser.add_argument('--output', '-o', help='Output database path')
    parser.add_argument('--test', '-t', action='store_true', help='Run verification test')
    
    args = parser.parse_args()
    
    try:
        result = main(args.input_file, args.output, args.test)
        print(f"\nResult: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
