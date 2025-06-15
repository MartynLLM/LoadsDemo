"""
SQLite Time Series Generator for Savja Flow Data

Creates a SQLite database with two tables:
1. timeseries_data: combination_id, date, input_value, generated_value
2. metadata: combination_id + all statistical parameters and quality metrics

This provides much better performance and query capabilities compared to CSV files.
"""

import pandas as pd
import numpy as np
import sqlite3
import math
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import os
import time

# Handle scipy import gracefully
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Note: scipy not available, using fallback methods")


class SQLiteTimeSeriesGenerator:
    """Generator that creates SQLite database with time series and metadata tables."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Parameter specifications
        self.mean_percentages = [80, 90, 100, 110, 120]
        self.variance_percentages = [70, 80, 90, 100, 110]
        self.correlations = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    def read_savja_data(self, csv_file_path: str = "SavjaForClaude.csv") -> Dict:
        """Read and analyze the Savja flow data."""
        print(f"Reading Savja flow data from: {csv_file_path}")
        
        try:
            df = pd.read_csv(csv_file_path)
            
            if 'Flow' not in df.columns or 'Date' not in df.columns:
                raise ValueError("Required columns 'Date' and 'Flow' not found")
            
            # Clean data
            df = df.dropna(subset=['Flow'])
            df['Flow'] = pd.to_numeric(df['Flow'], errors='coerce')
            df = df.dropna(subset=['Flow'])
            
            # Parse dates
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
            df = df.sort_values('Date')
            
            flow_data = df['Flow'].values
            dates = df['Date'].values
            
            original_stats = {
                'data': flow_data,
                'dates': dates,
                'date_strings': df['Date'].dt.strftime('%Y-%m-%d').values,  # SQLite date format
                'mean': np.mean(flow_data),
                'variance': np.var(flow_data, ddof=1),
                'std': np.std(flow_data, ddof=1),
                'min': np.min(flow_data),
                'max': np.max(flow_data),
                'autocorr': self._calculate_autocorr(flow_data),
                'n_samples': len(flow_data),
                'date_range': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
            }
            
            print(f"✓ Successfully loaded {len(flow_data)} data points")
            print(f"  Date range: {original_stats['date_range']}")
            print(f"  Mean: {original_stats['mean']:.4f}")
            print(f"  Variance: {original_stats['variance']:.4f}")
            print(f"  Min: {original_stats['min']:.4f}, Max: {original_stats['max']:.4f}")
            
            return original_stats
            
        except Exception as e:
            print(f"Error reading Savja data: {e}")
            return None
    
    def _calculate_autocorr(self, data: np.ndarray, lag: int = 1) -> float:
        """Calculate first-order autocorrelation."""
        if len(data) <= lag:
            return 0.0
        
        y = data[lag:]
        y_lag = data[:-lag]
        
        if len(y) <= 1 or np.var(y) == 0 or np.var(y_lag) == 0:
            return 0.0
            
        return np.corrcoef(y, y_lag)[0, 1]
    
    def generate_correlated_series_fixed(self, original_data: np.ndarray, 
                                       target_correlation: float,
                                       target_mean: float,
                                       target_variance: float,
                                       attempt_seed: int) -> np.ndarray:
        """Fixed correlation generation that handles perfect correlation correctly."""
        np.random.seed(attempt_seed)
        
        n = len(original_data)
        orig_mean = np.mean(original_data)
        orig_var = np.var(original_data, ddof=1)
        orig_std = np.sqrt(orig_var)
        
        # SPECIAL CASE: Perfect correlation (1.0)
        if abs(target_correlation - 1.0) < 1e-10:
            print(f"    Using perfect correlation path (correlation = 1.0)")
            
            if target_variance <= 0:
                new_series = np.full(n, target_mean)
            else:
                # Direct linear transformation: new = a * original + b
                a = np.sqrt(target_variance / orig_var) if orig_var > 0 else 1.0
                b = target_mean - a * orig_mean
                new_series = a * original_data + b
                
                print(f"    Linear transformation: a={a:.6f}, b={b:.6f}")
            
            # Handle non-negativity if needed
            if np.any(new_series < 0):
                print(f"    Handling negative values for perfect correlation...")
                min_val = np.min(new_series)
                if min_val < 0:
                    shift = -min_val + 0.001
                    new_series = new_series + shift
                    current_mean = np.mean(new_series)
                    new_series = new_series + (target_mean - current_mean)
                    new_series = np.maximum(new_series, 0.001)
                    print(f"    Applied shift to ensure non-negativity")
            
            return new_series
        
        # SPECIAL CASE: Near-perfect correlation (> 0.99)
        elif target_correlation > 0.99:
            print(f"    Using near-perfect correlation path (correlation = {target_correlation})")
            
            if orig_std > 0:
                standardized_orig = (original_data - orig_mean) / orig_std
            else:
                standardized_orig = np.zeros(n)
            
            noise_scale = np.sqrt(1 - target_correlation**2) * 0.1
            noise = np.random.normal(0, noise_scale, n)
            standardized_new = target_correlation * standardized_orig + noise
            
            target_std = np.sqrt(target_variance)
            new_series = standardized_new * target_std + target_mean
            
            if np.any(new_series < 0):
                new_series = self._handle_negativity_preserving_correlation(
                    new_series, original_data, target_correlation, target_mean, target_variance)
            
            return new_series
        
        # REGULAR CASE: Correlation < 0.99
        else:
            return self._generate_regular_correlated_series(
                original_data, target_correlation, target_mean, target_variance, attempt_seed)
    
    def _handle_negativity_preserving_correlation(self, series: np.ndarray, 
                                                original_data: np.ndarray,
                                                target_correlation: float,
                                                target_mean: float, 
                                                target_variance: float) -> np.ndarray:
        """Handle negative values while preserving correlation."""
        if target_correlation > 0.95:
            min_val = np.min(series)
            if min_val < 0:
                shift = -min_val + 0.001
                series_shifted = series + shift
                current_mean = np.mean(series_shifted)
                series_shifted = series_shifted + (target_mean - current_mean)
                return np.maximum(series_shifted, 0.001)
        
        return self._exponential_transform_to_nonnegative(series, target_mean, target_variance)
    
    def _exponential_transform_to_nonnegative(self, series: np.ndarray,
                                            target_mean: float,
                                            target_variance: float) -> np.ndarray:
        """Transform series to non-negative using exponential method."""
        current_mean = np.mean(series)
        current_std = np.std(series, ddof=1)
        
        if current_std > 0:
            standardized = (series - current_mean) / current_std
        else:
            standardized = np.zeros(len(series))
        
        min_std = np.min(standardized)
        if min_std < 0:
            shift = -min_std + 0.1
            exp_series = np.exp(standardized + shift)
        else:
            exp_series = np.exp(standardized)
        
        exp_mean = np.mean(exp_series)
        exp_std = np.std(exp_series, ddof=1)
        
        if exp_std > 0 and exp_mean > 0:
            target_std = np.sqrt(target_variance)
            final_series = (exp_series - exp_mean) / exp_std * target_std + target_mean
            return np.maximum(final_series, 0.001)
        else:
            return np.full(len(series), target_mean)
    
    def _generate_regular_correlated_series(self, original_data: np.ndarray,
                                          target_correlation: float,
                                          target_mean: float,
                                          target_variance: float,
                                          attempt_seed: int) -> np.ndarray:
        """Generate correlated series for regular correlations (< 0.99)."""
        n = len(original_data)
        orig_mean = np.mean(original_data)
        orig_std = np.std(original_data, ddof=1)
        
        if orig_std > 0:
            standardized_orig = (original_data - orig_mean) / orig_std
        else:
            standardized_orig = np.zeros(n)
        
        independent_noise = np.random.normal(0, 1, n)
        correlation_factor = np.sqrt(1 - target_correlation**2)
        
        standardized_new = (target_correlation * standardized_orig + 
                          correlation_factor * independent_noise)
        
        target_std = np.sqrt(target_variance)
        new_series = standardized_new * target_std + target_mean
        
        if np.any(new_series < 0):
            if target_mean > 0 and target_variance > 0 and SCIPY_AVAILABLE:
                new_series = self._gamma_transform_to_nonnegative(
                    standardized_new, target_mean, target_variance)
            else:
                new_series = np.maximum(new_series, 0.001)
        
        return new_series
    
    def _gamma_transform_to_nonnegative(self, standardized_series: np.ndarray,
                                      target_mean: float,
                                      target_variance: float) -> np.ndarray:
        """Transform using gamma distribution for non-negativity."""
        uniform_vals = stats.norm.cdf(standardized_series)
        shape_param = target_mean**2 / target_variance
        scale_param = target_variance / target_mean
        
        if shape_param > 0 and scale_param > 0:
            new_series = stats.gamma.ppf(uniform_vals, a=shape_param, scale=scale_param)
            return np.maximum(new_series, 0.001)
        
        return self._exponential_transform_to_nonnegative(
            standardized_series, target_mean, target_variance)
    
    def create_database_schema(self, db_path: str) -> bool:
        """Create SQLite database with required tables."""
        print(f"\nCreating SQLite database: {db_path}")
        
        try:
            # Remove existing database if it exists
            if os.path.exists(db_path):
                os.remove(db_path)
                print(f"  Removed existing database")
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create timeseries_data table
            cursor.execute('''
                CREATE TABLE timeseries_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    combination_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    input_value REAL NOT NULL,
                    generated_value REAL NOT NULL
                )
            ''')
            
            # Create metadata table
            cursor.execute('''
                CREATE TABLE metadata (
                    combination_id TEXT PRIMARY KEY,
                    mean_percentage INTEGER NOT NULL,
                    variance_percentage INTEGER NOT NULL,
                    target_correlation REAL NOT NULL,
                    target_mean REAL NOT NULL,
                    target_variance REAL NOT NULL,
                    target_std REAL NOT NULL,
                    achieved_mean REAL NOT NULL,
                    achieved_variance REAL NOT NULL,
                    achieved_std REAL NOT NULL,
                    achieved_correlation REAL NOT NULL,
                    achieved_autocorrelation REAL NOT NULL,
                    min_value REAL NOT NULL,
                    max_value REAL NOT NULL,
                    negative_values_count INTEGER NOT NULL,
                    mean_error_pct REAL NOT NULL,
                    var_error_pct REAL NOT NULL,
                    corr_error_pct REAL NOT NULL,
                    max_error_pct REAL NOT NULL,
                    is_perfect_case BOOLEAN NOT NULL,
                    non_negative_constraint_met BOOLEAN NOT NULL
                )
            ''')
            
            # Create indexes for better query performance
            cursor.execute('CREATE INDEX idx_timeseries_combination ON timeseries_data(combination_id)')
            cursor.execute('CREATE INDEX idx_timeseries_date ON timeseries_data(date)')
            cursor.execute('CREATE INDEX idx_metadata_correlation ON metadata(target_correlation)')
            cursor.execute('CREATE INDEX idx_metadata_mean_var ON metadata(mean_percentage, variance_percentage)')
            
            conn.commit()
            conn.close()
            
            print(f"✅ Database schema created successfully")
            print(f"  Tables: timeseries_data, metadata")
            print(f"  Indexes: combination_id, date, correlation, mean_percentage/variance_percentage")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating database schema: {e}")
            return False
    
    def generate_all_combinations(self, original_stats: Dict) -> List[Dict]:
        """Generate all combinations and return metadata."""
        print("\n" + "="*60)
        print("GENERATING ALL COMBINATIONS (SQLite OUTPUT)")
        print("="*60)
        
        original_data = original_stats['data']
        original_mean = original_stats['mean']
        original_variance = original_stats['variance']
        n_points = original_stats['n_samples']
        
        total_combinations = (len(self.mean_percentages) * 
                            len(self.variance_percentages) * 
                            len(self.correlations))
        
        print(f"Generating {total_combinations} combinations")
        print(f"Output will be saved to SQLite database:")
        print(f"  timeseries_data table: {total_combinations * n_points:,} rows")
        print(f"  metadata table: {total_combinations} rows")
        
        results = []
        combination_count = 0
        start_time = time.time()
        
        for mean_pct in self.mean_percentages:
            for var_pct in self.variance_percentages:
                for correlation in self.correlations:
                    combination_count += 1
                    
                    target_mean = original_mean * (mean_pct / 100.0)
                    target_variance = original_variance * (var_pct / 100.0)
                    combo_id = f"M{mean_pct}_V{var_pct}_C{correlation:.1f}"
                    
                    print(f"Combination {combination_count:3d}/{total_combinations}: {combo_id}")
                    
                    # Generate series
                    generated_series = self.generate_correlated_series_fixed(
                        original_data, correlation, target_mean, target_variance, 
                        self.random_seed + combination_count * 100)
                    
                    # Calculate achieved statistics
                    achieved_mean = np.mean(generated_series)
                    achieved_var = np.var(generated_series, ddof=1)
                    achieved_std = np.std(generated_series, ddof=1)
                    achieved_corr = np.corrcoef(original_data, generated_series)[0, 1]
                    achieved_autocorr = self._calculate_autocorr(generated_series)
                    min_value = np.min(generated_series)
                    max_value = np.max(generated_series)
                    negative_count = np.sum(generated_series < 0)
                    
                    # Calculate error percentages
                    mean_error_pct = abs(achieved_mean - target_mean) / target_mean * 100 if target_mean > 0 else 0
                    var_error_pct = abs(achieved_var - target_variance) / target_variance * 100 if target_variance > 0 else 0
                    corr_error_pct = abs(achieved_corr - correlation) / abs(correlation) * 100 if correlation != 0 else abs(achieved_corr) * 100
                    
                    # Check for perfect case
                    is_perfect_case = (mean_pct == 100 and var_pct == 100 and abs(correlation - 1.0) < 1e-10)
                    if is_perfect_case:
                        max_diff = np.max(np.abs(generated_series - original_data))
                        print(f"    🎯 PERFECT CASE - Max difference: {max_diff:.8f}")
                    
                    # Store metadata
                    result = {
                        'combination_id': combo_id,
                        'mean_percentage': mean_pct,
                        'variance_percentage': var_pct,
                        'target_correlation': correlation,
                        'target_mean': target_mean,
                        'target_variance': target_variance,
                        'target_std': np.sqrt(target_variance),
                        'achieved_mean': achieved_mean,
                        'achieved_variance': achieved_var,
                        'achieved_std': achieved_std,
                        'achieved_correlation': achieved_corr,
                        'achieved_autocorrelation': achieved_autocorr,
                        'min_value': min_value,
                        'max_value': max_value,
                        'negative_values_count': negative_count,
                        'mean_error_pct': mean_error_pct,
                        'var_error_pct': var_error_pct,
                        'corr_error_pct': corr_error_pct,
                        'max_error_pct': max(mean_error_pct, var_error_pct, corr_error_pct),
                        'is_perfect_case': is_perfect_case,
                        'non_negative_constraint_met': negative_count == 0,
                        'generated_series': generated_series  # Keep for saving
                    }
                    
                    results.append(result)
                    
                    # Progress update
                    if combination_count % 25 == 0 or combination_count == total_combinations:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / combination_count
                        remaining = (total_combinations - combination_count) * avg_time
                        print(f"    ⏱ Progress: {combination_count}/{total_combinations} "
                              f"({combination_count/total_combinations*100:.1f}%) "
                              f"ETA: {remaining/60:.1f} min")
        
        print(f"\n✅ All {total_combinations} combinations generated successfully!")
        return results
    
    def save_to_database(self, original_stats: Dict, results: List[Dict], 
                        db_path: str = "savja_timeseries.db"):
        """Save all data to SQLite database."""
        print(f"\n" + "="*60)
        print("SAVING TO SQLITE DATABASE")
        print("="*60)
        
        original_data = original_stats['data']
        date_strings = original_stats['date_strings']
        n_samples = original_stats['n_samples']
        n_combinations = len(results)
        
        print(f"Database: {db_path}")
        print(f"timeseries_data table: {n_samples * n_combinations:,} rows")
        print(f"metadata table: {n_combinations} rows")
        
        # Create database schema
        if not self.create_database_schema(db_path):
            return False
        
        start_time = time.time()
        
        try:
            conn = sqlite3.connect(db_path)
            
            # Save metadata first (smaller, faster)
            print(f"\nSaving metadata...")
            metadata_rows = []
            
            for result in results:
                metadata_row = (
                    result['combination_id'],
                    result['mean_percentage'],
                    result['variance_percentage'],
                    result['target_correlation'],
                    result['target_mean'],
                    result['target_variance'],
                    result['target_std'],
                    result['achieved_mean'],
                    result['achieved_variance'],
                    result['achieved_std'],
                    result['achieved_correlation'],
                    result['achieved_autocorrelation'],
                    result['min_value'],
                    result['max_value'],
                    result['negative_values_count'],
                    result['mean_error_pct'],
                    result['var_error_pct'],
                    result['corr_error_pct'],
                    result['max_error_pct'],
                    result['is_perfect_case'],
                    result['non_negative_constraint_met']
                )
                metadata_rows.append(metadata_row)
            
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT INTO metadata VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', metadata_rows)
            
            metadata_time = time.time() - start_time
            print(f"✅ Metadata saved: {len(metadata_rows)} rows in {metadata_time:.1f} seconds")
            
            # Save time series data in batches for better performance
            print(f"\nSaving time series data...")
            batch_size = 10000
            total_rows = 0
            
            for combo_idx, result in enumerate(results):
                combo_id = result['combination_id']
                generated_series = result['generated_series']
                
                # Prepare batch data for this combination
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
                cursor.executemany('''
                    INSERT INTO timeseries_data (combination_id, date, input_value, generated_value)
                    VALUES (?,?,?,?)
                ''', batch_data)
                
                total_rows += len(batch_data)
                
                # Commit periodically and show progress
                if (combo_idx + 1) % 10 == 0 or combo_idx == len(results) - 1:
                    conn.commit()
                    elapsed = time.time() - start_time
                    progress = (combo_idx + 1) / len(results)
                    eta = (elapsed / progress - elapsed) if progress > 0 else 0
                    print(f"  Progress: {combo_idx + 1}/{len(results)} combinations "
                          f"({progress*100:.1f}%) - {total_rows:,} rows - "
                          f"ETA: {eta/60:.1f} min")
            
            # Final commit and close
            conn.commit()
            conn.close()
            
            total_time = time.time() - start_time
            db_size = os.path.getsize(db_path) / 1024 / 1024  # MB
            
            print(f"\n✅ SQLite database saved successfully!")
            print(f"  File: {db_path}")
            print(f"  Total rows: {total_rows:,} (timeseries) + {len(metadata_rows)} (metadata)")
            print(f"  Database size: {db_size:.1f} MB")
            print(f"  Total time: {total_time/60:.1f} minutes")
            print(f"  Write speed: {total_rows/total_time:.0f} rows/second")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving to database: {e}")
            return False
    
    def create_summary_report(self, results: List[Dict]) -> str:
        """Create summary report for SQLite generation."""
        print(f"\n" + "="*80)
        print("SQLITE GENERATION SUMMARY")
        print("="*80)
        
        total_combinations = len(results)
        
        # Quality metrics
        mean_errors = [r['mean_error_pct'] for r in results]
        var_errors = [r['var_error_pct'] for r in results]
        corr_errors = [r['corr_error_pct'] for r in results]
        max_errors = [r['max_error_pct'] for r in results]
        
        success_1pct = sum(1 for e in max_errors if e <= 1.0)
        success_5pct = sum(1 for e in max_errors if e <= 5.0)
        negative_violations = sum(1 for r in results if r['negative_values_count'] > 0)
        perfect_cases = sum(1 for r in results if r['is_perfect_case'])
        
        print(f"Generation Summary:")
        print(f"  Total combinations: {total_combinations}")
        print(f"  Perfect cases (M100_V100_C1.0): {perfect_cases}")
        print(f"  Quality within 1% error: {success_1pct}/{total_combinations} ({success_1pct/total_combinations*100:.1f}%)")
        print(f"  Quality within 5% error: {success_5pct}/{total_combinations} ({success_5pct/total_combinations*100:.1f}%)")
        print(f"  Constraint violations: {negative_violations}/{total_combinations}")
        
        print(f"\nSQLite Database Benefits:")
        print(f"  ✅ Efficient storage and indexing")
        print(f"  ✅ Fast SQL queries for filtering and analysis")
        print(f"  ✅ Better performance than CSV for large datasets")
        print(f"  ✅ Relational structure for joining tables")
        print(f"  ✅ Built-in data integrity and consistency")
        
        print(f"\nSample SQL queries:")
        print(f"  -- Get perfect case data")
        print(f"  SELECT * FROM timeseries_data WHERE combination_id = 'M100_V100_C1.0';")
        print(f"  ")
        print(f"  -- Get high correlation metadata")
        print(f"  SELECT * FROM metadata WHERE target_correlation >= 0.9;")
        print(f"  ")
        print(f"  -- Join tables for specific analysis")
        print(f"  SELECT t.date, t.input_value, t.generated_value, m.achieved_correlation")
        print(f"  FROM timeseries_data t JOIN metadata m ON t.combination_id = m.combination_id")
        print(f"  WHERE m.mean_percentage = 100 AND m.variance_percentage = 100;")
        
        return (f"Generated {total_combinations} combinations in SQLite database. "
                f"Quality: {success_1pct} within 1% error, {success_5pct} within 5% error.")


def main_sqlite(csv_file_path: str = "SavjaForClaude.csv", 
                db_path: str = "savja_timeseries.db",
                random_seed: int = 42):
    """Main function for SQLite database generation."""
    print("="*80)
    print("SAVJA SQLite TIME SERIES GENERATOR")
    print("🔧 Fixed correlation handling + SQLite database output")
    print("="*80)
    print(f"Input: {csv_file_path}")
    print(f"Output: {db_path} (SQLite database)")
    print(f"Random seed: {random_seed}")
    print("="*80)
    
    if not os.path.exists(csv_file_path):
        print(f"❌ Error: Input file '{csv_file_path}' not found!")
        return "Input file not found"
    
    # Initialize generator
    generator = SQLiteTimeSeriesGenerator(random_seed=random_seed)
    
    # Read data
    print("\nStep 1: Reading Savja data...")
    original_stats = generator.read_savja_data(csv_file_path)
    if original_stats is None:
        return "Failed to read data"
    
    # Generate combinations
    print("\nStep 2: Generating all combinations...")
    results = generator.generate_all_combinations(original_stats)
    
    # Save to SQLite
    print("\nStep 3: Saving to SQLite database...")
    success = generator.save_to_database(original_stats, results, db_path)
    if not success:
        return "Failed to save database"
    
    # Create summary
    print("\nStep 4: Creating summary...")
    summary = generator.create_summary_report(results)
    
    print(f"\n🎉 SQLITE GENERATION COMPLETED!")
    print(f"📊 Database: {db_path}")
    print(f"📋 Tables: timeseries_data, metadata")
    
    return summary


if __name__ == "__main__":
    import sys
    
    # Command line arguments
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "SavjaForClaude.csv"
    db_file = sys.argv[2] if len(sys.argv) > 2 else "savja_timeseries.db"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    
    result = main_sqlite(csv_file, db_file, seed)
    
    print(f"\nFinal Result: {result}")
    print("\nUsage examples:")
    print("# Connect to database")
    print("import sqlite3")
    print(f"conn = sqlite3.connect('{db_file}')")
    print()
    print("# Query perfect case")
    print("perfect_data = pd.read_sql('''")
    print("    SELECT * FROM timeseries_data WHERE combination_id = 'M100_V100_C1.0'")
    print("''', conn)")
    print()
    print("# Get metadata summary")
    print("metadata = pd.read_sql('SELECT * FROM metadata', conn)")
    print()
    print("# Join tables for analysis")
    print("combined = pd.read_sql('''")
    print("    SELECT t.date, t.input_value, t.generated_value, m.achieved_correlation")
    print("    FROM timeseries_data t JOIN metadata m ON t.combination_id = m.combination_id")
    print("    WHERE m.target_correlation = 1.0")
    print("''', conn)")
    print()
    print("conn.close()")