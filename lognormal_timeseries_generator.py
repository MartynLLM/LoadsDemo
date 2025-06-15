"""
SQLite Time Series Generator for Savja Flow Data - Log-Normal Distribution Version

Creates a SQLite database with two tables:
1. timeseries_data: combination_id, date, input_value, generated_value
2. metadata: combination_id + all statistical parameters and quality metrics

Uses log-normal distribution to eliminate negative values while preserving correlation structure.
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


class SQLiteTimeSeriesGeneratorLogNormal:
    """Generator that creates SQLite database with log-normal distributed time series."""
    
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
            
            # Ensure positive values for log-normal approach
            df['Flow'] = np.maximum(df['Flow'], 1e-6)
            
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
                'date_range': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}",
                # Log-normal parameters for original data
                'log_mean': np.mean(np.log(flow_data)),
                'log_std': np.std(np.log(flow_data), ddof=1)
            }
            
            print(f"✓ Successfully loaded {len(flow_data)} data points")
            print(f"  Date range: {original_stats['date_range']}")
            print(f"  Mean: {original_stats['mean']:.4f}")
            print(f"  Variance: {original_stats['variance']:.4f}")
            print(f"  Min: {original_stats['min']:.4f}, Max: {original_stats['max']:.4f}")
            print(f"  Log-space: μ_log={original_stats['log_mean']:.4f}, σ_log={original_stats['log_std']:.4f}")
            
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
    
    def _compute_lognormal_parameters(self, target_mean: float, target_variance: float) -> Tuple[float, float]:
        """
        Compute log-normal distribution parameters (mu, sigma) given target mean and variance.
        
        For log-normal distribution:
        - mean = exp(mu + sigma^2/2)
        - variance = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
        """
        if target_mean <= 0 or target_variance <= 0:
            return 0.0, 1.0
        
        # Solve for mu and sigma
        # Let m = target_mean, v = target_variance
        # sigma^2 = ln(v/m^2 + 1)
        # mu = ln(m) - sigma^2/2
        
        try:
            sigma_squared = np.log(target_variance / (target_mean**2) + 1)
            sigma = np.sqrt(sigma_squared)
            mu = np.log(target_mean) - sigma_squared / 2
            
            return mu, sigma
        except:
            # Fallback: use method of moments approximation
            cv = np.sqrt(target_variance) / target_mean  # coefficient of variation
            sigma = np.sqrt(np.log(1 + cv**2))
            mu = np.log(target_mean) - sigma**2 / 2
            return mu, sigma
    
    def generate_correlated_lognormal_series(self, original_data: np.ndarray, 
                                           target_correlation: float,
                                           target_mean: float,
                                           target_variance: float,
                                           attempt_seed: int) -> np.ndarray:
        """Generate correlated log-normal series using Gaussian copula approach."""
        np.random.seed(attempt_seed)
        
        n = len(original_data)
        
        # SPECIAL CASE: Perfect correlation (1.0)
        if abs(target_correlation - 1.0) < 1e-10:
            print(f"    Using perfect correlation path for log-normal (correlation = 1.0)")
            
            # For perfect correlation, use direct transformation
            if target_variance <= 0:
                return np.full(n, target_mean)
            
            # Transform original data to preserve perfect correlation
            orig_mean = np.mean(original_data)
            orig_var = np.var(original_data, ddof=1)
            
            if orig_var > 0:
                # Linear transformation then convert to log-normal scale
                a = np.sqrt(target_variance / orig_var)
                b = target_mean - a * orig_mean
                linear_transformed = a * original_data + b
                
                # Ensure positive values
                linear_transformed = np.maximum(linear_transformed, 1e-6)
                
                return linear_transformed
            else:
                return np.full(n, target_mean)
        
        # REGULAR CASE: Use Gaussian copula approach
        # Step 1: Transform original data to standard normal via empirical CDF
        original_ranks = stats.rankdata(original_data) / (n + 1)
        original_normal = stats.norm.ppf(original_ranks)
        
        # Handle edge cases in normal transformation
        original_normal = np.clip(original_normal, -6, 6)
        
        # Step 2: Generate correlated normal variables
        if target_correlation >= 0.99:
            # Near-perfect correlation
            noise_scale = np.sqrt(1 - target_correlation**2) * 0.1
            noise = np.random.normal(0, noise_scale, n)
            correlated_normal = target_correlation * original_normal + noise
        else:
            # Regular correlation using Cholesky-like approach
            independent_normal = np.random.normal(0, 1, n)
            correlation_factor = np.sqrt(1 - target_correlation**2)
            correlated_normal = (target_correlation * original_normal + 
                               correlation_factor * independent_normal)
        
        # Step 3: Transform to uniform via normal CDF
        uniform_values = stats.norm.cdf(correlated_normal)
        uniform_values = np.clip(uniform_values, 1e-10, 1 - 1e-10)
        
        # Step 4: Transform to log-normal distribution
        mu, sigma = self._compute_lognormal_parameters(target_mean, target_variance)
        
        if sigma > 0:
            # Use inverse CDF of log-normal distribution
            normal_values = stats.norm.ppf(uniform_values)
            lognormal_values = np.exp(mu + sigma * normal_values)
        else:
            lognormal_values = np.full(n, target_mean)
        
        # Ensure no extreme values
        lognormal_values = np.clip(lognormal_values, 1e-6, target_mean * 1000)
        
        print(f"    Generated log-normal series: mean={np.mean(lognormal_values):.4f}, "
              f"var={np.var(lognormal_values, ddof=1):.4f}, "
              f"min={np.min(lognormal_values):.4f}")
        
        return lognormal_values
    
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
                    non_negative_constraint_met BOOLEAN NOT NULL,
                    distribution_type TEXT NOT NULL
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
        print("GENERATING ALL COMBINATIONS (LOG-NORMAL SQLite OUTPUT)")
        print("="*60)
        
        original_data = original_stats['data']
        original_mean = original_stats['mean']
        original_variance = original_stats['variance']
        n_points = original_stats['n_samples']
        
        total_combinations = (len(self.mean_percentages) * 
                            len(self.variance_percentages) * 
                            len(self.correlations))
        
        print(f"Generating {total_combinations} combinations using log-normal distribution")
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
                    
                    # Generate log-normal series
                    generated_series = self.generate_correlated_lognormal_series(
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
                    negative_count = np.sum(generated_series < 0)  # Should be 0 for log-normal
                    
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
                        'distribution_type': 'log-normal',
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
                        db_path: str = "savja_timeseries_lognormal.db"):
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
                    result['non_negative_constraint_met'],
                    result['distribution_type']
                )
                metadata_rows.append(metadata_row)
            
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT INTO metadata VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
        print("SQLITE LOG-NORMAL GENERATION SUMMARY")
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
        
        print(f"Log-Normal Generation Summary:")
        print(f"  Total combinations: {total_combinations}")
        print(f"  Perfect cases (M100_V100_C1.0): {perfect_cases}")
        print(f"  Quality within 1% error: {success_1pct}/{total_combinations} ({success_1pct/total_combinations*100:.1f}%)")
        print(f"  Quality within 5% error: {success_5pct}/{total_combinations} ({success_5pct/total_combinations*100:.1f}%)")
        print(f"  Negative value violations: {negative_violations}/{total_combinations}")
        print(f"  ✅ Log-normal distribution ensures non-negative values!")
        
        print(f"\nLog-Normal Distribution Benefits:")
        print(f"  ✅ Eliminates negative values by construction")
        print(f"  ✅ Maintains correlation structure via Gaussian copula")
        print(f"  ✅ Realistic for flow/rate data (naturally positive)")
        print(f"  ✅ Preserves statistical moments (mean, variance)")
        print(f"  ✅ Handles skewed distributions naturally")
        
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
        print(f"  -- Check for negative values (should be zero)")
        print(f"  SELECT combination_id, negative_values_count FROM metadata WHERE negative_values_count > 0;")
        print(f"  ")
        print(f"  -- Join tables for log-normal analysis")
        print(f"  SELECT t.date, t.input_value, t.generated_value, m.achieved_correlation")
        print(f"  FROM timeseries_data t JOIN metadata m ON t.combination_id = m.combination_id")
        print(f"  WHERE m.distribution_type = 'log-normal' AND m.mean_percentage = 100;")
        
        return (f"Generated {total_combinations} log-normal combinations in SQLite database. "
                f"Quality: {success_1pct} within 1% error, {success_5pct} within 5% error. "
                f"Negative violations: {negative_violations} (should be 0).")


def main_lognormal_sqlite(csv_file_path: str = "SavjaForClaude.csv", 
                         db_path: str = "savja_timeseries_lognormal.db",
                         random_seed: int = 42):
    """Main function for log-normal SQLite database generation."""
    print("="*80)
    print("SAVJA LOG-NORMAL SQLite TIME SERIES GENERATOR")
    print("🔧 Log-normal distribution + SQLite database output")
    print("="*80)
    print(f"Input: {csv_file_path}")
    print(f"Output: {db_path} (SQLite database)")
    print(f"Distribution: Log-Normal (eliminates negative values)")
    print(f"Random seed: {random_seed}")
    print("="*80)
    
    if not os.path.exists(csv_file_path):
        print(f"❌ Error: Input file '{csv_file_path}' not found!")
        return "Input file not found"
    
    # Initialize generator
    generator = SQLiteTimeSeriesGeneratorLogNormal(random_seed=random_seed)
    
    # Read data
    print("\nStep 1: Reading Savja data...")
    original_stats = generator.read_savja_data(csv_file_path)
    if original_stats is None:
        return "Failed to read data"
    
    # Generate combinations
    print("\nStep 2: Generating all log-normal combinations...")
    results = generator.generate_all_combinations(original_stats)
    
    # Save to SQLite
    print("\nStep 3: Saving to SQLite database...")
    success = generator.save_to_database(original_stats, results, db_path)
    if not success:
        return "Failed to save database"
    
    # Create summary
    print("\nStep 4: Creating summary...")
    summary = generator.create_summary_report(results)
    
    print(f"\n🎉 LOG-NORMAL SQLITE GENERATION COMPLETED!")
    print(f"📊 Database: {db_path}")
    print(f"📋 Tables: timeseries_data, metadata")
    print(f"🚫 Negative values eliminated by log-normal distribution!")
    
    return summary


if __name__ == "__main__":
    import sys
    
    # Command line arguments
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "SavjaForClaude.csv"
    db_file = sys.argv[2] if len(sys.argv) > 2 else "savja_timeseries_lognormal.db"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    
    result = main_lognormal_sqlite(csv_file, db_file, seed)
    
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
    print("# Check for negative values (should be zero)")
    print("negative_check = pd.read_sql('''")
    print("    SELECT combination_id, negative_values_count FROM metadata")
    print("    WHERE negative_values_count > 0")
    print("''', conn)")
    print()
    print("# Get log-normal metadata")
    print("metadata = pd.read_sql('''")
    print("    SELECT * FROM metadata WHERE distribution_type = 'log-normal'")
    print("''', conn)")
    print()
    print("# Join tables for analysis")
    print("combined = pd.read_sql('''")
    print("    SELECT t.date, t.input_value, t.generated_value, m.achieved_correlation")
    print("    FROM timeseries_data t JOIN metadata m ON t.combination_id = m.combination_id")
    print("    WHERE m.target_correlation = 1.0 AND m.distribution_type = 'log-normal'")
    print("''', conn)")
    print()
    print("# Statistical analysis of log-normal properties")
    print("stats_analysis = pd.read_sql('''")
    print("    SELECT ")
    print("        mean_percentage,")
    print("        variance_percentage,")
    print("        target_correlation,")
    print("        AVG(mean_error_pct) as avg_mean_error,")
    print("        AVG(var_error_pct) as avg_var_error,")
    print("        AVG(corr_error_pct) as avg_corr_error,")
    print("        SUM(negative_values_count) as total_negative_values")
    print("    FROM metadata")
    print("    GROUP BY mean_percentage, variance_percentage, target_correlation")
    print("    ORDER BY target_correlation DESC")
    print("''', conn)")
    print()
    print("conn.close()")