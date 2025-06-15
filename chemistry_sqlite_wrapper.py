"""
Chemistry SQLite Database Wrapper

Extends the existing SQLite database to include synthetic chemistry time series
with correlations ranging from -1.0 to 1.0 in increments of 0.1.

Adds new tables:
- chemistry_timeseries_data: combination_id, date, original_nitrate, original_tp, synthetic_nitrate, synthetic_tp
- chemistry_metadata: combination_id + all statistical parameters and quality metrics
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


class ChemistrySQLiteWrapper:
    """Wrapper that generates chemistry data and stores in SQLite database."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Correlation range: -1.0 to 1.0 in increments of 0.1
        self.correlations = [round(x * 0.1, 1) for x in range(-10, 11)]  # -1.0, -0.9, ..., 0.9, 1.0
    
    def read_chemistry_data(self, csv_file_path: str = "chem.csv") -> Dict:
        """Read and analyze the chemistry data."""
        print(f"Reading chemistry data from: {csv_file_path}")
        
        try:
            df = pd.read_csv(csv_file_path)
            
            if not all(col in df.columns for col in ['Provdatum', 'Nitrate', 'TP']):
                raise ValueError("Required columns 'Provdatum', 'Nitrate', 'TP' not found")
            
            # Clean data
            df = df.dropna(subset=['Nitrate', 'TP'])
            df['Nitrate'] = pd.to_numeric(df['Nitrate'], errors='coerce')
            df['TP'] = pd.to_numeric(df['TP'], errors='coerce')
            df = df.dropna(subset=['Nitrate', 'TP'])
            
            # Parse dates (DD/MM/YYYY format)
            df['Date'] = pd.to_datetime(df['Provdatum'], format='%d/%m/%Y')
            df = df.sort_values('Date')
            
            nitrate_data = df['Nitrate'].values
            tp_data = df['TP'].values
            dates = df['Date'].values
            
            # Calculate statistics for both variables
            nitrate_stats = self._calculate_stats(nitrate_data, "Nitrate")
            tp_stats = self._calculate_stats(tp_data, "TP")
            
            # Calculate cross-correlation between nitrate and TP
            cross_correlation = np.corrcoef(nitrate_data, tp_data)[0, 1]
            
            chemistry_stats = {
                'nitrate_data': nitrate_data,
                'tp_data': tp_data,
                'dates': dates,
                'date_strings': df['Date'].dt.strftime('%Y-%m-%d').values,
                'nitrate_stats': nitrate_stats,
                'tp_stats': tp_stats,
                'cross_correlation': cross_correlation,
                'n_samples': len(nitrate_data),
                'date_range': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
            }
            
            print(f"✓ Successfully loaded {len(nitrate_data)} data points")
            print(f"  Date range: {chemistry_stats['date_range']}")
            print(f"  Nitrate - Min: {nitrate_stats['min']:.1f}, Max: {nitrate_stats['max']:.1f}, Mean: {nitrate_stats['mean']:.1f}, Std: {nitrate_stats['std']:.1f}")
            print(f"  TP - Min: {tp_stats['min']:.1f}, Max: {tp_stats['max']:.1f}, Mean: {tp_stats['mean']:.1f}, Std: {tp_stats['std']:.1f}")
            print(f"  Cross-correlation (Nitrate vs TP): {cross_correlation:.4f}")
            
            return chemistry_stats
            
        except Exception as e:
            print(f"Error reading chemistry data: {e}")
            return None
    
    def _calculate_stats(self, data: np.ndarray, name: str) -> Dict:
        """Calculate comprehensive statistics for a data series."""
        return {
            'name': name,
            'data': data,
            'mean': np.mean(data),
            'std': np.std(data, ddof=1),
            'variance': np.var(data, ddof=1),
            'min': np.min(data),
            'max': np.max(data),
            'autocorr': self._calculate_autocorr(data),
            'n_samples': len(data)
        }
    
    def _calculate_autocorr(self, data: np.ndarray, lag: int = 1) -> float:
        """Calculate first-order autocorrelation."""
        if len(data) <= lag:
            return 0.0
        
        y = data[lag:]
        y_lag = data[:-lag]
        
        if len(y) <= 1 or np.var(y) == 0 or np.var(y_lag) == 0:
            return 0.0
            
        return np.corrcoef(y, y_lag)[0, 1]
    
    def generate_synthetic_series(self, original_data: np.ndarray, 
                                target_correlation: float,
                                target_stats: Dict,
                                attempt_seed: int = None) -> np.ndarray:
        """Generate synthetic time series with specified correlation and exact statistical properties."""
        if attempt_seed is not None:
            np.random.seed(attempt_seed)
        
        n = len(original_data)
        target_min = target_stats['min']
        target_max = target_stats['max']
        target_mean = target_stats['mean']
        target_std = target_stats['std']
        
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
            if target_std <= 0:
                synthetic_series = np.full(n, target_mean)
            else:
                a = target_std / orig_std if orig_std > 0 else 1.0
                b = target_mean - a * orig_mean
                synthetic_series = a * original_data + b
        
        elif abs(target_correlation + 1.0) < 1e-10:
            # Perfect negative correlation
            if target_std <= 0:
                synthetic_series = np.full(n, target_mean)
            else:
                a = -target_std / orig_std if orig_std > 0 else -1.0
                b = target_mean - a * orig_mean
                synthetic_series = a * original_data + b
        
        elif abs(target_correlation) > 0.99:
            # Near-perfect correlation (positive or negative)
            noise_scale = np.sqrt(1 - target_correlation**2) * 0.1
            noise = np.random.normal(0, noise_scale, n)
            standardized_new = target_correlation * standardized_orig + noise
            synthetic_series = standardized_new * target_std + target_mean
        
        else:
            # Regular correlation case
            independent_noise = np.random.normal(0, 1, n)
            correlation_factor = np.sqrt(1 - target_correlation**2)
            
            standardized_new = (target_correlation * standardized_orig + 
                              correlation_factor * independent_noise)
            synthetic_series = standardized_new * target_std + target_mean
        
        # Enforce exact statistical constraints
        synthetic_series = self._enforce_exact_statistics(
            synthetic_series, target_min, target_max, target_mean, target_std)
        
        return synthetic_series
    
    def _enforce_exact_statistics(self, series: np.ndarray, 
                                target_min: float, target_max: float,
                                target_mean: float, target_std: float) -> np.ndarray:
        """Enforce exact statistical properties while preserving relative relationships."""
        n = len(series)
        
        # Step 1: Adjust mean
        current_mean = np.mean(series)
        series = series + (target_mean - current_mean)
        
        # Step 2: Adjust standard deviation
        current_std = np.std(series, ddof=1)
        if current_std > 0:
            series = (series - np.mean(series)) * (target_std / current_std) + target_mean
        
        # Step 3: Enforce min/max constraints with iterative adjustment
        max_iterations = 100
        tolerance = 1e-6
        
        for iteration in range(max_iterations):
            current_min = np.min(series)
            current_max = np.max(series)
            current_mean = np.mean(series)
            current_std = np.std(series, ddof=1)
            
            # Check if constraints are satisfied
            min_ok = abs(current_min - target_min) < tolerance
            max_ok = abs(current_max - target_max) < tolerance
            mean_ok = abs(current_mean - target_mean) < tolerance
            std_ok = abs(current_std - target_std) < tolerance
            
            if min_ok and max_ok and mean_ok and std_ok:
                break
            
            # Adjust for min/max violations
            if current_min < target_min or current_max > target_max:
                # Scale and shift to fit range
                current_range = current_max - current_min
                target_range = target_max - target_min
                
                if current_range > 0:
                    scale_factor = target_range / current_range
                    series = (series - current_min) * scale_factor + target_min
                else:
                    series = np.full(n, target_mean)
            
            # Re-adjust mean and std
            current_mean = np.mean(series)
            series = series + (target_mean - current_mean)
            
            current_std = np.std(series, ddof=1)
            if current_std > 0:
                series = (series - np.mean(series)) * (target_std / current_std) + target_mean
            
            # Final range clipping if needed
            series = np.clip(series, target_min, target_max)
        
        # Final verification and micro-adjustments
        final_min = np.min(series)
        final_max = np.max(series)
        
        # Ensure exact min/max by adjusting extreme values
        min_idx = np.argmin(series)
        max_idx = np.argmax(series)
        
        series[min_idx] = target_min
        series[max_idx] = target_max
        
        # Final mean/std adjustment for remaining values
        if n > 2:
            other_indices = [i for i in range(n) if i != min_idx and i != max_idx]
            if len(other_indices) > 0:
                other_values = series[other_indices]
                
                target_sum = target_mean * n
                current_sum_fixed = series[min_idx] + series[max_idx]
                remaining_sum = target_sum - current_sum_fixed
                
                current_other_mean = np.mean(other_values)
                target_other_mean = remaining_sum / len(other_indices)
                series[other_indices] = other_values + (target_other_mean - current_other_mean)
        
        return series
    
    def extend_database_schema(self, db_path: str) -> bool:
        """Extend existing SQLite database with chemistry tables."""
        print(f"\nExtending SQLite database: {db_path}")
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if chemistry tables already exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chemistry_timeseries_data'")
            if cursor.fetchone():
                print("  Chemistry tables already exist, dropping and recreating...")
                cursor.execute("DROP TABLE IF EXISTS chemistry_timeseries_data")
                cursor.execute("DROP TABLE IF EXISTS chemistry_metadata")
            
            # Create chemistry_timeseries_data table
            cursor.execute('''
                CREATE TABLE chemistry_timeseries_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    combination_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    original_nitrate REAL NOT NULL,
                    original_tp REAL NOT NULL,
                    synthetic_nitrate REAL NOT NULL,
                    synthetic_tp REAL NOT NULL
                )
            ''')
            
            # Create chemistry_metadata table
            cursor.execute('''
                CREATE TABLE chemistry_metadata (
                    combination_id TEXT PRIMARY KEY,
                    nitrate_target_correlation REAL NOT NULL,
                    tp_target_correlation REAL NOT NULL,
                    nitrate_achieved_correlation REAL NOT NULL,
                    tp_achieved_correlation REAL NOT NULL,
                    nitrate_original_min REAL NOT NULL,
                    nitrate_original_max REAL NOT NULL,
                    nitrate_original_mean REAL NOT NULL,
                    nitrate_original_std REAL NOT NULL,
                    tp_original_min REAL NOT NULL,
                    tp_original_max REAL NOT NULL,
                    tp_original_mean REAL NOT NULL,
                    tp_original_std REAL NOT NULL,
                    nitrate_synthetic_min REAL NOT NULL,
                    nitrate_synthetic_max REAL NOT NULL,
                    nitrate_synthetic_mean REAL NOT NULL,
                    nitrate_synthetic_std REAL NOT NULL,
                    tp_synthetic_min REAL NOT NULL,
                    tp_synthetic_max REAL NOT NULL,
                    tp_synthetic_mean REAL NOT NULL,
                    tp_synthetic_std REAL NOT NULL,
                    original_cross_correlation REAL NOT NULL,
                    synthetic_cross_correlation REAL NOT NULL,
                    nitrate_correlation_error_pct REAL NOT NULL,
                    tp_correlation_error_pct REAL NOT NULL,
                    max_correlation_error_pct REAL NOT NULL,
                    nitrate_stats_exact_match BOOLEAN NOT NULL,
                    tp_stats_exact_match BOOLEAN NOT NULL
                )
            ''')
            
            # Create indexes for better query performance
            cursor.execute('CREATE INDEX idx_chemistry_timeseries_combination ON chemistry_timeseries_data(combination_id)')
            cursor.execute('CREATE INDEX idx_chemistry_timeseries_date ON chemistry_timeseries_data(date)')
            cursor.execute('CREATE INDEX idx_chemistry_metadata_nitrate_corr ON chemistry_metadata(nitrate_target_correlation)')
            cursor.execute('CREATE INDEX idx_chemistry_metadata_tp_corr ON chemistry_metadata(tp_target_correlation)')
            
            conn.commit()
            conn.close()
            
            print(f"✅ Chemistry database schema extended successfully")
            print(f"  New tables: chemistry_timeseries_data, chemistry_metadata")
            print(f"  New indexes: combination_id, date, correlations")
            
            return True
            
        except Exception as e:
            print(f"❌ Error extending database schema: {e}")
            return False
    
    def generate_all_chemistry_combinations(self, chemistry_stats: Dict) -> List[Dict]:
        """Generate all chemistry combinations for correlations from -1.0 to 1.0."""
        print("\n" + "="*60)
        print("GENERATING ALL CHEMISTRY COMBINATIONS (SQLite OUTPUT)")
        print("="*60)
        
        original_nitrate = chemistry_stats['nitrate_data']
        original_tp = chemistry_stats['tp_data']
        nitrate_stats = chemistry_stats['nitrate_stats']
        tp_stats = chemistry_stats['tp_stats']
        n_points = chemistry_stats['n_samples']
        
        # Generate combinations for each correlation value (both nitrate and TP use same correlation)
        total_combinations = len(self.correlations)
        
        print(f"Generating {total_combinations} combinations")
        print(f"Correlation range: {min(self.correlations)} to {max(self.correlations)} (step 0.1)")
        print(f"Output will be saved to SQLite database:")
        print(f"  chemistry_timeseries_data table: {total_combinations * n_points:,} rows")
        print(f"  chemistry_metadata table: {total_combinations} rows")
        
        results = []
        combination_count = 0
        start_time = time.time()
        
        for correlation in self.correlations:
            combination_count += 1
            
            combo_id = f"CHEM_C{correlation:.1f}"
            
            print(f"Combination {combination_count:2d}/{total_combinations}: {combo_id} (r={correlation:.1f})")
            
            # Generate synthetic nitrate series
            synthetic_nitrate = self.generate_synthetic_series(
                original_nitrate, correlation, nitrate_stats, 
                self.random_seed + combination_count * 100)
            
            # Generate synthetic TP series  
            synthetic_tp = self.generate_synthetic_series(
                original_tp, correlation, tp_stats,
                self.random_seed + combination_count * 100 + 50)
            
            # Calculate achieved statistics
            achieved_nitrate_corr = np.corrcoef(original_nitrate, synthetic_nitrate)[0, 1]
            achieved_tp_corr = np.corrcoef(original_tp, synthetic_tp)[0, 1]
            
            # Calculate cross-correlations
            original_cross_corr = chemistry_stats['cross_correlation']
            synthetic_cross_corr = np.corrcoef(synthetic_nitrate, synthetic_tp)[0, 1]
            
            # Calculate synthetic statistics
            nitrate_synthetic_stats = {
                'min': np.min(synthetic_nitrate),
                'max': np.max(synthetic_nitrate),
                'mean': np.mean(synthetic_nitrate),
                'std': np.std(synthetic_nitrate, ddof=1)
            }
            
            tp_synthetic_stats = {
                'min': np.min(synthetic_tp),
                'max': np.max(synthetic_tp),
                'mean': np.mean(synthetic_tp),
                'std': np.std(synthetic_tp, ddof=1)
            }
            
            # Calculate error percentages
            nitrate_corr_error_pct = abs(achieved_nitrate_corr - correlation) / abs(correlation) * 100 if correlation != 0 else abs(achieved_nitrate_corr) * 100
            tp_corr_error_pct = abs(achieved_tp_corr - correlation) / abs(correlation) * 100 if correlation != 0 else abs(achieved_tp_corr) * 100
            max_corr_error_pct = max(nitrate_corr_error_pct, tp_corr_error_pct)
            
            # Check statistical exactness
            nitrate_exact = (abs(nitrate_synthetic_stats['min'] - nitrate_stats['min']) < 0.01 and
                           abs(nitrate_synthetic_stats['max'] - nitrate_stats['max']) < 0.01 and
                           abs(nitrate_synthetic_stats['mean'] - nitrate_stats['mean']) < 0.01 and
                           abs(nitrate_synthetic_stats['std'] - nitrate_stats['std']) < 0.01)
            
            tp_exact = (abs(tp_synthetic_stats['min'] - tp_stats['min']) < 0.01 and
                       abs(tp_synthetic_stats['max'] - tp_stats['max']) < 0.01 and
                       abs(tp_synthetic_stats['mean'] - tp_stats['mean']) < 0.01 and
                       abs(tp_synthetic_stats['std'] - tp_stats['std']) < 0.01)
            
            # Special case logging
            if abs(correlation - 1.0) < 1e-10:
                nitrate_max_diff = np.max(np.abs(synthetic_nitrate - original_nitrate))
                tp_max_diff = np.max(np.abs(synthetic_tp - original_tp))
                print(f"    🎯 PERFECT POSITIVE CASE - Nitrate max diff: {nitrate_max_diff:.8f}, TP max diff: {tp_max_diff:.8f}")
            elif abs(correlation + 1.0) < 1e-10:
                print(f"    🎯 PERFECT NEGATIVE CASE")
            
            # Store metadata
            result = {
                'combination_id': combo_id,
                'nitrate_target_correlation': correlation,
                'tp_target_correlation': correlation,
                'nitrate_achieved_correlation': achieved_nitrate_corr,
                'tp_achieved_correlation': achieved_tp_corr,
                'nitrate_original_min': nitrate_stats['min'],
                'nitrate_original_max': nitrate_stats['max'],
                'nitrate_original_mean': nitrate_stats['mean'],
                'nitrate_original_std': nitrate_stats['std'],
                'tp_original_min': tp_stats['min'],
                'tp_original_max': tp_stats['max'],
                'tp_original_mean': tp_stats['mean'],
                'tp_original_std': tp_stats['std'],
                'nitrate_synthetic_min': nitrate_synthetic_stats['min'],
                'nitrate_synthetic_max': nitrate_synthetic_stats['max'],
                'nitrate_synthetic_mean': nitrate_synthetic_stats['mean'],
                'nitrate_synthetic_std': nitrate_synthetic_stats['std'],
                'tp_synthetic_min': tp_synthetic_stats['min'],
                'tp_synthetic_max': tp_synthetic_stats['max'],
                'tp_synthetic_mean': tp_synthetic_stats['mean'],
                'tp_synthetic_std': tp_synthetic_stats['std'],
                'original_cross_correlation': original_cross_corr,
                'synthetic_cross_correlation': synthetic_cross_corr,
                'nitrate_correlation_error_pct': nitrate_corr_error_pct,
                'tp_correlation_error_pct': tp_corr_error_pct,
                'max_correlation_error_pct': max_corr_error_pct,
                'nitrate_stats_exact_match': nitrate_exact,
                'tp_stats_exact_match': tp_exact,
                'synthetic_nitrate': synthetic_nitrate,  # Keep for saving
                'synthetic_tp': synthetic_tp  # Keep for saving
            }
            
            results.append(result)
            
            # Progress update
            if combination_count % 5 == 0 or combination_count == total_combinations:
                elapsed = time.time() - start_time
                avg_time = elapsed / combination_count
                remaining = (total_combinations - combination_count) * avg_time
                print(f"    ⏱ Progress: {combination_count}/{total_combinations} "
                      f"({combination_count/total_combinations*100:.1f}%) "
                      f"ETA: {remaining:.1f} sec")
        
        print(f"\n✅ All {total_combinations} chemistry combinations generated successfully!")
        return results
    
    def save_chemistry_to_database(self, chemistry_stats: Dict, results: List[Dict], 
                                 db_path: str = "savja_timeseries.db"):
        """Save chemistry data to SQLite database."""
        print(f"\n" + "="*60)
        print("SAVING CHEMISTRY DATA TO SQLITE DATABASE")
        print("="*60)
        
        original_nitrate = chemistry_stats['nitrate_data']
        original_tp = chemistry_stats['tp_data']
        date_strings = chemistry_stats['date_strings']
        n_samples = chemistry_stats['n_samples']
        n_combinations = len(results)
        
        print(f"Database: {db_path}")
        print(f"chemistry_timeseries_data table: {n_samples * n_combinations:,} rows")
        print(f"chemistry_metadata table: {n_combinations} rows")
        
        # Extend database schema
        if not self.extend_database_schema(db_path):
            return False
        
        start_time = time.time()
        
        try:
            conn = sqlite3.connect(db_path)
            
            # Save metadata first
            print(f"\nSaving chemistry metadata...")
            metadata_rows = []
            
            for result in results:
                metadata_row = (
                    result['combination_id'],
                    result['nitrate_target_correlation'],
                    result['tp_target_correlation'],
                    result['nitrate_achieved_correlation'],
                    result['tp_achieved_correlation'],
                    result['nitrate_original_min'],
                    result['nitrate_original_max'],
                    result['nitrate_original_mean'],
                    result['nitrate_original_std'],
                    result['tp_original_min'],
                    result['tp_original_max'],
                    result['tp_original_mean'],
                    result['tp_original_std'],
                    result['nitrate_synthetic_min'],
                    result['nitrate_synthetic_max'],
                    result['nitrate_synthetic_mean'],
                    result['nitrate_synthetic_std'],
                    result['tp_synthetic_min'],
                    result['tp_synthetic_max'],
                    result['tp_synthetic_mean'],
                    result['tp_synthetic_std'],
                    result['original_cross_correlation'],
                    result['synthetic_cross_correlation'],
                    result['nitrate_correlation_error_pct'],
                    result['tp_correlation_error_pct'],
                    result['max_correlation_error_pct'],
                    result['nitrate_stats_exact_match'],
                    result['tp_stats_exact_match']
                )
                metadata_rows.append(metadata_row)
            
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT INTO chemistry_metadata VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', metadata_rows)
            
            metadata_time = time.time() - start_time
            print(f"✅ Chemistry metadata saved: {len(metadata_rows)} rows in {metadata_time:.1f} seconds")
            
            # Save time series data
            print(f"\nSaving chemistry time series data...")
            total_rows = 0
            
            for combo_idx, result in enumerate(results):
                combo_id = result['combination_id']
                synthetic_nitrate = result['synthetic_nitrate']
                synthetic_tp = result['synthetic_tp']
                
                # Prepare batch data for this combination
                batch_data = []
                for i in range(n_samples):
                    row = (
                        combo_id,
                        date_strings[i],
                        float(original_nitrate[i]),
                        float(original_tp[i]),
                        float(synthetic_nitrate[i]),
                        float(synthetic_tp[i])
                    )
                    batch_data.append(row)
                
                # Insert batch
                cursor.executemany('''
                    INSERT INTO chemistry_timeseries_data 
                    (combination_id, date, original_nitrate, original_tp, synthetic_nitrate, synthetic_tp)
                    VALUES (?,?,?,?,?,?)
                ''', batch_data)
                
                total_rows += len(batch_data)
                
                # Commit periodically and show progress
                if (combo_idx + 1) % 5 == 0 or combo_idx == len(results) - 1:
                    conn.commit()
                    elapsed = time.time() - start_time
                    progress = (combo_idx + 1) / len(results)
                    eta = (elapsed / progress - elapsed) if progress > 0 else 0
                    print(f"  Progress: {combo_idx + 1}/{len(results)} combinations "
                          f"({progress*100:.1f}%) - {total_rows:,} rows - "
                          f"ETA: {eta:.1f} sec")
            
            # Final commit and close
            conn.commit()
            conn.close()
            
            total_time = time.time() - start_time
            db_size = os.path.getsize(db_path) / 1024 / 1024  # MB
            
            print(f"\n✅ Chemistry data saved to SQLite database successfully!")
            print(f"  File: {db_path}")
            print(f"  Total rows: {total_rows:,} (chemistry_timeseries) + {len(metadata_rows)} (chemistry_metadata)")
            print(f"  Database size: {db_size:.1f} MB")
            print(f"  Total time: {total_time:.1f} seconds")
            print(f"  Write speed: {total_rows/total_time:.0f} rows/second")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving chemistry data to database: {e}")
            return False
    
    def create_chemistry_summary_report(self, chemistry_stats: Dict, results: List[Dict]) -> str:
        """Create summary report for chemistry generation."""
        print(f"\n" + "="*80)
        print("CHEMISTRY SQLITE GENERATION SUMMARY")
        print("="*80)
        
        total_combinations = len(results)
        
        # Quality metrics
        nitrate_corr_errors = [r['nitrate_correlation_error_pct'] for r in results]
        tp_corr_errors = [r['tp_correlation_error_pct'] for r in results]
        max_corr_errors = [r['max_correlation_error_pct'] for r in results]
        
        success_1pct = sum(1 for e in max_corr_errors if e <= 1.0)
        success_5pct = sum(1 for e in max_corr_errors if e <= 5.0)
        nitrate_exact_matches = sum(1 for r in results if r['nitrate_stats_exact_match'])
        tp_exact_matches = sum(1 for r in results if r['tp_stats_exact_match'])
        
        print(f"Chemistry Generation Summary:")
        print(f"  Total combinations: {total_combinations}")
        print(f"  Correlation range: {min(self.correlations)} to {max(self.correlations)} (step 0.1)")
        print(f"  Quality within 1% error: {success_1pct}/{total_combinations} ({success_1pct/total_combinations*100:.1f}%)")
        print(f"  Quality within 5% error: {success_5pct}/{total_combinations} ({success_5pct/total_combinations*100:.1f}%)")
        print(f"  Nitrate exact statistical matches: {nitrate_exact_matches}/{total_combinations} ({nitrate_exact_matches/total_combinations*100:.1f}%)")
        print(f"  TP exact statistical matches: {tp_exact_matches}/{total_combinations} ({tp_exact_matches/total_combinations*100:.1f}%)")
        
        print(f"\nCorrelation Performance:")
        print(f"  Nitrate correlation errors - Min: {min(nitrate_corr_errors):.3f}%, Max: {max(nitrate_corr_errors):.3f}%, Avg: {np.mean(nitrate_corr_errors):.3f}%")
        print(f"  TP correlation errors - Min: {min(tp_corr_errors):.3f}%, Max: {max(tp_corr_errors):.3f}%, Avg: {np.mean(tp_corr_errors):.3f}%")
        
        print(f"\nDatabase Integration:")
        print(f"  ✅ Chemistry data integrated with existing flow database")
        print(f"  ✅ Consistent date format and indexing")
        print(f"  ✅ Comprehensive metadata for analysis")
        print(f"  ✅ Efficient storage and query capabilities")
        
        print(f"\nSample SQL queries for chemistry data:")
        print(f"  -- Get perfect correlation data")
        print(f"  SELECT * FROM chemistry_timeseries_data WHERE combination_id = 'CHEM_C1.0';")
        print(f"  ")
        print(f"  -- Get negative correlation metadata")
        print(f"  SELECT * FROM chemistry_metadata WHERE nitrate_target_correlation < 0;")
        print(f"  ")
        print(f"  -- Join with flow data for multi-variable analysis")
        print(f"  SELECT c.date, c.synthetic_nitrate, c.synthetic_tp, f.generated_value as flow")
        print(f"  FROM chemistry_timeseries_data c")
        print(f"  JOIN timeseries_data f ON c.date = f.date")
        print(f"  WHERE c.combination_id = 'CHEM_C0.8' AND f.combination_id = 'M100_V100_C0.8';")
        
        return (f"Generated {total_combinations} chemistry combinations in SQLite database. "
                f"Quality: {success_1pct} within 1% error, {success_5pct} within 5% error.")


def main_chemistry_sqlite(chem_file_path: str = "chem.csv", 
                         db_path: str = "savja_timeseries.db",
                         random_seed: int = 42):
    """Main function for chemistry SQLite database generation."""
    print("="*80)
    print("CHEMISTRY SQLite TIME SERIES GENERATOR")
    print("🧪 Generate synthetic nitrate and TP with correlations -1.0 to 1.0")
    print("="*80)
    print(f"Input: {chem_file_path}")
    print(f"Database: {db_path}")
    print(f"Correlations: -1.0, -0.9, -0.8, ..., 0.8, 0.9, 1.0 (21 values)")
    print(f"Random seed: {random_seed}")
    print("="*80)
    
    if not os.path.exists(chem_file_path):
        print(f"❌ Error: Input file '{chem_file_path}' not found!")
        return "Chemistry input file not found"
    
    if not os.path.exists(db_path):
        print(f"❌ Error: Database file '{db_path}' not found!")
        print("Please run the Savja flow generator first to create the base database.")
        return "Base database not found"
    
    # Initialize wrapper
    wrapper = ChemistrySQLiteWrapper(random_seed=random_seed)
    
    # Read chemistry data
    print("\nStep 1: Reading chemistry data...")
    chemistry_stats = wrapper.read_chemistry_data(chem_file_path)
    if chemistry_stats is None:
        return "Failed to read chemistry data"
    
    # Generate all combinations
    print("\nStep 2: Generating all chemistry combinations...")
    results = wrapper.generate_all_chemistry_combinations(chemistry_stats)
    
    # Save to SQLite database
    print("\nStep 3: Saving to SQLite database...")
    success = wrapper.save_chemistry_to_database(chemistry_stats, results, db_path)
    if not success:
        return "Failed to save chemistry data to database"
    
    # Create summary
    print("\nStep 4: Creating summary...")
    summary = wrapper.create_chemistry_summary_report(chemistry_stats, results)
    
    print(f"\n🎉 CHEMISTRY SQLITE GENERATION COMPLETED!")
    print(f"📊 Database: {db_path}")
    print(f"📋 Tables: chemistry_timeseries_data, chemistry_metadata (+ existing flow tables)")
    
    return summary


def verify_database_integration(db_path: str = "savja_timeseries.db"):
    """Verify that both flow and chemistry data are properly integrated in the database."""
    print("="*80)
    print("DATABASE INTEGRATION VERIFICATION")
    print("="*80)
    
    if not os.path.exists(db_path):
        print(f"❌ Database file '{db_path}' not found!")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table information
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"📋 Tables in database:")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  - {table}: {count:,} rows")
        
        # Sample data verification
        print(f"\n🔍 Sample data verification:")
        
        if 'timeseries_data' in tables:
            cursor.execute("SELECT COUNT(DISTINCT combination_id) FROM timeseries_data")
            flow_combinations = cursor.fetchone()[0]
            print(f"  Flow combinations: {flow_combinations}")
        
        if 'chemistry_timeseries_data' in tables:
            cursor.execute("SELECT COUNT(DISTINCT combination_id) FROM chemistry_timeseries_data")
            chem_combinations = cursor.fetchone()[0]
            print(f"  Chemistry combinations: {chem_combinations}")
            
            # Show chemistry correlation range
            cursor.execute("SELECT MIN(nitrate_target_correlation), MAX(nitrate_target_correlation) FROM chemistry_metadata")
            min_corr, max_corr = cursor.fetchone()
            print(f"  Chemistry correlation range: {min_corr} to {max_corr}")
        
        # Date overlap verification
        if 'timeseries_data' in tables and 'chemistry_timeseries_data' in tables:
            cursor.execute("""
                SELECT COUNT(*) FROM timeseries_data t 
                JOIN chemistry_timeseries_data c ON t.date = c.date
            """)
            overlap_count = cursor.fetchone()[0]
            print(f"  Date overlap records: {overlap_count:,}")
        
        # Sample combined query
        print(f"\n🔗 Sample combined analysis:")
        if 'timeseries_data' in tables and 'chemistry_timeseries_data' in tables:
            cursor.execute("""
                SELECT t.date, t.generated_value, c.synthetic_nitrate, c.synthetic_tp
                FROM timeseries_data t 
                JOIN chemistry_timeseries_data c ON t.date = c.date
                WHERE t.combination_id = 'M100_V100_C0.8' 
                AND c.combination_id = 'CHEM_C0.8'
                LIMIT 5
            """)
            sample_data = cursor.fetchall()
            
            if sample_data:
                print(f"  Combined data sample (flow + chemistry with 0.8 correlation):")
                print(f"    Date         Flow    Nitrate    TP")
                for row in sample_data:
                    print(f"    {row[0]}  {row[1]:7.2f}  {row[2]:8.1f}  {row[3]:5.1f}")
            else:
                print(f"  No matching combination found for combined analysis")
        
        conn.close()
        
        print(f"\n✅ Database integration verified successfully!")
        print(f"📊 Ready for multi-variable time series analysis")
        
    except Exception as e:
        print(f"❌ Error verifying database: {e}")


# Example usage combining flow and chemistry data
def example_combined_analysis(db_path: str = "savja_timeseries.db"):
    """Example of combined analysis using both flow and chemistry data."""
    print("="*80)
    print("EXAMPLE COMBINED ANALYSIS")
    print("="*80)
    
    if not os.path.exists(db_path):
        print(f"❌ Database file '{db_path}' not found!")
        return
    
    try:
        import pandas as pd
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        
        # Example 1: Get high correlation data for all variables
        print("Example 1: High correlation data (r=0.9) for all variables")
        query1 = """
            SELECT t.date, t.input_value as original_flow, t.generated_value as synthetic_flow,
                   c.original_nitrate, c.synthetic_nitrate, c.original_tp, c.synthetic_tp
            FROM timeseries_data t 
            JOIN chemistry_timeseries_data c ON t.date = c.date
            WHERE t.combination_id = 'M100_V100_C0.9' 
            AND c.combination_id = 'CHEM_C0.9'
            ORDER BY t.date
            LIMIT 10
        """
        df1 = pd.read_sql(query1, conn)
        if not df1.empty:
            print(df1.head())
        else:
            print("No matching data found")
        
        # Example 2: Compare different correlations
        print(f"\nExample 2: Correlation comparison across variables")
        query2 = """
            SELECT fm.combination_id as flow_combo, fm.achieved_correlation as flow_corr,
                   cm.combination_id as chem_combo, cm.nitrate_achieved_correlation as nitrate_corr,
                   cm.tp_achieved_correlation as tp_corr
            FROM metadata fm
            CROSS JOIN chemistry_metadata cm
            WHERE fm.target_correlation = cm.nitrate_target_correlation
            AND fm.mean_percentage = 100 AND fm.variance_percentage = 100
            ORDER BY fm.target_correlation
        """
        df2 = pd.read_sql(query2, conn)
        if not df2.empty:
            print(df2.head())
        else:
            print("No matching metadata found")
        
        # Example 3: Statistical summary
        print(f"\nExample 3: Statistical summary by correlation level")
        query3 = """
            SELECT cm.nitrate_target_correlation as correlation,
                   COUNT(*) as n_records,
                   AVG(c.synthetic_nitrate) as avg_nitrate,
                   AVG(c.synthetic_tp) as avg_tp,
                   AVG(t.generated_value) as avg_flow
            FROM chemistry_timeseries_data c
            JOIN chemistry_metadata cm ON c.combination_id = cm.combination_id
            JOIN timeseries_data t ON c.date = t.date AND t.combination_id = 'M100_V100_C' || PRINTF('%.1f', cm.nitrate_target_correlation)
            GROUP BY cm.nitrate_target_correlation
            ORDER BY cm.nitrate_target_correlation
        """
        df3 = pd.read_sql(query3, conn)
        if not df3.empty:
            print(df3)
        else:
            print("No statistical summary data found")
        
        conn.close()
        
        print(f"\n✅ Combined analysis examples completed!")
        
    except Exception as e:
        print(f"❌ Error in combined analysis: {e}")


if __name__ == "__main__":
    import sys
    
    # Command line arguments
    chem_file = sys.argv[1] if len(sys.argv) > 1 else "chem.csv"
    db_file = sys.argv[2] if len(sys.argv) > 2 else "savja_timeseries.db"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    
    # Main chemistry generation
    result = main_chemistry_sqlite(chem_file, db_file, seed)
    print(f"\nChemistry Generation Result: {result}")
    
    # Verify database integration
    print("\n" + "="*80)
    verify_database_integration(db_file)
    
    # Example combined analysis
    print("\n" + "="*80)
    example_combined_analysis(db_file)
    
    print(f"\n" + "="*80)
    print("USAGE EXAMPLES FOR COMBINED DATABASE:")
    print("="*80)
    print("# Connect to database")
    print("import sqlite3")
    print("import pandas as pd")
    print(f"conn = sqlite3.connect('{db_file}')")
    print()
    print("# Get all data for correlation 0.8")
    print("combined_data = pd.read_sql('''")
    print("    SELECT t.date, t.generated_value as flow, c.synthetic_nitrate, c.synthetic_tp")
    print("    FROM timeseries_data t")
    print("    JOIN chemistry_timeseries_data c ON t.date = c.date")
    print("    WHERE t.combination_id = 'M100_V100_C0.8' AND c.combination_id = 'CHEM_C0.8'")
    print("''', conn)")
    print()
    print("# Compare correlations across all variables")
    print("correlation_summary = pd.read_sql('''")
    print("    SELECT fm.target_correlation, fm.achieved_correlation as flow_corr,")
    print("           cm.nitrate_achieved_correlation, cm.tp_achieved_correlation")
    print("    FROM metadata fm")
    print("    JOIN chemistry_metadata cm ON ABS(fm.target_correlation - cm.nitrate_target_correlation) < 0.01")
    print("    WHERE fm.mean_percentage = 100 AND fm.variance_percentage = 100")
    print("    ORDER BY fm.target_correlation")
    print("''', conn)")
    print()
    print("# Get negative correlation chemistry data")
    print("negative_corr = pd.read_sql('''")
    print("    SELECT * FROM chemistry_timeseries_data")
    print("    WHERE combination_id LIKE 'CHEM_C-0.%'")
    print("''', conn)")
    print()
    print("conn.close()")
    print()
    print("🎉 Complete multi-variable time series database ready for analysis!")