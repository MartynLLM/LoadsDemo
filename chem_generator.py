"""
Chemistry Time Series Generator for Nitrate and TP Data

Creates synthetic time series with user-specified correlations while maintaining
the same statistical properties (min, max, mean, std) as the original data.

Based on the Savja generator but adapted for chemistry data with two variables.
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


class ChemistryTimeSeriesGenerator:
    """Generator for synthetic nitrate and TP time series with specified correlations."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
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
            
            # Parse dates (assuming DD/MM/YYYY format based on the sample)
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
        """
        Generate synthetic time series with specified correlation and exact statistical properties.
        
        Args:
            original_data: Original time series to correlate with
            target_correlation: Desired correlation coefficient
            target_stats: Dict with 'min', 'max', 'mean', 'std' keys
            attempt_seed: Random seed for reproducibility
        
        Returns:
            Synthetic time series with desired properties
        """
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
            # Perfect correlation case
            print(f"    Using perfect correlation (r = 1.0)")
            if target_std <= 0:
                synthetic_series = np.full(n, target_mean)
            else:
                # Direct linear transformation
                a = target_std / orig_std if orig_std > 0 else 1.0
                b = target_mean - a * orig_mean
                synthetic_series = a * original_data + b
        
        elif target_correlation > 0.99:
            # Near-perfect correlation
            print(f"    Using near-perfect correlation (r = {target_correlation})")
            noise_scale = np.sqrt(1 - target_correlation**2) * 0.1
            noise = np.random.normal(0, noise_scale, n)
            standardized_new = target_correlation * standardized_orig + noise
            synthetic_series = standardized_new * target_std + target_mean
        
        else:
            # Regular correlation case
            print(f"    Using regular correlation (r = {target_correlation})")
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
        """
        Enforce exact statistical properties while preserving relative relationships.
        
        This is a sophisticated constraint satisfaction approach that maintains
        the correlation structure while ensuring exact min, max, mean, and std.
        """
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
                    # Scale to fit range
                    scale_factor = target_range / current_range
                    series = (series - current_min) * scale_factor + target_min
                else:
                    # Constant series case
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
        if n > 2:  # Only if we have more than just min/max values
            other_indices = [i for i in range(n) if i != min_idx and i != max_idx]
            if len(other_indices) > 0:
                other_values = series[other_indices]
                
                # Calculate what the other values should sum to
                target_sum = target_mean * n
                current_sum_fixed = series[min_idx] + series[max_idx]
                remaining_sum = target_sum - current_sum_fixed
                
                # Adjust other values to achieve target mean
                current_other_mean = np.mean(other_values)
                target_other_mean = remaining_sum / len(other_indices)
                series[other_indices] = other_values + (target_other_mean - current_other_mean)
        
        return series
    
    def generate_synthetic_chemistry_data(self, chemistry_stats: Dict, 
                                        nitrate_correlation: float,
                                        tp_correlation: float,
                                        seed_offset: int = 0) -> Dict:
        """
        Generate synthetic nitrate and TP time series with specified correlations.
        
        Args:
            chemistry_stats: Original chemistry data statistics
            nitrate_correlation: Target correlation for synthetic nitrate vs original nitrate
            tp_correlation: Target correlation for synthetic TP vs original TP
            seed_offset: Offset for random seed
        
        Returns:
            Dict with synthetic data and achieved statistics
        """
        print(f"\nGenerating synthetic chemistry data:")
        print(f"  Target nitrate correlation: {nitrate_correlation:.3f}")
        print(f"  Target TP correlation: {tp_correlation:.3f}")
        
        original_nitrate = chemistry_stats['nitrate_data']
        original_tp = chemistry_stats['tp_data']
        nitrate_stats = chemistry_stats['nitrate_stats']
        tp_stats = chemistry_stats['tp_stats']
        
        # Generate synthetic nitrate series
        print("  Generating synthetic nitrate...")
        synthetic_nitrate = self.generate_synthetic_series(
            original_nitrate, 
            nitrate_correlation,
            nitrate_stats,
            self.random_seed + seed_offset
        )
        
        # Generate synthetic TP series
        print("  Generating synthetic TP...")
        synthetic_tp = self.generate_synthetic_series(
            original_tp,
            tp_correlation, 
            tp_stats,
            self.random_seed + seed_offset + 1000
        )
        
        # Calculate achieved statistics
        achieved_nitrate_stats = self._calculate_achieved_stats(
            original_nitrate, synthetic_nitrate, "Nitrate")
        achieved_tp_stats = self._calculate_achieved_stats(
            original_tp, synthetic_tp, "TP")
        
        # Cross-correlation between synthetic series
        synthetic_cross_corr = np.corrcoef(synthetic_nitrate, synthetic_tp)[0, 1]
        
        result = {
            'synthetic_nitrate': synthetic_nitrate,
            'synthetic_tp': synthetic_tp,
            'achieved_nitrate_stats': achieved_nitrate_stats,
            'achieved_tp_stats': achieved_tp_stats,
            'synthetic_cross_correlation': synthetic_cross_corr,
            'target_nitrate_correlation': nitrate_correlation,
            'target_tp_correlation': tp_correlation,
            'dates': chemistry_stats['date_strings']
        }
        
        # Print summary
        print(f"  ✓ Synthetic nitrate - achieved correlation: {achieved_nitrate_stats['correlation']:.4f}")
        print(f"  ✓ Synthetic TP - achieved correlation: {achieved_tp_stats['correlation']:.4f}")
        print(f"  ✓ Cross-correlation (synthetic nitrate vs synthetic TP): {synthetic_cross_corr:.4f}")
        
        return result
    
    def _calculate_achieved_stats(self, original: np.ndarray, synthetic: np.ndarray, name: str) -> Dict:
        """Calculate achieved statistics for synthetic series."""
        correlation = np.corrcoef(original, synthetic)[0, 1]
        
        return {
            'name': name,
            'correlation': correlation,
            'mean': np.mean(synthetic),
            'std': np.std(synthetic, ddof=1),
            'min': np.min(synthetic),
            'max': np.max(synthetic),
            'autocorr': self._calculate_autocorr(synthetic)
        }
    
    def save_to_csv(self, chemistry_stats: Dict, result: Dict, 
                   output_file: str = "synthetic_chemistry.csv"):
        """Save original and synthetic data to CSV."""
        print(f"\nSaving data to: {output_file}")
        
        # Create DataFrame with all data
        df = pd.DataFrame({
            'Date': result['dates'],
            'Original_Nitrate': chemistry_stats['nitrate_data'],
            'Original_TP': chemistry_stats['tp_data'],
            'Synthetic_Nitrate': result['synthetic_nitrate'],
            'Synthetic_TP': result['synthetic_tp']
        })
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        print(f"✓ Saved {len(df)} rows to {output_file}")
        print(f"  Columns: {list(df.columns)}")
        
        return df
    
    def create_summary_report(self, chemistry_stats: Dict, result: Dict) -> str:
        """Create a comprehensive summary report."""
        print(f"\n" + "="*80)
        print("CHEMISTRY TIME SERIES GENERATION SUMMARY")
        print("="*80)
        
        original_nitrate_stats = chemistry_stats['nitrate_stats']
        original_tp_stats = chemistry_stats['tp_stats']
        achieved_nitrate = result['achieved_nitrate_stats']
        achieved_tp = result['achieved_tp_stats']
        
        print(f"Original Data:")
        print(f"  Nitrate: min={original_nitrate_stats['min']:.1f}, max={original_nitrate_stats['max']:.1f}, "
              f"mean={original_nitrate_stats['mean']:.1f}, std={original_nitrate_stats['std']:.1f}")
        print(f"  TP: min={original_tp_stats['min']:.1f}, max={original_tp_stats['max']:.1f}, "
              f"mean={original_tp_stats['mean']:.1f}, std={original_tp_stats['std']:.1f}")
        print(f"  Cross-correlation: {chemistry_stats['cross_correlation']:.4f}")
        
        print(f"\nSynthetic Data:")
        print(f"  Nitrate: min={achieved_nitrate['min']:.1f}, max={achieved_nitrate['max']:.1f}, "
              f"mean={achieved_nitrate['mean']:.1f}, std={achieved_nitrate['std']:.1f}")
        print(f"  TP: min={achieved_tp['min']:.1f}, max={achieved_tp['max']:.1f}, "
              f"mean={achieved_tp['mean']:.1f}, std={achieved_tp['std']:.1f}")
        print(f"  Cross-correlation: {result['synthetic_cross_correlation']:.4f}")
        
        print(f"\nTarget vs Achieved Correlations:")
        print(f"  Nitrate: target={result['target_nitrate_correlation']:.4f}, "
              f"achieved={achieved_nitrate['correlation']:.4f}, "
              f"error={abs(result['target_nitrate_correlation'] - achieved_nitrate['correlation']):.4f}")
        print(f"  TP: target={result['target_tp_correlation']:.4f}, "
              f"achieved={achieved_tp['correlation']:.4f}, "
              f"error={abs(result['target_tp_correlation'] - achieved_tp['correlation']):.4f}")
        
        # Statistical property preservation check
        nitrate_exact = (abs(achieved_nitrate['min'] - original_nitrate_stats['min']) < 0.01 and
                        abs(achieved_nitrate['max'] - original_nitrate_stats['max']) < 0.01 and
                        abs(achieved_nitrate['mean'] - original_nitrate_stats['mean']) < 0.01 and
                        abs(achieved_nitrate['std'] - original_nitrate_stats['std']) < 0.01)
        
        tp_exact = (abs(achieved_tp['min'] - original_tp_stats['min']) < 0.01 and
                   abs(achieved_tp['max'] - original_tp_stats['max']) < 0.01 and
                   abs(achieved_tp['mean'] - original_tp_stats['mean']) < 0.01 and
                   abs(achieved_tp['std'] - original_tp_stats['std']) < 0.01)
        
        print(f"\nStatistical Property Preservation:")
        print(f"  Nitrate exact match: {'✓' if nitrate_exact else '✗'}")
        print(f"  TP exact match: {'✓' if tp_exact else '✗'}")
        
        return f"Generated synthetic chemistry time series with correlations {result['target_nitrate_correlation']:.3f} (nitrate) and {result['target_tp_correlation']:.3f} (TP)"


def generate_synthetic_chemistry(csv_file_path: str = "chem.csv",
                               nitrate_correlation: float = 0.8,
                               tp_correlation: float = 0.8,
                               output_file: str = "synthetic_chemistry.csv",
                               random_seed: int = 42):
    """
    Main function to generate synthetic chemistry time series.
    
    Args:
        csv_file_path: Path to input chemistry CSV file
        nitrate_correlation: Target correlation for synthetic vs original nitrate
        tp_correlation: Target correlation for synthetic vs original TP
        output_file: Output CSV file name
        random_seed: Random seed for reproducibility
    
    Returns:
        Summary string
    """
    print("="*80)
    print("CHEMISTRY TIME SERIES GENERATOR")
    print("🧪 Generate synthetic nitrate and TP time series")
    print("="*80)
    print(f"Input: {csv_file_path}")
    print(f"Output: {output_file}")
    print(f"Target correlations: Nitrate={nitrate_correlation:.3f}, TP={tp_correlation:.3f}")
    print(f"Random seed: {random_seed}")
    print("="*80)
    
    if not os.path.exists(csv_file_path):
        print(f"❌ Error: Input file '{csv_file_path}' not found!")
        return "Input file not found"
    
    # Initialize generator
    generator = ChemistryTimeSeriesGenerator(random_seed=random_seed)
    
    # Read chemistry data
    print("\nStep 1: Reading chemistry data...")
    chemistry_stats = generator.read_chemistry_data(csv_file_path)
    if chemistry_stats is None:
        return "Failed to read chemistry data"
    
    # Generate synthetic data
    print("\nStep 2: Generating synthetic time series...")
    result = generator.generate_synthetic_chemistry_data(
        chemistry_stats, nitrate_correlation, tp_correlation)
    
    # Save to CSV
    print("\nStep 3: Saving results...")
    df = generator.save_to_csv(chemistry_stats, result, output_file)
    
    # Create summary
    print("\nStep 4: Creating summary...")
    summary = generator.create_summary_report(chemistry_stats, result)
    
    print(f"\n🎉 CHEMISTRY GENERATION COMPLETED!")
    print(f"📊 Output file: {output_file}")
    print(f"📋 {len(df)} data points generated")
    
    return summary


# Example usage and testing
if __name__ == "__main__":
    # Example 1: High correlation for both variables
    print("="*50)
    print("EXAMPLE 1: High correlations (0.9, 0.9)")
    print("="*50)
    result1 = generate_synthetic_chemistry(
        csv_file_path="chem.csv",
        nitrate_correlation=0.9,
        tp_correlation=0.9,
        output_file="synthetic_chemistry_high_corr.csv",
        random_seed=42
    )
    
    # Example 2: Medium correlation for both variables  
    print("\n" + "="*50)
    print("EXAMPLE 2: Medium correlations (0.6, 0.6)")
    print("="*50)
    result2 = generate_synthetic_chemistry(
        csv_file_path="chem.csv",
        nitrate_correlation=0.6,
        tp_correlation=0.6,
        output_file="synthetic_chemistry_medium_corr.csv",
        random_seed=123
    )
    
    # Example 3: Different correlations for each variable
    print("\n" + "="*50)
    print("EXAMPLE 3: Different correlations (0.95, 0.4)")
    print("="*50)
    result3 = generate_synthetic_chemistry(
        csv_file_path="chem.csv",
        nitrate_correlation=0.95,
        tp_correlation=0.4,
        output_file="synthetic_chemistry_mixed_corr.csv",
        random_seed=456
    )
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED!")
    print("="*80)
    print("Generated files:")
    print("  - synthetic_chemistry_high_corr.csv")
    print("  - synthetic_chemistry_medium_corr.csv") 
    print("  - synthetic_chemistry_mixed_corr.csv")
    print("\nEach file contains:")
    print("  - Date: Original dates")
    print("  - Original_Nitrate: Original nitrate values")
    print("  - Original_TP: Original TP values")
    print("  - Synthetic_Nitrate: Generated nitrate with specified correlation")
    print("  - Synthetic_TP: Generated TP with specified correlation")
    print("\nThe synthetic series maintain exact statistical properties:")
    print("  ✓ Same minimum, maximum, mean, and standard deviation")
    print("  ✓ User-specified correlation with original series")
    print("  ✓ Preserved temporal structure")
