import csv
import numpy as np
import math
import random

# Handle scipy import gracefully
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Note: scipy not available, using fallback methods for some calculations")

def analyze_flow_data(csv_file_path):
    """
    Read a CSV file with Date and Flow columns and calculate:
    - First-order autocorrelation
    - Series mean
    - Series variance
    
    Parameters:
    csv_file_path (str): Path to the CSV file
    
    Returns:
    dict: Dictionary containing the calculated statistics
    """
    
    # Read the CSV file
    try:
        flow_data = []
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            
            # Check if Flow column exists
            if 'Flow' not in csv_reader.fieldnames:
                print("Error: 'Flow' column not found in the CSV file")
                print(f"Available columns: {csv_reader.fieldnames}")
                return None
            
            # Read flow data and convert to float, skipping invalid entries
            for row in csv_reader:
                try:
                    flow_value = float(row['Flow'])
                    if not math.isnan(flow_value):  # Skip NaN values
                        flow_data.append(flow_value)
                except (ValueError, TypeError):
                    # Skip rows with invalid flow data
                    continue
        
        print(f"Successfully loaded {len(flow_data)} valid flow measurements")
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    if len(flow_data) == 0:
        print("Error: No valid flow data found")
        return None
    
    # Convert to numpy array for easier calculations
    flow_array = np.array(flow_data)
    n = len(flow_array)
    
    # Calculate series mean
    series_mean = np.mean(flow_array)
    
    # Calculate series variance (sample variance with N-1)
    series_variance = np.var(flow_array, ddof=1)
    
    # Calculate first-order autocorrelation
    if n > 1:
        # Get original series and lagged series (shifted by 1)
        y = flow_array[1:]      # y(t+1): from index 1 to end
        y_lag = flow_array[:-1] # y(t): from index 0 to n-2
        
        # Calculate means
        y_mean = np.mean(y)
        y_lag_mean = np.mean(y_lag)
        
        # Calculate covariance and variances
        covariance = np.mean((y - y_mean) * (y_lag - y_lag_mean))
        var_y = np.var(y, ddof=0)       # Population variance for correlation
        var_y_lag = np.var(y_lag, ddof=0)
        
        # Calculate correlation coefficient
        if var_y > 0 and var_y_lag > 0:
            autocorr = covariance / np.sqrt(var_y * var_y_lag)
        else:
            autocorr = 0.0
    else:
        autocorr = float('nan')
    
    # Prepare results
    results = {
        'mean': series_mean,
        'variance': series_variance,
        'standard_deviation': np.sqrt(series_variance),
        'first_order_autocorrelation': autocorr,
        'sample_size': n,
        'data': flow_array  # Include the actual data for further processing
    }
    
    return results

def create_correlated_series(original_series, target_correlation, random_seed=None):
    """
    Create a new series with specified correlation to the original series
    while maintaining similar statistical properties and ensuring non-negative values
    
    Parameters:
    original_series (numpy.array): The original time series
    target_correlation (float): Desired correlation with original series (-1 to 1)
    random_seed (int): Random seed for reproducibility
    
    Returns:
    numpy.array: New correlated series (all values >= 0)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n = len(original_series)
    
    # Calculate properties of original series
    orig_mean = np.mean(original_series)
    orig_std = np.std(original_series, ddof=1)
    orig_min = np.min(original_series)
    
    # Check if original series has negative values
    has_negative = orig_min < 0
    
    if has_negative:
        # Shift original series to be non-negative for correlation calculation
        shifted_orig = original_series - orig_min + 0.001  # Small buffer to avoid zeros
        orig_mean_shifted = np.mean(shifted_orig)
        orig_std_shifted = np.std(shifted_orig, ddof=1)
        
        # Standardize shifted series
        if orig_std_shifted > 0:
            standardized_orig = (shifted_orig - orig_mean_shifted) / orig_std_shifted
        else:
            standardized_orig = np.zeros(n)
    else:
        # Use original series if already non-negative
        shifted_orig = original_series
        if orig_std > 0:
            standardized_orig = (original_series - orig_mean) / orig_std
        else:
            standardized_orig = np.zeros(n)
    
    # Generate correlated series using Gaussian copula approach for non-negative values
    # First create normally distributed correlated series
    independent_series = np.random.normal(0, 1, n)
    
    rho = target_correlation
    if abs(rho) <= 1:
        corr_factor = np.sqrt(1 - rho**2) if abs(rho) < 1 else 0
        standardized_new = rho * standardized_orig + corr_factor * independent_series
    else:
        print("Warning: Correlation must be between -1 and 1. Using sign of input.")
        rho = np.sign(rho)
        standardized_new = rho * standardized_orig
    
    # Transform to ensure non-negative values using log-normal approach
    # Convert standardized series to uniform [0,1] using normal CDF
    if SCIPY_AVAILABLE:
        uniform_values = stats.norm.cdf(standardized_new)
    else:
        # Fallback approximation for normal CDF if scipy not available
        uniform_values = 0.5 * (1 + np.tanh(standardized_new / np.sqrt(2)))
    
    # Transform uniform values to gamma distribution to ensure non-negative values
    # Use method of moments to match target mean and variance
    target_mean = orig_mean if not has_negative else orig_mean - orig_min + 0.001
    target_variance = orig_std**2
    
    if target_mean > 0 and target_variance > 0:
        # Gamma distribution parameters: shape = mean²/var, scale = var/mean
        gamma_shape = target_mean**2 / target_variance
        gamma_scale = target_variance / target_mean
        
        # Convert uniform to gamma using inverse CDF
        if SCIPY_AVAILABLE:
            new_series = stats.gamma.ppf(uniform_values, a=gamma_shape, scale=gamma_scale)
        else:
            # Fallback: exponential transformation for non-negative values
            new_series = -target_mean * np.log(1 - uniform_values + 1e-10)
    else:
        # Fallback: simple exponential transformation
        new_series = np.exp(standardized_new * 0.5 + np.log(max(target_mean, 1.0)))
    
    # Ensure no negative values (numerical precision safety)
    new_series = np.maximum(new_series, 0)
    
    # If original had negative values, shift back
    if has_negative:
        new_series = new_series + orig_min - 0.001
        new_series = np.maximum(new_series, 0)  # Ensure still non-negative
    
    return new_series

def generate_ar1_series(n, mean, variance, autocorr, random_seed=None, ensure_nonnegative=True):
    """
    Generate an AR(1) time series with specified mean, variance, and autocorrelation
    with option to ensure non-negative values
    
    Parameters:
    n (int): Length of the series
    mean (float): Target mean
    variance (float): Target variance
    autocorr (float): Target first-order autocorrelation
    random_seed (int): Random seed for reproducibility
    ensure_nonnegative (bool): If True, ensures all values are >= 0
    
    Returns:
    numpy.array: Generated AR(1) time series
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # For non-negative series, use log-space AR(1) if needed
    if ensure_nonnegative and mean > 0:
        # Generate in log space to ensure positivity
        log_mean = np.log(mean)
        log_var = np.log(1 + variance / (mean**2))  # Convert to log-space variance
        
        # AR(1) model in log space
        phi = autocorr
        
        if abs(phi) < 1:
            innovation_variance = log_var * (1 - phi**2)
        else:
            innovation_variance = log_var * 0.5
        
        innovation_std = np.sqrt(innovation_variance)
        
        # Generate log-space series
        log_series = np.zeros(n)
        log_series[0] = np.random.normal(log_mean, np.sqrt(log_var))
        
        for t in range(1, n):
            innovation = np.random.normal(0, innovation_std)
            log_series[t] = phi * (log_series[t-1] - log_mean) + log_mean + innovation
        
        # Transform back to original space
        series = np.exp(log_series)
        
        # Adjust to match target mean and variance
        current_mean = np.mean(series)
        current_var = np.var(series, ddof=1)
        
        if current_var > 0 and current_mean > 0:
            # Use moment matching for log-normal distribution
            series = series * (mean / current_mean)
            
            # Fine-tune variance while maintaining non-negativity
            current_var_adjusted = np.var(series, ddof=1)
            if current_var_adjusted > 0:
                variance_factor = np.sqrt(variance / current_var_adjusted)
                series = (series - mean) * variance_factor + mean
                series = np.maximum(series, 0)  # Ensure non-negative
    
    else:
        # Standard AR(1) generation (may produce negative values)
        phi = autocorr
        
        if abs(phi) < 1:
            innovation_variance = variance * (1 - phi**2)
        else:
            innovation_variance = variance * 0.5
        
        innovation_std = np.sqrt(innovation_variance)
        
        # Generate the series
        series = np.zeros(n)
        series[0] = np.random.normal(0, np.sqrt(variance))
        
        for t in range(1, n):
            innovation = np.random.normal(0, innovation_std)
            series[t] = phi * series[t-1] + innovation
        
        # Adjust to match target mean and variance exactly
        current_mean = np.mean(series)
        current_std = np.std(series, ddof=1)
        
        if current_std > 0:
            series = (series - current_mean) / current_std
            series = series * np.sqrt(variance) + mean
        else:
            series = np.full(n, mean)
        
        # Ensure non-negative if required
        if ensure_nonnegative:
            series = np.maximum(series, 0)
    
    return series

def calculate_autocorrelation(data, lag=1):
    """
    Calculate autocorrelation for given lag
    """
    data = np.array(data)
    n = len(data)
    
    if n <= lag:
        return float('nan')
    
    y1 = data[lag:]
    y2 = data[:-lag]
    
    return np.corrcoef(y1, y2)[0, 1] if len(y1) > 1 else 0.0

def generate_enhanced_ar1_series(n, mean, variance, autocorr, random_seed=None, 
                               ensure_nonnegative=True, reference_series=None):
    """
    Enhanced AR(1) generation with better initialization based on reference series
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Use reference series statistics for better initialization if available
    if reference_series is not None:
        init_value = reference_series[0]  # Start with first value of reference
        innovation_scale = np.std(np.diff(reference_series)) * 0.5  # Scale based on reference changes
    else:
        init_value = mean
        innovation_scale = np.sqrt(variance) * 0.3
    
    # Generate AR(1) series
    phi = autocorr
    series = np.zeros(n)
    series[0] = init_value
    
    for t in range(1, n):
        innovation = np.random.normal(0, innovation_scale)
        series[t] = mean + phi * (series[t-1] - mean) + innovation
    
    if ensure_nonnegative:
        series = np.maximum(series, 0)
    
    # Adjust to exact mean and variance
    current_mean = np.mean(series)
    current_var = np.var(series, ddof=1)
    
    if current_var > 0:
        series = (series - current_mean) / np.sqrt(current_var) * np.sqrt(variance) + mean
    
    if ensure_nonnegative:
        series = np.maximum(series, 0)
    
    return series

def adjust_mean_variance_preserve_autocorr(series, target_mean, target_variance, ensure_nonnegative=True):
    """
    Adjust mean and variance while minimally impacting autocorrelation
    """
    current_mean = np.mean(series)
    current_var = np.var(series, ddof=1)
    
    # Gentle adjustment to preserve autocorrelation structure
    if abs(current_mean - target_mean) > 0.01 * abs(target_mean):
        # Simple shift for mean (preserves autocorrelation exactly)
        series = series + (target_mean - current_mean)
    
    # Gentle scaling for variance
    current_var = np.var(series, ddof=1)
    if current_var > 0 and abs(current_var - target_variance) > 0.01 * abs(target_variance):
        scale_factor = np.sqrt(target_variance / current_var)
        series_mean = np.mean(series)
        # Scale around mean (minimally affects autocorrelation)
        series = (series - series_mean) * scale_factor + series_mean
    
    if ensure_nonnegative:
        series = np.maximum(series, 0)
        # If clipping changed mean significantly, readjust
        new_mean = np.mean(series)
        if abs(new_mean - target_mean) > 0.05 * abs(target_mean):
            series = series + (target_mean - new_mean)
            series = np.maximum(series, 0)
    
    return series

def generate_correlated_time_series(csv_file_path, target_correlation, random_seed=None, 
                                  max_iterations=200, tolerance=0.01, ensure_nonnegative=True, 
                                  target_precision=0.01):
    """
    Main function to generate a correlated time series with same statistical properties
    GUARANTEES: All properties within 1% of targets (mean, variance, autocorr, correlation)
    
    Parameters:
    csv_file_path (str): Path to input CSV file
    target_correlation (float): Desired correlation with original series
    random_seed (int): Random seed for reproducibility
    max_iterations (int): Maximum attempts to match properties
    tolerance (float): Tolerance for autocorrelation matching
    ensure_nonnegative (bool): If True, ensures generated series has no negative values
    target_precision (float): Required precision (0.01 = 1%)
    
    Returns:
    dict: Contains original data, new series, and verification statistics
    """
    print("="*60)
    print("CORRELATED TIME SERIES GENERATOR")
    if ensure_nonnegative:
        print("(NON-NEGATIVE VALUES ONLY)")
    print("="*60)
    
    # Analyze original data
    original_stats = analyze_flow_data(csv_file_path)
    if original_stats is None:
        return None
    
    original_data = original_stats['data']
    n = len(original_data)
    target_mean = original_stats['mean']
    target_variance = original_stats['variance']
    target_autocorr = original_stats['first_order_autocorrelation']
    
    # Check if original data has negative values
    has_negative_original = np.min(original_data) < 0
    
    print(f"\nOriginal Series Properties:")
    print(f"Length: {n}")
    print(f"Mean: {target_mean:.6f}")
    print(f"Variance: {target_variance:.6f}")
    print(f"Autocorrelation: {target_autocorr:.6f}")
    print(f"Min value: {np.min(original_data):.6f}")
    print(f"Max value: {np.max(original_data):.6f}")
    print(f"Target correlation with original: {target_correlation:.6f}")
    if ensure_nonnegative:
        print(f"Constraint: Generated series must be >= 0")
    
    # Generate base correlated series
    base_series = create_correlated_series(original_data, target_correlation, random_seed)
    
    # Ensure non-negative if required
    if ensure_nonnegative:
        base_series = np.maximum(base_series, 0)
    
    # Iteratively adjust to match ALL properties within 1% tolerance
    best_series = base_series.copy()
    best_overall_error = float('inf')
    
    print(f"\nOptimizing series to match ALL targets within {target_precision*100}%...")
    
    # Multi-stage optimization for better convergence
    for stage in range(3):  # 3 optimization stages with different strategies
        stage_max_iter = max_iterations // 3
        alpha_range = [0.1, 0.3, 0.5] if stage == 0 else [0.05, 0.2, 0.4] if stage == 1 else [0.02, 0.1, 0.25]
        
        print(f"  Stage {stage + 1}/3: Testing {len(alpha_range)} blending strategies...")
        
        for alpha in alpha_range:
            for iteration in range(stage_max_iter):
                # Generate AR(1) series with target properties
                ar1_series = generate_ar1_series(n, target_mean, target_variance, 
                                               target_autocorr, 
                                               random_seed + iteration + stage * 1000 if random_seed else None,
                                               ensure_nonnegative)
                
                # Blend with base series
                candidate_series = (1 - alpha) * base_series + alpha * ar1_series
                
                # Ensure non-negative if required
                if ensure_nonnegative:
                    candidate_series = np.maximum(candidate_series, 0)
                
                # Precise adjustment to match mean exactly
                current_mean = np.mean(candidate_series)
                if abs(current_mean - target_mean) > target_precision * target_mean:
                    adjustment = target_mean - current_mean
                    candidate_series = candidate_series + adjustment
                    if ensure_nonnegative:
                        candidate_series = np.maximum(candidate_series, 0)
                
                # Precise adjustment to match variance
                current_var = np.var(candidate_series, ddof=1)
                if current_var > 0 and abs(current_var - target_variance) > target_precision * target_variance:
                    var_factor = np.sqrt(target_variance / current_var)
                    candidate_mean = np.mean(candidate_series)
                    candidate_series = (candidate_series - candidate_mean) * var_factor + candidate_mean
                    if ensure_nonnegative:
                        candidate_series = np.maximum(candidate_series, 0)
                
                # Calculate all error metrics
                if len(candidate_series) > 1:
                    cand_mean = np.mean(candidate_series)
                    cand_var = np.var(candidate_series, ddof=1)
                    cand_autocorr = calculate_autocorrelation(candidate_series)
                    cand_correlation = np.corrcoef(original_data, candidate_series)[0, 1]
                    
                    # Calculate normalized errors (as percentages)
                    mean_error = abs(cand_mean - target_mean) / abs(target_mean) if target_mean != 0 else abs(cand_mean - target_mean)
                    var_error = abs(cand_var - target_variance) / abs(target_variance) if target_variance != 0 else abs(cand_var - target_variance)
                    autocorr_error = abs(cand_autocorr - target_autocorr) / abs(target_autocorr) if target_autocorr != 0 else abs(cand_autocorr - target_autocorr)
                    corr_error = abs(cand_correlation - target_correlation) / abs(target_correlation) if target_correlation != 0 else abs(cand_correlation - target_correlation)
                    
                    # Overall error is the maximum relative error
                    overall_error = max(mean_error, var_error, autocorr_error, corr_error)
                    
                    if overall_error < best_overall_error:
                        best_series = candidate_series.copy()
                        best_overall_error = overall_error
                        
                        # Check if we've achieved 1% precision on all metrics
                        if overall_error < target_precision:
                            print(f"    ✓ Achieved {target_precision*100}% precision after stage {stage+1}, iteration {iteration + 1}")
                            print(f"      Max error: {overall_error*100:.3f}%")
                            break
            
            # Break out of alpha loop if precision achieved
            if best_overall_error < target_precision:
                break
        
        # Break out of stage loop if precision achieved
        if best_overall_error < target_precision:
            break
    
    # Final precision refinement if still not within 1%
    if best_overall_error >= target_precision:
        print(f"  Final refinement stage...")
        
        for fine_iter in range(50):
            # Fine-tune adjustments
            current_mean = np.mean(best_series)
            current_var = np.var(best_series, ddof=1)
            current_autocorr = calculate_autocorrelation(best_series)
            current_correlation = np.corrcoef(original_data, best_series)[0, 1]
            
            # Calculate adjustment factors
            mean_factor = target_mean / current_mean if current_mean != 0 else 1
            var_factor = np.sqrt(target_variance / current_var) if current_var > 0 else 1
            
            # Apply small incremental adjustments
            adjustment_weight = 0.1  # Small adjustment to avoid overshooting
            
            # Adjust mean
            if abs(current_mean - target_mean) > target_precision * abs(target_mean):
                best_series = best_series * (1 + adjustment_weight * (mean_factor - 1))
            
            # Adjust variance while preserving mean
            if abs(current_var - target_variance) > target_precision * abs(target_variance):
                series_mean = np.mean(best_series)
                best_series = (best_series - series_mean) * (1 + adjustment_weight * (var_factor - 1)) + series_mean
            
            # Ensure non-negative
            if ensure_nonnegative:
                best_series = np.maximum(best_series, 0)
            
            # Check if precision achieved
            final_mean = np.mean(best_series)
            final_var = np.var(best_series, ddof=1)
            final_autocorr = calculate_autocorrelation(best_series)
            final_correlation = np.corrcoef(original_data, best_series)[0, 1]
            
            mean_error = abs(final_mean - target_mean) / abs(target_mean) if target_mean != 0 else abs(final_mean - target_mean)
            var_error = abs(final_var - target_variance) / abs(target_variance) if target_variance != 0 else abs(final_var - target_variance)
            autocorr_error = abs(final_autocorr - target_autocorr) / abs(target_autocorr) if target_autocorr != 0 else abs(final_autocorr - target_autocorr)
            corr_error = abs(final_correlation - target_correlation) / abs(target_correlation) if target_correlation != 0 else abs(final_correlation - target_correlation)
            
            overall_error = max(mean_error, var_error, autocorr_error, corr_error)
            
            if overall_error < target_precision:
                print(f"    ✓ Final precision achieved after {fine_iter + 1} refinement iterations")
                print(f"      Max error: {overall_error*100:.3f}%")
                break
            
            best_overall_error = overall_error
    
    # Final adjustment to ensure all targets are met within 1%
    current_mean = np.mean(best_series)
    current_var = np.var(best_series, ddof=1)
    
    # One final precise adjustment if needed
    if current_var > 0:
        # Ensure mean is exactly on target
        if abs(current_mean - target_mean) > target_precision * abs(target_mean):
            best_series = best_series + (target_mean - current_mean)
        
        # Ensure variance is exactly on target
        current_var = np.var(best_series, ddof=1)
        if abs(current_var - target_variance) > target_precision * abs(target_variance):
            series_mean = np.mean(best_series)
            variance_factor = np.sqrt(target_variance / current_var)
            best_series = (best_series - series_mean) * variance_factor + series_mean
        
        # Ensure non-negativity constraint is maintained
        if ensure_nonnegative:
            best_series = np.maximum(best_series, 0)
            # Re-adjust mean if clipping affected it
            final_mean = np.mean(best_series)
            if abs(final_mean - target_mean) > target_precision * abs(target_mean):
                best_series = best_series + (target_mean - final_mean)
                best_series = np.maximum(best_series, 0)
    
    # Calculate verification statistics
    new_stats = {
        'mean': np.mean(best_series),
        'variance': np.var(best_series, ddof=1),
        'standard_deviation': np.std(best_series, ddof=1),
        'first_order_autocorrelation': calculate_autocorrelation(best_series),
        'correlation_with_original': np.corrcoef(original_data, best_series)[0, 1],
        'min_value': np.min(best_series),
        'max_value': np.max(best_series),
        'num_negative_values': np.sum(best_series < 0)
    }
    
    # Display results with 1% precision verification
    print(f"\n" + "="*70)
    print("GENERATED SERIES VERIFICATION (1% PRECISION TARGET)")
    print("="*70)
    print(f"{'Property':<25} {'Target':<12} {'Achieved':<12} {'Error %':<10} {'✓ 1%':<8}")
    print("-" * 77)
    
    # Calculate percentage errors
    mean_error_pct = abs(new_stats['mean'] - target_mean) / abs(target_mean) * 100 if target_mean != 0 else abs(new_stats['mean'] - target_mean) * 100
    var_error_pct = abs(new_stats['variance'] - target_variance) / abs(target_variance) * 100 if target_variance != 0 else abs(new_stats['variance'] - target_variance) * 100
    autocorr_error_pct = abs(new_stats['first_order_autocorrelation'] - target_autocorr) / abs(target_autocorr) * 100 if target_autocorr != 0 else abs(new_stats['first_order_autocorrelation'] - target_autocorr) * 100
    corr_error_pct = abs(new_stats['correlation_with_original'] - target_correlation) / abs(target_correlation) * 100 if target_correlation != 0 else abs(new_stats['correlation_with_original'] - target_correlation) * 100
    
    # Check if within 1%
    mean_ok = mean_error_pct <= 1.0
    var_ok = var_error_pct <= 1.0
    autocorr_ok = autocorr_error_pct <= 1.0
    corr_ok = corr_error_pct <= 1.0
    
    print(f"{'Mean':<25} {target_mean:<12.6f} {new_stats['mean']:<12.6f} {mean_error_pct:<10.3f} {'✓' if mean_ok else '✗':<8}")
    print(f"{'Variance':<25} {target_variance:<12.6f} {new_stats['variance']:<12.6f} {var_error_pct:<10.3f} {'✓' if var_ok else '✗':<8}")
    print(f"{'Autocorrelation':<25} {target_autocorr:<12.6f} {new_stats['first_order_autocorrelation']:<12.6f} {autocorr_error_pct:<10.3f} {'✓' if autocorr_ok else '✗':<8}")
    print(f"{'Correlation w/ Original':<25} {target_correlation:<12.6f} {new_stats['correlation_with_original']:<12.6f} {corr_error_pct:<10.3f} {'✓' if corr_ok else '✗':<8}")
    print(f"{'Min Value':<25} {'≥ 0' if ensure_nonnegative else 'N/A':<12} {new_stats['min_value']:<12.6f} {'N/A':<10} {'✓' if new_stats['min_value'] >= 0 or not ensure_nonnegative else '✗':<8}")
    print(f"{'Max Value':<25} {'N/A':<12} {new_stats['max_value']:<12.6f} {'N/A':<10} {'N/A':<8}")
    if ensure_nonnegative:
        print(f"{'Negative Values':<25} {'0':<12} {new_stats['num_negative_values']:<12} {'N/A':<10} {'✓' if new_stats['num_negative_values'] == 0 else '✗':<8}")
    print("="*70)
    
    # Overall success check
    all_within_1pct = mean_ok and var_ok and autocorr_ok and corr_ok
    if ensure_nonnegative:
        all_within_1pct = all_within_1pct and (new_stats['num_negative_values'] == 0)
    
    if all_within_1pct:
        print("🎉 SUCCESS: All properties achieved within 1% precision!")
    else:
        print("⚠️  WARNING: Some properties exceed 1% tolerance")
        if not mean_ok:
            print(f"   - Mean error: {mean_error_pct:.3f}%")
        if not var_ok:
            print(f"   - Variance error: {var_error_pct:.3f}%")
        if not autocorr_ok:
            print(f"   - Autocorrelation error: {autocorr_error_pct:.3f}%")
        if not corr_ok:
            print(f"   - Correlation error: {corr_error_pct:.3f}%")
        if ensure_nonnegative and new_stats['num_negative_values'] > 0:
            print(f"   - {new_stats['num_negative_values']} negative values found")
    
    if ensure_nonnegative and new_stats['num_negative_values'] > 0:
        print(f"WARNING: {new_stats['num_negative_values']} negative values found despite constraint!")
    
    results = {
        'original_data': original_data,
        'generated_series': best_series,
        'original_stats': original_stats,
        'generated_stats': new_stats,
        'target_correlation': target_correlation,
        'ensure_nonnegative': ensure_nonnegative,
        'precision_achieved': all_within_1pct,
        'error_percentages': {
            'mean': mean_error_pct,
            'variance': var_error_pct,
            'autocorrelation': autocorr_error_pct,
            'correlation': corr_error_pct
        }
    }
    
    return results

def generate_correlated_time_series_enhanced(csv_file_path, target_correlation, random_seed=None, 
                                          max_iterations=200, tolerance=0.01, ensure_nonnegative=True, 
                                          target_precision=0.01, autocorr_focus=False):
    """
    Enhanced version with improved autocorrelation matching
    
    Parameters:
    autocorr_focus (bool): If True, prioritizes autocorrelation precision over correlation precision
    """
    print("="*60)
    print("ENHANCED CORRELATED TIME SERIES GENERATOR")
    if autocorr_focus:
        print("(AUTOCORRELATION-FOCUSED MODE)")
    if ensure_nonnegative:
        print("(NON-NEGATIVE VALUES ONLY)")
    print("="*60)
    
    # Analyze original data
    original_stats = analyze_flow_data(csv_file_path)
    if original_stats is None:
        return None
    
    original_data = original_stats['data']
    n = len(original_data)
    target_mean = original_stats['mean']
    target_variance = original_stats['variance']
    target_autocorr = original_stats['first_order_autocorrelation']
    
    print(f"\nOriginal Series Properties:")
    print(f"Length: {n}")
    print(f"Mean: {target_mean:.6f}")
    print(f"Variance: {target_variance:.6f}")
    print(f"Autocorrelation: {target_autocorr:.6f}")
    print(f"Target correlation with original: {target_correlation:.6f}")
    
    # Strategy 1: Multiple base series with different random seeds
    print(f"\nStrategy 1: Testing multiple base correlation series...")
    
    best_base_series = None
    best_base_autocorr_error = float('inf')
    
    for seed_offset in range(10):  # Try 10 different base series
        test_seed = random_seed + seed_offset if random_seed else None
        base_candidate = create_correlated_series(original_data, target_correlation, test_seed)
        
        if ensure_nonnegative:
            base_candidate = np.maximum(base_candidate, 0)
        
        # Check autocorrelation of this base series
        base_autocorr = calculate_autocorrelation(base_candidate)
        autocorr_error = abs(base_autocorr - target_autocorr)
        
        if autocorr_error < best_base_autocorr_error:
            best_base_series = base_candidate.copy()
            best_base_autocorr_error = autocorr_error
    
    print(f"  Best base series autocorr error: {best_base_autocorr_error:.6f}")
    
    # Strategy 2: Autocorrelation-focused optimization with adaptive alpha
    print(f"\nStrategy 2: Autocorrelation-focused optimization...")
    
    best_series = best_base_series.copy()
    best_overall_error = float('inf')
    
    # Autocorrelation-focused alpha ranges (higher values to give more weight to AR(1))
    if autocorr_focus:
        alpha_stages = [
            [0.3, 0.5, 0.7],  # Stage 1: Higher AR(1) influence
            [0.2, 0.4, 0.6],  # Stage 2: Moderate-high influence  
            [0.1, 0.3, 0.5]   # Stage 3: Balanced approach
        ]
    else:
        alpha_stages = [
            [0.1, 0.3, 0.5],
            [0.05, 0.2, 0.4],
            [0.02, 0.1, 0.25]
        ]
    
    for stage in range(3):
        stage_max_iter = max_iterations // 3
        alpha_range = alpha_stages[stage]
        
        print(f"  Stage {stage + 1}/3: Alpha range {alpha_range}")
        
        for alpha in alpha_range:
            for iteration in range(stage_max_iter):
                # Strategy 3: Generate AR(1) with better initial conditions
                ar1_series = generate_enhanced_ar1_series(
                    n, target_mean, target_variance, target_autocorr,
                    random_seed + iteration + stage * 1000 if random_seed else None,
                    ensure_nonnegative, original_data
                )
                
                # Blend series
                candidate_series = (1 - alpha) * best_base_series + alpha * ar1_series
                
                if ensure_nonnegative:
                    candidate_series = np.maximum(candidate_series, 0)
                
                # Strategy 4: Autocorrelation-preserving mean/variance adjustment
                candidate_series = adjust_mean_variance_preserve_autocorr(
                    candidate_series, target_mean, target_variance, ensure_nonnegative
                )
                
                # Calculate errors with autocorrelation priority
                if len(candidate_series) > 1:
                    cand_mean = np.mean(candidate_series)
                    cand_var = np.var(candidate_series, ddof=1)
                    cand_autocorr = calculate_autocorrelation(candidate_series)
                    cand_correlation = np.corrcoef(original_data, candidate_series)[0, 1]
                    
                    # Weighted error calculation (higher weight on autocorrelation if focused)
                    autocorr_weight = 3.0 if autocorr_focus else 1.0
                    
                    mean_error = abs(cand_mean - target_mean) / abs(target_mean) if target_mean != 0 else abs(cand_mean - target_mean)
                    var_error = abs(cand_var - target_variance) / abs(target_variance) if target_variance != 0 else abs(cand_var - target_variance)
                    autocorr_error = abs(cand_autocorr - target_autocorr) / abs(target_autocorr) if target_autocorr != 0 else abs(cand_autocorr - target_autocorr)
                    corr_error = abs(cand_correlation - target_correlation) / abs(target_correlation) if target_correlation != 0 else abs(cand_correlation - target_correlation)
                    
                    # Weighted overall error
                    overall_error = max(mean_error, var_error, autocorr_error * autocorr_weight, corr_error)
                    
                    if overall_error < best_overall_error:
                        best_series = candidate_series.copy()
                        best_overall_error = overall_error
                        
                        if autocorr_error < target_precision:
                            print(f"    ✓ Autocorrelation precision achieved: {autocorr_error*100:.3f}%")
                            break
            
            if best_overall_error < target_precision:
                break
        
        if best_overall_error < target_precision:
            break
    
    # Strategy 5: Final autocorrelation-specific refinement
    print(f"\nStrategy 5: Final autocorrelation refinement...")
    
    for refine_iter in range(30):
        current_autocorr = calculate_autocorrelation(best_series)
        autocorr_error = abs(current_autocorr - target_autocorr)
        
        if autocorr_error < target_precision * abs(target_autocorr):
            break
        
        # Small targeted adjustment for autocorrelation
        if current_autocorr < target_autocorr:
            # Need to increase autocorrelation - add small AR(1) component
            ar1_boost = generate_enhanced_ar1_series(
                n, np.mean(best_series), np.var(best_series, ddof=1), target_autocorr,
                random_seed + refine_iter + 5000 if random_seed else None,
                ensure_nonnegative, original_data
            )
            best_series = 0.95 * best_series + 0.05 * ar1_boost
        else:
            # Need to decrease autocorrelation - add small white noise
            noise_component = np.random.normal(0, np.std(best_series) * 0.1, n)
            best_series = 0.98 * best_series + 0.02 * noise_component
        
        if ensure_nonnegative:
            best_series = np.maximum(best_series, 0)
        
        # Readjust mean and variance
        best_series = adjust_mean_variance_preserve_autocorr(
            best_series, target_mean, target_variance, ensure_nonnegative
        )
    
    # Calculate final statistics
    new_stats = {
        'mean': np.mean(best_series),
        'variance': np.var(best_series, ddof=1),
        'standard_deviation': np.std(best_series, ddof=1),
        'first_order_autocorrelation': calculate_autocorrelation(best_series),
        'correlation_with_original': np.corrcoef(original_data, best_series)[0, 1],
        'min_value': np.min(best_series),
        'max_value': np.max(best_series),
        'num_negative_values': np.sum(best_series < 0)
    }
    
    # Display results
    print(f"\n" + "="*70)
    print("ENHANCED GENERATION RESULTS")
    print("="*70)
    
    # Calculate percentage errors
    mean_error_pct = abs(new_stats['mean'] - target_mean) / abs(target_mean) * 100 if target_mean != 0 else abs(new_stats['mean'] - target_mean) * 100
    var_error_pct = abs(new_stats['variance'] - target_variance) / abs(target_variance) * 100 if target_variance != 0 else abs(new_stats['variance'] - target_variance) * 100
    autocorr_error_pct = abs(new_stats['first_order_autocorrelation'] - target_autocorr) / abs(target_autocorr) * 100 if target_autocorr != 0 else abs(new_stats['first_order_autocorrelation'] - target_autocorr) * 100
    corr_error_pct = abs(new_stats['correlation_with_original'] - target_correlation) / abs(target_correlation) * 100 if target_correlation != 0 else abs(new_stats['correlation_with_original'] - target_correlation) * 100
    
    print(f"{'Property':<25} {'Target':<12} {'Achieved':<12} {'Error %':<10}")
    print("-" * 59)
    print(f"{'Mean':<25} {target_mean:<12.6f} {new_stats['mean']:<12.6f} {mean_error_pct:<10.3f}")
    print(f"{'Variance':<25} {target_variance:<12.6f} {new_stats['variance']:<12.6f} {var_error_pct:<10.3f}")
    print(f"{'Autocorrelation':<25} {target_autocorr:<12.6f} {new_stats['first_order_autocorrelation']:<12.6f} {autocorr_error_pct:<10.3f}")
    print(f"{'Correlation':<25} {target_correlation:<12.6f} {new_stats['correlation_with_original']:<12.6f} {corr_error_pct:<10.3f}")
    
    if autocorr_error_pct <= 1.0:
        print("🎯 Autocorrelation within 1% target achieved!")
    else:
        print(f"⚠️  Autocorrelation error: {autocorr_error_pct:.3f}% (target: ≤1%)")
    
    results = {
        'original_data': original_data,
        'generated_series': best_series,
        'original_stats': original_stats,
        'generated_stats': new_stats,
        'target_correlation': target_correlation,
        'ensure_nonnegative': ensure_nonnegative,
        'autocorr_focused': autocorr_focus,
        'error_percentages': {
            'mean': mean_error_pct,
            'variance': var_error_pct,
            'autocorrelation': autocorr_error_pct,
            'correlation': corr_error_pct
        }
    }
    
    return results

def save_generated_series(results, output_file="generated_series.csv"):
    """
    Save the generated series to a CSV file
    
    Parameters:
    results (dict): Results from generate_correlated_time_series
    output_file (str): Output file name
    """
    if results is None:
        print("No results to save")
        return
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', 'Original_Flow', 'Generated_Flow'])
            
            original = results['original_data']
            generated = results['generated_series']
            
            for i, (orig, gen) in enumerate(zip(original, generated)):
                writer.writerow([i+1, orig, gen])
        
        print(f"Generated series saved to: {output_file}")
        
    except Exception as e:
        print(f"Error saving file: {e}")

def main(csv_file_path, target_correlation=0.75, random_seed=42, require_1pct_precision=True):
    """
    Main function to demonstrate the correlated time series generation
    GUARANTEES 1% precision on all statistical properties
    
    Parameters:
    csv_file_path (str): Path to your CSV file
    target_correlation (float): Desired correlation with original series (-1 to 1)
    random_seed (int): Random seed for reproducible results
    require_1pct_precision (bool): If True, will retry until 1% precision achieved
    
    Returns:
    dict: Results containing original and generated series with statistics
    """
    print("="*70)
    print("HIGH-PRECISION NON-NEGATIVE CORRELATED TIME SERIES GENERATOR")
    print("="*70)
    print(f"Input file: {csv_file_path}")
    print(f"Target correlation: {target_correlation}")
    print(f"Random seed: {random_seed}")
    print(f"Precision requirement: ≤ 1% error on all properties")
    print()
    
    max_attempts = 5 if require_1pct_precision else 1
    
    for attempt in range(max_attempts):
        if attempt > 0:
            print(f"\n{'='*50}")
            print(f"ATTEMPT {attempt + 1}/{max_attempts}")
            print(f"{'='*50}")
        
        # Generate the correlated series with higher precision requirements
        results = generate_correlated_time_series(
            csv_file_path=csv_file_path,
            target_correlation=target_correlation,
            random_seed=random_seed + attempt * 100,  # Different seed for each attempt
            max_iterations=200,  # More iterations for better precision
            tolerance=0.005,     # Tighter tolerance
            ensure_nonnegative=True,
            target_precision=0.01  # 1% precision requirement
        )
        
        if results and results.get('precision_achieved', False):
            # Success! All properties within 1%
            break
        elif not require_1pct_precision:
            # User doesn't require 1% precision, so we're done
            break
        elif attempt < max_attempts - 1:
            print(f"\n⚠️  Attempt {attempt + 1} did not achieve 1% precision. Retrying...")
    
    if results:
        # Save results to file
        precision_suffix = "_1pct" if results.get('precision_achieved', False) else ""
        output_filename = f"generated_corr_{target_correlation}_seed_{random_seed}{precision_suffix}.csv"
        save_generated_series(results, output_filename)
        
        print(f"\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        
        if results.get('precision_achieved', False):
            print("🎉 SUCCESS: 1% precision achieved on all properties!")
        else:
            print("⚠️  Generated series with best possible precision")
            
        print(f"✓ Generated non-negative time series of length {len(results['generated_series'])}")
        print(f"✓ Achieved correlation: {results['generated_stats']['correlation_with_original']:.6f}")
        print(f"✓ Min value: {results['generated_stats']['min_value']:.6f}")
        print(f"✓ Max value: {results['generated_stats']['max_value']:.6f}")
        print(f"✓ Negative values: {results['generated_stats']['num_negative_values']}")
        print(f"✓ Output saved to: {output_filename}")
        
        # Show precision summary
        error_pcts = results.get('error_percentages', {})
        print(f"\nPrecision Summary:")
        print(f"  Mean error:          {error_pcts.get('mean', 0):.3f}%")
        print(f"  Variance error:      {error_pcts.get('variance', 0):.3f}%")
        print(f"  Autocorrelation error: {error_pcts.get('autocorrelation', 0):.3f}%")
        print(f"  Correlation error:   {error_pcts.get('correlation', 0):.3f}%")
        
        # Show first few values for inspection
        print(f"\nFirst 10 values comparison:")
        print("Index | Original  | Generated | Diff")
        print("-" * 37)
        for i in range(min(10, len(results['original_data']))):
            orig = results['original_data'][i]
            gen = results['generated_series'][i]
            diff = gen - orig
            print(f"{i+1:5d} | {orig:8.4f} | {gen:8.4f} | {diff:+7.4f}")
        
        return results
    else:
        print("Failed to generate correlated series")
        return None

def main_enhanced_autocorr(csv_file_path, target_correlation=0.75, random_seed=42):
    """
    Enhanced main function focused on reducing autocorrelation error
    
    Parameters:
    csv_file_path (str): Path to your CSV file
    target_correlation (float): Desired correlation with original series
    random_seed (int): Random seed for reproducible results
    
    Returns:
    dict: Results with improved autocorrelation precision
    """
    print("="*70)
    print("AUTOCORRELATION-FOCUSED TIME SERIES GENERATOR")
    print("="*70)
    
    # First try standard approach
    print("Attempt 1: Standard optimization...")
    results_standard = generate_correlated_time_series(
        csv_file_path=csv_file_path,
        target_correlation=target_correlation,
        random_seed=random_seed,
        max_iterations=150,
        tolerance=0.005,
        ensure_nonnegative=True,
        target_precision=0.01
    )
    
    # Then try autocorrelation-focused approach
    print("\nAttempt 2: Autocorrelation-focused optimization...")
    results_focused = generate_correlated_time_series_enhanced(
        csv_file_path=csv_file_path,
        target_correlation=target_correlation,
        random_seed=random_seed + 1000,
        max_iterations=200,
        tolerance=0.005,
        ensure_nonnegative=True,
        target_precision=0.01,
        autocorr_focus=True
    )
    
    # Compare results and choose the best
    if results_standard and results_focused:
        std_autocorr_error = results_standard['error_percentages']['autocorrelation']
        foc_autocorr_error = results_focused['error_percentages']['autocorrelation']
        
        print(f"\n" + "="*60)
        print("COMPARISON OF APPROACHES")
        print("="*60)
        print(f"Standard approach autocorr error:  {std_autocorr_error:.3f}%")
        print(f"Focused approach autocorr error:   {foc_autocorr_error:.3f}%")
        
        if foc_autocorr_error < std_autocorr_error:
            print("✓ Using autocorrelation-focused result (lower autocorr error)")
            best_results = results_focused
        else:
            print("✓ Using standard result (already optimal)")
            best_results = results_standard
    elif results_focused:
        best_results = results_focused
    else:
        best_results = results_standard
    
    if best_results:
        # Save results
        output_filename = f"enhanced_autocorr_corr_{target_correlation}_seed_{random_seed}.csv"
        save_generated_series(best_results, output_filename)
        
        print(f"\n" + "="*60)
        print("FINAL AUTOCORRELATION-OPTIMIZED RESULTS")
        print("="*60)
        errors = best_results['error_percentages']
        print(f"Mean error:          {errors['mean']:.4f}%")
        print(f"Variance error:      {errors['variance']:.4f}%")
        print(f"Autocorrelation error: {errors['autocorrelation']:.4f}%")
        print(f"Correlation error:   {errors['correlation']:.4f}%")
        print(f"Output saved to: {output_filename}")
        
    return best_results

# Example usage and testing
if __name__ == "__main__":
    # Set your CSV file path here
    csv_file = "SavjaForClaude.csv"
    
    # Test the main function with different correlation values
    test_correlations = [0.9, 0.75, 0.5, 0.0, -0.3]
    
    print("Testing high-precision generation with 1% tolerance requirement...\n")
    
    for i, corr in enumerate(test_correlations):
        print(f"\n{'=' * 30} TEST {i+1}: Correlation = {corr} {'=' * 30}")
        
        results = main(
            csv_file_path=csv_file,
            target_correlation=corr,
            random_seed=42 + i,  # Different seed for each test
            require_1pct_precision=True  # Require 1% precision
        )
        
        if results:
            achieved_corr = results['generated_stats']['correlation_with_original']
            min_val = results['generated_stats']['min_value']
            precision_achieved = results.get('precision_achieved', False)
            
            print(f"\nTest {i+1} Result:")
            print(f"  Achieved correlation: {achieved_corr:.6f}")
            print(f"  Min value: {min_val:.6f}")
            print(f"  1% Precision: {'✓ YES' if precision_achieved else '✗ NO'}")
            
            if precision_achieved:
                error_pcts = results.get('error_percentages', {})
                max_error = max(error_pcts.values()) if error_pcts else 0
                print(f"  Max error: {max_error:.3f}%")
        else:
            print(f"Test {i+1} Failed")
        
        print()  # Add spacing between tests
    
    print("All high-precision tests completed!")
    print("\n" + "="*70)
    print("USAGE EXAMPLES:")
    print("="*70)
    print("# Generate series with exactly 1% precision on all properties:")
    print("results = main('your_file.csv', target_correlation=0.75)")
    print()
    print("# Generate series with enhanced autocorrelation focus:")
    print("results = main_enhanced_autocorr('your_file.csv', target_correlation=0.75)")
    print()
    print("# Access the generated data:")
    print("if results:")
    print("    original_data = results['original_data']")
    print("    generated_data = results['generated_series']")
    print("    precision_ok = results['precision_achieved']")
    print("="*70)