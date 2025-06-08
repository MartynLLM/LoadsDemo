"""
Data Input/Output Operations Module

Handles reading CSV files, parsing dates, and saving generated series.
"""

import csv
import numpy as np
import math
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Union


def read_flow_data(csv_file_path: str) -> Optional[Dict]:
    """
    Read a CSV file with Date and Flow columns.
    
    Parameters:
    csv_file_path (str): Path to the CSV file
    
    Returns:
    dict: Dictionary containing flow data, dates, and metadata, or None if error
    """
    try:
        flow_data = []
        date_data = []
        
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            
            # Check if required columns exist
            if 'Flow' not in csv_reader.fieldnames:
                print("Error: 'Flow' column not found in the CSV file")
                print(f"Available columns: {csv_reader.fieldnames}")
                return None
            
            has_dates = 'Date' in csv_reader.fieldnames
            if not has_dates:
                print("Warning: 'Date' column not found - using index for time axis")
            
            # Read flow data and convert to float, skipping invalid entries
            for row in csv_reader:
                try:
                    flow_value = float(row['Flow'])
                    if not math.isnan(flow_value):  # Skip NaN values
                        flow_data.append(flow_value)
                        
                        # Try to parse date if available
                        if has_dates:
                            date_obj = _parse_date(row['Date'].strip())
                            date_data.append(date_obj if date_obj else len(flow_data) - 1)
                        else:
                            date_data.append(len(flow_data) - 1)
                            
                except (ValueError, TypeError):
                    # Skip rows with invalid flow data
                    continue
        
        print(f"Successfully loaded {len(flow_data)} valid flow measurements")
        
        # Sort data by date if we have valid dates
        if has_dates and date_data and isinstance(date_data[0], datetime):
            print("Sorting data by date for proper time series order...")
            combined_data = list(zip(date_data, flow_data))
            combined_data.sort(key=lambda x: x[0])
            date_data, flow_data = zip(*combined_data)
            date_data = list(date_data)
            flow_data = list(flow_data)
            print(f"Data sorted from {date_data[0].strftime('%d/%m/%Y')} to {date_data[-1].strftime('%d/%m/%Y')}")
        
        return {
            'flow_data': np.array(flow_data),
            'dates': date_data,
            'has_dates': has_dates,
            'sample_size': len(flow_data)
        }
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def _parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse date string using multiple formats, prioritizing DD/MM/YYYY.
    
    Parameters:
    date_str (str): Date string to parse
    
    Returns:
    datetime: Parsed date object or None if parsing fails
    """
    date_formats = ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d', '%m-%d-%Y']
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None


def save_generated_series(original_data: np.ndarray, generated_data: np.ndarray, 
                         output_file: str, dates: Optional[List] = None) -> None:
    """
    Save the original and generated series to a CSV file.
    
    Parameters:
    original_data (np.ndarray): Original time series data
    generated_data (np.ndarray): Generated time series data
    output_file (str): Output file name
    dates (List, optional): Date information for each data point
    """
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Write header
            if dates and isinstance(dates[0], datetime):
                writer.writerow(['Date', 'Original_Flow', 'Generated_Flow'])
            else:
                writer.writerow(['Index', 'Original_Flow', 'Generated_Flow'])
            
            # Write data
            for i, (orig, gen) in enumerate(zip(original_data, generated_data)):
                if dates and isinstance(dates[0], datetime):
                    date_str = dates[i].strftime('%d/%m/%Y')
                    writer.writerow([date_str, orig, gen])
                else:
                    writer.writerow([i+1, orig, gen])
        
        print(f"Generated series saved to: {output_file}")
        
    except Exception as e:
        print(f"Error saving file: {e}")


def create_output_filename(target_correlation: float, random_seed: int, 
                          precision_achieved: bool = False, 
                          method: str = "corr_focused") -> str:
    """
    Create standardized output filename based on parameters.
    
    Parameters:
    target_correlation (float): Target correlation value
    random_seed (int): Random seed used
    precision_achieved (bool): Whether 1% precision was achieved
    method (str): Generation method used
    
    Returns:
    str: Formatted filename
    """
    correlation_str = str(target_correlation).replace('.', 'p').replace('-', 'neg')
    precision_suffix = "_1pct" if precision_achieved else ""
    
    return f"generated_{method}_corr_{correlation_str}_seed_{random_seed}{precision_suffix}.csv"
