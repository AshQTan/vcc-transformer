#!/usr/bin/env python3
"""
GPU Monitoring Script for VCC Transformer

This script provides real-time GPU monitoring during training
with logging and alerting capabilities.

Usage:
    python scripts/monitor_gpu.py --log-file gpu_monitoring.log
    python scripts/monitor_gpu.py --alert-temp 85 --alert-power 350
"""

import argparse
import sys
import time
import json
import csv
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vcc_transformer.utils.gpu_optimization import GPUOptimizer


def monitor_with_logging(optimizer: GPUOptimizer, args):
    """Monitor GPUs with comprehensive logging"""
    
    # Setup logging files
    if args.log_file:
        log_file = Path(args.log_file)
        csv_file = log_file.with_suffix('.csv')
        json_file = log_file.with_suffix('.json')
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(f"gpu_monitoring_{timestamp}.log")
        csv_file = Path(f"gpu_monitoring_{timestamp}.csv")
        json_file = Path(f"gpu_monitoring_{timestamp}.json")
    
    # Initialize CSV file
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'gpu_id', 'name', 'temperature', 'power_usage',
            'memory_used', 'memory_total', 'utilization', 'power_limit'
        ])
    
    # Initialize JSON log
    json_data = {
        'start_time': datetime.now().isoformat(),
        'monitoring_data': []
    }
    
    print(f"GPU Monitoring Started")
    print(f"CSV Log: {csv_file}")
    print(f"JSON Log: {json_file}")
    print(f"Alert Thresholds: Temp={args.alert_temp}°C, Power={args.alert_power}W")
    print("-" * 80)
    
    try:
        while True:
            timestamp = datetime.now()
            
            # Collect data for all GPUs
            gpu_data = []
            
            for gpu_id in range(optimizer.device_count):
                status = optimizer.get_gpu_status(gpu_id)
                if status:
                    # Check for alerts
                    alerts = []
                    if status.temperature > args.alert_temp:
                        alerts.append(f"HIGH TEMP ({status.temperature}°C)")
                    if status.power_usage > args.alert_power:
                        alerts.append(f"HIGH POWER ({status.power_usage:.1f}W)")
                    if status.memory_used / status.memory_total > 0.95:
                        alerts.append(f"HIGH MEMORY ({status.memory_used/status.memory_total*100:.1f}%)")
                    
                    # Console output
                    alert_str = f" ⚠️  {', '.join(alerts)}" if alerts else ""
                    print(f"GPU {gpu_id:2d}: {status.temperature:5.1f}°C | "
                          f"{status.power_usage:6.1f}W | "
                          f"{status.utilization:3d}% | "
                          f"{status.memory_used:5.1f}/{status.memory_total:5.1f}GB"
                          f"{alert_str}")
                    
                    # CSV logging
                    with open(csv_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            timestamp.isoformat(),
                            status.gpu_id,
                            status.name,
                            status.temperature,
                            status.power_usage,
                            status.memory_used,
                            status.memory_total,
                            status.utilization,
                            status.power_limit
                        ])
                    
                    # Collect for JSON
                    gpu_data.append({
                        'gpu_id': status.gpu_id,
                        'name': status.name,
                        'temperature': status.temperature,
                        'power_usage': status.power_usage,
                        'memory_used': status.memory_used,
                        'memory_total': status.memory_total,
                        'utilization': status.utilization,
                        'power_limit': status.power_limit,
                        'alerts': alerts
                    })
            
            # JSON logging
            json_data['monitoring_data'].append({
                'timestamp': timestamp.isoformat(),
                'gpus': gpu_data
            })
            
            # Save JSON periodically
            if len(json_data['monitoring_data']) % 10 == 0:
                with open(json_file, 'w') as f:
                    json.dump(json_data, f, indent=2)
            
            print("-" * 80)
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        
        # Final JSON save
        json_data['end_time'] = datetime.now().isoformat()
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Final logs saved:")
        print(f"  CSV: {csv_file}")
        print(f"  JSON: {json_file}")


def show_summary(log_file: str):
    """Show summary statistics from log file"""
    try:
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        print(f"GPU Monitoring Summary: {log_file}")
        print(f"Start Time: {data['start_time']}")
        print(f"End Time: {data.get('end_time', 'In Progress')}")
        print(f"Data Points: {len(data['monitoring_data'])}")
        print("-" * 50)
        
        # Calculate statistics per GPU
        gpu_stats = {}
        
        for entry in data['monitoring_data']:
            for gpu in entry['gpus']:
                gpu_id = gpu['gpu_id']
                if gpu_id not in gpu_stats:
                    gpu_stats[gpu_id] = {
                        'name': gpu['name'],
                        'temperatures': [],
                        'power_usage': [],
                        'utilization': [],
                        'memory_usage': [],
                        'alert_count': 0
                    }
                
                gpu_stats[gpu_id]['temperatures'].append(gpu['temperature'])
                gpu_stats[gpu_id]['power_usage'].append(gpu['power_usage'])
                gpu_stats[gpu_id]['utilization'].append(gpu['utilization'])
                gpu_stats[gpu_id]['memory_usage'].append(
                    gpu['memory_used'] / gpu['memory_total'] * 100
                )
                gpu_stats[gpu_id]['alert_count'] += len(gpu['alerts'])
        
        # Print statistics
        for gpu_id, stats in gpu_stats.items():
            print(f"GPU {gpu_id} ({stats['name']}):")
            print(f"  Temperature: {min(stats['temperatures']):.1f}°C - "
                  f"{max(stats['temperatures']):.1f}°C "
                  f"(avg: {sum(stats['temperatures'])/len(stats['temperatures']):.1f}°C)")
            print(f"  Power: {min(stats['power_usage']):.1f}W - "
                  f"{max(stats['power_usage']):.1f}W "
                  f"(avg: {sum(stats['power_usage'])/len(stats['power_usage']):.1f}W)")
            print(f"  Utilization: {min(stats['utilization'])}% - "
                  f"{max(stats['utilization'])}% "
                  f"(avg: {sum(stats['utilization'])/len(stats['utilization']):.1f}%)")
            print(f"  Memory Usage: {min(stats['memory_usage']):.1f}% - "
                  f"{max(stats['memory_usage']):.1f}% "
                  f"(avg: {sum(stats['memory_usage'])/len(stats['memory_usage']):.1f}%)")
            print(f"  Alerts: {stats['alert_count']}")
            print()
    
    except Exception as e:
        print(f"Error reading log file: {e}")


def main():
    parser = argparse.ArgumentParser(description="GPU Monitoring for VCC Transformer")
    parser.add_argument("--log-file", type=str, help="Log file path (without extension)")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
    parser.add_argument("--alert-temp", type=float, default=85.0, 
                       help="Temperature alert threshold (°C)")
    parser.add_argument("--alert-power", type=float, default=350.0,
                       help="Power alert threshold (W)")
    parser.add_argument("--summary", type=str, help="Show summary from JSON log file")
    
    args = parser.parse_args()
    
    if args.summary:
        show_summary(args.summary)
        return 0
    
    # Create GPU optimizer for monitoring
    optimizer = GPUOptimizer()
    
    if not optimizer.enabled:
        print("GPU monitoring not available. Please install nvidia-ml-py3:")
        print("pip install nvidia-ml-py3")
        return 1
    
    print(f"Detected {optimizer.device_count} GPU(s)")
    
    # Start monitoring
    monitor_with_logging(optimizer, args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
