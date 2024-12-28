import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def plot_metrics(log_file):
    # Read CSV log file
    df = pd.read_csv(log_file)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot loss
    ax1.plot(df['epoch'], df['epoch_val_loss'], label='Validation Loss')
    ax1.set_title('Training Progress')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(df['epoch'], df['epoch_val_acc_top1'], label='Top-1 Accuracy')
    ax2.plot(df['epoch'], df['epoch_val_acc_top5'], label='Top-5 Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def print_latest_metrics(log_file):
    df = pd.read_csv(log_file)
    latest = df.iloc[-1]
    
    print("\n" + "="*50)
    print(f"Latest Metrics (Epoch {int(latest['epoch'])})")
    print("-"*50)
    print(f"Validation Loss:    {latest['epoch_val_loss']:.4f}")
    print(f"Top-1 Accuracy:     {latest['epoch_val_acc_top1']:.4f}")
    print(f"Top-5 Accuracy:     {latest['epoch_val_acc_top5']:.4f}")
    print("="*50 + "\n")

def main():
    # Get latest log file
    log_dir = Path('logs')
    csv_logs = list(log_dir.glob('*.csv'))
    
    if not csv_logs:
        print("No log files found!")
        return
        
    latest_log = max(csv_logs, key=lambda p: p.stat().st_mtime)
    
    print(f"Analyzing log file: {latest_log}")
    print_latest_metrics(latest_log)
    
    # Plot if requested
    if len(sys.argv) > 1 and sys.argv[1] == '--plot':
        plot_metrics(latest_log)

if __name__ == '__main__':
    main() 