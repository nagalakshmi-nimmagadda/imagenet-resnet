import time
import psutil
import torch
from datetime import datetime

class CostMonitor:
    def __init__(self, instance_type, cost_per_hour):
        self.start_time = time.time()
        self.instance_type = instance_type
        self.cost_per_hour = cost_per_hour
        
    def log_metrics(self):
        elapsed_hours = (time.time() - self.start_time) / 3600
        cost_usd = elapsed_hours * self.cost_per_hour
        gpu_util = torch.cuda.utilization()
        
        metrics = {
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cost_usd': cost_usd,
            'gpu_utilization': gpu_util,
            'memory_used': f"{torch.cuda.memory_allocated() / 1e9:.2f}GB"
        }
        return metrics 