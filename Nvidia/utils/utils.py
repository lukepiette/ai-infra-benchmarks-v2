import psutil
import pynvml



# Function to monitor the resources of the system
def monitor_resources():
    # CPU utilization
    cpu_usage = psutil.cpu_percent(interval=1)

    # GPU utilization
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

    return {
        'cpu_usage': cpu_usage,
        'gpu_usage': gpu_utilization.gpu,
        'gpu_memory_used': mem_info.used / (1024 * 1024 * 1024),  # Convert to MB
        'gpu_memory_total': mem_info.total / (1024 * 1024 * 1024)  # Convert to MB
    }
    
    
    
    