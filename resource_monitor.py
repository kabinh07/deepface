import psutil
import GPUtil
import torch
import time

# Define global lists to store resource usage
cpu_usage_list = []
ram_usage_list = []
gpu_usage_list = []
torch_mem_list = []

def get_cpu_ram_usage():
    """ Get current CPU and RAM utilization """
    cpu_usage = psutil.cpu_percent(interval=0.5)  # 0.5s sampling interval
    ram_info = psutil.virtual_memory()
    ram_usage = ram_info.used / (1024 ** 3)  # Convert bytes to GB
    return cpu_usage, ram_usage

def get_gpu_usage():
    """ Get current GPU utilization using GPUtil """
    gpus = GPUtil.getGPUs()
    if not gpus:
        return 0, 0  # No GPU detected

    gpu = gpus[0]  # Assume first GPU (modify if multi-GPU needed)
    return gpu.load * 100, gpu.memoryUsed  # GPU Load (%) and Memory Used (MB)

def get_torch_gpu_usage():
    """ Get current PyTorch-specific GPU memory usage """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1e6  # Convert bytes to MB
    return 0

def monitor_resources(stop_event):
    """ Continuously monitor CPU, RAM, and GPU usage during inference """
    global cpu_usage_list, ram_usage_list, gpu_usage_list, torch_mem_list
    while not stop_event.is_set():  # Check the threading event instead of stop_monitoring variable
        cpu, ram = get_cpu_ram_usage()
        gpu_load, gpu_mem = get_gpu_usage()
        torch_mem = get_torch_gpu_usage()

        # Store values
        cpu_usage_list.append(cpu)
        ram_usage_list.append(ram)
        gpu_usage_list.append(gpu_load)
        torch_mem_list.append(torch_mem)

        time.sleep(0.1)  # Sample every 100ms