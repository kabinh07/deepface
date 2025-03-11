# from resource_monitor import monitor_resources, cpu_usage_list, ram_usage_list, gpu_usage_list, torch_mem_list
from deepface import DeepFace
# import threading
# import time
# import torch

# print("CUDA Available:", torch.cuda.is_available())

def run_deepface_inference(model_name="Facenet512"):
    """ Run DeepFace model inference and measure resource utilization """
    img1 = "data/image_1.jpg"  # Change to your image path
    img2 = "data/image_2.jpg"  # Change to your image path

    print(f"Running DeepFace model: {model_name}...\n")

    # # Reset global lists
    # cpu_usage_list.clear()
    # ram_usage_list.clear()
    # gpu_usage_list.clear()
    # torch_mem_list.clear()

    # Start resource monitoring in a separate thread
    # stop_event = threading.Event()
    # monitor_thread = threading.Thread(target=monitor_resources, args=(stop_event,))
    # monitor_thread.start()

    # Start timer
    # start_time = time.time()

    # Run DeepFace model
    result = DeepFace.verify(img1, img2, model_name=model_name, detector_backend="retinaface", enforce_detection=False)

    # # Stop monitoring
    # stop_event.set()  # Signal the monitoring thread to stop
    # monitor_thread.join()  # Wait for the thread to finish

    # # Compute average and max usage
    # avg_cpu = sum(cpu_usage_list) / len(cpu_usage_list) if cpu_usage_list else 0
    # max_cpu = max(cpu_usage_list, default=0)
    
    # avg_ram = sum(ram_usage_list) / len(ram_usage_list) if ram_usage_list else 0
    # max_ram = max(ram_usage_list, default=0)

    # avg_gpu = sum(gpu_usage_list) / len(gpu_usage_list) if gpu_usage_list else 0
    # max_gpu = max(gpu_usage_list, default=0)

    # avg_torch_mem = sum(torch_mem_list) / len(torch_mem_list) if torch_mem_list else 0
    # max_torch_mem = max(torch_mem_list, default=0)

    # # Compute inference time
    # elapsed_time = time.time() - start_time

    # Print results
    print("\n=== DeepFace Results ===")
    print(result)

    # print("\n=== CPU & RAM Usage ===")
    # print(f"Avg CPU Usage (%): {avg_cpu:.2f}, Max CPU Usage (%): {max_cpu:.2f}")
    # print(f"Avg RAM Usage (GB): {avg_ram:.2f}, Max RAM Usage (GB): {max_ram:.2f}")

    # print("\n=== GPU Stats ===")
    # print(f"Avg GPU Load (%): {avg_gpu:.2f}, Max GPU Load (%): {max_gpu:.2f}")

    # print("\n=== PyTorch GPU Memory ===")
    # print(f"Avg Allocated (MB): {avg_torch_mem:.2f}, Max Allocated (MB): {max_torch_mem:.2f}")

    # print(f"\nTotal Inference Time: {elapsed_time:.2f} seconds\n")

if __name__ == "__main__":
    run_deepface_inference()
