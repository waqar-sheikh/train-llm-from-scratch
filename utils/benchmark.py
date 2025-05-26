import torch
import time
import statistics

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(func, *args, warmup_steps=5, num_steps=5):
    if warmup_steps > 0:
        for i in range(warmup_steps):
            func(*args)
            cuda_sync()

    times = []
    for i in range(num_steps):
        start = time.time()
        func(*args)
        cuda_sync()
        times.append(time.time() - start)
    
    return statistics.mean(times), statistics.variance(times)