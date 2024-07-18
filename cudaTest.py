import os
import torch




os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print("env variable set to ",os.environ['CUDA_VISIBLE_DEVICES'])

print("Device Count: ", torch.cuda.device_count())
# Check if CUDA is available
print("CUDA Available: ", torch.cuda.is_available())

torch.cuda.set_device(0);

print("cuda device set to 0")

print("Device Count", torch.cuda.device_count())

print("Current Device: ", torch.cuda.current_device())

print("CUDA Available: ", torch.cuda.is_available())

