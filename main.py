import numpy as np
import random
import torch
import utils

# seed 고정
utils.set_seed(42)

# 디바이스 설정
device = utils.get_device()

# 디바이스 확인
x = torch.ones(5, device=device)
print(x)
print(f"Device: {device}")

# seed 고정 확인
fruit = ['apple', 'banana', 'pear', 'strawberry', 'pineapple']
random.shuffle(fruit)
print(fruit)
print(random.uniform(10.5, 20.5))
print(np.random.rand(5))