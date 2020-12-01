
import time
import numpy as np
import torch

from random_environment_test import Environment

print('So far:')
print('seed 0: worked in 59 steps')
print('seed 1: worked in 78 steps')
print('seed 2: Final distance = 0.79219455')
print('seed 3: worked in 62 steps')
print('seed 4: worked in 57 steps')
print('seed 5: Final distance = 0.22068891')
print('seed 6: worked in 59 steps')
print('seed 7: worked in 69 steps')
print('seed 8: Final distance = 0.8885965')
print('seed 9: Final distance = 0.8017202')
print('seed 10: worked in 68 steps')

all_states = np.load('/Users/oliviagallup/Downloads/s1_all_states.npy')
state = [0.9099997, 0.4453198]

np.random.seed(1)  # 1606064318  # goodbye 3
environment = Environment(magnification=500)
environment.show(state, all_states, 'every')