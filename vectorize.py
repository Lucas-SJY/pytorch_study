import torch
import d2l
import numpy as np
import pandas as pd
import matplotlib
import time


class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


n = 10000
a = torch.ones([n])
b = torch.ones([n])
c = torch.zeros(n)
timer = Timer()
'''for i in range(n):
    c[i] = a[i] + b[i]
print(f'{timer.stop():.5f} sec')'''

timer.start()
d = a + b
print(f'{timer.stop():.5f} sec')