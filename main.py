# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import numpy as np
import matplotlib
from pathlib import Path
import requests
import os
import pandas as pd
import d2l

print(torch.cuda.is_available())
print(torch.__version__)

X = torch.randn(10,10)
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)
inputs, outputs = data.iloc[:, 0:2], data.iloc[:,2]
inputs = inputs.fillna(inputs.mean())
print("inputs\n", inputs)
print("outputs\n", outputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print("new inputs\n",inputs)
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X, y)
print(X.T)
#print(X)
#print(X[1, 2])
#Y = X.numpy()
#print(type(Y), Y)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
