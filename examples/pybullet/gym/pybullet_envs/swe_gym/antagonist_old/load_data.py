#!/usr/bin/env python3

import numpy as np

data = np.load('test_control.npy')
print(data[0])
print("------------")
data_2 = np.load('test_joints.npy')
print(data_2[0])