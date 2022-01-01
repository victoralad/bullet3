import time
import math
import numpy as np
import copy

from datetime import datetime
from attrdict import AttrDict
from collections import namedtuple

np.random.seed(0)
mean_dist = [0.0]*6
cov_dist_vec = [0.08]*6
cov_dist = np.diag(cov_dist_vec)
for _ in range(5):
  disturbance = np.random.multivariate_normal(mean_dist, cov_dist)
  print(disturbance)
