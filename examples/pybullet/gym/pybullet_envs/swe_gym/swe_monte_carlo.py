import pickle
import copy
import matplotlib.pyplot as plt
import numpy as np

mean_dist = [0.0, 0.4, 0.3, 0.05, 0.05, 0.05] 
cov_dist = np.diag([0.01, 0.01, 0.01, 0.0001, 0.0001, 0.0001])

num_simulations = 10
goal_poses = [None]*num_simulations
count = 0

def isValid(goal_poses):
    goal_poses = copy.copy(goal_poses)
    goal_upper_bound = [0.2, 0.6, 0.6, 0.1, 0.1, 0.1]
    # Also use the distance from initial pose to goal pose to check validity
    return True

while count < num_simulations:
    goal_poses[count] = np.random.multivariate_normal(mean_dist, cov_dist)
    if isValid(goal_poses[count]):
        count += 1


goal_poses = np.array(goal_poses)
print(goal_poses)
# Turn this whole script into a class.
