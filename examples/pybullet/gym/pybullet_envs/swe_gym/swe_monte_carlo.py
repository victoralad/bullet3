import pickle
import copy
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

class MonteCarlo:
    def __init__(self, num_simulations):
        self.num_simulations = num_simulations
        self.mean_dist = [0.0, 0.4, 0.3, 0.05, 0.05, 0.05] 
        self.cov_dist = np.diag([0.01, 0.04, 0.04, 0.0001, 0.0001, 0.0001])
        self.goal_poses = [None]*num_simulations
    
    # Run the simulation
    def RunSimulation(self):
        count = 0
        while count < self.num_simulations:
            self.goal_poses[count] = np.random.multivariate_normal(self.mean_dist, self.cov_dist)
            if self.IsValid(self.goal_poses[count]):
                count += 1

    # Check if the sampled goal pose is a valid pose.
    def IsValid(self, goal_poses):
        goal_poses = np.array(copy.copy(goal_poses))
        obj_init_pose = np.array([0.0, 0.7, 0.02, 0.0, 0.0, 0.0])
        goal_upper_bound = np.array([0.2, 0.6, 0.6, 0.1, 0.1, 0.1])
        goal_lower_bound = np.array([-0.2, 0.05, 0.05, -0.1, -0.1, -0.1])
        # Check if sampled goal pose is within the bounds of a valid goal pose.
        if all(goal_poses < goal_upper_bound) and all(goal_poses > goal_lower_bound):
            return True
        return False
    
    # Return the goalposes and ensure that the goal poses have been updated.
    def GetGoalPoses(self):
        assert self.goal_poses[-1] is not None
        return np.array(self.goal_poses)

monte_c = MonteCarlo(10)
monte_c.RunSimulation()
goal_poses = monte_c.GetGoalPoses()
goal_poses = np.array(goal_poses)
# print(goal_poses)