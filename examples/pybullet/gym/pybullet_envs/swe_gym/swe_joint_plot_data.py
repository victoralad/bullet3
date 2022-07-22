import pickle
import matplotlib.pyplot as plt
import numpy as np

seed = 6
gauss = 5
exp_run = 107
baseline_exp_run = 107

with open('data/rl/obj_pose_error_{}_seed_{}_gauss_{}.data'.format(exp_run, seed, gauss), 'rb') as filehandle:
    # read the data as binary data stream
    obj_pose_error_data = pickle.load(filehandle)

with open('data/no_rl/obj_pose_error_{}_gauss_{}.data'.format(exp_run, gauss), 'rb') as filehandle:
    # read the data as binary data stream
    obj_pose_error_data_no_rl = pickle.load(filehandle)

with open('data/baseline/obj_pose_error_baseline_{}.data'.format(baseline_exp_run), 'rb') as filehandle:
    # read the data as binary data stream
    obj_pose_error_data_baseline = pickle.load(filehandle)

print(np.mean(obj_pose_error_data_baseline))
plt.plot(list(range(len(obj_pose_error_data))), obj_pose_error_data)
plt.plot(list(range(len(obj_pose_error_data_no_rl))), obj_pose_error_data_no_rl)
plt.plot(list(range(len(obj_pose_error_data_baseline))), obj_pose_error_data_baseline, 'm-')
# naming the x axis
plt.xlabel('Num-time-steps')
# naming the y axis
plt.ylabel('Obj-pose-error-norm')

plt.legend(["Residual RL", "Standard", "Baseline"])
  
# giving a title to my graph
plt.title('Plot of Object pose error norm over a single episode \n Avg object pose error norm: Residual RL = {:.3f}, Standard = {:.3f}, \n Baseline = {:.3f}'.format(np.mean(obj_pose_error_data), np.mean(obj_pose_error_data_no_rl), np.mean(obj_pose_error_data_baseline)))

# Saving the figure.
plt.savefig("data/joint_plot/trial_{}/eps_OPEN_plot_{}_seed_{}_gauss_{}.svg".format(exp_run, exp_run, seed, gauss))

plt.show()
