import numpy as np
from scipy.spatial.transform import Rotation as R
from HandEyeCalibration import Solution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generateSampleData(n=100, mode = 'dynamic', noise_translation=0.005, noise_rotation=3.0, randomness_translation = 0.005, randomness_rotation = 4.0):
    if mode not in ['dynamic', 'static']:
        raise ValueError('Mode not recognized')
    
    R_dynamic = np.asarray(R.from_euler('xyz', [5, 5, 90], degrees=True).as_matrix()) # initially unknown but identified via calibrateRotation.py
    t_dynamic = np.array([[0.1], [0.1], [0.1]])  # in TCP COS (unknown)
    R_static = np.asarray(R.from_euler('xyz', [-90, 90, 5], degrees=True).as_matrix())
    t_static = np.array([[-0.4], [-3.8], [0.54]]) # in base COS (unknown)
    if mode == 'dynamic':
        R_cam = R_dynamic
        t_cam = t_dynamic
        t_target = t_static
        R_target = R_static
    else:
        R_cam = R_static
        t_cam = t_static
        t_target = t_dynamic
        R_target = R_dynamic
    print('target:')
    print(t_target)
    print(R_target)
    
    print('cam:')
    print(t_cam)
    print(R_cam)
    
    t_TCP_list = np.zeros((3, n))
    R_TCP_list = np.zeros((3, 3, n))
    t_meas_list = np.zeros((3, n))
    R_meas_list = np.zeros((3, 3, n))
    for i in range(n):
        # generate random measurments
        t_random = randomness_translation * (2 * np.random.random((3,1))-1)
        R_random = np.asarray(R.from_euler('xyz', randomness_rotation * (2*np.random.random(3)-1), degrees=True).as_matrix())
        R_TCP = np.asarray(R.from_rotvec([np.pi/2, 0, -0.9*np.sin(i*2*np.pi/n)]).as_matrix()) # random TCP rotation
        t_TCP = np.array([[np.sin(i*2*np.pi/n)-1.3], [-1], [0.5*np.sin(2*i*2*np.pi/n)+0.5]]) # random TCP position
        t_TCP = t_TCP + t_random
        R_TCP = R_TCP @ R_random
        t_noise = noise_translation * (2 * np.random.random((3,1))-1)
        R_noise = np.asarray(R.from_euler('xyz', noise_rotation * (2*np.random.random(3)-1), degrees=True).as_matrix())
        if mode == 'dynamic':
            t_meas = R_cam.T @ (R_TCP.T @ (t_target - t_TCP) - t_cam) + t_noise # noisy measurement
            R_meas = R_cam.T @ R_TCP.T @ R_target @ R_noise # noisy measurement
        else:
            t_meas = R_cam.T @ (t_TCP + R_TCP @ t_target - t_cam) + t_noise
            R_meas = R_cam.T @ R_TCP @ R_target @ R_noise
    
        # assemble matrices
        t_TCP_list[:,i] = t_TCP.T
        R_TCP_list[:,:,i] = R_TCP
        t_meas_list[:,i] = t_meas.T
        R_meas_list[:,:,i] = R_meas

    return t_TCP_list, R_TCP_list,t_meas_list, R_meas_list, R_cam, t_cam, R_target, t_target

def plotSolution(solution, connectTCP = True, plotTCPline = True, plotDynamicLine = True, plotViewLine = True, filename=None):
    n = solution.t_TCP_list.shape[1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if connectTCP:
        ax.plot(solution.t_TCP_list[0,:], solution.t_TCP_list[1,:], solution.t_TCP_list[2,:], c='k', alpha=0.8, linestyle='dashed')
        ax.scatter(solution.t_TCP_list[0,:], solution.t_TCP_list[1,:], solution.t_TCP_list[2,:], c='k', s=10)
    
    if solution.mode == 'dynamic':
        static_model = 'Target'
        dynamic_model = 'Cam'
        t_dynamic = solution.t_cam
        R_dynamic = solution.R_cam
        t_static = solution.t_target
        R_static = solution.R_target
    else:
        static_model = 'Cam'
        dynamic_model = 'Target'
        t_dynamic = solution.t_target
        R_dynamic = solution.R_target
        t_static = solution.t_cam
        R_static = solution.R_cam
        
    # Plot the static model
    plotModel(ax, t_static, R_static, model = static_model)
    # Plot the dynamic model and the TCP COS    
    for i in range(n):
        plotModel(ax, solution.t_TCP_list[:,i], solution.R_TCP_list[:,:,i], model = 'COS')
        t_dynamic_i = (solution.t_TCP_list[:,i].T + (solution.R_TCP_list[:,:,i] @ t_dynamic).T).T
        if solution.mode == 'dynamic':
            # from dynamic cam to static target
            t_cam_i = t_dynamic_i
            t_meas_i = solution.R_TCP_list[:,:,i] @ solution.R_cam @ solution.t_meas_list[:,i]
        else:
            # from static cam to dynamic target
            t_cam_i = solution.t_cam
            t_meas_i = solution.R_cam @ solution.t_meas_list[:,i]
        plotModel(ax, t_dynamic_i, solution.R_TCP_list[:,:,i] @ R_dynamic, model = dynamic_model)
        if plotTCPline:
            ax.plot([0, solution.t_TCP_list[0,i]], [0, solution.t_TCP_list[1,i]], [0, solution.t_TCP_list[2,i]], c='k', alpha=1.0)
        if plotDynamicLine:
            ax.plot([t_dynamic_i[0][0], solution.t_TCP_list[0,i]], [t_dynamic_i[1][0], solution.t_TCP_list[1,i]], [t_dynamic_i[2][0], solution.t_TCP_list[2,i]], c='k', alpha=1.0)
        if plotViewLine:
            ax.plot([t_cam_i[0][0], t_cam_i[0][0]+t_meas_i[0]], [t_cam_i[1][0], t_cam_i[1][0]+t_meas_i[1]], [t_cam_i[2][0], t_cam_i[2][0]+t_meas_i[2]], c='k', alpha=0.3)
            ax.scatter(t_cam_i[0][0]+t_meas_i[0], t_cam_i[1][0]+t_meas_i[1], t_cam_i[2][0]+t_meas_i[2], c='r', s=20)
        

    # Origin
    plotModel(ax, [0,0,0], np.eye(3), model = 'COS')
    # Set labels
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.view_init(elev=45, azim=-40, roll=0)
    ax.set_aspect('equal', 'box')
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight',transparent=True)
    plt.show()
    
def plotSolution2D(solution, filename=None):
    n = solution.t_TCP_list.shape[1]
    t_target_computed = np.zeros((3, n))
    r_target_computed = np.zeros((3, n))
    r_target = R.from_matrix(solution.R_target).as_rotvec()
    for i in range(n):
        if solution.mode == 'dynamic':
            A = solution.R_TCP_list[:,:,i]
            b = solution.R_TCP_list[:,:,i] @ solution.R_cam @ solution.t_meas_list[:,i] + solution.t_TCP_list[:,i] # this is the target in base COS
            R_target = solution.R_TCP_list[:,:,i] @ solution.R_cam @ solution.R_meas_list[:,:,i]
        else:
            A = solution.R_TCP_list[:,:,i].T
            b = solution.R_TCP_list[:,:,i].T @ (solution.R_cam @ solution.t_meas_list[:,i] - solution.t_TCP_list[:,i])
            R_target = solution.R_TCP_list[:,:,i].T @ solution.R_cam @ solution.R_meas_list[:,:,i]
        b.shape = (3,1)
        t_target_computed[:,i] = (A @ solution.t_cam + b).T
        r_target_computed[:,i] = R.from_matrix(R_target).as_rotvec()
    
    fig, axs = plt.subplots(2)
    for i in range(3):
        col = ['r', 'g', 'b'][i]
        axs[0].plot(range(n),t_target_computed[i,:], c=col)
        axs[0].plot([0, n-1],[solution.t_target[i], solution.t_target[i]], c='k', linestyle='dashed')
    axs[0].set(ylabel='target position / m')
    
    for i in range(3):
        col = ['r', 'g', 'b'][i]
        axs[1].plot(range(n),r_target_computed[i,:], c=col)
        axs[1].plot([0, n-1],[r_target[i], r_target[i]], c='k', linestyle='dashed')
    axs[1].set(xlabel='time step', ylabel='target rotation / rad')
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight',transparent=True)
    
def plotModel(ax, t, R, model = 'Cam'):
    if model == 'Cam':
        cam_length = 0.3
        cam_width = 0.3
        cam_height = cam_width * 1080 / 720
        cam_model = np.array([[0, 0, 0],[cam_width/2, cam_height/2, cam_length],[cam_width/2, -cam_height/2, cam_length],[-cam_width/2, -cam_height/2, cam_length],[-cam_width/2, cam_height/2, cam_length]])
        cam_model = cam_model[[0, 1, 2, 3, 4, 1, 2, 0, 3, 4, 0],:]
        cam_model_transformed = (R @ cam_model.T) + t
        ax.plot(cam_model_transformed[0,:], cam_model_transformed[1,:], cam_model_transformed[2,:], c='r')
    elif model == 'Target':
        target_width = 0.2
        target_coords = np.array([[target_width, target_width, 0], [target_width, -target_width, 0], [-target_width, -target_width, 0], [-target_width, target_width, 0], [target_width, target_width, 0]])
        target_coords = np.dot(R, target_coords.T) + t
        ax.plot(target_coords[0,:], target_coords[1,:], target_coords[2,:], c='k')
        ax.scatter(t[0], t[1], t[2], c='k')
    elif model == 'COS':
        ax_len = 0.4
        for i in range(3):
            col = ['r', 'g', 'b'][i]
            ax.plot([ax_len*R[0,i]+t[0], t[0]], [ax_len*R[1,i]+t[1], t[1]], [ax_len*R[2,i]+t[2], t[2]], c=col)
    else:
        raise ValueError('Model not recognized')