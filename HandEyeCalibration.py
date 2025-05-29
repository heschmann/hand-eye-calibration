import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.optimize as sopt
import time

class Solution:
    def __init__(self, t_cam, R_cam, t_target, R_target, mode, residual_translation, residual_rotation, time_translation, time_rotation):
        self.t_cam = t_cam
        self.R_cam = R_cam
        self.t_target = t_target
        self.R_target = R_target
        self.mode = mode
        self.residual_translation = residual_translation
        self.residual_rotation = residual_rotation
        self.time_translation = time_translation
        self.time_rotation = time_rotation
        self.t_TCP_list = None
        self.R_TCP_list = None
        self.t_meas_list = None
        self.R_meas_list = None
        self.jointly = None
        self.x0 = None
        self.x = None
        self.alpha = None
        self.constraints = []
        self.residual_constraints = None
    
    def printSolution(self):
        print('Camera Translation:')
        print(self.t_cam)
        print('Camera Rotation:')
        print(self.R_cam)
        print('Target Translation:')
        print(self.t_target)
        print('Target Rotation:')
        print(self.R_target)
        
    def printDiagnostics(self):
        if self.jointly:
            print(f'Translation and rotation were solved jointly with alpha = {self.alpha}.')
        else:
            print('Translation and rotation were solved sequentially.')
        print('Time for translation calibration:')
        print(self.time_translation)
        print('Translation Residual:')
        print(self.residual_translation)
        print('Time for rotation calibration:')
        print(self.time_rotation)
        print('Rotation Residual:')
        print(self.residual_rotation)
        if len(self.constraints) > 0:
            print('Constraint Residual:')
            print(self.residual_constraints)

class Constraint:
    def __init__(self, quantity, type, mean, tol = 0.005, factor = 100.0):
        if quantity not in ['camera', 'target']:
            raise ValueError('Constraint quantity not recognized')
        if type not in ['norm', 'position']:
            raise ValueError('Constraint type not recognized')
        if not isinstance(mean, (float, int)) and type == 'norm':
            raise ValueError('Mean must be a scalar when type is "norm"')
        elif not isinstance(mean, (np.ndarray)) and type == 'position':
            raise ValueError('Mean must be a column vector when type is "position"')
        if tol < 0.0:
            raise ValueError('Tolerance must be positive')
        if type == 'position' and mean.shape[0] != 3:
            print('Warning: Mean for position constraint was not a column vector. Reshaping to (3,1).')
            mean.shape = (3,1) # ensure mean is a column vector
        self.quantity = quantity # 'cam' or 'target'
        self.type = type # 'norm' or 'position'
        self.mean = mean # guess for the position (column vector) or norm (positive scalar)
        self.tol = tol # tolerance for the constraint (positive scalar) in which the constraint is not active
        self.factor = factor # factor for the constraint (positive scalar) which is multiplied with the residual if the constraint is active

def getTranslationResiduals(t_TCP_list, R_TCP_list, t_meas_list, R_cam, mode):
    n = t_TCP_list.shape[1]
    A_all = np.empty(shape=(0,3))
    b_all = np.empty(shape=(0,1))
    for i in range(n):
        # assemble matrices
        if mode == 'dynamic':
            # this is the target in base COS
            A = R_TCP_list[:,:,i]
            b = R_TCP_list[:,:,i] @ R_cam @ t_meas_list[:,i] + t_TCP_list[:,i]
        else:
            # this is the target in TCP COS
            A = R_TCP_list[:,:,i].T
            b = R_TCP_list[:,:,i].T @ (R_cam @ t_meas_list[:,i] - t_TCP_list[:,i])
        b.shape = (3,1)
        A_all = np.append(A_all, A, axis=0)
        b_all = np.append(b_all, b, axis=0)

    result = np.linalg.lstsq(np.hstack((A_all, -np.tile(np.identity(3), (n, 1)))), -b_all,rcond=None)
    t_cam = result[0][0:3]
    t_cam.shape = (3,1)
    residual = result[1][0]
    t_target = result[0][3:6]
    t_target.shape = (3,1)
    return residual, t_cam, t_target

def getRotationResiduals(x, R_TCP_list, R_meas_list, mode):
    R_cam = np.asarray(R.from_rotvec(x[0:3]).as_matrix())
    R_target = np.asarray(R.from_rotvec(x[3:6]).as_matrix())
    n = R_TCP_list.shape[2]
    vars = np.zeros(n)
    for i in range(n):
        R_TCP = R_TCP_list[:,:,i]
        R_meas = R_meas_list[:,:,i]
        if mode == 'dynamic':
            diff = R_TCP @ R_cam @ R_meas @ R_target.T
        else:
            diff = R_TCP.T @ R_cam @ R_meas @ R_target.T
        vars[i] = np.trace(diff) # = 1 + 2 * cos(theta)
    return np.arccos(np.clip((vars - 1) / 2, -1.0, 1.0))

def getConstraintResidual(constraints, t_cam, t_target):
    constraint_residual = np.array([])
    for constr in constraints:
        if constr.type == 'norm':
            if constr.quantity == 'camera':
                constraint_residual = np.append(constraint_residual, constr.factor*np.maximum(np.abs(np.linalg.norm(t_cam)-constr.mean)-constr.tol,0))
            else:
                # target
                constraint_residual = np.append(constraint_residual, constr.factor*np.maximum(np.abs(np.linalg.norm(t_target)-constr.mean)-constr.tol,0))
        else:
            # position
            if constr.quantity == 'camera':
                constraint_residual = np.append(constraint_residual, constr.factor*np.maximum(np.abs(np.linalg.norm(t_cam-constr.mean))-constr.tol,0))
            else:
                # target
                constraint_residual = np.append(constraint_residual, constr.factor*np.maximum(np.abs(np.linalg.norm(t_target-constr.mean))-constr.tol,0))
    return constraint_residual

def getJointResiduals(x, t_TCP_list, R_TCP_list,t_meas_list, R_meas_list, mode, constraints, alpha):
    residual_rotation = getRotationResiduals(x, R_TCP_list, R_meas_list, mode)
    R_cam = np.asarray(R.from_rotvec(x[0:3]).as_matrix())
    R_target = np.asarray(R.from_rotvec(x[3:6]).as_matrix())
    
    residual_translation, t_cam, t_target = getTranslationResiduals(t_TCP_list, R_TCP_list, t_meas_list, R_cam, mode)
    joint_residual = np.append(residual_rotation, alpha*residual_translation)

    constraint_residual = getConstraintResidual(constraints, t_cam, t_target)
    joint_residual = np.append(joint_residual, constraint_residual)
    return  joint_residual

def solveHandEyeCalibration(t_TCP_list, R_TCP_list,t_meas_list, R_meas_list, mode = 'dynamic', jointly = False, x0 = np.zeros(6), constraints = [], alpha = 1.0):
    if mode not in ['dynamic', 'static']:
        raise ValueError('Mode not recognized')
    # Solve for rotation
    loss = 'linear' # linear soft_l1
    start = time.time()
    if jointly:
        if len(constraints) > 0:
            print(f'{len(constraints)} constraints were found.')
        result = sopt.least_squares(getJointResiduals, x0, args=(t_TCP_list, R_TCP_list,t_meas_list, R_meas_list, mode, constraints, alpha), loss=loss)
        residual_rotation = 0.5 * np.sum(np.square(getRotationResiduals(result.x, R_TCP_list, R_meas_list, mode)))
    else:
        if len(constraints) > 0:
            print('Constraints are only available for joint optimization. They will be ignored in the following.')
        result = sopt.least_squares(getRotationResiduals, x0, args=(R_TCP_list, R_meas_list, mode), loss=loss)
        residual_rotation = result.cost
    end = time.time()
    time_rotation = end - start
    R_cam = R.from_rotvec(result.x[0:3]).as_matrix()
    R_target = R.from_rotvec(result.x[3:6]).as_matrix()
    
    # Solve for translation
    start = time.time()
    residual_translation, t_cam, t_target = getTranslationResiduals(t_TCP_list, R_TCP_list, t_meas_list, R_cam, mode)
    end = time.time()
    time_translation = end - start
    
    print('Solution found.')
    sol = Solution(t_cam, R_cam, t_target, R_target, mode, residual_translation, residual_rotation, time_translation, time_rotation)
    sol.t_TCP_list = t_TCP_list
    sol.R_TCP_list = R_TCP_list
    sol.t_meas_list = t_meas_list
    sol.R_meas_list = R_meas_list
    sol.jointly = jointly
    sol.x0 = x0
    sol.x = result.x
    sol.alpha = alpha
    sol.constraints = constraints
    sol.residual_constraints = getConstraintResidual(constraints, t_cam, t_target)
    return sol