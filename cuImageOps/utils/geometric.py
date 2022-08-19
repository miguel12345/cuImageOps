import numpy as np
import math

def shear_mat(x:float, y:float):
    return np.array([
        [1,                 x,                  x],
        [y,                 1,                  y],
        [0,                 0,                  1],
    ],dtype=np.float32)

def translate_mat(x:float, y:float):
    return np.array([
        [1,                 0,                  x],
        [0,                 1,                  y],
        [0,                 0,                  1],
    ],dtype=np.float32)

def scale_mat(scale_x: float, scale_y: float):
    return np.array([
        [scale_x,           0,                  0],
        [0,                 scale_y,            0],
        [0,                 0,                  1],
    ],dtype=np.float32)

def rotation_mat_rad(theta: float):

    return np.array([
        [math.cos(theta),   -math.sin(theta),   0],
        [math.sin(theta),   math.cos(theta),    0],
        [0,                 0,                  1],
    ],dtype=np.float32)

def rotation_mat_deg(theta: float):
    return rotation_mat_rad(math.radians(theta))