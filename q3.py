from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import *
import numpy.linalg as la
import cv2 as cv

def Gaussian_Blur(sigma,kernel_size):
# This can use array to boost access speed. #
    half_kernel = (kernel_size-1) // 2
    Gaussian_matrix = np.zeros((kernel_size,kernel_size),dtype=float)
    for i in range(kernel_size):
        for j in range(kernel_size):
            Gaussian_matrix[i][j] = Gaussian_Model(sigma,(j-half_kernel,half_kernel-i))
    return Gaussian_matrix
    
def Gaussian_Model(sigma,point):
    x, y = point
    ratio = 1/(2*math.pi*sigma**2)
# Slightly different for the expression from the lecture. #
    e_part = math.exp(-(x**2+y**2)/(2*sigma**2))
    return ratio * e_part


def optical_flow(kernel_size, src_1, src_2):
    window = Gaussian_Blur(kernel_size/6,kernel_size)
    src_1= cv.filter2D(src_1,-1,window)
    src_2 = cv.filter2D(src_2,-1,window)
    height, width = src_1.shape
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    grad_x = cv.filter2D(src_1,-1,sobel_x)
    grad_y = cv.filter2D(src_1,-1,sobel_y)
    I_xy = grad_x * grad_y
    I_x = grad_x * grad_x
    I_y = grad_y * grad_y
    I_t = src_2 - src_1
    midpoint = kernel_size // 2
    U = np.zeros((height, width),dtype="float")
    V = np.zeros((height, width),dtype="float")
    for i in range(height):
        for j in range(width):
            bound_x_left = j - midpoint - 1
            if j - midpoint - 1 < 0: bound_x_left = 0
            bound_x_right = j + midpoint
            if j + midpoint >= width: bound_x_right = width - 1
            bound_y_left = i - midpoint - 1
            if i - midpoint - 1 < 0: bound_y_left = 0
            bound_y_right = i + midpoint
            if i + midpoint >= height: bound_y_right = height - 1
            window_Ix = I_x[bound_x_left:bound_x_right, bound_y_left:bound_y_right]
            window_Iy = I_y[bound_x_left:bound_x_right, bound_y_left:bound_y_right]
            window_It = -I_t[bound_x_left:bound_x_right, bound_y_left:bound_y_right]
            A = np.vstack((window_Ix.flatten(), window_Iy.flatten()))
            A = A.T
            U[i][j], V[i][j] = np.dot(np.linalg.pinv(A), window_It.flatten())
    t = 10
    U_first = U[::t, ::t]
    V_first = V[::t, ::t]
    
    r, c = src_1.shape
    cols, rows = np.meshgrid(np.linspace(0, c - 1, c), np.linspace(0, r - 1, r))
    cols = cols[::t,::t]
    rows = rows[::t, ::t]
    plt.figure(figsize=(9,9))
    plt.imshow(src_1)
    plt.quiver(cols, rows, U_first, V_first, color='white')
    plt.show()
            
            
if __name__ == '__main__':
    img_1 = cv.imread("./Q3_optical_flow/Backyard/frame07.png", cv.IMREAD_GRAYSCALE)
    img_2 = cv.imread("./Q3_optical_flow/Backyard/frame08.png", cv.IMREAD_GRAYSCALE)
    optical_flow(15,img_1, img_2)
            
            
