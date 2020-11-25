from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import *
import numpy.linalg as la
import cv2 as cv
from scipy.ndimage import *

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
    
def gauss1d(sigma, kernel_size):
    center = kernel_size/2
    x_dist = np.arange(-center, center + 1)
    gauss1d = np.exp(-(x_dist**2)/(2*sigma**2))
    gauss1d = gauss1d/np.sum(gauss1d)
    return gauss1d

def optical_flow(kernel_size, src_1, src_2):
    height, width = src_1.shape
    blur_matrix = Gaussian_Blur(kernel_size/6, kernel_size)
    src_1 = cv.filter2D(src_1,-1,blur_matrix)
    src_2 = cv.filter2D(src_2,-1,blur_matrix)
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    I_x = cv.filter2D(src_1,-1,sobel_x)
    I_y = cv.filter2D(src_1,-1,sobel_y)
    U = np.zeros((height, width),dtype="float")
    V = np.zeros((height, width),dtype="float")
    mid =  kernel_size // 2
    for i in range(height):
        for j in range(width):
            i_buf = i
            j_buf = j
            max_loop = 10
            left_height_bound = i - mid
            right_height_bound = i + mid + 1
            left_width_bound = j - mid
            right_width_bound = j + mid + 1
            window_Ix = np.zeros((kernel_size,kernel_size),dtype='float')
            window_Iy = np.zeros((kernel_size,kernel_size),dtype='float')
            window_I = np.zeros((kernel_size,kernel_size),dtype='float')
            I_xx = 0.0
            I_yy = 0.0
            I_xy = 0.0
            height_index = 0
            for h in range(left_height_bound, right_height_bound):
                width_index = 0
                for w in range(left_width_bound, right_width_bound):
                    if h < 0 or w < 0 or h >= height or w >= width: continue
                    else:
                        I_xx += (I_x[h][w] * I_x[h][w])
                        I_yy += (I_y[h][w] * I_y[h][w])
                        I_xy += (I_x[h][w] * I_y[h][w])
                        window_Ix[height_index][width_index] = I_x[h][w]
                        window_Iy[height_index][width_index] = I_y[h][w]
                        window_I[height_index][width_index] = src_1[h][w]
                    width_index += 1
                height_index += 1
            M = np.array([[I_xx,I_xy],[I_xy,I_yy]])
            while 1:
                M_inv = None
                try: M_inv = la.inv(M)
                except: break
                left_height_bound_t = i_buf - mid
                right_height_bound_t = i_buf + mid + 1
                left_width_bound_t = j_buf - mid
                right_width_bound_t = j_buf + mid + 1
                window_It = np.zeros((kernel_size,kernel_size),dtype='float')
                height_index = 0
                for h in range(left_height_bound_t, right_height_bound_t):
                    width_index = 0
                    for w in range(left_width_bound_t, right_width_bound_t):
                        if h < 0 or w < 0 or h >= height or w >= width: window_It[height_index][width_index] -= window_I[height_index][width_index]
                        else: window_It[height_index][width_index] = src_2[h][w] - window_I[height_index][width_index]
                        width_index += 1
                    height_index += 1
                window_It = cv.filter2D(window_It,-1,gauss1d(kernel_size/6, kernel_size))
                I_xt = 0.0
                I_yt = 0.0
                for h in range(kernel_size):
                    for w in range(kernel_size):
                        I_xt += window_Ix[h][w] * window_It[h][w]
                        I_yt += window_Iy[h][w] * window_It[h][w]
                N = np.array([-I_xt, -I_yt])
                res = np.dot(M_inv, N)
                u, v = res[0], res[1]
                max_loop -= 1
                if (u + v < 0.5) or not max_loop: break
                j_buf += int(u)
                i_buf += int(v)
                U[i][j] = u
                V[i][j] = v
    return U, V


def getGreyImge(img):
# Change the rgb value to grey. #
    rgb_weights = np.array([0.2989, 0.5870, 0.1140])
    return np.dot(img[...,:3], rgb_weights)
    
def main(path_1, path_2, kernel_size):
    img_1 = plt.imread(path_1)
    img_2 = plt.imread(path_2)
    U, V = optical_flow(5,getGreyImge(img_1), getGreyImge(img_2))
    t = 10
    U_first = U[::t, ::t]
    V_first = V[::t, ::t]
    r, c = getGreyImge(img_1).shape
    cols, rows = np.meshgrid(np.linspace(0, c - 1, c), np.linspace(0, r - 1, r))
    cols = cols[::t,::t]
    rows = rows[::t, ::t]
    plt.figure(figsize=(9,9))
    plt.imshow(img_1)
    plt.quiver(cols, rows, U_first, V_first, color='red')
    plt.show()
        
if __name__ == '__main__':
    main("./Q3_optical_flow/Backyard/frame07.png", "./Q3_optical_flow/Backyard/frame11.png", 25)
