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
    if (kernel_size%2 == 0): length = length + 1
    center = kernel_size/2
    x_dist = np.arange(-center, center + 1)
    gauss1d = np.exp(-(x_dist**2)/(2*sigma**2))
    gauss1d = gauss1d/np.sum(gauss1d)
    return gauss1d
'''
def optical_flow(kernel_size, src_1, src_2):
    m, n = src_1.shape
    window = Gaussian_Blur(kernel_size/6,kernel_size)
# Decrease the noise by using Gaussian
    src_1 = cv.filter2D(src_1,-1,window)
    src_2 = cv.filter2D(src_2,-1,window)
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
# Gradient x
    I_x = cv.filter2D(src_1,-1,sobel_x)
# Gradient y
    I_y = cv.filter2D(src_1,-1,sobel_y)
# Gradient I x y
    I_xy = I_x * I_y
# Two array to save the displacement
    U = np.zeros((m, n),dtype="float")
    V = np.zeros((m, n),dtype="float")
# Mid of windows
    midpoint = kernel_size // 2
    for i in range(m):
        for j in range(n):
# Initialize best_u and best_v with infinity value
            best_u = 99999999.9
            best_v = 99999999.9
# Set i_buff and j_buff same with the original pixel position
            i_buff = i
            j_buff = j
# Maximum time for iterations
            counter = 0
# Bound for windows
            left_bound_x = i - midpoint
            right_bound_x = i + midpoint + 1
            left_bound_y = j - midpoint
            right_bound_y = j + midpoint + 1
# Element of second moment matrix
            window_Ix = np.zeros((kernel_size,kernel_size),dtype='float')
            window_Iy = np.zeros((kernel_size,kernel_size),dtype='float')
            window_I = np.zeros((kernel_size,kernel_size),dtype='float')
# Sigma value of each position of second moment matrix
            I_xx = 0.0
            I_yy = 0.0
            I_xy = 0.0
# Set the value for each windows and calculate the sigma value
            count_x = 0
            for x in range(left_bound_x, right_bound_x):
                count_y = 0
                for y in range(left_bound_y, right_bound_y):
                    if x<0 or x>=m or y<0 or y>=n:
                        I_xx += 0.0
                        I_yy += 0.0
                        I_xy += 0.0
                        window_Ix[count_x][count_y] = 0.0
                        window_Iy[count_x][count_y] = 0.0
                        window_I[count_x][count_y] = 0.0
                    else:
                        I_xx += (I_x[x][y] * I_x[x][y])
                        I_yy += (I_y[x][y] * I_y[x][y])
                        I_xy += (I_x[x][y] * I_y[x][y])
                        window_Ix[count_x][count_y] = I_x[x][y]
                        window_Iy[count_x][count_y] = I_y[x][y]
                        window_I[count_x][count_y] = src_1[x][y]
                    count_y += 1
                count_x += 1
### Second Moment Matrix
            M = np.array([[I_xx,I_xy],[I_xy,I_yy]])
            while 1:
# Computer inverse of Second Moment Matrix
                M_inv = None
# Handle singluar matrix
                try: M_inv = la.inv(M)
                except: break
# Compute bound for t
                left_bound_x_t = i_buff - midpoint
                right_bound_x_t = i_buff + midpoint + 1
                left_bound_y_t = j_buff - midpoint
                right_bound_y_t = j_buff + midpoint + 1
# Construct window I_t
                window_It = np.zeros((kernel_size,kernel_size),dtype='float')
# Give value for window I_t
                count_x = 0
                for x in range(left_bound_x_t, right_bound_x_t):
                    count_y = 0
                    for y in range(left_bound_y_t, right_bound_y_t):
                        if x<0 or x>=m or y<0 or y>=n: window_It[count_x][count_y] = 0.0
                        else: window_It[count_x][count_y] = src_2[x][y] - window_I[count_x][count_y]
                window_It = cv.filter2D(window_It,-1,gauss1d(kernel_size/6, kernel_size))
                I_xt = 0.0
                I_yt = 0.0
                for a in range(kernel_size):
                    for b in range(kernel_size):
                        I_xt += window_Ix[a][b] * window_It[a][b]
                        I_yt += window_Iy[a][b] * window_It[a][b]
                N = np.array([-I_xt, -I_yt])
                res = np.dot(M_inv, N).reshape((2,1))
                buff_u, buff_v = res[0], res[1]
                counter += 1
                if buff_u + buff_v < 0.1 or counter == 5: break
                i_buff += int(buff_u)
                j_buff += int(buff_v)
                U[i][j] += buff_u
                V[i][j] += buff_v
    return U, V
'''

def optical_flow(kernel_size, src_1, src_2):
    

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
    main("./Q3_optical_flow/Evergreen/frame08.png", "./Q3_optical_flow/Evergreen/frame09.png", 15)
