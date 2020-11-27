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
    
def optical_flow(kernel_size, src_1, src_2):
# Get height and width
    height, width = src_1.shape
# Get blur matrix
    blur_matrix = Gaussian_Blur(0.83, 5)
# Decrease the noise
    src_1 = cv.filter2D(src_1, -1, blur_matrix)
    src_2 = cv.filter2D(src_2, -1, blur_matrix)
# Get sobel operator.
    sobel_x = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    sobel_y = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
# Get gradient X
    I_x = cv.filter2D(src_1, -1, sobel_x)
# Get gradient Y
    I_y = cv.filter2D(src_1, -1, sobel_y)
# Save vector U
    U = np.zeros((height, width), dtype="float")
# Save vector V
    V = np.zeros((height, width), dtype="float")
# Calculate mid of window
    mid =  kernel_size // 2
    for i in range(height):
        for j in range(width):
# Set the original i_buf = i
            i_buf = i
# Set the original j_buf = j
            j_buf = j
# Set maximum loop number
            max_loop = 5
# Calculate left bound of x for the window
            left_height_bound = i - mid
# Calculate right bound of x for the window
            right_height_bound = i + mid + 1
# Calculate left bound of y for the window
            left_width_bound = j - mid
# Calculate right bound of y for the window
            right_width_bound = j + mid + 1
# Cut boundary cases
            if left_height_bound < 0 or left_width_bound < 0 or right_height_bound >= height or right_width_bound >= width: continue
# Set window for gradient x
            window_Ix = I_x[left_height_bound:right_height_bound, left_width_bound:right_width_bound]
# Set window for gradient y
            window_Iy = I_y[left_height_bound:right_height_bound, left_width_bound:right_width_bound]
# Set window for I(t)
            window_I = src_1[left_height_bound:right_height_bound, left_width_bound:right_width_bound]
# Construct Matrix A
            A = np.vstack((window_Ix.flatten(),window_Iy.flatten()))
# Get the transcope of A
            A_T = A.T
# Calculate the inverse of A
            inv_A = None
# Handele exception
            try: inv_A = np.linalg.pinv(A_T)
            except: continue
# Loop until small displacement
            while 1:
# Boundary for I(t+1)
                left_height_bound_t = i_buf - mid
                right_height_bound_t = i_buf + mid + 1
                left_width_bound_t = j_buf - mid
                right_width_bound_t = j_buf + mid + 1
# Cut boundary cases
                if left_height_bound_t < 0 or left_width_bound_t < 0 or right_height_bound_t >= height or right_width_bound_t >= width: break
# Concadinate window_I(t) and window_I(t+1)
                con_src = np.vstack((window_I.flatten(), src_2[left_height_bound_t:right_height_bound_t, left_width_bound_t:right_width_bound_t].flatten()))
# t-axis blur
                con_src = gaussian_filter1d(con_src, sigma = 0.1, axis = 0)
# Construct Matrix b
                window_It = con_src[1] - con_src[0]
# Get the res
                res = np.dot(inv_A, -window_It.T)
# Get u, v displacement
                u, v = res[0], res[1]
# Break when displacement is small or exceed max loop
                if ((abs(u) + abs(v)) < 0.5) or not max_loop: break
# Update j_buf and i_buf
                j_buf += int(u)
                i_buf += int(v)
# Add displacement to U and V
                U[i][j] += u
                V[i][j] += v
# Update upper bound of max_loop
                max_loop -= 1
    return U, V

def getGreyImge(img):
# Change the rgb value to grey. #
    rgb_weights = np.array([0.2989, 0.5870, 0.1140])
    return np.dot(img[...,:3], rgb_weights)
    
def main(path_1, path_2, kernel_size):
    img_1 = plt.imread(path_1)
    img_2 = plt.imread(path_2)
    try: U, V = optical_flow(kernel_size, getGreyImge(img_1), getGreyImge(img_2))
    except: U, V = optical_flow(kernel_size, img_1, img_2)
    t = 10
    U_first = U[::t, ::t]
    V_first = V[::t, ::t]
    try: r, c = getGreyImge(img_1).shape
    except: r, c = img_1.shape
    cols, rows = np.meshgrid(np.linspace(0, c - 1, c), np.linspace(0, r - 1, r))
    cols = cols[::t,::t]
    rows = rows[::t, ::t]
    plt.figure(figsize=(9,9))
    plt.imshow(img_1)
    plt.quiver(cols, rows, U_first, -V_first, color='red')
    plt.show()
        
if __name__ == '__main__':
    main("./Q3_optical_flow/Basketball/frame07.png", "./Q3_optical_flow/Basketball/frame08.png", 25)
