import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
from scipy.ndimage import *

def getGradient(src, threshold):
    m, n = src.shape
    directions = np.zeros((m, n), dtype=float)
    gradients = np.zeros((m, n), dtype=float)
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    grad_x = convolve(src, sobel_x)
    grad_y = convolve(src, sobel_y)
    for i in range(m):
        for j in range(n):
            gradients[i][j] = math.sqrt(grad_x[i][j] * grad_x[i][j] + grad_y[i][j] * grad_y[i][j])
            if gradients[i][j] < threshold: gradients[i][j] = 0
            if not grad_x[i][j]: directions[i][j] = np.pi/2
            else: directions[i][j] = math.atan(grad_y[i][j] / grad_x[i][j])
            directions[i][j] *= (180 / math.pi)
            if directions[i][j] < 0: directions[i][j] += 180
    return gradients, directions
    
def HOG(src, tao, threshold, block_size):
    x, y = src.shape
    gradients, directions = getGradient(src, threshold)
    m = x // tao
    n = y // tao
    histogram = np.zeros((m, n, 6), dtype=float)
    for i in range(m * tao):
        for j in range(n * tao):
            if 15 <= directions[i][j] and directions[i][j] < 45: histogram[i // tao][j // tao][1] += gradients[i][j]
            elif 45 <= directions[i][j] and directions[i][j] < 75: histogram[i // tao][j // tao][2] += gradients[i][j]
            elif 75 <= directions[i][j] and directions[i][j] < 105: histogram[i // tao][j // tao][3] += gradients[i][j]
            elif 105 <= directions[i][j] and directions[i][j] < 135: histogram[i // tao][j // tao][4] += gradients[i][j]
            elif 135 <= directions[i][j] and directions[i][j] < 165: histogram[i // tao][j // tao][5] += gradients[i][j]
            else: histogram[i // tao][j // tao][0] += gradients[i][j]
            
    length = (m-1) * (n-1) * 24
    descriptor = np.zeros((m-1,n-1,24), dtype=float)
    e = 0.001
    for i in range(m-1):
        for j in range(n-1):
            sigma = 0
            for k in range(0,6):
                sigma += histogram[i][j][k] ** 2
                descriptor[i][j][k] = histogram[i][j][k]
            for k in range(0,6):
                sigma += histogram[i+1][j][k] ** 2
                descriptor[i][j][6+k] = histogram[i+1][j][k]
            for k in range(0,6):
                sigma += histogram[i][j+1][k] ** 2
                descriptor[i][j][12+k] = histogram[i][j+1][k]
            for k in range(0,6):
                sigma += histogram[i+1][j+1][k] ** 2
                descriptor[i][j][18+k] = histogram[i+1][j+1][k]
                
            for k in range(24):
                descriptor[i][j][k] = descriptor[i][j][k] / math.sqrt(sigma + e ** 2)
    
    src = src.astype('float') / 255.0
    normalizition = descriptor.reshape((m - 1, n - 1, 4, 6))
    vis = np.sum(normalizition ** 2, axis = 2) * 7
    angles_segs = np.arange(0, 180, 180 / 6)
    meshX, meshY = np.meshgrid(np.r_[int(tao * 2 / 2) : tao * (y // tao) - (tao * 2 / 2) + 1: tao], np.r_[int(tao * 2 /2) : tao * (x // tao) - (tao * 2 / 2) + 1: tao])
    meshU = vis * np.sin(angles_segs).reshape((1, 1, 6))
    meshV = -1 * vis * np.cos(angles_segs).reshape((1, 1, 6))
    plt.imshow(src, cmap='gray', vmin=0, vmax=1)
    for i in range(6):
        plt.quiver(meshX - 0.5 * meshU[:, :, i], meshY - 0.5 * meshV[:, :, i], meshU[:, :, i], meshV[:, :, i], color="red", headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()
    
def getGreyImge(img):
# Change the rgb value to grey. #
    rgb_weights = [0.2989, 0.5870, 0.1140]
    return np.dot(img[...,:3], rgb_weights)
    
if __name__ == '__main__':
    img = plt.imread("./Q4/2.jpg")
    src = None
    if len(img.shape) == 2: src = img
    else: src = getGreyImge(img)
    HOG(src, 8, 0.01, 2)
