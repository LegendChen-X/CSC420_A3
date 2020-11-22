import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv

def getGradient(src, threshold):
    m, n = src.shape
    src = src.astype('float') / 255.0
    directions = np.zeros((m, n), dtype=float)
    gradients = np.zeros((m, n), dtype=float)
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    grad_x = cv.filter2D(src,-1,sobel_x)
    grad_y = cv.filter2D(src,-1,sobel_y)
    gradients = grad_x**2 + grad_y**2
    for i in range(m):
        for j in range(n):
            gradients[i][j] = math.sqrt(gradients[i][j])
            if gradients[i][j] < threshold:
                gradients[i][j] = 0
            directions[i][j] = math.atan(grad_y[i][j] / grad_x[i][j])
            directions[i][j] *= (180 / math.pi)
            if directions[i][j]<0: directions[i][j] += 180
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
                sigma += histogram[i][j][k] ** 2
                descriptor[i][j][6+k] = histogram[i+1][j][k]
            for k in range(0,6):
                sigma += histogram[i][j][k] ** 2
                descriptor[i][j][12+k] = histogram[i][j+1][k]
            for k in range(0,6):
                sigma += histogram[i][j][k] ** 2
                descriptor[i][j][18+k] = histogram[i+1][j+1][k]
            for k in range(24):
                descriptor[i][j][k] = descriptor[i][j][k] / math.sqrt(sigma+e**2)
    
    src = src.astype('float') / 255.0
    normalizition = descriptor.reshape((m - block_size + 1, n - block_size + 1, block_size ** 2, 6))
    vis = np.sum(normalizition ** 2, axis = 2) * 7
    angles_segs = np.arange(0, 180, 180 / 6)
    meshX, meshY = np.meshgrid(np.r_[int(tao * block_size / 2) : tao * (y // tao) - (tao * block_size / 2) + 1: tao], np.r_[int(tao * block_size /2) : tao * (x // tao) - (tao * block_size / 2) + 1: tao])
    meshU = vis * np.sin(angles_segs).reshape((1, 1, 6))
    meshV = -1 * vis * np.cos(angles_segs).reshape((1, 1, 6))
    plt.imshow(src, cmap='gray', vmin=0, vmax=1)
    for i in range(6):
        plt.quiver(meshX - 0.5 * meshU[:, :, i], meshY - 0.5 * meshV[:, :, i], meshU[:, :, i], meshV[:, :, i], color="red", headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()
    
    
if __name__ == '__main__':
    src = plt.imread("./Q4/1.jpg")
    HOG(src, 8, 0.00001, 2)
