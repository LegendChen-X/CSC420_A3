import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

def find_max(sigma):
    img = np.ones((20, 20, 1), dtype = np.float64)
    cv.rectangle(img, (6, 6), (14, 14), (0.0), -1)
    LoG = (sigma ** 2) * ndimage.gaussian_laplace(img, sigma = sigma)
    return np.amax(LoG)

if __name__ == '__main__':
    x = np.linspace(1, 10, 1500)
    y = []
    for i in x:
        t = find_max(i)
        y.append(t)
    plt.figure()
    plt.plot(x,y,color="red")
    plt.xlabel("sigma")
    plt.ylabel("value")
    plt.show()
