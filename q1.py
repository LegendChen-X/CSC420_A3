import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

def find_maxmag(sigma):
    img = np.zeros((20, 20, 1), dtype = np.float64)
    cv.rectangle(img, (5, 5), (10, 10), (1.0), -1)
    LoG = (sigma ** 2) * ndimage.gaussian_laplace(img, sigma = sigma)
    return np.amin(LoG)

if __name__ == '__main__':
    x = np.linspace(1, 5, 1500)
    y = []
    for i in x:
        t = find_maxmag(i)
        y.append(t)
    plt.figure()
    plt.plot(x,y,color="red")
    plt.xlabel("sigma")
    plt.ylabel("maximum")
    plt.show()
