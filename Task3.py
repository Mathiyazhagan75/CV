import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2



def plot_filters(gx,gy):
  img = plt.imread('cameraman3.tif',0)
  GX = cv2.filter2D(img, -1, gx)
  GY = cv2.filter2D(img, -1, gy)
  G = np.sqrt(np.square(GX) + np.square(GY))
  A = np.arctan2(GY, GX)
  plt.figure(1)
  plt.imshow(img,cmap='gray')
  plt.title('Input Image')
  plt.figure(2)
  plt.subplot(2,2,1),plt.imshow(GX, cmap='gray')
  plt.subplot(2,2,2),plt.imshow(GY, cmap='gray')
  plt.subplot(2,2,3),plt.imshow(G, cmap='gray')
  plt.subplot(2,2,4),plt.imshow(A, cmap='gray')

    # # Robert operator
    # gx = np.array([[1, 0 ],[0,-1 ]])
    # gy = gx.T
    # plot_filters(gx,gy)
    
    # # Prewitt operator
    # gx = np.array([[ -1 ,0 ,1],[ -1, 0 , 1],[ -1, 0, 1]])
    # gy = gx.T
    # plot_filters(gx,gy)
    
    # #Sobel Operator
    # gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    # gy = gx.T
    # plot_filters(gx,gy)
    
def KirschCompass(image):
    lena = plt.imread('Lena.jpeg',0)
    plt.imshow(lena,cmap='gray')
    m1 = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])
    m2 = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])
    m3 = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
    m6 = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
    m9 = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]])
    m8 = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])
    m7 = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])
    m4 = np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
    kirsch = {1:m1, 2:m2, 3:m3, 4:m4, 6:m6, 7:m7, 8:m8, 9:m9}
    for i in range(1, 10):
        plt.subplot(3, 3, i)
        plt.axis('off')
        if i==5:
            plt.plot()
            continue
        plt.imshow(cv2.filter2D(lena, -1, kirsch[i]), cmap='gray')
    
def CannyEdge(image):
    def gaussian_smoothing(image, kernel_size=(5, 5), sigma=1.4):
        return cv2.GaussianBlur(image, kernel_size, sigma)
    
    def prewitt(image):
        gx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        gy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        Gx = cv2.filter2D(image, -1, gx)
        Gy = cv2.filter2D(image, -1, gy)
        magnitude = np.sqrt(Gx**2 + Gy**2)
        direction = np.arctan2(Gy, Gx)
        return magnitude, direction
    
    def non_maximum_suppression(magnitude, direction):
        nms_image = np.zeros_like(magnitude)
        angle = np.degrees(direction) % 180
    
        for i in range(1, magnitude.shape[0] - 1):
            for j in range(1, magnitude.shape[1] - 1):
                q, r = 255, 255
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]
    
                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    nms_image[i, j] = magnitude[i, j]
    
        return nms_image
    
    def hysteresis_thresholding(nms_image, low_ratio=0.1, high_ratio=0.2):
        high_threshold = np.max(nms_image) * high_ratio
        low_threshold = high_threshold * low_ratio
    
        strong_edges = (nms_image >= high_threshold)
        weak_edges = ((nms_image >= low_threshold) & (nms_image < high_threshold))
    
        output_image = np.zeros_like(nms_image, dtype=np.uint8)
        output_image[strong_edges] = 255
    
        def track_edges(i, j):
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    if 0 <= i + dx < nms_image.shape[0] and 0 <= j + dy < nms_image.shape[1]:
                        if weak_edges[i + dx, j + dy]:
                            output_image[i + dx, j + dy] = 255
                            weak_edges[i + dx, j + dy] = False
                            track_edges(i + dx, j + dy)
    
        for i in range(1, nms_image.shape[0] - 1):
            for j in range(1, nms_image.shape[1] - 1):
                if output_image[i, j] == 255:
                    track_edges(i, j)
    
        return output_image


    image = cv2.imread('cameraman3.tif', 0)
    smoothed_image = gaussian_smoothing(image)
    gradient_magnitude, gradient_direction = prewitt(smoothed_image)
    nms_image = non_maximum_suppression(gradient_magnitude, gradient_direction)
    edges = hysteresis_thresholding(nms_image)
    plt.figure(figsize=(14, 10))
    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title('Gradient Magnitude')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(nms_image, cmap='gray')
    plt.title('Non-Maximum Suppression')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')
    
    plt.show()

def LoG(image, sigma, size):
    x, y = np.meshgrid(np.arange(-size//2 + 1, size//2 + 1), np.arange(-size//2 + 1, size//2 + 1))
    kernel = -(1 / (np.pi * sigma**4)) * (1 - ((x**2 + y**2) / (2 * sigma**2))) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(np.abs(kernel))  # Normalize the kernel
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image

    image = cv2.imread('cameraman3.tif', 0)
    LoG3 = LoG(image, sigma=0.5, size=3)
    LoG5 = LoG(image, sigma=0.8, size=5)
    LoG7 = LoG(image, sigma=0.8, size=7)
    LoG9 = LoG(image, sigma=0.8, size=9)
    plt.figure(figsize=(16, 10))
    plt.subplot(1, 5, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image'), plt.axis('off')
    plt.subplot(1, 5, 2), plt.imshow(LoG3, cmap='gray'), plt.title('LoG 3x3'), plt.axis('off')
    plt.subplot(1, 5, 3), plt.imshow(LoG5, cmap='gray'), plt.title('LoG 5x5'), plt.axis('off')
    plt.subplot(1, 5, 4), plt.imshow(LoG7, cmap='gray'), plt.title('LoG 7x7'), plt.axis('off')
    plt.subplot(1, 5, 5), plt.imshow(LoG9, cmap='gray'), plt.title('LoG 9x9'), plt.axis('off')
    plt.show()

def LPFbyBlock(image):
    img = cv2.imread('cameraman3.tif', cv2.IMREAD_GRAYSCALE)
    shape = img.shape
    blockSize = 8
    lpf = (1/9) * np.ones((3, 3))
    newImg = np.zeros(shape)
    for i in range(0, shape[0], blockSize):
        for j in range(0, shape[1], blockSize):
            block = img[i:i+blockSize, j:j+blockSize]
            filtered_block = cv2.filter2D(block, -1, lpf)
            newImg[i:i+blockSize, j:j+blockSize] = filtered_block
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Input')
    
    plt.subplot(1, 2, 2)
    plt.imshow(newImg, cmap='gray')
    plt.axis('off')
    plt.title('3x3 LPF by Sub-block')
    
    plt.show()
