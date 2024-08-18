import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sounddevice
import cv2
import itertools

def getBlurredImage(img):
    img = cv2.imread('cameraman3.tif',0)
    print(img.shape)
    M=3
    low_pass_filter = 1/(M*M)*np.ones([M,M])
    img_blur = cv2.filter2D(img,-1,low_pass_filter)
    plt.imshow(img_blur, cmap='gray')
    img_reduced = img-img_blur
    plt.imshow(img_reduced,cmap='gray')

def getHPFNoise(img):
    img = plt.imread('cameraman3.tif',0)
    img = img/255
    plt.imshow(img,cmap='gray')
    gaussian_noise = np.random.normal(0,1,img.shape)
    noisy_img = np.clip(img + 0.3 * gaussian_noise,0,1)
    plt.imshow(noisy_img,cmap = 'gray')
    high_pass_filter = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    img_after_filter = cv2.filter2D(noisy_img,-1,high_pass_filter)
    plt.imshow(img_after_filter, cmap='gray')

def Quantization(image):
    # image = cv2.imread('cameraman3.tif', 0)
    
    vector_length = 2  
    bit_rate = 3  
    
    dynamic_range = np.max(image) - np.min(image)
    quantization_interval = np.ceil(dynamic_range / (2 ** bit_rate))
    codebook = {}
    midpoints = []
    
    for i in range(0, np.max(image), int(quantization_interval)):
        midpoints.append(i + quantization_interval // 2)
    
    code_vectors = list(itertools.product(midpoints, repeat=vector_length))
    
    for idx, vector in enumerate(code_vectors):
        codebook[vector] = idx
    
    def find_closest_code_vector(codebook, vector):
        vector = np.array(vector)
        min_distance = np.inf
        closest_vector = None
        
        for code_vector in codebook.keys():
            distance = np.linalg.norm(np.array(code_vector) - vector)
            if distance < min_distance:
                min_distance = distance
                closest_vector = code_vector
                
        return closest_vector
    
    image_flatten = np.reshape(image, (-1))
    encoded_image = np.zeros(image_flatten.shape[0] // vector_length).astype(int)
    
    for i in range(0, image_flatten.shape[0], vector_length):
        vector = tuple(image_flatten[i:i + vector_length])
        closest_vector = find_closest_code_vector(codebook, vector)
        encoded_image[i // vector_length] = codebook[closest_vector]
    
    decoded_image = np.ones(image.shape)
    decoded_image_flatten = np.reshape(decoded_image, (-1))
    
    for i in range(0, decoded_image_flatten.shape[0], vector_length):
        code_vector = list(codebook.keys())[encoded_image[i // vector_length]]
        decoded_image_flatten[i:i + vector_length] = np.array(code_vector)
    
    decoded_image = np.reshape(decoded_image_flatten, image.shape)
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].imshow(image / 255, cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(decoded_image / 255, cmap='gray')
    axs[1].set_title(f'Reconstructed Image (R={bit_rate}, L={vector_length})')
    plt.show()

def getNeighbors(input_image):
    input_image = cv2.imread('cameraman3.tif', 0)
    
    scale_factor = 2
    new_dimensions = (input_image.shape[1] * scale_factor, input_image.shape[0] * scale_factor)
    
    
    resized_images = {
        "Original Image": input_image,
        "Nearest Neighbor": cv2.resize(input_image, new_dimensions, interpolation=cv2.INTER_NEAREST),
        "Bilinear": cv2.resize(input_image, new_dimensions, interpolation=cv2.INTER_LINEAR),
        "Bicubic": cv2.resize(input_image, new_dimensions, interpolation=cv2.INTER_CUBIC)
    }
    
    fig, axs = plt.subplots(1, len(resized_images), figsize=(20, 5))
    for idx, (title, img) in enumerate(resized_images.items()):
        axs[idx].imshow(img, cmap='gray')
        axs[idx].set_title(title)
        axs[idx].axis('off')
    plt.show()

def get_resize(input_image):
    input_image = cv2.imread('cameraman3.tif', 0)
    input_image = cv2.resize(input_image, (256, 256))
    
    resized_images = {
        "Original Image (256x256)": input_image,
        "Resized to (128x128)": cv2.resize(input_image, (128, 128)),
        "Resized to (512x512)": cv2.resize(cv2.resize(input_image, (128, 128)), (512, 512))
    }
    
    fig, axs = plt.subplots(1, len(resized_images), figsize=(15, 5))
    
    for idx, (title, img) in enumerate(resized_images.items()):
        axs[idx].imshow(img, cmap='gray')
        axs[idx].set_title(title)
        axs[idx].axis('off')
    plt.show()

def get_filters(image):
    image = np.abs(np.subtract.outer(np.arange(8), np.arange(8))).astype(np.float32)
    # image = plt.imread('cameraman3.tif')
    average_filter = 1/9 * np.ones((3, 3)) 
    weighted_average_filter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    
    average_filtered_image = cv2.filter2D(image, -1, average_filter)
    weighted_filtered_image = cv2.filter2D(image, -1, weighted_average_filter)
    
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    images = {
        'Generated Image': image,
        '3x3 Average Filtered': average_filtered_image,
        '3x3 Weighted Average Filtered': weighted_filtered_image
    }
    
    for idx, (title, img) in enumerate(images.items()):
        axs[idx].imshow(img, cmap='gray')
        axs[idx].set_title(title)
        axs[idx].axis('off')
    plt.show()

def generateImage():
    N = 16
    a = np.zeros((N, N), dtype=int)
    for i in range((N + 1) // 2):
        for j in range((N + 1) // 2):
            value = min(i, j)
            a[i, j] = value
            a[N - i - 1, j] = value
            a[i, N - j - 1] = value
            a[N - i - 1, N - j - 1] = value
    print(a)
    plt.imshow(a, cmap="gray")
    plt.show()