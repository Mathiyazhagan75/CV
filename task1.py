import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sounddevice

def recorAndDisplayVoice(duration, fs):
    # duration = 5 
    # fs = 44100  
    
    print("Recording...")
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    print("Recording finished")
    
    write('hello.wav', fs, myrecording)
    print("File saved as hello.wav")
    
    fs, data = read('hello.wav')
    print(f"Sample rate: {fs}")
    print(f"Data shape: {data.shape}")
    
    # sd.play(data,fs)
    plt.figure(figsize=(10, 4))
    plt.plot(data)
    plt.title('Waveform of the recorded audio')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.show()


def lloyd_max_quantizer(signal, num_levels, max_iter=100, tolerance=1e-6):
    min_val, max_val = np.min(signal), np.max(signal)
    levels = np.linspace(min_val, max_val, num_levels)

    for iteration in range(max_iter):
        boundaries = (levels[:-1] + levels[1:]) / 2

        quantized_signal = np.zeros_like(signal)
        for i, sample in enumerate(signal):
            quantized_signal[i] = levels[np.argmin(np.abs(sample - levels))]
            
        new_levels = np.zeros(num_levels)
        for i in range(num_levels):
            if i == 0:
                region = signal[signal <= boundaries[0]]
            elif i == num_levels - 1:
                region = signal[signal > boundaries[-1]]
            else:
                region = signal[(signal > boundaries[i-1]) & (signal <= boundaries[i])]
            new_levels[i] = np.mean(region) if len(region) > 0 else levels[i]

        if np.max(np.abs(new_levels - levels)) < tolerance:
            break

        levels = new_levels

    # np.random.seed(42)
    # signal = np.random.randn(1000)  
    # num_levels = 4  
    
    # quantized_signal, levels, boundaries = lloyd_max_quantizer(signal, num_levels)
    
    # plt.figure(figsize=(12, 6))
    
    # plt.subplot(2, 1, 1)
    # plt.plot(signal, label='Original Signal', alpha=0.7)
    # plt.scatter(range(len(signal)), quantized_signal, color='red', label='Quantized Signal', s=10)
    # plt.title('Original vs. Quantized Signal')
    # plt.legend()
    
    # plt.subplot(2, 1, 2)
    # plt.hist(signal, bins=50, alpha=0.7, label='Original Signal')
    # plt.hist(quantized_signal, bins=50, alpha=0.7, label='Quantized Signal', color='red')
    # plt.title('Histogram of Original and Quantized Signals')
    # plt.legend()
    
    # plt.tight_layout()
    # plt.show()
    return quantized_signal, levels, boundaries

def scrambleImage(image):
    # image = cv2.imread('cameraman.jpg')
    
    flattened_image = image.flatten()
    
    np.random.shuffle(flattened_image)
    
    scrambled_image = flattened_image.reshape(image.shape)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Scrambled Image')
    plt.imshow(scrambled_image, cmap='gray')
    plt.axis('off')
    
    plt.show()