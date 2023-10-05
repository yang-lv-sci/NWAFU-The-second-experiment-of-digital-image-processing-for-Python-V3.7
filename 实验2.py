# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 19:34:28 2023

@author: lvyan
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import sys
import math
from numba import jit

@jit(nopython=True)
def normalized_mse(imageA, imageB):
    mse = np.sum((imageA - imageB) ** 2)
    return mse / float(imageA.shape[0] * imageA.shape[1])

@jit(nopython=True)
def psnr(imageA, imageB):
    mse_value = normalized_mse(imageA, imageB)
    if mse_value == 0:
        return 100
    return 20 * math.log10(255.0 / math.sqrt(mse_value))

def entropy(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    logs = np.log2(hist + 0.00001)
    return -1 * (hist * logs).sum()


def average_gradient(image):
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    d = np.sqrt(dx**2 + dy**2)
    return np.mean(d)


def standard_deviation(image):
    return np.std(image)


def sharpness(image):
    return np.max(cv2.convertScaleAbs(cv2.Laplacian(image, cv2.CV_64F)))

def add_gaussian_noise(image, sigma, weight):
    row, col, ch = image.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.clip(image + gauss * weight, 0, 255)
    return noisy.astype(np.uint8)


def homomorphic_filter(img, gamma_l, gamma_h, c, d0):
    img_log = np.log1p(np.array(img, dtype="float"))
    rows, cols = img_log.shape
    rh, rl, cutoff = gamma_h, gamma_l, d0

    X = np.linspace(-cols//2, cols//2, cols)
    Y = np.linspace(-rows//2, rows//2, rows)
    X, Y = np.meshgrid(X, Y)
    H = np.exp(-c * (X**2 + Y**2) / (2 * cutoff**2))
    H = rl + (rh - rl) * (1 - H)

    img_fft = np.fft.fftshift(np.fft.fft2(img_log))
    img_fft_filt = img_fft * H
    img_filt = np.fft.ifft2(np.fft.ifftshift(img_fft_filt))
    img_exp = np.expm1(np.real(img_filt))
    img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp)) * 255
    img_out = img_exp.astype("uint8")

    return img_out

def reset_parameters():
    gamma_l_slider.set(0)  # 设置为初始值
    gamma_h_slider.set(0)  # 设置为初始值
    c_slider.set(1)  # 设置为初始值
    d0_slider.set(1)  # 设置为初始值
    noise_weight_slider.set(0)  # 设置为初始值
    noise_sigma_slider.set(0)  # 设置为初始值
    update_image()  # 调用 update_image 函数来重置图像和标签


def update_image():
    noise_weight = noise_weight_slider.get() / 100
    noise_sigma = noise_sigma_slider.get()
    
    noisy_img = add_gaussian_noise(img_rgb, noise_sigma, noise_weight)
    
    gamma_l = gamma_l_slider.get() / 10
    gamma_h = gamma_h_slider.get() / 10
    c = c_slider.get() / 10
    d0 = d0_slider.get()

    filtered_channels = [homomorphic_filter(noisy_img[:,:,i], gamma_l, gamma_h, c, d0) for i in range(3)]
    filtered = np.stack(filtered_channels, axis=2)

    noisy_float = noisy_img.astype('float')
    filtered_float = filtered.astype('float')

    nmse_value = normalized_mse(noisy_float, filtered_float)
    psnr_value = psnr(noisy_float, filtered_float)
    entropy_value = entropy(filtered)
    avg_grad_value = average_gradient(filtered)
    std_dev_value = standard_deviation(filtered)
    sharpness_value = sharpness(filtered)

    nmse_label.config(text=f"NMSE: {nmse_value:.2f}")
    psnr_label.config(text=f"PSNR: {psnr_value:.2f}")
    entropy_label.config(text=f"Entropy: {entropy_value:.2f}")
    avg_grad_label.config(text=f"Avg Gradient: {avg_grad_value:.2f}")
    std_dev_label.config(text=f"Std Dev: {std_dev_value:.2f}")
    sharpness_label.config(text=f"Sharpness: {sharpness_value:.2f}")

    im = Image.fromarray(filtered)
    imgtk = ImageTk.PhotoImage(image=im)
    label_filtered.imgtk = imgtk
    label_filtered.configure(image=imgtk)


root = tk.Tk()
root.title("Homomorphic Filter GUI")

file_path = filedialog.askopenfilename()
img = cv2.imread(file_path)
if img is None:
    print(f"Error: Cannot load image at {file_path}")
    sys.exit(1)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gamma_l_slider = tk.Scale(root, from_=1, to=20, label='Gamma L', orient=tk.HORIZONTAL, command=lambda x: update_image(), resolution=0.1)
gamma_l_slider.pack(fill="both")

gamma_h_slider = tk.Scale(root, from_=1, to=20, label='Gamma H', orient=tk.HORIZONTAL, command=lambda x: update_image(), resolution=0.1)
gamma_h_slider.pack(fill="both")

c_slider = tk.Scale(root, from_=1, to=10, label='C', orient=tk.HORIZONTAL, command=lambda x: update_image(), resolution=0.1)
c_slider.pack(fill="both")

d0_slider = tk.Scale(root, from_=1, to=100, label='D0', orient=tk.HORIZONTAL, command=lambda x: update_image(), resolution=0.1)
d0_slider.pack(fill="both")

noise_weight_slider = tk.Scale(root, from_=1, to=100, label='Noise Weight', orient=tk.HORIZONTAL, resolution=1,command=lambda x: update_image())
noise_weight_slider.pack(fill="both")

noise_sigma_slider = tk.Scale(root, from_=1, to=100, label='Noise Sigma', orient=tk.HORIZONTAL, resolution=1,command=lambda x: update_image())
noise_sigma_slider.pack(fill="both")

reset_button = tk.Button(root, text="Reset", command=reset_parameters)
reset_button.pack()

nmse_label = tk.Label(root, text="NMSE:")
nmse_label.pack()
psnr_label = tk.Label(root, text="PSNR:")
psnr_label.pack()
entropy_label = tk.Label(root, text="Entropy:")
entropy_label.pack()
avg_grad_label = tk.Label(root, text="Average Gradient:")
avg_grad_label.pack()
std_dev_label = tk.Label(root, text="Standard Deviation:")
std_dev_label.pack()
sharpness_label = tk.Label(root, text="Sharpness:")
sharpness_label.pack()

frame = tk.Frame(root)
frame.pack()

im_original = Image.fromarray(img_rgb)
imgtk_original = ImageTk.PhotoImage(image=im_original)
label_original = tk.Label(frame, image=imgtk_original)
label_original.imgtk = imgtk_original
label_original.pack(side="left")

im_filtered = Image.fromarray(img_rgb)
imgtk_filtered = ImageTk.PhotoImage(image=im_filtered)
label_filtered = tk.Label(frame, image=imgtk_filtered)
label_filtered.imgtk = imgtk_filtered
label_filtered.pack(side="right")

root.mainloop()






