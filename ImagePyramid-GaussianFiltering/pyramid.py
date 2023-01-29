import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

#done using linear interpolation
def Upsample(image):
    spike = np.zeros((image.shape[0] * 2, image.shape[1] * 2))
    spike[:] = 200
    for i in range(0, spike.shape[0], 2):
        for j in range(0, spike.shape[1], 2):
            a = int(i / 2)
            b = int(j / 2)
            spike[i][j] = image[a][b]
    linIntKernel = [[1 / 4, 1 / 2, 1 / 4], [1 / 2, 1, 1 / 2], [1 / 4, 1 / 2, 1 / 4]]
    upsampled = convolve(spike, linIntKernel)
    return upsampled

def Downsample(image):
    blurred = gaussian_filter(image, sigma=3)
    rowRemoved = np.delete(blurred, list(range(1, image.shape[0], 2)), axis=0)
    colRemoved = np.delete(rowRemoved, list(range(1, image.shape[1], 2)), axis=1)

    return colRemoved

img = cv2.imread("lena_gray_256_noisy.png")
img = rgb2gray(img)
plt.imshow(img, cmap='gray')
plt.show()


downsampled = Downsample(img)
plt.imshow(downsampled, cmap='gray')
plt.show()
downsampled = Downsample(downsampled)
plt.imshow(downsampled, cmap='gray')
plt.show()


upsampled = Upsample(downsampled)
plt.imshow(upsampled, cmap='gray')
plt.show()
upsampled = Upsample(upsampled)
plt.imshow(upsampled, cmap='gray')
plt.show()


residual = cv2.subtract(img, upsampled)
plt.imshow(residual, cmap='gray')
plt.show()


freqSpectrum = np.fft.fft2(residual)
# freqSpectrum2 = np.fft.fftshift(freqSpectrum)
max = np.log(1+np.abs(np.amax(freqSpectrum)))
plt.imshow(np.log(1+np.abs(freqSpectrum)), cmap='gray')
plt.show()
# Tried to remove high freq comps from fft, didnt work
# so used gaussian filter instead

blurred = gaussian_filter(residual, sigma=3)
original = cv2.add(blurred, upsampled)
plt.imshow(original, cmap='gray')
plt.show()


# def PyramidOpencv():
#     img = cv2.imread("lena_gray_256_noisy.png")
#     img = rgb2gray(img)
#     plt.imshow(img, cmap='gray')
#     plt.show()
#
#     imgk1 = cv2.pyrDown(img)
#     imgk2 = cv2.pyrDown(imgk1)
#
#
#     plt.imshow(imgk1, cmap='gray')
#     plt.show()
#
#
#     imgUp = cv2.pyrUp(imgk1, dstsize=(256,256))
#     residual = cv2.subtract(img, imgUp)
#
#     plt.imshow(residual, cmap='gray')
#     plt.show()
#
#     blurred = gaussian_filter(residual, sigma=5)
#
#     original = cv2.add(blurred, imgUp)
#
#     plt.imshow(original, cmap='gray')
#     plt.show()
#
#     plt.subplot(151),plt.imshow(img, cmap = 'gray')
#     plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#     plt.subplot(152),plt.imshow(imgk1, cmap = 'gray')
#     plt.title('DFT Mag'), plt.xticks([]), plt.yticks([])
#     plt.subplot(153),plt.imshow(imgUp, cmap = 'gray')
#     plt.title('DFT Phase'), plt.xticks([]), plt.yticks([])
#     plt.subplot(154),plt.imshow(residual, cmap = 'gray')
#     plt.title('DFT Mag'), plt.xticks([]), plt.yticks([])
#     plt.subplot(155),plt.imshow(original, cmap = 'gray')
#     plt.title('DFT Phase'), plt.xticks([]), plt.yticks([])
#     plt.show()




