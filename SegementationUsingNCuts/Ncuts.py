from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
from scipy.linalg import eigh

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# Creating the graph
#Calculate the distance affinity
# noOfNodes = img.shape[0] * img.shape[1]
# distAff = np.empty((noOfNodes,noOfNodes), dtype=float)
#
# for i in range(0, imgRows):
#     for j in range(0, imgCols):
#         for x in range(0, imgRows):
#             for y in range(0, imgCols):
#                 col = int((x*imgRows) + y)
#                 row = int((i*imgRows) + j)
#                 print(i,j,x,y)
#                 distAff[row,col] = ((x-i)*(x-i)) + ((y-j)*(y-j))
# sigmad = np.std(distAff)
#
# print(distAff)
# print(sigmad)
# np.set_printoptions(suppress=True)
# distAff = np.exp(-(distAff)/(2*sigmad*sigmad))
# print(distAff)


def GetDistanceAffinityMatrix(imgRows, imgCols):
    noOfNodes = imgRows * imgCols
    distAff = np.empty((noOfNodes, noOfNodes), dtype=float)
    for i in range(0, imgRows):
        for j in range(0, imgCols):
            for x in range(0, imgRows):
                for y in range(0, imgCols):
                    col = int((x * imgRows) + y)
                    row = int((i * imgRows) + j)
                    distAff[row, col] = ((x - i) * (x - i)) + ((y - j) * (y - j))
    sigmad = np.std(distAff)
    np.set_printoptions(suppress=True)
    distAff = np.exp(-(distAff) / (2 * sigmad * sigmad))
    return distAff

def GetIntensityAffinityMatrix(img, imgRows, imgCols):
    noOfNodes = imgRows * imgCols
    intAff = np.empty((noOfNodes, noOfNodes), dtype=float)
    for i in range(0, imgRows):
        for j in range(0, imgCols):
            for x in range(0, imgRows):
                for y in range(0, imgCols):
                    col = int((x * imgRows) + y)
                    row = int((i * imgRows) + j)
                    intAff[row, col] = (img[i,j]-img[x,y]) * (img[i,j]-img[x,y])
    sigmai = np.std(intAff)
    np.set_printoptions(suppress=True)
    intAff = np.exp(-(intAff) / (2 * sigmai * sigmai))
    return intAff

def GetDegreeMatrix(affinity, imgRows, imgCols):
    noOfNodes = imgRows * imgCols
    deg = np.zeros((noOfNodes, noOfNodes), dtype=float)
    sum = np.sum(affinity, axis=1)
    for i in range(0, noOfNodes):
        deg[i,i] = sum[i]
    return deg





img = cv2.imread('peppers_color.tif')
img = rgb2gray(img)
img = cv2.resize(img, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)


imgRows = img.shape[0]
imgCols = img.shape[1]
distAff = GetDistanceAffinityMatrix(imgRows, imgCols)
print("dist done")
intAff = GetIntensityAffinityMatrix(img, imgRows, imgCols)
print("int done")
# Calculating the affinity matrix - which is the sum of the above two
# We are doing this coz - when segmenting if two corners have a color and
# the center has a different color, the corners will be in a different
# segment based on their distance
affinity = distAff + intAff
plt.imshow(affinity, cmap='gray')
plt.show()
# Building the degree matrix
degree = GetDegreeMatrix(affinity, imgRows, imgCols)
#D-A
DminusA = degree - affinity

#compute eigen vectors
eigvals, eigvecs = eigh(DminusA, degree, eigvals_only=False) # get sorted from min to max eigenVals &eighenVector
eig2 = eigvecs[:,1] #get 2nd smallest eigenvector
seg1 = img.copy()
seg2 = img.copy()
seg1[eig2.reshape(img.shape) <= 0] = 0
seg2[eig2.reshape(img.shape) > 0] = 0

plt.imshow(seg1, cmap='gray')
plt.show()
plt.imshow(seg2, cmap='gray')
plt.show()











