import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

# cv2.imshow('image',img)
# cv2.waitKey(0)

A = np.empty((0,4), int)

r1 = np.array([[1,2,3,4]])
r2 = np.array([[1,1,1,1]])

A = np.append(A, r1, axis=0)
A = np.append(A, r2, axis=0)
print(A)

def GetCorners(img):
    # found, corners = cv2.findChessboardCornersSB(img, (8,6))
    found, corners = cv2.findChessboardCornersSB(img, (13,12))

    print(corners)
    print("###############################################################################")
    imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgpoints = [] # 2d points in image plane.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    ## Checking
    if found == True:
        corners2 = cv2.cornerSubPix(imgG, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, (1, 1), corners2[0:1, 0:1, 0:2], found)
        cv2.drawChessboardCorners(img, (1, 1), corners2[155:156, 0:1, 0:2], found)

        cv2.imshow('img', img)
        cv2.waitKey(0)

    return corners

def GetMatrixA(corners1, corners2):
    # take n points into consideration - that gives 2n equations => 2n rows in matrix A

    n = 30
    A = np.empty((0, 9), float)
    for i in range(0, n): # adding two equations each iteration
        point1 = corners1[i*5:(i*5)+1, 0:1, 0:2]
        point2 = corners2[i*5:(i*5)+1, 0:1, 0:2]

        point1 = point1.reshape((2))
        point2 = point2.reshape((2))
        print(point1)
        print(point2)

        row1 = np.array([[point1[0], point1[1], 1, 0, 0, 0, -(point2[0]*point1[0]), -point2[0]*point1[1], -point2[0]]])
        row2 = np.array([[0, 0, 0, point1[0], point1[1], 1, -point2[1]*point1[0], -point2[1]*point1[0], -point2[1]]])

        A = np.append(A, row1, axis=0)
        A = np.append(A, row2, axis=0)
    return A

img1 = cv2.imread('images/Image1.tif')
img9 = cv2.imread('images/Image9.tif')

corners1 = GetCorners(img1)
corners9 = GetCorners(img9)

A = GetMatrixA(corners1, corners9)
np.set_printoptions(suppress=True)


vals, vecs = LA.eig(A.T @ A)
# eigen val of the smallest eigen vector
smallestEigenvector = vecs[:, np.argmin(vals)]
print("answer")
print(smallestEigenvector)

## Checking
## Corresponding points
## 213.18077 181.05115
## 321.4193    49.56148


p1 = np.array([[167], [178], [1]])
H = np.array(smallestEigenvector)
H = H.reshape((3,3))
print(H)
print(p1.shape)
print(H.shape)
p2 =  H @ p1

p2 = p2.reshape((3))
print(p2)
p2 = p2/p2[2]
print(p2)

# Actual answer 461, 435
# but got 460.97790097 431.13289117   1.


# n= 8
# [-0.00118319  0.00100296 -0.99623029 -0.00043829 -0.00252958  0.08662857
#   0.00000185  0.00000008 -0.00342003]
# [463.19614608 435.33896054   1.        ]
# Adding more points will improve the accuracy

#n=30
# [ 0.00030427  0.00002809  0.83221893  0.00070068  0.00341851 -0.55442877
#  -0.00000436  0.0000034   0.00284953]
# gives the best answer









