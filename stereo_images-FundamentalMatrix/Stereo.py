import random
from numpy import linalg as LA
import numpy as np
import cv2
import matplotlib.pyplot as plt


def GenerateFundamentalMatrix(randomList, img1Points, img2Points):
    # the 8 point algorithm

    F = np.empty((0, 9), float)
    for i in randomList:
        pt1 = img1Points[i]
        pt2 = img2Points[i]
        row = np.array(
            [[pt1[0] * pt2[0], pt1[0] * pt2[1], pt1[0], pt1[1] * pt2[0], pt1[1] * pt2[1], pt1[1], pt2[0], pt2[1], 1]])

        F = np.append(F, row, axis=0)

    return F

def ComputeSampsonsDistance(pt1, pt2, F):
    pt1 = np.array([[pt1[0], pt1[1], 1]])
    pt2 = np.array([[pt2[0], pt2[1], 1]])

    num = pt1 @ F @ pt2.T
    deno1 = F.T @ pt1.T
    deno2 = F @ pt2.T
    deno1 = LA.norm(deno1)
    deno2 = LA.norm(deno2)
    dist = (num*num)/((deno1*deno1) + (deno2*deno2))

    return dist


img1 = cv2.imread('Q3Image/viprectification_deskLeft.png',cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('Q3Image/viprectification_deskRight.png',cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keyPoints1, descriptors1 = sift.detectAndCompute(img1, None)
keyPoints2, descriptors2 = sift.detectAndCompute(img2, None)

print(len(keyPoints1))
print(len(keyPoints2))


# img_1 = cv2.drawKeypoints(img1,keyPoints1,img1)
# plt.imshow(img_1)
# plt.show()


# bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
# matches = bf.match(descriptors1, descriptors2)
# matches = sorted(matches, key = lambda x:x.distance)
# matchingImag = cv2.drawMatches(img1, keyPoints1, img2, keyPoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(matchingImag)
# plt.show()



FLANN_INDEX_KDTREE = 1
flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
matcher = cv2.FlannBasedMatcher(flann_params, {})

img1Points=[]
img2Points=[]
matches = matcher.knnMatch(descriptors1, descriptors2, 2)
matchesMask = [[0,0] for i in range(len(matches))]
for i, (m1,m2) in enumerate(matches):
    if m1.distance < 0.7 * m2.distance:
        matchesMask[i] = [1,0]
        pt1 = keyPoints1[m1.queryIdx].pt
        pt2 = keyPoints2[m1.trainIdx].pt
        print(i, pt1,pt2)
        img1Points.append(pt1)
        img2Points.append(pt2)

for i in range(0, 5):
    noOfMatches = len(img1Points)
    randomList = random.sample(range(0, noOfMatches), 8)

    F = GenerateFundamentalMatrix(randomList, img1Points, img2Points)
    np.set_printoptions(suppress=True)
    vals, vecs = LA.eig(F.T @ F)
    # eigen val of the smallest eigen vector
    smallestEigenvector = vecs[:, np.argmin(vals)]
    F = smallestEigenvector.reshape((3, 3))
    print(F)

    #Compute the mean error using Sampsons Distance
    count = 0
    dist = 0
    for j in range(0, noOfMatches):
        if j not in randomList:
            count += 1;
            dist += ComputeSampsonsDistance(img1Points[j], img2Points[j], F)
    print("error is : " , dist/count)
