
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random as rng
import math

items1 = ['col1', 'col2']

items2 = ['a', 'b']
items3 = [1, 2]
data = zip(items2, items3)
print(items1)
print(list(data))

# cap = cv2.VideoCapture('atrium.mp4')
# cap = cv2.VideoCapture(0)
# while(True):
#     ret, frame = cap.read()
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

def rgb2gray(rgb):
  r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return gray


## sum of difference between N frames  to
## highlight the moving objects

def FindSumOfDifference(frames):
  rows = frames[0].shape[0]
  cols = frames[0].shape[1]
  D = np.zeros((rows, cols))

  for m in range(0, rows):
    for n in range(0, cols):
      for i in range(0, len(frames) - 1):
        for j in range(i+1, len(frames)):
          framei = frames[i]
          framej = frames[j]
          D[m,n] += np.abs(framei[m,n] - framej[m,n])
  # plt.imshow(D)
  # plt.show()
  return D

## Get Detections - Segmentation and region props
## finding blobbs and returning bounding boxes around them
## regionprops finds connected components in your image and
# puts bounding boxes around them - GetDetections

# Segmentation
#
def PerformSegmentation(D):
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = 2
  attempts=10

  Z = D.reshape((-1,2))
  Z = np.float32(Z)
  print(Z.shape)

  ret,label,center=cv2.kmeans(Z,K,None,criteria,attempts,cv2.KMEANS_RANDOM_CENTERS)

  center = np.uint8(center)
  res = center[label.flatten()]
  segmentedImg = res.reshape((D.shape))

  # plt.imshow(segmentedImg)
  # plt.show()
  return segmentedImg


def PerformImageProcessing(segmentedImg):
  # Dialation
  kernel = np.ones((10,10), np.uint8)
  dialated1 = cv2.dilate(segmentedImg, kernel, iterations=1)

  # Blur the image
  ksize = (10,10)
  blurred = cv2.blur(dialated1, ksize)

  # apply binary thresholding
  ret, thresh = cv2.threshold(dialated1, 150, 255, cv2.THRESH_BINARY)

  # Do Morphological processing - Dialation
  kernel = np.ones((10,10), np.uint8)
  dialated = cv2.dilate(thresh, kernel, iterations=1)

  return dialated
# visualize
# cv2.imshow('Dia1 image', dialated1)
# cv2.waitKey(0)
# cv2.imshow('blurred image', blurred)
# cv2.waitKey(0)
# cv2.imshow('Binary image', thresh)
# cv2.waitKey(0)
# cv2.imshow('Dia image', dialated)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## you have got the people now - use getFeature to get the HoG features - extracting info about the ones
## got in the previous step - GetFeatures - getting the centers and associating the contours in appropriate frames
## using the distances
contour_color_map = {}
def ShowDetections(processedImg, baseImg):
  contours, _ = cv2.findContours(processedImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  min = 100
  ind = 0
  l = contour_color_map.keys()
  for i in range(0, len(l)):
    for j in range(len(contours)):
      dis = math.dist(cv2.moments(contours[j]), l[i])
      if (dis < min):
        min = dis
        ind = j
        del contours[j]

  contours_poly = [None] * len(contours)
  boundRect = [None] * len(contours)
  for i, c in enumerate(contours):
    contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    boundRect[i] = cv2.boundingRect(contours_poly[i])

  drawing = np.zeros((processedImg.shape[0], processedImg.shape[1], 3), dtype=np.uint8)

  for i in range(len(contours)):
    # print(cv2.contourArea(contours[i]))
    if(cv2.contourArea(contours[i]) > 3000):
        # cv2.drawContours(drawing, contours_poly, i, color)
        cv2.rectangle(baseImg, (int(boundRect[i][0]), int(boundRect[i][1])), \
                     (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)


  plt.imshow(baseImg)
  plt.show()

def ReadVideo():
  vidcap = cv2.VideoCapture('atrium.mp4.avi')
  success,image = vidcap.read()
  count = 0
  video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
  print("Number of frames: ", video_length)
  frames = []
  while success:
    image = rgb2gray(image)
    frames.append(image)
    success, image = vidcap.read()
    colorImage = image
    count += 1
    if(len(frames) == 3):
      D = FindSumOfDifference(frames)
      segmentedImg = PerformSegmentation(D)
      processedImg = PerformImageProcessing(segmentedImg)
      ShowDetections(processedImg, colorImage)
      frames = []

ReadVideo()

#
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# K = 2
# attempts=10
#
# Z = D.reshape((-1,2))
# Z = np.float32(Z)
# print(Z.shape)
#
# ret,label,center=cv2.kmeans(Z,K,None,criteria,attempts,cv2.KMEANS_RANDOM_CENTERS)
#
# center = np.uint8(center)
# res = center[label.flatten()]
# result_image = res.reshape((D.shape))
#
# plt.imshow(result_image, cmap='gray')
# plt.show()


from skimage import measure
from skimage.filters import threshold_otsu
from skimage.color import label2rgb
#
# threshold = threshold_otsu(D)
#
# segmentedImg = measure.label(D<threshold)
#
# plt.imshow(segmentedImg, cmap='gray')
# plt.show()

# Regionprops
# props = measure.regionprops(segmentedImg)
# imgCopy = np.asarray(D)
# for prop in props:
#   print(prop.centroid)

  # if prop.area_bbox > 0:
  #   cv2.rectangle(imgCopy, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (200,0,0))
# prop = props[1]
# if prop.area_bbox > 0:
#     cv2.rectangle(imgCopy, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255,0,0), 4)
# cropped_image = imgCopy[prop.bbox[0]:prop.bbox[2],prop.bbox[1]:prop.bbox[3]]
#
# plt.imshow(cropped_image, cmap='gray')
# plt.show()


# for i in range(0, 3):
#   image = rgb2gray(image)
#   plt.imshow(image)
#   plt.show()
#   frames.append(image)
#   # cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
# success, image = vidcap.read()
# count += 1

# contour_color_map = {}
#
#
# def ShowDetections(processedImg, baseImg):
#   contours, _ = cv2.findContours(processedImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#   contours_poly = [None] * len(contours)
#   boundRect = [None] * len(contours)
#   for i, c in enumerate(contours):
#     contours_poly[i] = cv2.approxPolyDP(c, 3, True)
#     boundRect[i] = cv2.boundingRect(contours_poly[i])
#
#   drawing = np.zeros((processedImg.shape[0], processedImg.shape[1], 3), dtype=np.uint8)
#
#   for i in range(len(contours)):
#     # print(cv2.contourArea(contours[i]))
#     if (cv2.contourArea(contours[i]) > 3000):
#       # cv2.drawContours(drawing, contours_poly, i, color)
#       cv2.rectangle(baseImg, (int(boundRect[i][0]), int(boundRect[i][1])), \
#                     (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
#
#   plt.imshow(baseImg)
#   plt.show()
