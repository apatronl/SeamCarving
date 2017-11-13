# Project 2: Seam Carving
# Alejandrina Patron Lopez
# apl7@gatech.edu
# GT ID 903075226

import numpy as np
import cv2

# Returns energy of all pixels by computing derivative (color image, each color channel
# is computed separately)
# e = |dI/dx| + |dI/dy|
def energyRGB(img):
    out = energyGray(img[:, :, 0]).astype(np.float64) + energyGray(img[:, :, 1]).astype(np.float64) + energyGray(img[:, :, 2]).astype(np.float64)
    return scaleTo8Bit(out)

# Returns energy of all pixels by computing derivative
# e = |dI/dx| + |dI/dy|
def energyGray(img):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    out = abs(sobelx) + abs(sobely)
    return scaleTo8Bit(out)

# Returns accumulated cost matrix of the given energy map
def costMatrix(energyMap):
    w = energyMap.shape[1]
    h = energyMap.shape[0]
    matrix = energyMap.copy()
    matrix = matrix.astype(np.float64)
    for y in range(1, h):
        for x in range(w):
            if (x == 0):
                matrix[y,x] = energyMap[y, x] + min(matrix[y - 1, x], matrix[y - 1, x + 1])
            elif (x == w - 1):
                matrix[y,x] = energyMap[y, x] + min(matrix[y - 1, x], matrix[y - 1, x - 1])
            else:
                matrix[y,x] = energyMap[y, x] + min(matrix[y - 1, x - 1], matrix[y - 1, x], matrix[y - 1, x + 1])
    return scaleTo8Bit(matrix)

# Creates the vertical path of the seam to be removed from the image
def verticalSeam(matrix):
    w = matrix.shape[1]
    h = matrix.shape[0]
    minLastRow = minList(matrix[h - 1, :])
    aList = list(matrix[h - 1, :])
    path = [minLastRow]
    j = 0
    for y in range(h - 1, 0, -1):
        x = path[j]
        if (x == 0):
            nextMin = minList(matrix[y - 1, x : x + 2])
            path.append(x + nextMin)
        elif (x == w - 1):
            nextMin = minList(matrix[y - 1, x - 1 : x + 1])
            path.append(x + nextMin - 1)
        else:
            nextMin = minList(matrix[y - 1, x - 1 : x + 2])
            path.append(x + nextMin - 1)

        j += 1
    return path

# Removes vertical seam (path) from the provided image
def removeVerticalSeam(path, colorImg):
    output = cv2.resize(colorImg, (colorImg.shape[1] - 1, colorImg.shape[0]), interpolation = cv2.INTER_LINEAR)
    currentY = colorImg.shape[0] - 1
    for x in range(colorImg.shape[0]):
        skipX = path[x]
        output[currentY, : skipX] = colorImg[currentY, : skipX]
        output[currentY, skipX :] = colorImg[currentY, skipX + 1 :]
        currentY = currentY - 1
    return output

# Returns x coordinate of minimum value in given row
def minList(row):
    aList = list(row)
    minNum = min(row)
    return aList.index(minNum)

# Reduces the width of an image by the number of pixels provided
def resize(img, pixels):
    checkSize(img, pixels)
    output = img.copy()
    for i in range(pixels):
        energy = energyRGB(output)
        cost = costMatrix(energy)
        path = verticalSeam(cost)
        output = removeVerticalSeam(path, output)
    return output

# Makes sure image's width is larger than pixels to be removed from it
def checkSize(img, pixels):
    if (img.shape[1] - pixels <= pixels):
        raise Exception ("Image's width is smaller than number of pixels to be removed.")

# Draws a red seam on a given image following the provided path
def drawSeam(path, colorImg):
    h = colorImg.shape[0]
    output = colorImg.copy()
    currentY = colorImg.shape[0] - 1
    for y in range(h):
        skipX = path[y]
        output[currentY, skipX, 0] = 0
        output[currentY, skipX, 1] = 0
        output[currentY, skipX, 2] = 255
        currentY = currentY - 1
    return output

# Scales image to uint8
def scaleTo8Bit(image, displayMin = None, displayMax = None):
    if displayMin == None:
        displayMin = np.min(image)

    if displayMax == None:
        displayMax = np.max(image)

    np.clip(image, displayMin, displayMax, out = image)

    image = image - displayMin
    cf = 255. / (displayMax - displayMin)
    imageOut = (cf * image).astype(np.uint8)
    return imageOut

# img = cv2.imread("venice.jpg")
# final = resize(img, 100)
# cv2.imwrite("veniceOut.jpg", final)
