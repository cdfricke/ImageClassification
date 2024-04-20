# Programmer: Connor Fricke (fricke.59@osu.edu)
# File: functions.py
# Latest Revision: 19-April-2024 --> Created for CSE 5052 Final Project

# *** MODULES ***
import scipy.io
import numpy as np
from matplotlib import pyplot as plt

# *** FUNCTIONS ***

def loadDataFrom(matFileName):
    """
    Loads data from provided .mat file into numpy arrays, then
    returns a reference to those arrays. First return value
    is the array of images. Second return value is the array
    of the corresponding classifications of those arrays.
    """
    data = scipy.io.loadmat(matFileName)
    x = np.array(data['x'])
    y = np.array(data['y'][0])
    y[y==-1] = 0
    return x, y

def getImage(imgIndex, imgArray):
    """
    Returns image at location imgIndex within imgArray.
    Image is a 1D NumPy array of length 784
    """
    image = imgArray[:,imgIndex]
    return image

def getBinaryClassification(imgIndex, imgArray):
    return imgArray[imgIndex]

def getClassification(imgIndex, imgArray):
    # y = 0 -> img is a 4
    # y = 1 -> img is a 9
    return (9 if imgArray[imgIndex] else 4)

def showImage(imgIndex, imgArray):
    """
    Utilizes PyPlot to display images by reshaping
    them into a 28 x 28 array and plotting their values
    """
    image = getImage(imgIndex, imgArray).reshape(28,28)
    plt.imshow(image, interpolation='nearest')
    plt.show()

def distanceBetweenImages(image1, image2):
    """
    Calculates a theoretical distance between two images
    by summing the squared residuals of the image values
    elementwise. Does not reshape the arrays, and assumes
    they are 1D.
    """
    distance = 0.0
    if (image1.size != image2.size):
        print("Error! Images must be equal in size for comparison.")
        return 0.0
    for i in range(image1.size):
        distance += (image1[i] - image2[i])**2

    distance /= image1.size # normalize
    return distance