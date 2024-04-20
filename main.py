# Programmer: Connor Fricke (fricke.59@osu.edu)
# File: main.py
# Latest Revision: 19-April-2024 --> Created

# *** MODULES ***
import scipy.io
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt

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

# *** MAIN ***

# loads data from file to x and y ndarrays
# X IS THE IMAGE VECTOR, Y IS THE CLASSIFICATION
x, y = loadDataFrom('mnist_49_3000.mat')
print(f"We have {y.size} images available.")

# display first 3 images for visualization
for i in range(3):
    showImage(i, x)
    print(f"Image {i} is labeled as {getClassification(i, y)}")

# Perform calculations on a few examples
image0 = getImage(0, x)
image1 = getImage(1, x)
image2 = getImage(2, x)
print("Example Distances:")
print("Image 0 [4] and Image 1 [9]:", distanceBetweenImages(image0, image1))
print("Image 1 [9] and Image 2 [4]:", distanceBetweenImages(image1, image2))
print("Image 2 [4] and Image 0 [4]:", distanceBetweenImages(image2, image0))

# *** SET UP TRAINING AND TEST DATA ***
trainingData = []
testData = []

for i in range(3000):
    if i < 2000:
        trainingData.append(getImage(i, x))
    else:
        testData.append(getImage(i, x))

# *** KNN ALGORITHM ***
classificationAccuracy = 0  # reflects the accuracy of the algorithm
N = 1000                    # check this many test images (1000 to check all)
LTD = 2000                  # length of our training data
incorrectIndices = []       # indices of images that were incorrectly classified

for i in range(N):
    testImg = getImage(i + LTD, x)
    allNeighbors = []
    for j in range(LTD):
        trainImg = getImage(j, x)
        distance = distanceBetweenImages(testImg, trainImg)
        # each neighbor is a tuple containing the index of the image and it's distance
        # neighbor[0] gives us the index of that image within the entire set
        # neighbor[1] gives us the distance proxy of that image
        neighbor = (j, distance)
        allNeighbors.append(neighbor)

    # recommended K is sqrt(N), N is length of training data
    K = int(sqrt(LTD))
    if K % 2 == 0:
        K -= 1
    KNN = sorted(allNeighbors, key=lambda neighbor: neighbor[1])[:K]

    # calculate prediction based on KNNs
    sum = 0
    for neighbor in KNN:
        sum += getBinaryClassification(neighbor[0], y)
    prediction = 0
    if (float(sum) / float(K) < 0.5):
        prediction = 0
    else:
        prediction = 1

    if (prediction == getBinaryClassification(i + LTD, y)):
        print(f"Image {i + LTD} Prediction = {prediction}, Correct!")
        classificationAccuracy += 1
    else:
        print(f"Image {i + LTD} Prediction = {prediction}, Incorrect!")
        incorrectIndices.append(i + LTD)
    
print(classificationAccuracy / N)
print("Incorrect Classifications found at:", incorrectIndices)
showImage(2002, x)





