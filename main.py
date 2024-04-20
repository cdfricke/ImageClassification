# Programmer: Connor Fricke (fricke.59@osu.edu)
# File: main.py
# Latest Revision: 19-April-2024 --> Created

# *** MODULES ***
from math import sqrt
from functions import *

# *** MAIN ***

# X IS THE IMAGE VECTOR, Y IS THE CLASSIFICATION VECTOR
x, y = loadDataFrom('mnist_49_3000.mat')
print(f"We have {y.size} images available.")

# display first 3 images for visualization
for i in range(3):
    showImage(i, x)
    print(f"Image {i} is labeled as {getClassification(i, y)}")

# *** EXAMPLE IMAGES AND DISTANCES ***
image0 = getImage(0, x)
image1 = getImage(1, x)
image2 = getImage(2, x)
print("Example Distances:")
print("Image 0 [4] and Image 1 [9]:", distanceBetweenImages(image0, image1))
print("Image 1 [9] and Image 2 [4]:", distanceBetweenImages(image1, image2))
print("Image 2 [4] and Image 0 [4]:", distanceBetweenImages(image2, image0))

# *** KNN ALGORITHM ***
classificationAccuracy = 0  # reflects the accuracy of the algorithm
N = 41                      # check this many test images (1000 to check all)
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

    # test accuracy
    if (prediction == getBinaryClassification(i + LTD, y)):
        print(f"Image {i + LTD} Prediction = {prediction}, Correct!")
        classificationAccuracy += 1
    else:
        print(f"Image {i + LTD} Prediction = {prediction}, Incorrect!")
        incorrectIndices.append(i + LTD)
    
# results
print(classificationAccuracy / N)
print("Incorrect Classifications found at:", incorrectIndices)
for index in incorrectIndices:
    showImage(index, x)





