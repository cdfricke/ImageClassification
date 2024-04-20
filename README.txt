Programmer: Connor Fricke
File: README.txt (Image Classification Project)
Course: Survey of AI, CSE 5052

Project Description: Binary classifier algorithm implemented
in Python for classification of images of handwritten numbers
4 and 9 from the MNIST handwritten digit database.

See file "mnist_49_3000.mat". Images are stored as vectors.
Any machine learning algorithm discussed in class may be used.

The first 2000 images are to be used as training data. The last
1000 images will be classified by the algorithm and used as the
test dataset.

KNN (K-Nearest-Neighbor) Algorithm:
Each image is a vector of 784 floating point vlaues between 0 and 1.
They are reshapen into a 28 by 28 matrix to be drawn. We can
compare two images by comparing the difference between their matrix
values, element by element. By summing the squares of the residuals
elementwise, we get a "distance" between images. For more similar images,
we expect the distance to be shorter.

Now that we have an algorithm that calculates our "distance" proxy,
for each test set image, we need to find the nearest neighbors 
in our training data set and use those neighbors as a way of 
voting on the classification of our test image, majority rule.
The number of neighbors is an odd number sqrt(N) where N is the size
of our training data set.


