// Programmer: Connor Fricke (cd.fricke23@gmail.com)
// File: KNN.cpp
// Latest Revision: 19-June-2024
// Synopsis: File for performing K-Nearest-Neighbor machine learning algorithm for classification
// of images of handwritten digits 4 and 9.

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
using namespace std;
using chrono::high_resolution_clock, chrono::duration_cast, chrono::milliseconds;

typedef vector<double> image;

vector<image> readImagesFile(const string& FILE_IN);

int main()
{
    auto startRead = high_resolution_clock::now();
    vector<image> IMAGES = readImagesFile("images.csv");
    auto stopRead = high_resolution_clock::now();
    auto readDuration = duration_cast<milliseconds>(stopRead - startRead);
    cout << "Read file in " << readDuration.count() / 1000.0 << " seconds.\n";

    return 0;
}

vector<image> readImagesFile(const string& FILE_IN)
{

    ifstream fin;
    fin.open(FILE_IN, ios::in);
    if (!fin.is_open()) exit(1);
    cout << "Reading data from " << FILE_IN << "..." << endl;

    const int numImgs = 3000; // number of images (rows of data) in the file
    const int imgSize = 784;  // number of floating-point values for each image

    // allocate space for vector of images
    vector<image> IMAGES;
    for (int i = 0; i < numImgs; i++) 
    {
        image nextImage;
        for (int j = 0; j < imgSize; j++)
        {
            double pixelValue = 0.0;
            fin >> pixelValue;
            nextImage.push_back(pixelValue);
        }
        IMAGES.push_back(nextImage);
    }

    return IMAGES;
}