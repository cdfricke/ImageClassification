# Programmer: Connor Fricke
# File: incorrect.py
# Latest Revision: 20-April-2024
#
# Python script for visualizing images which were incorrectly
# classified by the KNN machine learning algorithm.

from functions import *

x, y = loadDataFrom('mnist_49_3000.mat')

with open('incorrect.dat', 'r', encoding='utf-8') as file:
    idx = file.readlines()
    for id in idx:
        showImage(int(id), x)
