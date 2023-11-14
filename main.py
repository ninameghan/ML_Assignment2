
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def main():
    # TASK 1
    load_data()

    # TASK 2

    # TASK 3

    # TASK 4

    # TASK 5

    # TASK 6

    # TASK 7


# TASK 1
def load_data():
    shoe_labels = [5, 7, 9]
    data = pd.read_csv("fashion-mnist_train.csv")

    # load all sandals (5), sneakers (7) and ankle boots (9) from the dataset
    shoes = data[data["label"].isin(shoe_labels)]

    # separate labels and feature vectors
    labels = shoes["label"]
    features = shoes.drop("label", axis=1)

    # display 1 image for each class
    pixels_sandals = features.iloc[[(labels == 5).argmax()]].to_numpy().reshape((28, 28))
    pixels_sneakers = features.iloc[[(labels == 7).argmax()]].to_numpy().reshape((28, 28))
    pixels_boots = features.iloc[[(labels == 9).argmax()]].to_numpy().reshape((28, 28))

    plt.imshow(pixels_sandals, cmap='gray')
    plt.show()
    plt.imshow(pixels_sneakers, cmap='gray')
    plt.show()
    plt.imshow(pixels_boots, cmap='gray')
    plt.show()


main()
