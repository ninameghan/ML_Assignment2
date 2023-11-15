
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
import time


def main():
    sample_sizes = [1000, 5000, 10000]
    # TASK 1
    labels, features = load_data()

    # TASK 2
    # accuracy, time_train, time_pred = evaluation(labels, features, DecisionTreeClassifier, 1000)

    # TASK 3
    perceptron(labels, features, sample_sizes)

    # TASK 4
    decision_tree(labels, features, sample_sizes)

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
    return labels, features


# TASK 2
def evaluation(labels, features, classifier, sample_size):
    # Create a K-Fold object to perform 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True)

    accuracy = []
    time_train = []
    time_pred = []

    labels = labels[:sample_size]
    features = features[:sample_size]

    # Cross validation loop splitting training & test data
    for train_index, test_index in kf.split(features):
        train_features = features.iloc[train_index]
        test_features = features.iloc[test_index]
        train_labels = labels.iloc[train_index]
        test_labels = labels.iloc[test_index]

        # Create instance of specified classifier
        clf = classifier()

        # Measure processing time for training
        start_time = time.time()
        clf = clf.fit(train_features, train_labels)
        training_time = time.time() - start_time
        time_train.append(training_time)

        # Measure processing time for prediction
        start_time = time.time()
        prediction = clf.predict(test_features)
        prediction_time = time.time() - start_time
        time_pred.append(prediction_time)

        # Determine confusion matrix
        c_matrix = metrics.confusion_matrix(test_labels, prediction)
        print("Confusion Matrix:\n", c_matrix)

        # Determine accuracy score
        accuracy.append(metrics.accuracy_score(test_labels, prediction))

    return accuracy, time_train, time_pred


def show_results(accuracy, time_train, time_pred):
    # Training time per training sample
    # Minimum
    min_time_train = np.min(time_train)
    print("Minimum processing time (Train):\n", min_time_train)
    # Maximum
    max_time_train = np.max(time_train)
    print("Maximum processing time (Train):\n", max_time_train)
    # Average
    avg_time_train = np.mean(time_train)
    print("Average processing time (Train):\n", avg_time_train)

    # Prediction time per evaluation sample
    # Minimum
    min_time_pred = np.min(time_pred)
    print("Minimum processing time (Prediction):\n", min_time_pred)
    # Maximum
    max_time_pred = np.max(time_pred)
    print("Maximum processing time (Prediction):\n", max_time_pred)
    # Average
    avg_time_pred = np.mean(time_pred)
    print("Average processing time (Prediction):\n", avg_time_pred)

    # Prediction accuracy
    # Minimum
    min_acc_test = np.min(accuracy)
    print("Minimum prediction accuracy (Test):\n", min_acc_test)
    # Maximum
    max_acc_test = np.max(accuracy)
    print("Maximum prediction accuracy (Test):\n", max_acc_test)
    # Average
    avg_acc_test = np.mean(accuracy)
    print("Average prediction accuracy (Test):\n", avg_acc_test)


def evaluate_classifier(labels, features, classifier, sample_sizes):
    accuracies = []
    times_train = []
    times_pred = []
    for sample_size in sample_sizes:
        accuracy, time_train, time_pred = evaluation(labels, features, classifier, sample_size)
        show_results(accuracy, time_train, time_pred)
        accuracies.append(np.mean(accuracy))
        times_train.append(np.mean(time_train))
        times_pred.append(np.mean(time_pred))

    return accuracies, times_train, times_pred


# TASK 3
def perceptron(labels, features, sample_sizes):
    accuracies, times_train, times_pred = evaluate_classifier(labels, features, Perceptron, sample_sizes)
    # Mean prediction accuracy for Perceptron classifier
    print("\nMean prediction accuracy for Perceptron classifier:\n", np.mean(accuracies))
    plt.figure()
    plt.plot(sample_sizes, times_pred)
    plt.title("Perceptron Classifier")
    plt.xlabel("Sample Size")
    plt.ylabel("Prediction Runtime")
    plt.show()


# TASK 4
def decision_tree(labels, features, sample_sizes):
    accuracies, times_train, times_pred = evaluate_classifier(labels, features, DecisionTreeClassifier, sample_sizes)
    # Mean prediction accuracy for Decision Tree classifier
    print("\nMean prediction accuracy for Decision Tree classifier:", np.mean(accuracies))
    plt.figure()
    plt.plot(sample_sizes, times_pred)
    plt.title("Decision Tree Classifier")
    plt.xlabel("Sample Size")
    plt.ylabel("Prediction Runtime")
    plt.show()


main()
