
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import time


def main():
    sample_sizes = [500, 1000, 1500]
    # TASK 1
    labels, features = load_data()

    # TASK 2
    # accuracy, time_train, time_pred = evaluation(labels, features, DecisionTreeClassifier, 1000)

    # TASK 3
    perceptron(labels, features, sample_sizes)

    # TASK 4
    decision_tree(labels, features, sample_sizes)

    # TASK 5
    k_nearest_neighbour(labels, features, sample_sizes)

    # TASK 6
    support_vector_machine(labels, features, sample_sizes)


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

    labels_sample = labels[:sample_size]
    features_sample = features[:sample_size]

    # Cross validation loop splitting training & test data
    for train_index, test_index in kf.split(features_sample):
        train_features = features_sample.iloc[train_index]
        test_features = features_sample.iloc[test_index]
        train_labels = labels_sample.iloc[train_index]
        test_labels = labels_sample.iloc[test_index]

        # Create instance of specified classifier
        clf = classifier

        # Measure processing time for training
        start_time_train = time.time()
        clf = clf.fit(train_features, train_labels)
        training_time = time.time() - start_time_train
        time_train.append(training_time)

        # Measure processing time for prediction
        start_time_pred = time.time()
        prediction = clf.predict(test_features)
        prediction_time = time.time() - start_time_pred
        time_pred.append(prediction_time)

        # Determine confusion matrix
        c_matrix = metrics.confusion_matrix(test_labels, prediction)
        # print("Confusion Matrix:\n", c_matrix)

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
    accuracies, times_train, times_pred = evaluate_classifier(labels, features, Perceptron(), sample_sizes)
    # Mean prediction accuracy for Perceptron classifier
    print("\nMean prediction accuracy for Perceptron classifier:", np.mean(accuracies))
    # Plot relationship between input data size and runtimes
    plt.figure()
    plt.plot(sample_sizes, times_pred)
    plt.title("Perceptron Classifier")
    plt.xlabel("Sample Size")
    plt.ylabel("Prediction Runtime")
    plt.show()


# TASK 4
def decision_tree(labels, features, sample_sizes):
    accuracies, times_train, times_pred = evaluate_classifier(labels, features, DecisionTreeClassifier(), sample_sizes)
    # Mean prediction accuracy for Decision Tree classifier
    print("\nMean prediction accuracy for Decision Tree classifier:", np.mean(accuracies))
    print(sample_sizes)
    print(times_pred)
    # Plot relationship between input data size and runtimes
    plt.figure()
    plt.plot(sample_sizes, times_pred)
    plt.title("Decision Tree Classifier")
    plt.xlabel("Sample Size")
    plt.ylabel("Prediction Runtime")
    plt.show()


# TASK 5
def k_nearest_neighbour(labels, features, sample_sizes):
    # Determine best k-value based on mean prediction accuracy
    acc_scores = []
    for k in range(1, 10):
        accuracies, times_train, times_pred = evaluate_classifier(labels, features, KNeighborsClassifier(n_neighbors=k),
                                                                  sample_sizes)
        acc_scores.append(np.mean(accuracies))

    # Evaluation with optimal k-value
    max_acc = np.max(acc_scores)
    optimal_k_value = acc_scores.index(max_acc) + 1
    accuracies_optimal, times_train_optimal, times_pred_optimal = evaluate_classifier(labels, features,
                                                              KNeighborsClassifier(n_neighbors=optimal_k_value),
                                                              sample_sizes)
    print("\nOptimal k-value:", optimal_k_value)

    # Best mean prediction accuracy for K-Nearest Neighbour classifier
    print("Best mean prediction accuracy for K-Nearest Neighbour classifier:", np.mean(accuracies_optimal))
    print()

    # Plot relationship between input data size and runtimes for optimal classifier
    plt.figure()
    plt.plot(sample_sizes, times_pred_optimal)
    plt.title("K-Nearest Neighbour Classifier")
    plt.xlabel("Sample Size")
    plt.ylabel("Prediction Runtime")
    plt.show()


# TASK 6
def support_vector_machine(labels, features, sample_sizes):
    # Determine best gamma value based on mean prediction accuracy
    gammas = [0.2, 0.4, 0.6, 0.8, 1.0]
    acc_scores = []
    for gamma in gammas:
        accuracies, times_train, times_pred = evaluate_classifier(labels, features, SVC(kernel="rbf", gamma=gamma),
                                                                  sample_sizes)
        acc_scores.append(np.mean(accuracies))

    # Evaluation with optimal gamma value
    max_acc = np.max(acc_scores)
    optimal_gamma_value = gammas[acc_scores.index(max_acc)]
    accuracies, times_train, times_pred = evaluate_classifier(labels, features,
                                                              SVC(kernel="rbf", gamma=optimal_gamma_value),
                                                              sample_sizes)
    print("\nOptimal gamma-value:", optimal_gamma_value)

    # Best mean prediction accuracy for Support Vector Machine classifier
    print("Best mean prediction accuracy for Support Vector Machine classifier:", np.mean(accuracies))
    print()

    # Plot relationship between input data size and runtimes for optimal classifier
    plt.figure()
    plt.plot(sample_sizes, times_pred)
    plt.title("Support Vector Machine Classifier")
    plt.xlabel("Sample Size")
    plt.ylabel("Prediction Runtime")
    plt.show()


main()
