# Machine Learning Classifier Visualizations

This repository contains a Python script (`visualize_classifiers.py`) that visualizes the decision boundaries of various machine learning classifiers on the Iris and make_moons datasets.

## Description

The script uses scikit-learn to train and visualize the following classifiers:

-   Logistic Regression
-   Linear Support Vector Machine (SVM)
-   Radial Basis Function (RBF) SVM
-   Decision Tree
-   Random Forest
-   K-Nearest Neighbors (KNN)

It visualizes the decision boundaries by plotting the predicted classes on a meshgrid and overlays the training and testing data points. Two figures are generated, one for the Iris dataset and one for the make_moons dataset, allowing for comparison of classifier performance on linear and non-linear data.

## Requirements

-   Python 3.x
-   scikit-learn (`sklearn`)
-   matplotlib
-   numpy

You can install the required packages using pip:

```bash
pip install scikit-learn matplotlib numpy