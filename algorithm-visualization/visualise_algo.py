import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

def visualize_classifiers(X, y, classifiers, classifier_names, title="Classifier Comparison"):
    """
    Visualizes the decision boundaries of different classifiers.

    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target vector.
        classifiers (list): List of classifier objects.
        classifier_names (list): List of classifier names (strings).
        title (str): Title of the plot.
    """

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    h = 0.02  # step size in the mesh

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    figure = plt.figure(figsize=(15, 5 * len(classifiers)))
    i = 1

    for name, clf in zip(classifier_names, classifiers):
        ax = plt.subplot(len(classifiers), 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.4)

        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k')
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=20, edgecolor='k', marker='x')

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name + f", Accuracy = {score:.2f}")
        i += 1

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # adjust subplot params so that the subplot(s) fits in to the figure area.
    plt.show()

# Example usage with the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Using only the first two features for visualization
y = iris.target

classifiers = [
    LogisticRegression(solver='lbfgs', multi_class='auto'),
    SVC(kernel='linear'),
    SVC(kernel='rbf'),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(3)
]

classifier_names = [
    "Logistic Regression",
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Random Forest",
    "K-Nearest Neighbors"
]

visualize_classifiers(X, y, classifiers, classifier_names)

# Example Usage with the make_moons dataset
X, y = datasets.make_moons(noise=0.3, random_state=0)

classifiers_moons = [
    LogisticRegression(solver='lbfgs'),
    SVC(kernel='linear'),
    SVC(kernel='rbf'),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(3)
]

classifier_names_moils = [
    "Logistic Regression",
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Random Forest",
    "K-Nearest Neighbors"
]

visualize_classifiers(X, y, classifiers_moons, classifier_names_moils, title = "Moons dataset classifier comparison")