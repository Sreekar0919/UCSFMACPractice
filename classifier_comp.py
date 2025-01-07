import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Importing the necessary machine learning classifiers and utility functions
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Names of classifiers to be used in the experiment (for display in the plot)
names = [
    "Nearest Neighbors",    # K-Nearest Neighbors classifier
    "Linear SVM",           # Support Vector Machine with linear kernel
    "RBF SVM",              # Support Vector Machine with Radial Basis Function kernel
    "Gaussian Process",     # Gaussian Process Classifier
    "Decision Tree",        # Decision Tree Classifier
    "Random Forest",        # Random Forest Classifier
    "Neural Net",           # Multi-layer Perceptron (MLP) Classifier
    "AdaBoost",             # AdaBoost Classifier
    "Naive Bayes",          # Naive Bayes Classifier
    "QDA",                  # Quadratic Discriminant Analysis
]

# List of classifiers corresponding to the names above
classifiers = [
    KNeighborsClassifier(3),                      # KNN classifier with 3 neighbors
    SVC(kernel="linear", C=0.025, random_state=42), # Linear Support Vector Machine (SVM)
    SVC(gamma=2, C=1, random_state=42),           # Radial Basis Function (RBF) kernel SVM
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),  # Gaussian Process Classifier with RBF kernel
    DecisionTreeClassifier(max_depth=5, random_state=42),        # Decision Tree classifier with max depth of 5
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42), # Random Forest with 10 trees
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),      # Neural Network (MLP) classifier with regularization parameter
    AdaBoostClassifier(random_state=42),                         # AdaBoost Classifier
    GaussianNB(),                                                # Gaussian Naive Bayes classifier
    QuadraticDiscriminantAnalysis(),                              # Quadratic Discriminant Analysis classifier
]

# Generate synthetic classification dataset (with 2 informative features)
X, y = make_classification(
    n_features=2,  # Number of features (input variables)
    n_redundant=0, # No redundant features
    n_informative=2, # Two informative features that help in classification
    random_state=1, # Set a seed for reproducibility
    n_clusters_per_class=1  # One cluster per class (no overlap in data)
)

# Introduce random noise to the dataset (shift values) to make it less ideal for linear separation
rng = np.random.RandomState(2)  
X += 2 * rng.uniform(size=X.shape)  # Shift X by a random amount

# Define the third dataset (linearly separable dataset)
linearly_separable = (X, y)

# List of datasets for experimentation
datasets = [
    make_moons(noise=0.3, random_state=0),    # Moons dataset with noise
    make_circles(noise=0.2, factor=0.5, random_state=1),   # Circles dataset with noise
    linearly_separable,  # A dataset that is linearly separable
]

# Create the figure for plotting, specifying the figure size
figure = plt.figure(figsize=(27, 9))
i = 1

# Iterate over each dataset for visualizing results
for ds_cnt, ds in enumerate(datasets):
    X, y = ds  # Extract features (X) and labels (y) from the dataset
    
    # Split the dataset into training (60%) and testing (40%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Define plot limits (based on min and max values of dataset with some padding)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Define a color map for the plots
    cm = plt.cm.RdBu   # RdBu is the Red-Blue color map
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])  # Bright red and blue colors for classes
    
    # Create a subplot for the dataset
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    
    # Plot title for the first dataset
    if ds_cnt == 0:
        ax.set_title("Input data")
    
    # Plot the training points, color-coded by their labels (using cm_bright color map)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    
    # Plot the testing points (with some transparency)
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    ax.set_xlim(x_min, x_max)  # Set limits for x-axis
    ax.set_ylim(y_min, y_max)  # Set limits for y-axis
    ax.set_xticks(())  # Hide x-ticks for a cleaner plot
    ax.set_yticks(())  # Hide y-ticks for a cleaner plot
    i += 1  # Move to next subplot position

    # Iterate over classifiers to display their decision boundaries
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        # Create a pipeline that first scales the features, then applies the classifier
        clf = make_pipeline(StandardScaler(), clf)
        
        # Train the classifier on the training data
        clf.fit(X_train, y_train)
        
        # Compute the accuracy score of the classifier on the test data
        score = clf.score(X_test, y_test)
        
        # Plot the decision boundary using the trained classifier
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        )

        # Plot the training points
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )
        
        # Plot the testing points (with some transparency)
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        # Set axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        
        # Display classifier name as title on the first row
        if ds_cnt == 0:
            ax.set_title(name)
        
        # Display the accuracy score in the plot (formatted to 2 decimal places)
        ax.text(
            x_max - 0.3, 
            y_min + 0.3, 
            ("%.2f" % score).lstrip("0"), 
            size=15,
            horizontalalignment="right",
        )
        
        i += 1  # Move to the next subplot

# Apply tight layout to prevent overlap between subplots and display the figure
plt.tight_layout()
plt.show()
