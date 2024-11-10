#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

# Create a DataFrame for easier manipulation
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# Split dataset into features (X) and target (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier (C4.5 uses 'entropy' as the criterion)
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Train the model on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Visualize the decision tree
plt.figure(figsize=(12,8))
tree.plot_tree(clf, feature_names=iris['feature_names'], class_names=iris['target_names'], filled=True)
plt.show()


# In[6]:


import csv
import random
import math

# stopping criterion
MINIMUM_SAMPLE_SIZE = 4
MAX_TREE_DEPTH = 3


class tree_node:

    def __init__(self, training_set, attribute_list, attribute_values, tree_depth, algorithm='ID3'):
        self.is_leaf = False
        self.dataset = training_set
        self.split_attribute = None
        self.split = None
        self.attribute_list = attribute_list
        self.attribute_values = attribute_values
        self.left_child = None
        self.right_child = None
        self.prediction = None
        self.depth = tree_depth
        self.algorithm = algorithm

    def build(self):

        training_set = self.dataset

        # only proceed building tree if stopping criterion isn't matched
        if self.depth < MAX_TREE_DEPTH and len(training_set) >= MINIMUM_SAMPLE_SIZE and len(set([elem["species"] for elem in training_set])) > 1:
            # get attribute and split with highest information gain (or gain ratio for C4.5)
            if self.algorithm == 'ID3':
                max_gain, attribute, split = max_information_gain(self.attribute_list, self.attribute_values, training_set)
            elif self.algorithm == 'C4.5':
                max_gain, attribute, split = max_gain_ratio(self.attribute_list, self.attribute_values, training_set)
            elif self.algorithm == 'CART':
                max_gain, attribute, split = max_gini_index(self.attribute_list, self.attribute_values, training_set)

            # test if gain is greater than 0
            if max_gain > 0:
                # split tree
                self.split = split
                self.split_attribute = attribute

                # create children
                training_set_l = [elem for elem in training_set if elem[attribute] < split]
                training_set_r = [elem for elem in training_set if elem[attribute] >= split]
                self.left_child = tree_node(training_set_l, self.attribute_list, self.attribute_values, self.depth + 1, self.algorithm)
                self.right_child = tree_node(training_set_r, self.attribute_list, self.attribute_values, self.depth + 1, self.algorithm)
                self.left_child.build()
                self.right_child.build()
            else:
                self.is_leaf = True
        else:
            self.is_leaf = True

        if self.is_leaf:
            # prediction of leaf is the most common class in training_set
            self.prediction = most_common_class(training_set)

    # test decision tree accuracy
    def predict(self, sample):
        if self.is_leaf:
            return self.prediction
        else:
            if sample[self.split_attribute] < self.split:
                return self.left_child.predict(sample)
            else:
                return self.right_child.predict(sample)

    def print(self, prefix):
        if self.is_leaf:
            print("\t" * self.depth + prefix + self.prediction)
        else:
            print("\t" * self.depth + prefix + self.split_attribute + "<" + str(self.split) + "?")
            self.left_child.print("[True] ")
            self.right_child.print("[False] ")

# calculate entropy (used by ID3 and C4.5)
def entropy(dataset):
    if len(dataset) == 0:
        return 0

    target_attribute_name = "species"
    target_attribute_values = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    data_entropy = 0
    for val in target_attribute_values:
        p = len([elem for elem in dataset if elem[target_attribute_name] == val]) / len(dataset)
        if p > 0:
            data_entropy += -p * math.log(p, 2)

    return data_entropy

# calculate Gini index (used by CART)
def gini_index(dataset):
    if len(dataset) == 0:
        return 0

    target_attribute_name = "species"
    target_attribute_values = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    gini = 1
    for val in target_attribute_values:
        p = len([elem for elem in dataset if elem[target_attribute_name] == val]) / len(dataset)
        gini -= p**2

    return gini

# calculate average entropy of split on an attribute (used by ID3)
def info_gain(attribute_name, split, dataset):
    set_smaller = [elem for elem in dataset if elem[attribute_name] < split]
    p_smaller = len(set_smaller) / len(dataset)
    set_greater_equals = [elem for elem in dataset if elem[attribute_name] >= split]
    p_greater_equals = len(set_greater_equals) / len(dataset)

    info_gain = entropy(dataset)
    info_gain -= p_smaller * entropy(set_smaller)
    info_gain -= p_greater_equals * entropy(set_greater_equals)

    return info_gain

# calculate gain ratio (used by C4.5)
def gain_ratio(attribute_name, split, dataset):
    set_smaller = [elem for elem in dataset if elem[attribute_name] < split]
    p_smaller = len(set_smaller) / len(dataset)
    set_greater_equals = [elem for elem in dataset if elem[attribute_name] >= split]
    p_greater_equals = len(set_greater_equals) / len(dataset)

    info_gain_value = info_gain(attribute_name, split, dataset)
    
    intrinsic_value = -p_smaller * math.log(p_smaller, 2) if p_smaller > 0 else 0
    intrinsic_value -= p_greater_equals * math.log(p_greater_equals, 2) if p_greater_equals > 0 else 0

    if intrinsic_value == 0:
        return 0

    return info_gain_value / intrinsic_value

# calculate Gini gain (used by CART)
def gini_gain(attribute_name, split, dataset):
    set_smaller = [elem for elem in dataset if elem[attribute_name] < split]
    set_greater_equals = [elem for elem in dataset if elem[attribute_name] >= split]

    gini_before = gini_index(dataset)
    gini_after = (len(set_smaller) / len(dataset)) * gini_index(set_smaller) + (len(set_greater_equals) / len(dataset)) * gini_index(set_greater_equals)

    return gini_before - gini_after

# get attribute and split with max information gain (ID3), gain ratio (C4.5), or Gini gain (CART)
def max_information_gain(attribute_list, attribute_values, dataset):
    max_gain = 0
    max_gain_attribute = None
    max_gain_split = None
    for attribute in attribute_list:
        for split in attribute_values[attribute]:
            split_gain = info_gain(attribute, split, dataset)
            if split_gain > max_gain:
                max_gain = split_gain
                max_gain_attribute = attribute
                max_gain_split = split
    return max_gain, max_gain_attribute, max_gain_split

def max_gain_ratio(attribute_list, attribute_values, dataset):
    max_gain_ratio_value = 0
    max_gain_attribute = None
    max_gain_split = None
    for attribute in attribute_list:
        for split in attribute_values[attribute]:
            split_gain_ratio = gain_ratio(attribute, split, dataset)
            if split_gain_ratio > max_gain_ratio_value:
                max_gain_ratio_value = split_gain_ratio
                max_gain_attribute = attribute
                max_gain_split = split
    return max_gain_ratio_value, max_gain_attribute, max_gain_split

def max_gini_index(attribute_list, attribute_values, dataset):
    max_gini_gain = 0
    max_gain_attribute = None
    max_gain_split = None
    for attribute in attribute_list:
        for split in attribute_values[attribute]:
            split_gini_gain = gini_gain(attribute, split, dataset)
            if split_gini_gain > max_gini_gain:
                max_gini_gain = split_gini_gain
                max_gain_attribute = attribute
                max_gain_split = split
    return max_gini_gain, max_gain_attribute, max_gain_split

def most_common_class(training_set):
    setosa_count = versicolor_count = virginica_count = 0
    for elem in training_set:
        if elem["species"] == "Iris-setosa":
            setosa_count += 1
        elif elem["species"] == "Iris-versicolor":
            versicolor_count += 1
        else:
            virginica_count += 1
    if setosa_count >= versicolor_count and setosa_count >= virginica_count:
        return "Iris-setosa"
    elif versicolor_count >= setosa_count and versicolor_count >= virginica_count:
        return "Iris-versicolor"
    else:
        return "Iris-virginica"

def read_iris_dataset():
    dataset = []
    attribute_values = {"sepal_length": [], "sepal_width": [], "petal_length": [], "petal_width": []}
    with open("IRIS.csv", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dataset.append({
                "sepal_length": float(row["sepal_length"]),
                "sepal_width": float(row["sepal_width"]),
                "petal_length": float(row["petal_length"]),
                "petal_width": float(row["petal_width"]),
                "species": row["species"]
            })
            for attribute in attribute_values:
                if float(row[attribute]) not in attribute_values[attribute]:
                    attribute_values[attribute].append(float(row[attribute]))

    for attribute in attribute_values:
        attribute_values[attribute] = sorted(attribute_values[attribute])
    
    return dataset, attribute_values

# MAIN FUNCTION
if __name__ == '__main__':

    # load dataset
    dataset, attribute_values = read_iris_dataset()

    # shuffle dataset randomly
    random.shuffle(dataset)

    # split dataset into training set (75%) and testing set (25%)
    split_point = int(len(dataset) * 0.75)
    training_set = dataset[:split_point]
    test_set = dataset[split_point:]

    # attributes
    attribute_list = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # build tree using ID3 algorithm
    tree = tree_node(training_set, attribute_list, attribute_values, 0, algorithm='C4.5')
    tree.build()

    print("Tree structure:")
    tree.print("")

    # test decision tree accuracy
    correct_predictions = 0
    for elem in test_set:
        if tree.predict(elem) == elem["species"]:
            correct_predictions += 1
    accuracy = correct_predictions / len(test_set)

    print("Accuracy:", accuracy)


# In[5]:


# Importing necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target, columns=["species"])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. CART using Gini impurity
cart = DecisionTreeClassifier(criterion='gini', random_state=42)
cart.fit(X_train, y_train)
cart_predictions = cart.predict(X_test)

# 2. ID3 using Entropy
id3 = DecisionTreeClassifier(criterion='entropy', random_state=42)
id3.fit(X_train, y_train)
id3_predictions = id3.predict(X_test)

# 3. C4.5 (Entropy + Pruning)
c45 = DecisionTreeClassifier(criterion='entropy', min_samples_split=10, random_state=42)
c45.fit(X_train, y_train)
c45_predictions = c45.predict(X_test)

# Function to visualize the decision tree
def visualize_tree(model, feature_names, model_name):
    plt.figure(figsize=(15,10))
    plot_tree(model, filled=True, feature_names=feature_names, class_names=iris.target_names)
    plt.title(f"{model_name} Decision Tree")
    plt.show()

# Visualize CART, ID3, and C4.5 decision trees
visualize_tree(cart, iris.feature_names, "CART")
visualize_tree(id3, iris.feature_names, "ID3")
visualize_tree(c45, iris.feature_names, "C4.5")

# Evaluation Function
def evaluate_model(y_test, predictions, model_name):
    print(f"\n{model_name} Performance:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions, target_names=iris.target_names))
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# Evaluate all models
evaluate_model(y_test, cart_predictions, "CART")
evaluate_model(y_test, id3_predictions, "ID3")
evaluate_model(y_test, c45_predictions, "C4.5")


# In[10]:


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree

# Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

# Create a DataFrame for easier manipulation
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# Split dataset into features (X) and target (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to train and evaluate a decision tree model
def train_and_evaluate_model(criterion_name, min_samples_split=None):
    # Initialize the Decision Tree Classifier
   
    if min_samples_split:
        clf = DecisionTreeClassifier(criterion=criterion_name, random_state=42, min_samples_split=min_samples_split)
    else:
        clf = DecisionTreeClassifier(criterion=criterion_name, random_state=42)
    # Train the model
    clf.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = clf.predict(X_test)
    
    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy with {criterion_name} (min_samples_split={min_samples_split}): {accuracy * 100:.2f}%')
    
    # Visualize the decision tree
    plt.figure(figsize=(12, 8))
    tree.plot_tree(clf, feature_names=iris['feature_names'], class_names=iris['target_names'].astype(str), filled=True)
    plt.title(f'Decision Tree using {criterion_name.upper()}')
    plt.show()

# 1. Implementing CART (Gini Index)
print("CART (Gini Index):")
train_and_evaluate_model('gini')

# 2. Implementing ID3 (Entropy)
print("ID3 (Entropy):")
train_and_evaluate_model('entropy')

# 3. Approximating C4.5 using Entropy + min_samples_split for pruning
print("C4.5 Approximation (Entropy + min_samples_split):")
train_and_evaluate_model('entropy', min_samples_split=10)


# In[20]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
iris_data = load_iris()
X, y = iris_data.data, iris_data.target

# Split data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers for CART, ID3, and C4.5
models = {
    "CART (Gini Index)": DecisionTreeClassifier(criterion='gini'),
    "ID3 (Information Gain)": DecisionTreeClassifier(criterion='entropy'),
    "C4.5 Approximation (Gain Ratio)": DecisionTreeClassifier(criterion='entropy',  splitter='best')
}

# Train, predict, and evaluate each model
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Visualize the decision tree
    plt.figure(figsize=(12, 8))
    plot_tree(clf, feature_names=iris_data.feature_names, class_names=iris_data.target_names, filled=True)
    plt.title(f"Decision Tree - {name}")
    plt.show()

    # Evaluate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Print metrics
    print(f"{name} Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")


# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target, columns=["species"])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert y_train and y_test to 1D arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# 1. CART using Gini impurity
cart = DecisionTreeClassifier(criterion='gini', random_state=42)
cart.fit(X_train, y_train)
cart_predictions = cart.predict(X_test)

# 2. ID3 using Entropy
id3 = DecisionTreeClassifier(criterion='entropy', random_state=42)
id3.fit(X_train, y_train)
id3_predictions = id3.predict(X_test)

# 3. C4.5 approximation (Entropy + Post-Pruning using ccp_alpha)
c45 = DecisionTreeClassifier(criterion='entropy',  splitter='best')
c45.fit(X_train, y_train)
c45_predictions = c45.predict(X_test)

# Function to visualize the decision tree
def visualize_tree(model, feature_names, model_name):
    plt.figure(figsize=(15, 10))
    plot_tree(model, filled=True, feature_names=feature_names, class_names=iris.target_names)
    plt.title(f"{model_name} Decision Tree")
    plt.show()

# Visualize CART, ID3, and C4.5 decision trees
visualize_tree(cart, iris.feature_names, "CART")
visualize_tree(id3, iris.feature_names, "ID3")
visualize_tree(c45, iris.feature_names, "C4.5")

# Evaluation Function
def evaluate_model(y_test, predictions, model_name):
    print(f"\n{model_name} Performance:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions, target_names=iris.target_names))
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%\n")

# Evaluate all models
evaluate_model(y_test, cart_predictions, "CART")
evaluate_model(y_test, id3_predictions, "ID3")
evaluate_model(y_test, c45_predictions, "C4.5")


# In[5]:


# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and predict with CART (Gini impurity), ID3 (Entropy), and C4.5 (Entropy with pruning)
cart = DecisionTreeClassifier(criterion='gini', random_state=42).fit(X_train, y_train)
id3 = DecisionTreeClassifier(criterion='entropy', random_state=42).fit(X_train, y_train)
c45 = DecisionTreeClassifier(criterion='entropy', splitter='best').fit(X_train, y_train)

cart_predictions = cart.predict(X_test)
id3_predictions = id3.predict(X_test)
c45_predictions = c45.predict(X_test)

# Function to visualize the decision tree
def visualize_tree(model, feature_names, model_name):
    plt.figure(figsize=(10, 7))
    plot_tree(model, filled=True, feature_names=feature_names, class_names=iris.target_names)
    plt.title(f"{model_name} Decision Tree")
    plt.show()

# Visualize CART, ID3, and C4.5 decision trees
visualize_tree(cart, iris.feature_names, "CART")
visualize_tree(id3, iris.feature_names, "ID3")
visualize_tree(c45, iris.feature_names, "C4.5")

# Function to display classification report
def evaluate_model(y_test, predictions, model_name):
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, predictions, target_names=iris.target_names))

# Evaluate all models
evaluate_model(y_test, cart_predictions, "CART")
evaluate_model(y_test, id3_predictions, "ID3")
evaluate_model(y_test, c45_predictions, "C4.5")


# ### 

# In[ ]:




