import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import necessary library 
# if you need more libraries, just import them
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import torch.nn.functional as F
import xgboost as xgb
import lightgbm as lgb
# ......
# --- end of task --- #
# ---------------------------------------------------------------------------
import torch.nn as nn
import torch
# number of features (len of X cols)

# ---------------------------------------------------------------------------

# load an imbalanced data set 
# there are 50 positive class instances 
# there are 500 negative class instances 
data = np.loadtxt('diabetes_new.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)
training_size = 0.75  # Set the training size to 0.75
num_train = int(n * training_size)  # Calculate number of training samples
sample_train = data[0:num_train, :-1]
label_train = data[0:num_train, -1].reshape(-1, 1)  # Reshape to 2D for compatibility

svm_params = [
    {'C': 0.1, 'kernel': 'linear'},
    {'C': 1.0, 'kernel': 'linear'},
    {'C': 10.0, 'kernel': 'linear'},
    {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'},
    {'C': 10.0, 'kernel': 'rbf', 'gamma': 'scale'},
]

accuracies = []

for params in svm_params:
    clf = svm.SVC(probability=True, **params)  # Use dictionary unpacking for parameters
    clf.fit(sample_train, label_train.ravel())  # Fit the model

    # Predict on training data
    y_pred = clf.predict(sample_train)
    acc_yours = accuracy_score(label_train, y_pred) * 100  # Calculate accuracy

    accuracies.append(acc_yours)  # Store the accuracy

param_labels = [f"C={p['C']}, kernel={p['kernel']}" for p in svm_params]
plt.figure(figsize=(10, 6))
plt.bar(param_labels, accuracies, color='skyblue')
plt.title('SVM Accuracy for Different Hyperparameters (Training Size = 0.75)')
plt.xlabel('Hyperparameters')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)  # Adjust y-axis for better visibility
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()  # Adjust layout
plt.show()
