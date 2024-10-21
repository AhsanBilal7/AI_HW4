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
print(n,p)
# always use last 25% data for testing 
num_test = int(0.25*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]
print(label_test.shape)
print("Total entries of 1 label", label_test[label_test[:,] == 1].shape)
print("Total entries of 0 label",label_test[label_test[:, ] == 0].shape)
# vary the percentage of data for training
num_train_per = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

acc_base_per = []
auc_base_per = []

acc_yours_per = []
auc_yours_per = []

svm_params = [
    {'C': 0.1, 'kernel': 'linear'},
    {'C': 1.0, 'kernel': 'linear'},
    {'C': 10.0, 'kernel': 'linear'},
    {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'},
    {'C': 10.0, 'kernel': 'rbf', 'gamma': 'scale'},
]

for  per in num_train_per: 
    # Move the model to CUDA

    # create training data and label
    num_train = int(n * per)
    sample_train = data[0:num_train, 0:-1]
    label_train = data[0:num_train, -1].reshape(-1, 1)  # Reshape to 2D for compatibility with BCE


    model = LogisticRegression()

    # --- Your Task --- #
    # Implement a baseline method that standardly trains 
    # the model using sample_train and label_train
    # ......
    # ......
    # print(sample_train[:3], label_train[:3])
    # ......
    model.fit(sample_train, label_train.ravel())  # Use ravel() to flatten the labels    # Before converting the tensor to numpy, move it to CPU

    if isinstance(sample_test, np.ndarray):  # Check if sample_test is a NumPy array
        test_prediction = model.predict(sample_test)
        # test_probs = model.predict_proba(sample_test)[:, 1]  # Get probabilities for the positive class
        test_probs = model.predict_proba(sample_test)  # Get probabilities for the positive class
    else:
        test_prediction = model.predict(sample_test.cpu().numpy())
        test_probs = model.predict_proba(sample_test.cpu().numpy())

    # evaluate model testing accuracy and stores it in "acc_base"
    # ......
    acc_base = accuracy_score(label_test, test_prediction) * 100
    max_indices = np.argmax(test_probs, axis=-1)  # axis=1 for probabilities over each class
    print(label_test.shape, test_prediction.shape, test_probs.shape, max_indices.shape, sample_train.shape)
    # print(label_test[0], np.argmax(test_probs[0], axis=-1))
    auc_base = roc_auc_score(label_test, max_indices)
    acc_base_per.append(acc_base)
    auc_base_per.append(auc_base)
    
    # evaluate model testing AUC score and stores it in "auc_base"
    # ......
    # --- end of task --- #
    
    
    # --- Your Task --- #

    clf = svm.SVC(probability=True, decision_function_shape='ovo')    # Fit the model to the training data
    clf.fit(sample_train, label_train.ravel())  # Ensure labels are flattened if necessary

    # Predict on training data (you might want to predict on test data instead)
    y_pred = clf.predict(sample_train)

    # Calculate accuracy
    acc_yours = accuracy_score(label_train, y_pred) * 100
    print(f"Accuracy: {acc_yours:.4f}")

    # Predict probabilities for the positive class
    y_pred_prob = clf.predict_proba(sample_train)[:, 1]  # Get probabilities for the positive class

    # Calculate AUC score
    auc_yours = roc_auc_score(label_train, y_pred_prob)
    print(f"AUC Score: {auc_yours:.4f}")
    # ......
    # evaluate model testing accuracy and stores it in "acc_yours"
    # ......
    acc_yours_per.append(acc_yours)
    # evaluate model testing AUC score and stores it in "auc_yours"
    # ......
    auc_yours_per.append(auc_yours)
    # --- end of task --- #
    

plt.figure()    
plt.plot(num_train_per,acc_base_per, label='Base Accuracy')
plt.plot(num_train_per,acc_yours_per, label='Your Accuracy')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Classification Accuracy')
plt.legend()


plt.figure()
plt.plot(num_train_per,auc_base_per, label='Base AUC Score')
plt.plot(num_train_per,auc_yours_per, label='Your AUC Score')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Classification AUC Score')
plt.legend()
plt.show()


