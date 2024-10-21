import numpy as np
import matplotlib.pyplot as plt


# --- Your Task --- #
# import necessary library 
# if you need more libraries, just import them
from sklearn.linear_model import LogisticRegression
from sklearn import  metrics

# ......
# --- end of task --- #


# load a data set for classification 
# in array "data", each row represents a patient 
# each column represents an attribute of patients 
# last column is the binary label: 1 means the patient has diabetes, 0 means otherwise
data = np.loadtxt('diabetes.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# always use last 25% data for testing 
num_test = int(0.25*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]


# --- Your Task --- #
# now, vary the percentage of data used for training 
# pick 8 values for array "num_train_per" e.g., 0.5 means using 50% of the available data for training 
# You should aim to observe overiftting (and normal performance) from these 8 values 
# Note: maximum percentage is 0.75
num_train_per = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
# --- end of task --- #

er_train_per = []
er_test_per = []
for per in num_train_per: 

    # create training data and label 
    num_train = int(n*per)
    sample_train = data[0:num_train,0:-1]
    label_train = data[0:num_train,-1]
    
    # we will use logistic regression model 
    model = LogisticRegression()
    
    # --- Your Task --- #
    # now, training your model using training data 
    model.fit(sample_train, label_train)
    train_prediction = model.predict(sample_train)
    training_accuracy = metrics.accuracy_score(label_train, train_prediction)*100
    training_error = 100 - training_accuracy
    # (sample_train, label_train)
    # ......
    # ......

    # now, evaluate training error (not MSE) of your model 
    test_prediction = model.predict(sample_test)
    testing_accuracy = metrics.accuracy_score(label_test, test_prediction)*100
    testing_error = 100 - testing_accuracy
    # store it in "er_train"
    # ......
    er_train_per.append(training_error)
    
    # now, evaluate testing error (not MSE) of your model 
    # store it in "er_test"
    # ......
    er_test_per.append(testing_error)
    # --- end of task --- #
    
overfitting_point = None
for i in range(1, len(er_test_per)):
    if er_test_per[i] > er_test_per[i-1]:
        overfitting_point = num_train_per[i-1]
        break
plt.figure()    
plt.plot(num_train_per,er_train_per, label='Training Error')
plt.plot(num_train_per,er_test_per, label='Testing Error')
# Highlight the overfitting point
if overfitting_point:
    plt.axvline(x=overfitting_point, color='r', linestyle='--', label=f'Overfitting starts at {overfitting_point * 100:.0f}%')
    plt.scatter(overfitting_point, er_test_per[num_train_per.index(overfitting_point)], color='red', zorder=5)
# print(overfitting_point)
plt.xlabel('Percentage of Training Data')
plt.ylabel('Classification Error')
plt.legend()
plt.show()


