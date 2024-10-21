import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import necessary library 
# if you need more libraries, just import them
from sklearn.linear_model import Ridge

# ......
# --- end of task --- #

# load a data set for regression
# in array "data", each row represents a community 
# each column represents an attribute of community 
# last column is the continuous label of crime rate in the community
data = np.loadtxt('crimerate.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# always use last 25% data for testing 
num_test = int(0.25*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]

def mse(predictions, targets):
    # return np.sqrt(((predictions - targets) ** 2).mean())
    return ((predictions - targets) ** 2).mean()

# --- Your Task --- #
# now, pick the percentage of data used for training 
# remember we should be able to observe overfitting with this pick 
# note: maximum percentage is 0.75 
per = 0.75
num_train = int(n*per)
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]
# --- end of task --- #


# --- Your Task --- #
# We will use a regression model called Ridge. 
# This model has a hyper-parameter alpha. Larger alpha means simpler model. 
# Pick 8 candidate values for alpha (in ascending order)
# Remember we should aim to observe both overfitting and underfitting from these values 
# Suggestion: the first value should be very small and the last should be large 
# alpha_vec = [0.001, 0.01, 0.1, 1.0, 1.5, 2.0, 2.1, 2.4]
alpha_vec = [0.001, 0.01, 0, 5, 10.0, 100.0, 1000.0, 10000.0]
# --- end of task --- #

er_train_alpha = []
er_test_alpha = []
for alpha in alpha_vec: 

    # pick ridge model, set its hyperparameter 
    model = Ridge(alpha = alpha)
    model.fit(sample_train, label_train)
    train_predictions = model.predict(sample_train)
    test_predictions = model.predict(sample_test)

    # Evaluate the model using Mean Squared Error (MSE) and R-squared (R²)

    # --- Your Task --- #
    # now, training your model using training data 
    # (sample_train, label_train)
    er_train = mse(label_train, train_predictions)
    er_test = mse(label_test, test_predictions)
    # ......
    # ......

    # now, evaluate training error (MSE) of your model 
    # store it in "er_train"
    # ......
    er_train_alpha.append(er_train)
    # now evaluate your testing error (MSE) and stores it in "er_test"
    # ......
    # ......
    er_test_alpha.append(er_test)
    # --- end of task --- #

    
plt.plot(alpha_vec,er_train_alpha, label='Training Error')
plt.plot(alpha_vec,er_test_alpha, label='Testing Error')

# Find and mark overfitting and underfitting points
min_test_error_idx = er_test_alpha.index(min(er_test_alpha))
overfitting_alpha = alpha_vec[min_test_error_idx]

plt.axvline(x=overfitting_alpha, color='r', linestyle='--', label=f'Min Test Error at α={overfitting_alpha}')
plt.scatter(overfitting_alpha, min(er_test_alpha), color='red', zorder=5)

plt.xlabel('Hyper-Parameter Alpha')
plt.ylabel('Prediction Error (MSE)')
plt.legend()
plt.show()
    


