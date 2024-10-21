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

def mse(predictions, targets):
    # return np.sqrt(((predictions - targets) ** 2).mean())
    return ((predictions - targets) ** 2).mean()

def manual_kfold_split(samples, labels, k):
    fold_size = len(samples) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        if i == k - 1:
            # Last fold includes any leftover data
            end = len(samples)
        else:
            end = (i + 1) * fold_size
        folds.append((samples[start:end], labels[start:end]))
    return folds

# always use last 25% data for testing 
num_test = int(0.25*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]

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
# Pick 5 candidate values for alpha (in ascending order)
# Remember we should aim to observe both overfitting and underfitting from these values 
# Suggestion: the first value should be very small and the last should be large 
# alpha_vec = [0.5, 0.5, 0.5, 0.5, 0.5]
# alpha_vec = [0.001, 0.01, 0.1, 0.5, 1.0]
alpha_vec = [0.5,1,1.5, 2,2.5]
# --- end of task --- #

er_train_alpha = []
er_test_alpha = []
er_valid_alpha = []
er_train_alpha = []

# generate the folds
k = 5
folds = manual_kfold_split(sample_train, label_train, k)
for alpha in alpha_vec: 

    # pick ridge model, set its hyperparameter 
    model = Ridge(alpha = alpha)
    validation_errors = []
    training_errors = []
    # --- Your Task --- #
    for i in range(k):
        # Create the validation set and the training set for this fold
        valid_samples, valid_labels = folds[i]

        # Combine the remaining folds to create the training set
        train_samples = np.concatenate([folds[j][0] for j in range(k) if j != i])
        train_labels = np.concatenate([folds[j][1] for j in range(k) if j != i])

        # Train the Ridge model with the current alpha
        model = Ridge(alpha=alpha)
        model.fit(train_samples, train_labels)

        # Predict on the validation set and compute the error
        valid_predictions = model.predict(valid_samples)
        valid_error = mse(valid_labels, valid_predictions)
        train_predictions = model.predict(train_samples)
        train_error = mse(train_labels, train_predictions)
        
        training_errors.append(train_error)
        validation_errors.append(valid_error)

    # Average validation error for the current alpha
    er_valid = np.mean(validation_errors)
    er_train = np.mean(training_errors)
    # er_valid_alpha.append(er_valid)
    # now implement k-fold cross validation 
    # on the training set (which means splitting 
    # training set into k-folds) to get the 
    # validation error for each candidate alpha value 
    # store it in "er_valid"
    # ......
    # ......
    # ......
    # ......
    er_valid_alpha.append(er_valid)
    er_train_alpha.append(er_train)
    # --- end of task --- #

# Identify overfitting by checking for points where testing error increases or remains high
# for i, (train_error, test_error, perc) in enumerate(zip(er_train_per, er_test_per, num_train_per)):
#     plt.text(perc, train_error, f"{train_error:.3f}", fontsize=8, ha='right')
plt.plot(alpha_vec, er_valid_alpha, label='Validation Error')
plt.plot(alpha_vec, er_train_alpha, label='Training Error')
optimal_point_alpha = alpha_vec[er_valid_alpha.index(min(er_valid_alpha))]
min_error_point =optimal_point_alpha, min(er_valid_alpha) 
plt.scatter(min_error_point[0], min_error_point[1], color='red', zorder=5)

plt.text(min_error_point[0], min_error_point[1], f"Minimum Error: {min(er_valid_alpha):.3f}", fontsize=8, ha='right')

plt.xlabel('Hyper Parameter')
plt.ylabel('Validation Error')
plt.title('Hyper parameter vs validation error')
plt.legend()
plt.grid(True)
plt.show()
print(alpha_vec)
print(er_valid_alpha)
# Now you should have obtained a validation error for each alpha value 
# In the homework, you just need to report these values

# The following practice is only for your own learning purpose.
# Compare the candidate values and pick the alpha that gives the smallest error 
# set it to "alpha_opt"

alpha_opt = ...

# now retrain your model on the entire training set using alpha_opt 
# then evaluate your model on the testing set 
model = Ridge(alpha = alpha_opt)
# ......
# ......
# ......
er_train = ...
er_test = ...


