import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import necessary library 
# if you need more libraries, just import them
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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

# --- Your Task --- #
# now, vary the percentage of data used for training 
# pick 8 values for array "num_train_per" e.g., 0.5 means using 50% of the available data for training 
# You should aim to observe overiftting (and normal performance) from these 8 values 
# Note: maximum percentage is 0.75
num_train_per = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
# --- end of task --- #

er_train_per = []
er_test_per = []
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

for per in num_train_per: 

    # create training data and label 
    num_train = int(n*per)
    sample_train = data[0:num_train,0:-1]
    label_train = data[0:num_train,-1]
    
    # we will use linear regression model 
    model = LinearRegression()
    model.fit(sample_train, label_train)
    train_predictions = model.predict(sample_train)
    test_predictions = model.predict(sample_test)

    # Evaluate the model using Mean Squared Error (MSE) and R-squared (R²)

    # --- Your Task --- #
    # now, training your model using training data 
    # (sample_train, label_train)
    er_train = rmse(label_train, train_predictions)
    er_test = rmse(label_test, test_predictions)
    # ......
    # ......

    # now, evaluate training error (MSE) of your model 
    # store it in "er_train"
    # ......
    er_train_per.append(er_train)
    
    # now, evaluate testing error (MSE) of your model 
    # store it in "er_test"
    # ......
    er_test_per.append(er_test)
    # --- end of task --- #
plt.plot(num_train_per,er_train_per, label='Training Error')
plt.plot(num_train_per,er_test_per, label='Testing Error')

# Identify overfitting point (where testing error starts increasing)
overfitting_point = None
for i in range(1, len(er_test_per)):
    if er_test_per[i] > er_test_per[i-1]:
        overfitting_point = num_train_per[i]
        break

# Highlight the overfitting point
if overfitting_point:
    plt.axvline(x=overfitting_point, color='r', linestyle='--', label=f'Overfitting starts at {overfitting_point * 100:.0f}%')
    plt.scatter(overfitting_point, er_test_per[num_train_per.index(overfitting_point)], color='red', zorder=5)

# Identify overfitting by checking for points where testing error increases or remains high
for i, (train_error, test_error, perc) in enumerate(zip(er_train_per, er_test_per, num_train_per)):
    plt.text(perc, train_error, f"{train_error:.3f}", fontsize=8, ha='right')
    plt.text(perc, test_error, f"{test_error:.3f}", fontsize=8, ha='right')

plt.xlabel('Percentage of Training Data')
plt.ylabel('Prediction Error (RMSE)')
plt.title('Training vs Testing Error to Identify Overfitting')
plt.legend()
plt.grid(True)
plt.show()

# plt.xlabel('Percentage of Training Data')
# plt.ylabel('Prediction Error (MSE)')
# plt.show()
# plt.legend()
    


