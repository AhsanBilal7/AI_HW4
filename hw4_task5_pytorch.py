import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import necessary library 
# if you need more libraries, just import them
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.nn.functional as F
# ......
# --- end of task --- #
# ---------------------------------------------------------------------------
import torch.nn as nn
import torch
# number of features (len of X cols)
input_dim = 8
# number of hidden layers
hidden_layers = 8
# number of classes (unique of y)
output_dim = 1
class Network(nn.Module):
    def __init__(self, input_dim=8, hidden_layers=64, dropout_rate=0.5):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_layers)
        self.linear2 = nn.Linear(hidden_layers, hidden_layers // 2)  # New layer with half the hidden size
        # self.linear3 = nn.Linear(hidden_layers // 2, 1)  # Output dimension is 1 for binary classification
        self.linear3 = nn.Linear(hidden_layers, 1)  # Output dimension is 1 for binary classification
        self.dropout = nn.Dropout(dropout_rate)  # Dropout to prevent overfitting

    def forward(self, x):
        x = F.relu(self.linear1(x))  # ReLU activation after the first layer
        x = self.dropout(x)  # Apply dropout after activation
        # x = F.relu(self.linear2(x))  # ReLU activation for the new layer
        x = self.linear3(x)  # No activation applied to the final output
        return x
    
my_method = Network()

criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogits for binary classification
optimizer = torch.optim.Adam(my_method.parameters(), lr=0.001)


def find_accuracy(inputs, labels, model):
    correct, total = 0, 0
    # no need to calculate gradients during inference
    with torch.no_grad():
        # calculate output by running through the network
        outputs = model(inputs)
        # get the predictions
        __, predicted = torch.max(outputs.data, 1)
        # update results
        total += labels.size(0)
        # print(predicted[15], labels[15])
        correct += (predicted == labels).sum().item()
    # print(f'Accuracy of the network on the {len(inputs)} test data: {100 * correct // total} %')
    accuracy = 100 * correct // total
    labels, outputs = labels.cpu().numpy(), outputs.cpu().numpy()
    auc = roc_auc_score(labels, outputs)
    return accuracy, auc
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

for  per in num_train_per: 
    # Move the model to CUDA

    # create training data and label
    num_train = int(n * per)
    sample_train = data[0:num_train, 0:-1]
    label_train = data[0:num_train, -1].reshape(-1, 1)  # Reshape to 2D for compatibility with BCE

    my_method = Network()

    model = LogisticRegression()
    my_method = my_method.cuda()

    # --- Your Task --- #
    # Implement a baseline method that standardly trains 
    # the model using sample_train and label_train
    # ......
    # ......
    # print(sample_train[:3], label_train[:3])
    # ......
    model.fit(sample_train, label_train.ravel())  # Use ravel() to flatten the labels    # Before converting the tensor to numpy, move it to CPU
    # try:
    #     print("try")
    #     sample_test = sample_test.to('cpu')           # Move tensor to GPU
    # except Exception as e:
    #     print("try")
    #     sample_test = sample_test           # Move tensor to GPU
    # Evaluate baseline model
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
    
    
    label_train_tensor = torch.tensor(label_train).float().cuda()
    sample_train_tensor = torch.tensor(sample_train).float().cuda()  # assumings
    sample_test_tensor = torch.tensor(sample_test).float().cuda()
    label_test_tensor = torch.tensor(label_test).float().cuda()
    # --- Your Task --- #
    epochs = 2
    for epoch in range(epochs):
        my_method.train()
        optimizer.zero_grad()
        outputs = my_method(sample_train_tensor)
        loss = criterion(outputs, label_train_tensor)  # Calculate the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

    acc_yours, auc_yours = find_accuracy(sample_test_tensor, label_test_tensor, my_method)
    # acc_yours_per.append(acc_yours)
    # auc_yours_per.append(auc_yours)  # display statistics
    # Now, implement your method 
    # Aim to improve AUC score of baseline 
    # while maintaining accuracy as much as possible 
    # ......
    # ......
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


