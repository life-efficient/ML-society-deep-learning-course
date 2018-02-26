import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd

#load dataset into pandas dataframe
df = pd.read_csv('cancer_data.csv')
df[['diagnosis']] = df['diagnosis'].map({'M': 0, 'B': 1}) #map into numeric data
df = df.sample(frac=1) #shuffle dataset

m = 450 #how many datapoints are we using as the training set

#put dataset into torch tensor so we can use then with torch
X = torch.Tensor(np.array(df[df.columns[2:-1]])) #pick our features from our dataset
Y = torch.Tensor(np.array(df[['diagnosis']])) #select out label

#split into training and testing data
x_train = Variable(X[:m])
y_train = Variable(Y[:m])

x_test = Variable(X[m:])
y_test = Variable(Y[m:])

#create model class - inherit useful functions and attributes from torch.nn.Module
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__() #call parent class initializer
        self.h1 = torch.nn.Linear(30, 10) #input layer to size 10 hidden layer
        self.out = torch.nn.Linear(10, 1) #hidden layer to single output

    #define the forward propagation/prediction equation of our model
    def forward(self, x):
        h1 = self.h1(x) #linear combination
        h1 = torch.nn.functional.relu(h1) #activation
        out = self.out(h1) #linear combination
        out = torch.nn.functional.sigmoid(out) #activation
        return out

#training hyper-parameters
no_epochs = 150
alpha = 0.003 #learning rate

mynet = Net() #create model from class
criterion = torch.nn.MSELoss() #define cost criterion
optimizer = torch.optim.Rprop(mynet.parameters(), lr=lr) #choose optimizer

#define graph for plotting costs
costs = [] #to store our calculated costs
plt.ion() #interactive update on
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_xlim(0, no_epochs)
plt.show()

#training loop
for epoch in range(no_epochs):

    #forward propagate - calulate our hypothesis
    h = mynet.forward(x_train)

    #calculate, plot and print cost
    cost = criterion(h, y_train)
    costs.append(cost.data[0])
    ax.plot(costs, 'b')
    fig.canvas.draw()    
    print('Epoch ', epoch, ' Cost: ', cost.data[0])

    #backpropagate + gradient descent step
    optimizer.zero_grad() #set gradients to zero
    cost.backward() #backpropagate to calculate derivatives
    optimizer.step() #update our weights

#test accuracy
test_h = mynet.forward(x_test) #predict values for out test set
test_h.data.round_() #round output probabilities to give us discrete predictions
correct = test_h.data.eq(y_test.data) #perform element-wise equality operation
accuracy = torch.sum(correct)/correct.shape[0] #calculate accuracy
print('Test accuracy: ', accuracy)

torch.save(mynet.state_dict(), 'mynet_trained') #save our model parameters

#mynet.load_state_dict(torch.load('mynet_trained')) #load in model parameters
