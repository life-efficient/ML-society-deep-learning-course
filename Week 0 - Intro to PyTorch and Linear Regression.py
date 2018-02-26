import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#dataset creation hyper-parameters
n = 2
m = 500

#training hyper-parameters
lr = 0.3
no_epochs = 150

#dataset creation
X = Variable(torch.rand(n, m))
Y = Variable(0.5*X.data[0, :] - 4.2*X.data[1, :] + 2)

w = Variable(torch.randn(1, n), requires_grad=True)
b = Variable(torch.randn(1), requires_grad=True)

#plot interactive mode on so it can update interactively
plt.ion()

#for plotting cost
costs = []
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_xlim(0, no_epochs)

#for plotting datapoints
ax1 = fig.add_subplot(122, projection='3d')
ax1.scatter(X[0, :].data, X[1, :].data, Y.data)
ax1.set_xlabel('Area of garden')
ax1.set_xlabel('Area of house')
ax1.set_xlabel('Price')

#for plotting hypothesis surface
x1 = np.arange(2)
x2 = np.arange(2)

x1, x2 = np.meshgrid(x1, x2)

plt.show()

for epoch in range(no_epochs):
    h = torch.mm(w, X) + b #calculate hypothesis
    cost = torch.sum((h-Y)**2)/(2*m) #calculate cost

    cost.backward() #calcualte partial derivatives

    w.data -= lr*w.grad.data #update weight values
    b.data -= lr*b.grad.data #update bias

    w.grad.data.zero_() #reset weight gradients to zero
    b.grad.data.zero_() #reset bias gradients to zero

    costs.append(cost.data[0]) #append epoch cost to the list of costs

    #calculate hypothesis surface
    ax1.view_init(azim=epoch) #set viewing angle of 3d plot
    y = x1*w.data[0][0] + x2*w.data[0][1] + b.data[0] #calculate predicted y values for surface
    s = ax1.plot_surface(x1, x2, y, color=(0, 1, 1, 0.5)) #plot hypothesis surface

    ax.plot(costs)#plot costs

    fig.canvas.draw() #update the axes on the figure

    s.remove() #remove the plotted surface
    
    print('Epoch: ', epoch, ' Cost: ', cost.data[0]) #print out loss for the epoch

print('Optimised weights: ', w.data) 
print('Optimised bias: ', b.data)

#make a prediction
test = torch.Tensor([[5], [10]])
out = torch.mm(w.data, test) + b.data
print('Output for [5, 10] = ', out)

