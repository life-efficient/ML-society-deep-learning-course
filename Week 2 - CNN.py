import torch
from torch.autograd import Variable
import torchvision
from torchvision import transforms, datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm

batch_size = 256
lr = 0.5
epochs = 1

training_data = datasets.MNIST(root='data/',
                                  transform=transforms.ToTensor(),
                                  train=True,
                                  download=True,
                               )

print(training_data[0])
# RUN here to show dataset format
plt.imshow(training_data[0][0][0])
plt.show()
# RUN here to visualise the data

test_data = datasets.MNIST(root='data/',
                           train=False,
                           transform=transforms.ToTensor())

# explain dataloader
# dataloader is a generator that can sample from the training set
training_samples = torch.utils.data.DataLoader(dataset=training_data,
                                          batch_size=batch_size,
                                          shuffle=True)

test_samples = torch.utils.data.DataLoader(dataset=test_data,
                                            batch_size=16,
                                           shuffle=False)

print('Number of training examples:', len(training_samples.dataset))
print('Number of test examples:', len(test_samples.dataset))

class convnet(torch.nn.Module):

    def __init__(self):
        super().__init__()
            # conv2d(in_channels, out_channels, kernel_size)
            # in_channels is the number of layers which it takes in (i.e.num color streams in 1st layer)
            # out_channels is the number of different filters that we use
            # kernel_size is the depthxwidthxheight of the kernel
        self.conv1 = torch.nn.Conv2d(1, 5, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(5, 10, kernel_size=3)
        #self.mp = torch.nn.MaxPool2d(2)    # takes the width of the kernel
        self.relu = F.relu
        self.dense = torch.nn.Linear(4840, 10)

    def forward(self, x):
        out1 = self.relu((self.conv1(x)))
        out2 = self.relu((self.conv2(out1)))

        todense = out2.view(x.size(0), -1)
        output = self.dense(todense)
        return F.log_softmax(output, dim=0)

m = convnet()


criterion = torch.nn.NLLLoss()
optimiser = torch.optim.SGD(m.parameters(), lr=lr)


def train():
    #m.train()
    fig = plt.figure(figsize=(10,20))
    ax = fig.add_subplot(111)
    ax.grid()
    plt.ion()
    plt.show()
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')

    costs = []
    loss = Variable(torch.Tensor([0]))

    for e in range(epochs):

        epoch_loss = 0

        # run through each batch, covering all data
        for i, (features, labels) in enumerate(training_samples):

            features, labels = Variable(features), Variable(labels)

            prediction = m(features)

            optimiser.zero_grad()
            loss = criterion(prediction, labels)
            loss.backward()
            optimiser.step()

            epoch_loss += loss

            # /batch size because this is summed
            costs.append(loss.data[0]/batch_size)
            ax.plot(costs, 'b')
            fig.canvas.draw()

            print('Epoch', e, '\tBatch', i, '\tLoss', loss.data[0])

            if i == 0:
                break
                #pass

        avg_loss = epoch_loss/(len(training_samples) * batch_size)
        #costs.append(avg_loss.data[0])

        print('Epoch', e, 'Average Loss:', avg_loss.data[0])
        #ax.plot(costs)
        #fig.canvas.draw()

train()

def test():
    print('\n\n\n')
    m.eval() # batchnorm and dropout behave differently
    correct = 0

    # test against training set
    for features, labels in training_samples:
        features, labels = Variable(features), Variable(labels)
        prediction = m(features)
        print(prediction.data.max(1))# max outputs over K
        pred = prediction.data.max(1)[1] # index of max output
        # gives us a binary vector with 1s where preds=labels
        correct += pred.eq(labels.data.view_as(pred)).sum()

    print('Training set accuracy:', correct/len(training_samples.dataset))


    correct = 0

    # test against test set
    for features, labels in test_samples:
        features, labels = Variable(features), Variable(labels)
        probabilities = m(features)
        prediction = probabilities.data.max(1)[1] # (1) - preserve the columns
                                                  # [1] get the argmax
        correct += prediction.eq(labels.data.view_as(prediction)).sum()

    print('Test set accuracy', correct/len(test_samples.dataset))

test()

#torch.save(m.state_dict(), 'latest_model.pt')
#m.eval()    # necessary after loading model weights back in because dropout and batch norm layers are in train mode by default
#m.load_state_dict('latest_model')