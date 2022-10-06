import torch
import torch.nn as nn
from torch.autograd import Variable

class MNIST_MLP(nn.Module):
    
    def __init__(self, layer_sizes=[784, 10]):
        super().__init__()
        self.layer_sizes = layer_sizes     
        self.fc1 = nn.Linear(layer_sizes[0], 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, layer_sizes[1])
        
        # Different activations that you can use in forward() method.
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # Switch from activation maps to vectors
        x = x.view(-1, self.layer_sizes[0])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        # x = self.relu(x) # For relu activation uncomment this line
        
        # You can add more layers here for stacked operations, e.g. x = self.fc2(x). 
        # You may have to define self.fc2 in init though.
        return x
    
    # This function maps the network on the device that is passed as argument.
    # If your device doesn't have a GPU, it set device='cpu'.
    def set_device(self, device):
        self.device = device
        self.to(self.device)
    
    # This function trains the model on the data passed as arguments.
    def fit(self, mnist_train_loader, num_epochs=1, mnist_valid_loader=None):
        train_loss_history = []
        train_acc_history = []
        valid_loss_history = []
        valid_acc_history = []
        
        for epoch in range(num_epochs):
            
            self.train() # Set to the training mode.
            iter_loss = 0
            iter_acc = 0
            for i, (items, classes) in enumerate(mnist_train_loader):
                items = Variable(items).to(self.device)
                classes = Variable(classes).to(self.device)

                self.optimizer.zero_grad()     # Clear off the gradients from any past operation
                outputs = self(items)      # Do the forward pass
                loss = self.criterion(outputs, classes) # Calculate the loss
                loss.backward()           # Calculate the gradients with help of back propagation
                self.optimizer.step()          # Ask the optimizer to adjust the parameters based on the gradients
                iter_loss += loss.data # Accumulate the loss
                iter_acc += (torch.max(outputs.data, 1)[1] == classes.data).sum()
                print("\r", i, "/", len(mnist_train_loader), ", Loss: ", loss.data/len(items), end="")
            train_loss_history += [iter_loss.cpu().detach().numpy()]
            train_acc_history += [(iter_acc/len(mnist_train_loader.dataset)).cpu().detach().numpy()]
            print("\tTrain: ", train_loss_history[-1], train_acc_history[-1], end="")
            
            self.eval() # Set to the evaluation mode.
            iter_loss = 0
            iter_acc = 0
            for i, (items, classes) in enumerate(mnist_valid_loader):
                items = Variable(items).to(self.device)
                classes = Variable(classes).to(self.device)

                outputs = self(items)      # Do the forward pass
                iter_loss += self.criterion(outputs, classes).data
                iter_acc += (torch.max(outputs.data, 1)[1] == classes.data).sum()
            valid_loss_history += [iter_loss.cpu().detach().numpy()]
            valid_acc_history += [(iter_acc/len(mnist_valid_loader.dataset)).cpu().detach().numpy()]
            print("\tValidation: ", valid_loss_history[-1], valid_acc_history[-1])
        
        return train_loss_history, train_acc_history, valid_loss_history, valid_acc_history
