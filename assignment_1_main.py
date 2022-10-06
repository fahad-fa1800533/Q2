import torch
import torch.nn as nn
# Torchvision module contains various utilities, classes, models and datasets 
# used towards computer vision usecases
from torchvision import datasets
from torchvision import transforms
from mlp_model import MNIST_MLP
import matplotlib.pyplot as plt

def mnist_loader(batch_size=512, classes=None):
    transform=transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_valid = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Select the classes which you want to train the classifier on.
    if classes is not None:
        mnist_train_idx = (mnist_train.targets == -1)
        mnist_valid_idx = (mnist_valid.targets == -1)
        for class_num in classes:
            mnist_train_idx |= (mnist_train.targets == class_num)
            mnist_valid_idx |= (mnist_valid.targets == class_num) 
        
        mnist_train.targets = mnist_train.targets[mnist_train_idx]
        mnist_valid.targets = mnist_valid.targets[mnist_valid_idx]
        mnist_train.data = mnist_train.data[mnist_train_idx]
        mnist_valid.data = mnist_valid.data[mnist_valid_idx]
    
    mnist_train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=1)
    mnist_valid_loader = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size, shuffle=True, num_workers=1)
    return mnist_train_loader, mnist_valid_loader

def main():
    # Load Data and Creat Data Loader based on Batch Size
    batch_size = 512 # Reduce this if you get out-of-memory error
    mnist_train_loader, mnist_valid_loader = mnist_loader(batch_size=batch_size)

    # Check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # The model
    model = MNIST_MLP(layer_sizes=[784, 10])
    model.set_device(device)
    # Our loss function and Optimizer
    model.criterion = nn.CrossEntropyLoss()
    model.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) #lr is the learning_rate

    # Train model for 2 epochs
    tlh, tah, vlh, vah = model.fit(mnist_train_loader, num_epochs=20, mnist_valid_loader=mnist_valid_loader)

    # Plot the results as a graph and save the figure.
    plt.figure()
    plt.plot(tah, label='Train Accuracy')
    plt.plot(vah, label='Validation Accuracy')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.savefig('Intial_Run.pdf')
    
    
if __name__=='__main__':
    main()