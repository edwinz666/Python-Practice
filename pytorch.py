# %%    ########## TENSORS ##########
import torch
import numpy as np



##### Initializing a Tensor #####

### Directly from data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

### From Numpy Array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

### From another Tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

### With random or constant values
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")




##### Attributes of a sensor
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")



##### Operations on Tensors

#  https://pytorch.org/docs/stable/torch.html

## Default to running on CPU for operations
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# numpy-like indexing/slicing
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
# all dimensions up to that point, works in the middle as well
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)


# concatenate tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1.size())
print(t1)
# another method torch.stack() introduces new dimension 
# https://pytorch.org/docs/stable/generated/torch.stack.html



### Arithmetic operations
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)


### Extract item (similar to polars .item())
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


## In-place operations denoted by _ suffix
# e.g. x.copy_(y), x.t_(), will change x.
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)




##### Bridge with Numpy

# Tensors on CPU and Numpy Arrays share memory locations
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# change in tensor changes the numpy array too
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# vice versa also applies
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")





# %%    ########## DATASETS & DATALOADERS ##########
import torch
from torch.utils.data import (Dataset, DataLoader)
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

### Load data
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

##### Example with polars
import polars as pl

# Step 1: Load the data into a Polars DataFrame
df = pl.scan_csv("data/Hitters.csv").collect()
df

# Step 2: Convert the DataFrame into numpy arrays and then tensors
features = df.select(['AtBat', 'HmRun', 'Walks']).to_numpy()
target = df.select('Hits').to_numpy()

features_tensor = torch.tensor(features, dtype=torch.float32)
target_tensor = torch.tensor(target, dtype=torch.float32)

# Step 3: Define a custom PyTorch dataset class
class CustomDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

# Step 4: Create a dataset and DataLoader instance
dataset = CustomDataset(features_tensor, target_tensor)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 5: Iterate through the DataLoader
for batch_features, batch_target in data_loader:
    print(batch_features, batch_target)
    # Place training loop here
   
   
   
   
### Iterating and visualizing dataset
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()




##### Creating a Custom Dataset for your files 
import os
import pandas as pd
from torchvision.io import read_image    

# must implement __init__, __len__, and __getitem__
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    



# prepare data for training with DataLoaders
# 'Dataset' abstract class retrieves dataset feature/labels one sample at a time
# 'DataLoader' allows samples in 'minibatches', reshuffling data at every epoch
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Iterate through DataLoader
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
# collapse the first dimension (from (1,28,28) to (28,28))
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")






# %%    ########## TRANSFORMS ##########
import torch
from torchvision import datasets
# torchvision.transforms module offers several commonly-used transforms out of the box
from torchvision.transforms import ToTensor, Lambda


# labels are integers --> require one-hot encoded tensors
# features are PIL Image format --> require normalized tensors
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    # converts NumPy array or PIL image into a FloatTensor
    # also scales image's pixel intensity values in range [0,1]
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float)
                            # scatter along 0th axis (of the zeros vec)
                            # index to set = torch.tensor(y)
                            # set the value of 1 there
                            .scatter_(0, torch.tensor(y), value=1))
)





# %%    ########## BUILD THE NEURAL NETWORK ##########

import os
import torch
# nn provides all building blocks required for a neural network
# every module in PyTorch subclasses this module
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Get Device for Training (use GPU/hardware accelerator if can)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")




##### Defining the Class

# subclasses nn.Module
class NeuralNetwork(nn.Module):
    # initialise neural network layers
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # nn.Linear(in_features, out_features, bias=True)
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    # required for subclasses of nn.Module
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# create instance of NeuralNetwork, move to device, and print structure
model = NeuralNetwork().to(device)
print(model)

# the '1' for dim=0 represents the batch num, which is the
# format generally required
X = torch.rand(1, 28, 28, device=device)
X
logits = model(X)
logits
# transform into class probabilities
pred_probab = nn.Softmax(dim=1)(logits)
pred_probab
# find index with highest probability
y_pred = pred_probab.argmax(1)
y_pred
print(f"Predicted class: {y_pred}")




##### Model Layers

# sample minibatch of 3 28x28 images
input_image = torch.rand(3,28,28)
print(input_image.size())

# initialize nn.Flatten layer
flatten = nn.Flatten()
# convert each 28x28 image into 784 pixel values
# (maintains the 'minibatch' dim=0)
flat_image = flatten(input_image)
print(flat_image.size())


# linear layer that applies linear transformation using
# its stored weights and biases
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
hidden1
print(hidden1.size())

# non-linear activations
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")


# nn.Sequential - 
# ordered container of modules
# data is passed through modules in defined order
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
pred_probab



##### Model Parameters, using nn.Module
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")






# %%    ########## AUTOMATIC DIFFERENTIATION - torch.autograd ##########
import torch


x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
# can also set x.requires_grad_(True) in future if not done here
w = torch.randn(5, 3, requires_grad=True)
w
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
z
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
loss

# compute function in forward direction,
# compute derivative during backpropagation step
# more info https://pytorch.org/docs/stable/autograd.html#function
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# how to get derivatives, need to call loss.backward() first
# need to pass retain_graph=True to .backward() if we need
# several backwards calls on same graph
loss.backward()
print(w.grad)
print(b.grad)


##### Disabling Gradient Tracking
### Mostly used for after we have trained model and just need
### it for forward computations
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# alternative way using .detach()
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)






# %%    ########## OPTIMIZING MODEL PARAMETERS ##########

#### Load previous code
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()


##### Hyperparameters


# Number of Epochs - the number times to iterate over the dataset
# Batch Size - the number of data samples propagated through the network 
#               before the parameters are updated
# Learning Rate - how much to update models parameters at each 
#               batch/epoch. Smaller values yield slow learning speed,
#                  while large values may result in unpredictable 
#                   behavior during training.
learning_rate = 1e-3
batch_size = 64
epochs = 5
l1_lambda = 0.01
l2_lambda = 0.01

###### Optimization loop
### Each iteration of the optimization loop is called an epoch.

# Each epoch consists of two main parts:
#   The Train Loop - iterate over the training dataset and try to 
#                       converge to optimal parameters.
#   The Validation/Test Loop - iterate over the test dataset to 
#                       check if model performance is improving.


### Loss Function
# Regression - nn.MSELoss(), 
# Classification - nn.NLLLoss() (negative loglikelihood),
#                  nn.LogSoftMax()
#                  nn.CrossEntropyLoss() (combines the above 2)

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

### Optimizer
# Defines how model parameters are adjusted at each training step to
# reduce model error
# different optimizers: https://pytorch.org/docs/stable/optim.html

# initialise Stochastic Gradient Descent optimizer
# with model parameters and learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# inside training loop, optimization occurs in 3 steps:
# 1. Call optimizer.zero_grad() to reset the gradients of model parameters. 
#       Gradients by default add up; to prevent double-counting, 
#       we explicitly zero them at each iteration.
# 2. Backpropagate the prediction loss with a call to loss.backward(). 
#       PyTorch deposits the gradients of the loss w.r.t. each parameter.
# 3. Once we have our gradients, we call optimizer.step() to adjust the 
#       parameters by the gradients collected in the backward pass.





######## Full Implementation ########

# loop over optimization code
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        ### in PyTorch, when you add terms to the loss function, they are treated as part of the 
        ### computational graph. This means that PyTorch can compute the derivative of the entire 
        ### loss function, including the regularization terms, when you call loss.backward()
        # Add L2 regularization (Ridge) term
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss += l2_lambda * l2_norm

        # Add L1 regularization (Lasso) term
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss += l1_lambda * l1_norm
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# evaluate model performance against test data
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")




# initialise loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# pass it into training loop
# can increase #epochs for potentially better performance.
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")






# %%    ########## SAVE AND LOAD THE MODEL ##########
import torch
import torchvision.models as models

# PyTorch models store the learned parameters in an internal state dictionary,
# called state_dict. These can be persisted via the torch.save method:
# model = models.vgg16(weights='IMAGENET1K_V1')

# save the model weights earlier
torch.save(model.state_dict(), 'model_weights.pth')
# save the model weights and the structure
torch.save(model, 'model.pth')

# Define model architecture (create instance of same model)
model_load = NeuralNetwork()

# Load parameters using load_state_dict
# weights_only=True is best practice, which limits functions executed to
# only those necessary to load the weights, rather than also the model structure
# Generally want to define model architecture explicitly (like we have just above),
# and then load the weights into that instantiated model structure, rather than
# save the model weights AND the structure and load with weights_only=False
model_load.load_state_dict(torch.load('model_weights.pth', weights_only=True))
# important to call this before prediction to set
# dropout and batch normalization layers to evaluation mode,
# otherwise inconsistent results potentially
model_load.eval()

# Extract a sample and predict on it
for batch_idx, (images, labels) in enumerate(train_dataloader):
    # images is a batch of images, labels is a batch of labels
    # To extract the first sample from this batch
    sample_image = images[0]  # First image in the batch
    sample_label = labels[0]  # First label in the batch

    # Add a batch dimension (not necessary in this case?)
    sample_image = sample_image.unsqueeze(0)
    
    print(f"Sample Image Shape: {sample_image.shape}")
    print(f"Sample Label: {sample_label.item()}")  # Convert tensor to Python number
    
    # Perform inference
    with torch.no_grad():  # No need to compute gradients for inference
        output = model(sample_image)
        _, predicted = torch.max(output, 1)  # Get the index of the max log-probability
    
    print(f"True Label: {sample_label.item()}")
    print(f"Predicted Label: {predicted.item()}")
    
    # Break after the first batch to get only one sample
    break

# rows correspond to each observation in the batch,
# cols correspond to output (10 categories)
output.shape
output
predicted.shape
predicted

# transform into class probabilities
pred_probab = nn.Softmax(dim=1)(output)
pred_probab
# find index with highest probability
y_pred = pred_probab.argmax(1)
y_pred.item()



# %%    ########## PYTORCH CUSTOM OPERATORS ##########
# https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html
