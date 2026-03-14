# TP: CNN for Binary Classification (1 vs 8)

_From [Dataflowr Module 6](https://dataflowr.github.io/website/modules/6-convolutional-neural-network/) by Marc Lelarge_

In this practical, you will build a Convolutional Neural Network (CNN) that learns filter weights to classify MNIST digits (1 vs 8).

## Setup


```python
%matplotlib inline
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
from matplotlib import pyplot as plt

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {device}")
```


```python
# plot multiple images
def plots(ims, interp=False, titles=None):
    if isinstance(ims, torch.Tensor):
        ims = ims.cpu().numpy()
    elif isinstance(ims, list) and len(ims) > 0 and isinstance(ims[0], torch.Tensor):
        ims = torch.stack(ims).cpu().numpy()
    mn, mx = ims.min(), ims.max()
    f = plt.figure(figsize=(12, 24))
    for i in range(len(ims)):
        sp = f.add_subplot(1, len(ims), i + 1)
        if not titles is None:
            sp.set_title(titles[i], fontsize=18)
        plt.imshow(ims[i], interpolation=None if interp else "none", vmin=mn, vmax=mx)


# plot a single image
def plot(im, interp=False):
    if isinstance(im, torch.Tensor):
        im = im.cpu().numpy()
    f = plt.figure(figsize=(3, 6), frameon=True)
    plt.imshow(im, interpolation=None if interp else "none")


plt.gray()
plt.close()
```

## Load MNIST Data


```python
root_dir = "./data/MNIST/"
train_set = torchvision.datasets.MNIST(root=root_dir, train=True, download=True)

# Extract images and labels
images = train_set.data.float() / 255
labels = train_set.targets

print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
```

The following lines of code implement a data loader for the train set and the test set. No modification is needed.


```python
eights = torch.stack([i for (i, l) in zip(images, labels) if l == 8])
ones = torch.stack([i for (i, l) in zip(images, labels) if l == 1])
```


```python
bs = 64

l8 = torch.tensor(0, dtype=torch.long)
eights_dataset = [[e.unsqueeze(0), l8] for e in eights]
l1 = torch.tensor(1, dtype=torch.long)
ones_dataset = [[e.unsqueeze(0), l1] for e in ones]
train_dataset = eights_dataset[1000:] + ones_dataset[1000:]
test_dataset = eights_dataset[:1000] + ones_dataset[:1000]

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=True)
```

## Task: Building a CNN

You will build a neural network that learns the weights of convolutional filters.

### Architecture:

The network should have:
1. **Convolutional layer**: 8 filters of size 3×3 (input: 1 channel, output: 8 channels)
2. **Max Pooling layer**: kernel size 7×7, stride 7 (reduces 28×28 to 4×4)
3. **Flatten**: converts [batch, 8, 4, 4] to [batch, 128]
4. **Linear layer**: maps 128 features to 2 classes (1 vs 8)

### TODO:

Complete the CNN class below. You'll need to:
- Set the correct `padding` value for the convolutional layer
- Implement the `forward` method with the correct sequence of operations

**Hints:**
- Use `F.max_pool2d(x, kernel_size=7, stride=7)` for max pooling
- Use `F.log_softmax(x, dim=1)` for the final output (works with NLLLoss)
- Don't forget to flatten between pooling and the linear layer!

**Documentation:**
- [nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [F.max_pool2d](https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool2d.html)


```python
class classifier(nn.Module):
    
    def __init__(self):
        super(classifier, self).__init__()
        # TODO: fill the missing padding value
        # Hint: what padding keeps the spatial dimensions unchanged?
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=?)
        self.fc = nn.Linear(in_features=128, out_features=2)
        
    def forward(self, x):
        # TODO: Implement your network here
        x = self.conv1(x)
        
        return x
```

## Test your model

Create an instance and test with a batch of 3 images.


```python
conv_class = classifier()
print(conv_class)
```


```python
# Test with a batch of 3 images
batch_3images = train_set.data[0:3].float().unsqueeze(1)
output = conv_class(batch_3images)
print(f"Input shape: {batch_3images.shape}")
print(f"Output shape: {output.shape}")
print(f"Output:\n{output}")
```

## Implement the Training Loop

Complete the training function below.

**Hints:**
- Use `model.train(True)` to set the model to training mode
- For each batch:
  1. Zero the gradients: `optimizer.zero_grad()`
  2. Forward pass: `outputs = model(inputs)`
  3. Compute loss: `loss = loss_fn(outputs, labels)`
  4. Backward pass: `loss.backward()`
  5. Update weights: `optimizer.step()`
- Track running loss and accuracy


```python
def train(model, data_loader, loss_fn, optimizer, n_epochs=1):
    model.train(True)
    loss_train = torch.zeros(n_epochs)
    acc_train = torch.zeros(n_epochs)

    for epoch_num in range(n_epochs):
        running_corrects = 0.0
        running_loss = 0.0
        size = 0

        for data in data_loader:
            inputs, labels = data
            bs = labels.size(0)

            # TODO: Implement the training step
            # 1. Zero gradients

            # 2. Forward pass

            # 3. Compute loss

            # 4. Backward pass

            # 5. Optimizer step

            # 6. Track statistics (running_loss and running_corrects)
            # Hint: use torch.max(outputs, 1) to get predictions

            size += bs

        epoch_loss = running_loss.item() / size
        epoch_acc = running_corrects.item() / size
        loss_train[epoch_num] = epoch_loss
        acc_train[epoch_num] = epoch_acc
        print(
            f"Epoch {epoch_num+1}/{n_epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}"
        )

    return loss_train, acc_train
```

## Setup Loss and Optimizer

Choose appropriate loss function and optimizer:
- For multi-class classification with log_softmax output, what do we need?
- Start with SGD optimizer with learning rate 1e-3


```python
conv_class = classifier()

# TODO: Choose the appropriate loss function
loss_fn = 

# TODO: Create SGD optimizer with learning_rate=1e-3
learning_rate = 1e-3
optimizer_cl = 
```

Train for 10 epochs


```python
l_t, a_t = train(conv_class, train_loader, loss_fn, optimizer_cl, n_epochs=10)
```

## Test Function

Evaluate the model on the test set.

**Hints:**
- Use `model.train(False)` or `model.eval()` to set evaluation mode
- Use `torch.no_grad()` to disable gradient computation
- Calculate test loss and accuracy


```python
def test(model, data_loader, loss_fn):
    model.train(False)

    running_corrects = 0.0
    running_loss = 0.0
    size = 0

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            bs = labels.size(0)

            # TODO: Implement testing

            size += bs

    print(
        f"Test - Loss: {running_loss / size:.4f} Acc: {running_corrects.item() / size:.4f}"
    )
```


```python
test(conv_class, test_loader, loss_fn)
```

Try [Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html) optimizer instead of SGD.

Reset the model and try training with Adam optimizer instead of SGD.


```python
# TODO: Create a new model and train with Adam optimizer
```

How many parameters did your network learn?


```python

```

## Analyze learned filters

Let's visualize what the network learned!


```python
# Count parameters
total_params = sum(p.numel() for p in conv_class.parameters())
print(f"Total parameters: {total_params}")

# Break down by layer
for name, param in conv_class.named_parameters():
    print(f"{name}: {param.numel()} parameters, shape {param.shape}")
```


```python
# View learned filters
for m in conv_class.children():
    print("Weights:", m.weight.data)
    print("Bias:", m.bias.data)
```


```python
# Extract and visualize the 8 learned filters
T_w = conv_class.conv1.weight.data
T_b = conv_class.conv1.bias.data

# Plot the 8 learned 3x3 filters
plots([T_w[i][0] for i in range(8)], titles=[f"Filter {i}" for i in range(8)])
```


```python

```
