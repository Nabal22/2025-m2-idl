# Stacking layers
_From [Dataflowr Module 5](https://dataflowr.github.io/website/modules/5-stacking-layers/) by Marc Lelarge_





```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

In TP-3, you implemented an MLP using PyTorch's `nn.Sequential`.

This works well for a simple sequence of operations, but for any model of reasonable complexity, it's best to write a subclass of `torch.nn.Module`.

Here is the same model written as a `torch.nn.Module`.


```python
import torch.nn as nn

model = torch.nn.Sequential(nn.Linear(2, 50), nn.ReLU(), nn.Linear(50, 1), nn.Sigmoid())
```

This is possible of a simple sequence of operations, but for any model of reasonable complexity, the best is to write a sub-class of `torch.nn.Module`.

Here is the same model written as a `torch.nn.Module`.


```python
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.sigmoid(x)
```

Inheriting from `torch.nn.Module` provides many mechanisms implemented in the superclass.

First, the `__call__` operator is redefined to call the forward method and run additional operations. The forward pass should thus be executed through this operator and not by calling forward explicitly.


```python
model = MLP()
input = torch.empty(12, 2).normal_()
output = model(input)
```

All Parameters added as class attributes are seen by Module.parameters().

As long as you use autograd-compliant operations, the backward pass is implemented automatically.

This is crucial to allow the update of the Parameters with your optimizer.

With a loss and an optimizer the training loop can use the usual four steps:
- forward
- loss
- backward
- step


```python
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
label = torch.zeros(12, 1)

output = model(input)
loss = loss_fn(output, label)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Now it's time to build more complex architectures!
