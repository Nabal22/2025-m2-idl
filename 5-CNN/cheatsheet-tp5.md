# TP-5 Cheatsheet — Réseaux Convolutifs (CNN)

---

## Sommaire

1. [Imports essentiels](#1-imports-essentiels)
2. [Opération de convolution 2D](#2-opération-de-convolution-2d)
3. [nn.Conv2d — syntaxe et paramètres](#3-nnconv2d--syntaxe-et-paramètres)
4. [Calcul de la taille de sortie](#4-calcul-de-la-taille-de-sortie)
5. [Pooling](#5-pooling)
6. [Architecture CNN typique](#6-architecture-cnn-typique)
7. [Batch Normalization](#7-batch-normalization)
8. [Empilage de couches avec nn.Sequential](#8-empilage-de-couches-avec-nnsequential)
9. [Flatten / view — du conv au linéaire](#9-flatten--view--du-conv-au-linéaire)
10. [Boucle d'entraînement complète](#10-boucle-dentraînement-complète)
11. [Référence rapide](#11-référence-rapide)

---

## 1. Imports essentiels

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

device = (
    "cuda:0" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
```

---

## 2. Opération de convolution 2D

La convolution 2D fait glisser un filtre (kernel) sur une image et calcule le **produit scalaire** entre le filtre et chaque patch d'image.

> Un filtre détecte un motif spécifique (bord horizontal, vertical, coin…). Un CNN apprend automatiquement les valeurs optimales de ces filtres.

| Paramètre | Description | Effet |
|---|---|---|
| `kernel_size` | Taille du filtre | Plus grand = champ récepteur plus large |
| `stride` | Pas de déplacement | Stride > 1 réduit la taille de sortie |
| `padding` | Zéros ajoutés en bordure | `padding=1` avec `kernel=3` préserve la taille |
| `in_channels` | Canaux en entrée | 1 (gris), 3 (RGB) |
| `out_channels` | Nombre de filtres appris | = nombre de feature maps en sortie |

```python
def conv2d(image, kernel, padding=1):
    img  = image.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
    kern = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, kH, kW]
    return F.conv2d(img, kern, padding=padding).squeeze()

# Filtre de détection de bord horizontal
top = torch.tensor([[-1, -1, -1], [1, 1, 1], [0, 0, 0]], dtype=torch.float32)
```

---

## 3. nn.Conv2d — syntaxe et paramètres

```python
nn.Conv2d(
    in_channels,   # canaux en entrée
    out_channels,  # nombre de filtres à apprendre
    kernel_size,   # int ou (H, W)
    stride=1,
    padding=0,
    bias=True
)

conv   = nn.Conv2d(1, 8, kernel_size=3, padding=1)              # préserve 28×28
conv_s = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)  # divise par 2
```

> Pour conserver H×W avec un kernel 3×3 : `padding = kernel_size // 2`.

### Nombre de paramètres

$$\text{params} = \text{out\_channels} \times (\text{in\_channels} \times k_H \times k_W + 1)$$

Exemple : `Conv2d(1, 8, 3)` → $8 \times (1 \times 9 + 1) = 80$ paramètres.

---

## 4. Calcul de la taille de sortie

$$\boxed{H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1}$$

avec $p$ = padding, $k$ = kernel\_size, $s$ = stride.

| $H_{in}$ | kernel | stride | padding | $H_{out}$ |
|---|---|---|---|---|
| 28 | 3 | 1 | 0 | 26 |
| 28 | 3 | 1 | 1 | **28** |
| 28 | 7 | 7 | 0 | 4 |
| 32 | 3 | 2 | 1 | 16 |

```python
def output_size(h_in, kernel, stride=1, padding=0):
    return (h_in + 2 * padding - kernel) // stride + 1
```

---

## 5. Pooling

### Max Pooling

Sélectionne la valeur **maximale** dans chaque fenêtre. Invariance aux petites translations, réduction spatiale.

```python
nn.MaxPool2d(kernel_size=2, stride=2)
F.max_pool2d(x, kernel_size=7, stride=7)   # 28×28 → 4×4
```

### Average et Global Average Pooling

```python
nn.AvgPool2d(kernel_size=2, stride=2)

# Global Average Pooling : [B, C, H, W] → [B, C]
x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).flatten(1)
```

> MaxPool préféré pour la classification. AvgPool utilisé dans ResNet, MobileNet.

---

## 6. Architecture CNN typique

### Flux

```
[B, C_in, H, W]
   ↓ Conv2d
[B, C_out, H', W']
   ↓ ReLU
[B, C_out, H', W']
   ↓ MaxPool2d
[B, C_out, H'', W'']
   ↓ Flatten
[B, C_out × H'' × W'']
   ↓ Linear
[B, num_classes]
```

### Implémentation TP-5 (MNIST 1 vs 8)

```python
class classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        # 28×28 → MaxPool(7,7) → 4×4 → 8×4×4 = 128
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))              # [B, 8, 28, 28]
        x = F.max_pool2d(x, kernel_size=7)     # [B, 8, 4, 4]
        x = x.view(x.size(0), -1)             # [B, 128]
        return F.log_softmax(self.fc(x), dim=1)
```

### CNN plus profond

```python
class DeepCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)   # [B, 32, 14, 14]
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)   # [B, 64, 7, 7]
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return F.log_softmax(self.fc2(x), dim=1)
```

---

## 7. Batch Normalization

Normalise les activations par mini-batch pour stabiliser l'entraînement.

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}, \quad y = \gamma \hat{x} + \beta$$

$\gamma$ et $\beta$ sont des paramètres appris. En inférence, PyTorch utilise les statistiques accumulées (`running_mean` / `running_var`).

```python
bn = nn.BatchNorm2d(num_features=32)  # = out_channels

# Ordre recommandé : Conv → BN → ReLU
x = F.relu(self.bn1(self.conv1(x)))
```

> Toujours appeler `model.eval()` avant l'inférence (active les stats mémorisées).

---

## 8. Empilage de couches avec nn.Sequential

```python
# Simple
model = nn.Sequential(nn.Linear(2, 50), nn.ReLU(), nn.Linear(50, 1))

# CNN avec blocs
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

> Préférer `nn.Module` dès que l'architecture a des connexions résiduelles ou des branches multiples.

---

## 9. Flatten / view — du conv au linéaire

```python
# Méthode 1 : view
x = x.view(x.size(0), -1)   # [B, C, H, W] → [B, C*H*W]

# Méthode 2 : flatten
x = x.flatten(1)

# Méthode 3 : dans Sequential
nn.Flatten()
```

> Astuce : `print(x.shape)` juste avant le flatten pour connaître `in_features` du Linear.

---

## 10. Boucle d'entraînement complète

```python
model     = classifier().to(device)
loss_fn   = nn.NLLLoss()   # avec log_softmax en sortie
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(model, loader, loss_fn, optimizer, n_epochs=1):
    model.train()
    for epoch in range(n_epochs):
        total_loss, correct, size = 0.0, 0, 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            total_loss += loss.item() * labels.size(0)
            correct    += (preds == labels).sum().item()
            size       += labels.size(0)
        print(f"Epoch {epoch+1}/{n_epochs} Loss: {total_loss/size:.4f} Acc: {correct/size:.4f}")

def test(model, loader, loss_fn):
    model.eval()
    total_loss, correct, size = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss    = loss_fn(outputs, labels)
            _, preds = torch.max(outputs, 1)
            total_loss += loss.item() * labels.size(0)
            correct    += (preds == labels).sum().item()
            size       += labels.size(0)
    print(f"Test Loss: {total_loss/size:.4f} Acc: {correct/size:.4f}")
```

---

## 11. Référence rapide

### Tableau des couches CNN

| Couche | Classe PyTorch | Forme entrée → sortie |
|---|---|---|
| Convolution 2D | `nn.Conv2d(Ci, Co, k, s, p)` | `[B,Ci,H,W]` → `[B,Co,H',W']` |
| Max Pooling | `nn.MaxPool2d(k, s)` | `[B,C,H,W]` → `[B,C,H/s,W/s]` |
| Batch Norm | `nn.BatchNorm2d(C)` | `[B,C,H,W]` → `[B,C,H,W]` |
| Aplatissement | `nn.Flatten()` | `[B,C,H,W]` → `[B,C·H·W]` |
| Couche linéaire | `nn.Linear(in, out)` | `[B,in]` → `[B,out]` |

### Formule taille de sortie

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1$$

### Règles mnémotechniques

| Objectif | Recette |
|---|---|
| Conserver H×W | `padding = kernel_size // 2` |
| Diviser H×W par 2 | `MaxPool2d(2, 2)` |
| Avec NLLLoss | sortie = `F.log_softmax(x, dim=1)` |
| Avec CrossEntropyLoss | sortie = logits bruts |

### Correspondance loss / sortie

| Tâche | Sortie | Loss |
|---|---|---|
| Multi-classes | `F.log_softmax(x, dim=1)` | `nn.NLLLoss()` |
| Multi-classes | logits bruts | `nn.CrossEntropyLoss()` |
| Binaire | `torch.sigmoid(x)` | `nn.BCELoss()` |
| Régression | scalaire | `nn.MSELoss()` |

### Flux de données TP-5

```
Input              [B,  1, 28, 28]
Conv2d(1,8,3,p=1)  [B,  8, 28, 28]   # 80 paramètres
ReLU               [B,  8, 28, 28]
MaxPool2d(7,7)     [B,  8,  4,  4]   # 28÷7 = 4
Flatten            [B, 128      ]    # 8×4×4 = 128
Linear(128, 2)     [B,   2      ]    # 258 paramètres
log_softmax        [B,   2      ]
```
