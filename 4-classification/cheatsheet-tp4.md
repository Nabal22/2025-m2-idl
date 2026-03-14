# TP-4 Cheatsheet — Classification & MLP

---

## Table des matières

1. [Perceptron multicouche (MLP) — Architecture](#1-perceptron-multicouche-mlp--architecture)
2. [Fonctions d'activation](#2-fonctions-dactivation)
3. [nn.Sequential et définition d'un réseau](#3-nnsequential-et-définition-dun-réseau)
4. [Fonctions de perte pour la classification](#4-fonctions-de-perte-pour-la-classification)
5. [Softmax et Log-Softmax](#5-softmax-et-log-softmax)
6. [One-hot encoding vs indices de classes](#6-one-hot-encoding-vs-indices-de-classes)
7. [Métriques de classification](#7-métriques-de-classification)
8. [Boucle d'entraînement complète avec validation](#8-boucle-dentraînement-complète-avec-validation)
9. [Référence rapide](#9-référence-rapide)

---

## 1. Perceptron multicouche (MLP) — Architecture

Un MLP est un réseau de neurones **entièrement connecté** (fully connected) composé de :
- Une **couche d'entrée** : reçoit les données aplaties
- Une ou plusieurs **couches cachées** : apprennent des représentations intermédiaires
- Une **couche de sortie** : produit les scores par classe

### Pourquoi des couches cachées ?

Sans non-linéarité, empiler des couches linéaires est équivalent à une seule couche linéaire :

$$L_1(L_2(x)) = W_1(W_2 x + b_2) + b_1 = (W_1 W_2)x + (W_1 b_2 + b_1) = W_3 x + b_3$$

Les **fonctions d'activation non linéaires** entre les couches donnent au réseau sa capacité d'expression.

### Architecture MLP pour CIFAR-10

```
Entrée : image 32x32x3 → Flatten → 3072 valeurs
   ↓
Linear(3072 → 1000)   [3 073 000 paramètres]
   ↓
ReLU
   ↓
Linear(1000 → 10)     [10 010 paramètres]
   ↓
Sortie : 10 scores (logits)
```

> **Total de paramètres** : environ 3,08 millions. Chaque couche `Linear(in, out)` a `in × out + out` paramètres (poids + biais).

### Dimensions des tenseurs (batch B)

| Étape | Shape |
|---|---|
| Image brute (batch) | `[B, 3, 32, 32]` |
| Après Flatten | `[B, 3072]` |
| Après Linear(3072, 1000) | `[B, 1000]` |
| Après Linear(1000, 10) | `[B, 10]` |

```python
total_params = sum(p.numel() for p in model.parameters())
print(f"Nombre total de paramètres : {total_params:,}")
```

---

## 2. Fonctions d'activation

### ReLU

$$\text{ReLU}(x) = \max(0, x)$$

```python
import torch.nn.functional as F
import torch.nn as nn

F.relu(x)
nn.ReLU()
```

Évite le problème de gradient évanescent. Fonction d'activation la plus utilisée.

### Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}} \in (0, 1)$$

```python
torch.sigmoid(x)
nn.Sigmoid()
```

Utilisée en **classification binaire** (sortie = probabilité). Décision : classe 1 si `σ(x) > 0.5`.

### Softmax

$$\text{softmax}(l)_k = \frac{e^{l_k}}{\sum_{j=0}^{K-1} e^{l_j}}$$

```python
F.softmax(logits, dim=1)     # pour un batch
nn.Softmax(dim=1)
```

Produit une **distribution de probabilité** sur $K$ classes. Toutes les sorties $\in (0,1)$ et leur somme vaut 1.

### Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \in (-1, 1)$$

```python
torch.tanh(x)
nn.Tanh()
```

### Comparaison

| Fonction | Sortie | Usage typique | Avantages |
|---|---|---|---|
| ReLU | $[0, +\infty)$ | Couches cachées | Simple, rapide, pas de saturation |
| Sigmoid | $(0, 1)$ | Sortie binaire | Interprétation probabiliste |
| Softmax | $(0,1)^K$, somme=1 | Sortie multi-classes | Distribution de probabilité valide |
| Tanh | $(-1, 1)$ | Couches cachées (RNN) | Centrée sur 0 |

---

## 3. nn.Sequential et définition d'un réseau

### Avec nn.Sequential

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Flatten(),              # [B, 3, 32, 32] → [B, 3072]
    nn.Linear(3072, 1000),
    nn.ReLU(),
    nn.Linear(1000, 10),       # logits — PAS de softmax ici !
)
```

### Avec nn.Module (architectures personnalisées)

```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)    # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)              # logits bruts
        return x
```

### Couche Linear

$$\text{output} = \text{input} \cdot W^T + b$$

```python
layer = nn.Linear(in_features=3072, out_features=1000)
print(layer.weight.shape)  # torch.Size([1000, 3072])
print(layer.bias.shape)    # torch.Size([1000])
```

> **Bonne pratique** : la couche de sortie ne doit **pas** avoir de softmax si vous utilisez `nn.CrossEntropyLoss`, qui l'applique en interne.

---

## 4. Fonctions de perte pour la classification

### Cross-Entropy Loss (multi-classes)

$$\mathcal{L}_{\text{CE}} = -\frac{1}{n} \sum_{i=1}^{n} \log \frac{e^{l_{y_i}}}{\sum_j e^{l_j}}$$

```python
criterion = nn.CrossEntropyLoss()
# Attend des logits bruts + indices entiers (torch.long)
loss = criterion(logits, targets)   # logits [B, K], targets [B]
```

### NLL Loss

```python
criterion = nn.NLLLoss()
# Attend des log-probabilités (après log_softmax)
log_probs = F.log_softmax(logits, dim=1)
loss      = criterion(log_probs, targets)
```

> **Equivalence** : `CrossEntropyLoss(logits, y)` == `NLLLoss(log_softmax(logits), y)`

### Tableau récapitulatif

| Fonction | Entrée attendue | Cible | Cas d'usage |
|---|---|---|---|
| `nn.MSELoss()` | Prédictions continues | Valeurs réelles | Régression |
| `nn.BCELoss()` | Probabilités (post-sigmoid) | 0 ou 1 | Classification binaire |
| `nn.BCEWithLogitsLoss()` | Logits bruts | 0 ou 1 | Binaire (stable) |
| `nn.NLLLoss()` | Log-probabilités | Indices entiers | Multi-classes |
| `nn.CrossEntropyLoss()` | Logits bruts | Indices entiers | **Multi-classes (recommandé)** |

---

## 5. Softmax et Log-Softmax

```python
# Softmax
probs = F.softmax(logits, dim=1)   # [B, K], valeurs dans (0,1), somme = 1

# Log-Softmax (plus stable numériquement)
log_probs = F.log_softmax(logits, dim=1)

# Classe prédite — softmax est monotone, donc argmax(logits) = argmax(probs)
predictions = logits.argmax(dim=1)   # [B]
```

### Astuce numerique

$$\text{softmax}(l)_k = \frac{e^{l_k - \max(l)}}{\sum_{j} e^{l_j - \max(l)}}$$

PyTorch applique cette normalisation automatiquement pour éviter les overflow.

---

## 6. One-hot encoding vs indices de classes

```python
# Indices (format natif PyTorch)
targets = torch.tensor([0, 3, 7, 1], dtype=torch.long)

# One-hot
one_hot = F.one_hot(targets, num_classes=10).float()   # [B, 10]

# One-hot → indices
indices = one_hot.argmax(dim=1)
```

| Format | Quand l'utiliser |
|---|---|
| Indices entiers (`torch.long`) | `CrossEntropyLoss`, `NLLLoss` |
| One-hot (`float`) | `BCELoss`, calculs manuels |

> **Attention** : `CrossEntropyLoss` attend des indices **entiers** (`torch.long`), **pas** du one-hot.

---

## 7. Métriques de classification

### Accuracy

```python
def compute_accuracy(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            predicted = model(inputs).argmax(dim=1)
            correct  += (predicted == labels).sum().item()
            total    += labels.size(0)
    return correct / total
```

### Matrice de confusion

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        preds = model(inputs).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm   = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
```

### Accuracy par classe

```python
for i, cls in enumerate(classes):
    mask    = (torch.tensor(all_labels) == i)
    correct = (torch.tensor(all_preds)[mask] == i).sum().item()
    print(f"{cls:12s} : {correct/mask.sum().item():.2%}")
```

---

## 8. Boucle d'entraînement complète avec validation

### Préparation

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

train_dataset = datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=100, shuffle=False, num_workers=2)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = nn.Sequential(nn.Flatten(), nn.Linear(3072, 1000), nn.ReLU(), nn.Linear(1000, 10)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### Fonction d'entraînement

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        correct    += (outputs.argmax(1) == labels).sum().item()
        total_loss += loss.item() * inputs.size(0)
        total      += inputs.size(0)

    return total_loss / total, correct / total
```

### Fonction d'évaluation

```python
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total_loss += loss.item() * inputs.size(0)
            total      += inputs.size(0)

    return total_loss / total, correct / total
```

### Boucle principale

```python
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    test_loss,  test_acc  = evaluate(model, test_loader, criterion, device)
    print(f"Epoch {epoch+1:02d} | Train {train_loss:.4f}/{train_acc:.2%} | Test {test_loss:.4f}/{test_acc:.2%}")
```

> **Overfitting** : si la perte de train baisse mais la perte de test remonte → régularisation (Dropout, weight decay), plus de données, architecture plus petite.

---

## 9. Référence rapide

### Blocs de construction PyTorch

| Composant | Code | Description |
|---|---|---|
| Couche linéaire | `nn.Linear(in, out)` | $y = xW^T + b$ |
| Aplatissement | `nn.Flatten()` | `[B,C,H,W]` → `[B,C·H·W]` |
| ReLU | `nn.ReLU()` | $\max(0, x)$ |
| Sigmoid | `nn.Sigmoid()` | $(0,1)$ |
| Softmax | `nn.Softmax(dim=1)` | Distribution de proba |
| Log-Softmax | `nn.LogSoftmax(dim=1)` | Log de softmax |
| Cross-Entropy | `nn.CrossEntropyLoss()` | Logits + indices |
| NLL Loss | `nn.NLLLoss()` | Log-proba + indices |
| SGD | `torch.optim.SGD(params, lr, momentum)` | — |
| Adam | `torch.optim.Adam(params, lr)` | — |

### Formules clés

| Concept | Formule |
|---|---|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ |
| Softmax | $p_k = \frac{e^{l_k}}{\sum_j e^{l_j}}$ |
| Cross-Entropy | $-\frac{1}{n}\sum \log p(y_i \mid x_i)$ |
| Argmax | $\hat{y} = \arg\max_k l_k$ |
| Accuracy | $\frac{\text{correct}}{\text{total}}$ |

### Checklist d'entraînement

- [ ] Données normalisées
- [ ] Modèle sur le bon device (`.to(device)`)
- [ ] `optimizer.zero_grad()` avant chaque backward
- [ ] `loss.backward()` puis `optimizer.step()`
- [ ] `model.train()` pendant l'entraînement
- [ ] `model.eval()` + `torch.no_grad()` pendant l'évaluation
- [ ] `CrossEntropyLoss` reçoit des **logits bruts** (pas de softmax)
- [ ] Cibles de type `torch.long`
