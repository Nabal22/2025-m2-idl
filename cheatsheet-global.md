# Cheatsheet Globale — Deep Learning 101 (TP 1 à 6)

> Synthèse complète de tous les TP. Chaque section renvoie au TP correspondant.

---

## Table des matières

- [TP1 — Transfer Learning](#tp1--transfer-learning)
- [TP2 — Tenseurs & Régression Linéaire](#tp2--tenseurs--régression-linéaire)
- [TP3 — Autodiff & Rétropropagation](#tp3--autodiff--rétropropagation)
- [TP4 — Classification & MLP](#tp4--classification--mlp)
- [TP5 — Réseaux Convolutifs (CNN)](#tp5--réseaux-convolutifs-cnn)
- [TP6 — Optimisation](#tp6--optimisation)
- [Référence rapide globale](#référence-rapide-globale)

---

## TP1 — Transfer Learning

> Adapter un modèle pré-entraîné sur ImageNet à une nouvelle tâche.

### Concept

Les premières couches d'un CNN apprennent des features génériques (bords, textures) réutilisables. Seule la tête de classification est spécifique à la tâche.

| Stratégie | Couches gelées | Quand l'utiliser |
|---|---|---|
| Feature Extraction | Toutes sauf la tête | Peu de données, domaine similaire |
| Fine-tuning partiel | Premières couches | Données moyennes |
| Fine-tuning complet | Aucune (lr faible) | Beaucoup de données |

### Chargement et adaptation

```python
from torchvision import models
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# VGG16
model = models.vgg16(weights='DEFAULT')
for param in model.parameters(): param.requires_grad = False
model.classifier._modules['6'] = nn.Linear(4096, N_CLASSES)

# MobileNetV3-small
model = models.mobilenet_v3_small(weights='DEFAULT')
for param in model.parameters(): param.requires_grad = False
model.classifier._modules['3'] = nn.Linear(1024, N_CLASSES)

model = model.to(device)
```

### Transformations ImageNet

$$x_{\text{norm}} = \frac{x - \mu}{\sigma}, \quad \mu=[0.485, 0.456, 0.406], \quad \sigma=[0.229, 0.224, 0.225]$$

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # si image en niveaux de gris
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Optimiseur sur la tête uniquement

```python
optimizer = torch.optim.SGD(model.classifier[-1].parameters(), lr=0.001)
```

### Compter les paramètres entraînables

```python
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total     = sum(p.numel() for p in model.parameters())
print(f"{n_trainable:,} entraînables / {n_total:,} total")
```

---

## TP2 — Tenseurs & Régression Linéaire

### Création de tenseurs

```python
import torch

torch.zeros(3, 4)           # zéros
torch.ones(5)               # uns
torch.rand(5)               # uniforme [0, 1)
torch.randn(5)              # normale N(0, 1)
torch.arange(0, 10, 2)      # [0, 2, 4, 6, 8]
torch.linspace(0, 1, 5)     # 5 valeurs équiréparties
torch.tensor([1, 2, 3])     # depuis liste Python
torch.from_numpy(arr)       # depuis NumPy (mémoire partagée !)
```

### Indexation & Slicing

```python
x[0], x[-1], x[2:5], x[::2]          # 1D
matrix[1, 2], matrix[:, 0]            # 2D
x[x > 5]                              # masque booléen
x[0, ...]                             # ellipsis : x[0, :, :, :]
x[5].item()                           # scalaire → Python float
```

> Les slices sont des **vues** — les modifier modifie le tenseur original.

### Broadcasting

```
Règles (droite à gauche) :
1. Compléter avec des 1 à gauche si nécessaire
2. Si une dim = 1, l'étendre
3. Dimensions incompatibles (≠1 des deux côtés) → ERREUR
```

```python
a = torch.arange(0, 40, 10).unsqueeze(1)  # (4, 1)
b = torch.arange(0, 4)                    # (4,)
a + b   # → (4, 4)
```

### Opérations utiles

```python
x.reshape(2, 6)     # changer la forme
x.unsqueeze(1)      # ajouter une dimension
x.permute(0,3,1,2)  # réordonner (BHWC → BCHW)
x.t()               # transposée 2D
x.numpy()           # vers NumPy (CPU uniquement)
x.add_(y)           # opération in-place (suffixe _)
```

### GPU

```python
x = x.to(device)
y = torch.ones_like(x, device=device)
arr = x.cpu().numpy()    # numpy requiert le CPU
```

### Régression Linéaire

**Modèle :** $\hat{y} = w \cdot x + b$

**MSE :** $L = \frac{1}{n}\sum(y_i - \hat{y}_i)^2$

**Gradients :**

$$\frac{\partial L}{\partial w} = -\frac{2}{n}\sum(y_i - \hat{y}_i)x_i \qquad \frac{\partial L}{\partial b} = -\frac{2}{n}\sum(y_i - \hat{y}_i)$$

### Standardisation

$$x_{\text{std}} = \frac{x - \mu_x}{\sigma_x}$$

```python
x_std = (x - x.mean()) / x.std()
y_std = (y - y.mean()) / y.std()

# Retour à l'échelle originale
w_orig = w * y.std() / x.std()
b_orig = b * y.std() + y.mean() - w_orig * x.mean()
```

### Autograd

```python
w = torch.tensor([0.5], requires_grad=True)
b = torch.tensor([0.5], requires_grad=True)

for _ in range(200):
    loss = ((w * x_std + b - y_std)**2).mean()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()
```

### nn.Linear (pipeline complet)

```python
model     = nn.Linear(1, 1)
loss_fn   = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for _ in range(200):
    optimizer.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    optimizer.step()
```

---

## TP3 — Autodiff & Rétropropagation

### Forward vs Backward mode

| Critère | Forward mode (nombres duaux) | Backward mode (PyTorch) |
|---|---|---|
| Direction | Entrées → Sorties | Sorties → Entrées |
| Coût | 1 passe / paramètre | **1 seule passe** (tous les params) |
| Idéal pour | Peu d'entrées | **Deep learning** (millions de params) |

### Nombres duaux

$$\varepsilon^2 = 0 \qquad f(x + \varepsilon) = f(x) + f'(x)\varepsilon$$

```python
def derivative(f, x):
    d = f(Dual(x, 1))   # partie duale = f'(x)
    return d.epsilon
```

### Graphe de calcul — exemple

Pour $\ell = (\exp(wx + b) - y)^2$ :

$$\frac{\partial \ell}{\partial w} = 2(\hat{y} - y) \cdot \hat{y} \cdot x \qquad \frac{\partial \ell}{\partial b} = 2(\hat{y} - y) \cdot \hat{y}$$

### PyTorch autograd

```python
w = torch.tensor(0.5, requires_grad=True)
y_hat = torch.exp(w * x + b)
loss  = (y_hat - y) ** 2
loss.backward()
print(w.grad)   # ∂loss/∂w
```

### `no_grad` et `detach`

```python
with torch.no_grad():
    y = model(x)        # pas de graphe (inférence)

z   = y.detach()        # détache du graphe
arr = y.detach().numpy()  # nécessaire avec requires_grad
```

### Dérivées locales communes

| Opération | Backward : `grad_in = grad_out ×` |
|---|---|
| $ax$ | `a` |
| $\exp(x)$ | `output` |
| $\ln(x)$ | `1/x` |
| $\sigma(x)$ | `output * (1 - output)` |
| $\text{ReLU}(x)$ | `(x > 0).float()` |
| $x \cdot y$ | `grad * y` pour x, `grad * x` pour y |

---

## TP4 — Classification & MLP

### Architecture MLP

```python
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3072, 1000),
    nn.ReLU(),
    nn.Linear(1000, 10),     # logits bruts — PAS de softmax ici
)
```

### Fonctions d'activation

| Fonction | Formule | Sortie | Usage |
|---|---|---|---|
| ReLU | $\max(0, x)$ | $[0, +\infty)$ | Couches cachées |
| Sigmoid | $\frac{1}{1+e^{-x}}$ | $(0, 1)$ | Sortie binaire |
| Softmax | $\frac{e^{l_k}}{\sum e^{l_j}}$ | $(0,1)^K$, somme=1 | Sortie multi-classes |
| Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1, 1)$ | RNN |

### Fonctions de perte

| Loss | Entrée | Cible | Cas d'usage |
|---|---|---|---|
| `nn.MSELoss()` | Continue | Réelle | Régression |
| `nn.BCEWithLogitsLoss()` | Logits | 0/1 float | Binaire |
| `nn.CrossEntropyLoss()` | **Logits bruts** | **Indices long** | **Multi-classes** |
| `nn.NLLLoss()` | Log-softmax | Indices long | Multi-classes |

> `CrossEntropyLoss` = `LogSoftmax` + `NLLLoss` en une opération.

### Classification

```python
# Prédiction : argmax des logits
predictions = logits.argmax(dim=1)

# Probabilités
probs = F.softmax(logits, dim=1)

# One-hot → indices
indices = F.one_hot(targets, num_classes=10).float()
```

### Boucle d'entraînement + validation

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()

def evaluate(model, loader, criterion, device):
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            # ... métriques
```

---

## TP5 — Réseaux Convolutifs (CNN)

### Conv2d

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
```

$$\text{params} = \text{out\_ch} \times (\text{in\_ch} \times k_H \times k_W + 1)$$

> Conserver H×W avec kernel 3×3 : `padding = 1`

### Taille de sortie

$$\boxed{H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1}$$

| $H_{in}$ | k | s | p | $H_{out}$ |
|---|---|---|---|---|
| 28 | 3 | 1 | 1 | **28** |
| 28 | 7 | 7 | 0 | 4 |
| 32 | 3 | 2 | 1 | 16 |

### Pooling

```python
nn.MaxPool2d(2, 2)                                    # divise H,W par 2
F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)           # Global Average Pooling
```

### Architecture CNN typique

```
Conv2d → BN → ReLU → MaxPool → (répéter) → Flatten → Linear → Softmax
```

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU(), nn.Linear(128, 10)
        )
    def forward(self, x): return self.classifier(self.features(x))
```

### Batch Normalization

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}, \quad y = \gamma \hat{x} + \beta$$

```python
nn.BatchNorm2d(num_features)   # après Conv2d, avant ReLU
```

### Flatten

```python
x = x.view(x.size(0), -1)   # [B,C,H,W] → [B, C·H·W]
x = x.flatten(1)             # identique
nn.Flatten()                 # dans Sequential
```

### Correspondance loss / sortie

| Sortie | Loss |
|---|---|
| `F.log_softmax(x, dim=1)` | `nn.NLLLoss()` |
| Logits bruts | `nn.CrossEntropyLoss()` |

---

## TP6 — Optimisation

### Algorithmes (formules)

| Algorithme | Formule de mise à jour |
|---|---|
| SGD | $\theta \leftarrow \theta - \eta g$ |
| Momentum | $v \leftarrow \gamma v + \eta g$ ; $\theta \leftarrow \theta - v$ |
| RMSProp | $s \leftarrow \gamma s + (1-\gamma)g^2$ ; $\theta \leftarrow \theta - \frac{\eta}{\sqrt{s+\epsilon}} g$ |
| Adam | $\theta \leftarrow \theta - \frac{\eta}{\sqrt{\hat{v}}+\epsilon}\hat{m}$ (avec bias correction) |
| AMSGrad | Idem Adam avec $\hat{v} \leftarrow \max(\hat{v}_{\text{prev}}, v)$ |

### Optimiseurs PyTorch

```python
torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)
torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)   # Transformers
torch.optim.RMSprop(params, lr=0.001, alpha=0.9)         # RNN
```

### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

StepLR(optimizer, step_size=30, gamma=0.1)               # paliers fixes
CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)    # décroissance douce
ReduceLROnPlateau(optimizer, factor=0.5, patience=10)    # sur plateau

# Appeler APRÈS optimizer.step()
scheduler.step()           # StepLR, CosineAnnealingLR
scheduler.step(val_loss)   # ReduceLROnPlateau
```

### Gradient Clipping

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### Recommandations

| Contexte | Optimiseur recommandé | lr |
|---|---|---|
| Cas général | Adam | `1e-3` |
| Vision (ResNet, VGG) | SGD + Momentum | `0.01`–`0.1` |
| Transformers, fine-tuning | AdamW | `5e-4` |
| RNN, RL | RMSProp | `1e-3` |

---

## Référence rapide globale

### Pipeline Deep Learning complet

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 1. DONNÉES
transform    = transforms.Compose([transforms.ToTensor(), transforms.Normalize(...)])
dataset      = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader   = DataLoader(dataset, batch_size=64, shuffle=True)

# 2. MODÈLE
model = nn.Sequential(nn.Flatten(), nn.Linear(3072, 512), nn.ReLU(), nn.Linear(512, 10))
model = model.to(device)

# 3. LOSS + OPTIMISEUR
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 4. ENTRAÎNEMENT
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
    scheduler.step()

    # 5. ÉVALUATION
    model.eval()
    with torch.no_grad():
        # ... calcul accuracy, loss validation
        pass
```

### Tableau de synthèse des TP

| TP | Thème | Concepts clés | Loss recommandée |
|---|---|---|---|
| **TP1** | Transfer Learning | VGG, MobileNet, feature extraction, fine-tuning | `CrossEntropyLoss` |
| **TP2** | Tenseurs & Régression | Tenseurs PyTorch, broadcasting, MSE, standardisation | MSE |
| **TP3** | Autodiff | Nombres duaux, graphe de calcul, backward, `requires_grad` | Toute |
| **TP4** | Classification MLP | `nn.Linear`, ReLU, Softmax, CrossEntropy, accuracy | `CrossEntropyLoss` |
| **TP5** | CNN | `Conv2d`, MaxPool, BN, Flatten, architecture CNN | `NLLLoss` / `CrossEntropyLoss` |
| **TP6** | Optimisation | SGD, Momentum, Adam, schedulers, gradient clipping | Toute |

### Les 5 étapes de la boucle d'entraînement

```
1. optimizer.zero_grad()    ← toujours en premier
2. outputs = model(inputs)  ← forward pass
3. loss = criterion(...)    ← calcul de la perte
4. loss.backward()          ← backward pass (gradients)
5. optimizer.step()         ← mise à jour des poids
```

### Modes train / eval

| | `model.train()` | `model.eval()` |
|---|---|---|
| Dropout | Actif | Désactivé |
| BatchNorm | Stats mini-batch | Stats mémorisées |
| `torch.no_grad()` | Non | **Oui** |

### Shapes importantes

| Contexte | Shape |
|---|---|
| Batch d'images RGB | `[B, 3, H, W]` |
| Après Flatten | `[B, C·H·W]` |
| Logits sortie | `[B, num_classes]` |
| Labels | `[B]` (indices `torch.long`) |
| Tenseur scalaire | `[]` ou `[1]` → `.item()` |

### Checklist universelle

- [ ] Device défini (`cuda` / `mps` / `cpu`)
- [ ] Modèle sur le device (`.to(device)`)
- [ ] Données normalisées
- [ ] `optimizer.zero_grad()` avant chaque batch
- [ ] `model.train()` / `model.eval()` aux bons moments
- [ ] `torch.no_grad()` pendant l'évaluation
- [ ] `CrossEntropyLoss` → logits bruts + `torch.long`
- [ ] `NLLLoss` → `log_softmax` + `torch.long`
- [ ] Gradient clipping si RNN / Transformer
- [ ] `scheduler.step()` après chaque epoch
