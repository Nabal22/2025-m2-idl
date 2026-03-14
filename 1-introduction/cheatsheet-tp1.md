# TP-1 Cheatsheet — Introduction & Transfer Learning

> **Objectif du TP :** Adapter un modèle pré-entraîné sur ImageNet (VGG16, MobileNetV3) à une nouvelle tâche de classification (chiens/chats ou chiffres MNIST) en utilisant le Transfer Learning.

---

## Table des matières

1. [Concept : Transfer Learning](#1-concept--transfer-learning)
2. [Feature Extraction vs Fine-tuning](#2-feature-extraction-vs-fine-tuning)
3. [Chargement d'un modèle pré-entraîné](#3-chargement-dun-modèle-pré-entraîné)
4. [Gel et dégel des couches](#4-gel-et-dégel-des-couches-requiresgrad)
5. [DataLoader et transformations d'images](#5-dataloader-et-transformations-dimages)
6. [Boucle d'entraînement](#6-boucle-dentraînement-standard)
7. [Évaluation du modèle](#7-évaluation-du-modèle)
8. [Référence rapide](#8-référence-rapide)

---

## 1. Concept : Transfer Learning

Le **Transfer Learning** consiste à réutiliser un modèle entraîné sur une grande tâche source (ex. ImageNet, 1,2 M images, 1000 classes) pour résoudre une tâche cible plus petite (ex. 2 classes chiens/chats, 10 chiffres MNIST).

### Pourquoi ça fonctionne ?

Les premières couches d'un CNN apprennent des caractéristiques génériques (bords, textures, formes) qui sont utiles pour de nombreuses tâches de vision. Seules les dernières couches sont spécifiques à la tâche d'origine.

```
ImageNet (1000 classes)           Tâche cible (N classes)
┌─────────────────────────────┐   ┌────────────────────────────┐
│  Conv layers (features)     │ → │  Conv layers (gelées)      │
│  Bords, textures, formes... │   │  Même extracteur de traits │
├─────────────────────────────┤   ├────────────────────────────┤
│  Classifier (1000 sorties)  │ → │  Nouveau classifier (N)    │
│  Spécifique à ImageNet      │   │  Appris de zéro            │
└─────────────────────────────┘   └────────────────────────────┘
```

### Avantages

- Beaucoup moins de données nécessaires
- Entraînement beaucoup plus rapide
- Meilleures performances en général

---

## 2. Feature Extraction vs Fine-tuning

| Stratégie | Couches gelées | Couches entraînées | Quand l'utiliser |
|---|---|---|---|
| **Feature Extraction** | Toutes sauf la tête | Tête de classification uniquement | Peu de données, domaine similaire |
| **Fine-tuning partiel** | Premières couches | Dernières couches + tête | Données moyennes, domaine similaire |
| **Fine-tuning complet** | Aucune | Tout le réseau (lr faible) | Beaucoup de données, domaine différent |

> **Règle pratique :** Commencer toujours par le Feature Extraction. Si les performances sont insuffisantes, dégeler progressivement des couches supplémentaires avec un learning rate très faible (ex. 10x à 100x plus faible que pour la tête).

---

## 3. Chargement d'un modèle pré-entraîné

### Setup et détection du device

```python
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, datasets

device = ("cuda:0" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
print(f"Utilisation de : {device}")
```

### VGG16 (Dogs vs Cats)

```python
# Chargement avec les poids pré-entraînés sur ImageNet
model_vgg = models.vgg16(weights='DEFAULT')
print(model_vgg)
```

Architecture VGG16 :

```
VGG(
  (features): Sequential(
    Conv2d(3, 64, 3x3) → ReLU → ... → MaxPool2d
    Conv2d(64, 128, 3x3) → ReLU → ... → MaxPool2d
    ...
    Conv2d(512, 512, 3x3) → ReLU → MaxPool2d
  )
  (avgpool): AdaptiveAvgPool2d(7x7)
  (classifier): Sequential(
    Linear(25088, 4096) → ReLU → Dropout(0.5)
    Linear(4096, 4096)  → ReLU → Dropout(0.5)
    Linear(4096, 1000)   ← À remplacer
  )
)
```

### MobileNetV3-small (MNIST)

```python
# Léger (~2.5M params), efficace sur mobile
model = models.mobilenet_v3_small(weights='DEFAULT')
print(model)
```

Architecture MobileNetV3-small :

```
MobileNetV3(
  (features): Sequential(...)   # Couches convolutives InvertedResidual
  (avgpool): AdaptiveAvgPool2d(1)
  (classifier): Sequential(
    Linear(576, 1024) → Hardswish → Dropout(0.2)
    Linear(1024, 1000)   ← À remplacer
  )
)
```

### Transférer le modèle sur le device

```python
model = model.to(device)
```

---

## 4. Gel et dégel des couches (`requires_grad`)

### Geler tous les paramètres (Feature Extraction)

```python
# Désactive le calcul de gradient pour tous les paramètres
for param in model.parameters():
    param.requires_grad = False
```

### Remplacer la tête de classification

```python
# Pour VGG16 : remplacer Linear(4096, 1000) → Linear(4096, 2)
model_vgg.classifier._modules['6'] = nn.Linear(4096, 2)

# Pour MobileNetV3-small : remplacer Linear(1024, 1000) → Linear(1024, 10)
model.classifier._modules['3'] = nn.Linear(1024, 10)
```

> **Important :** La nouvelle couche `nn.Linear` créée a `requires_grad=True` par défaut, même si les autres paramètres sont gelés. Seule cette couche sera mise à jour par l'optimiseur.

### Vérifier les paramètres entraînables

```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
frozen    = total - trainable

print(f"Total       : {total:,}")
print(f"Entraînable : {trainable:,}")
print(f"Gelé        : {frozen:,}")
```

Exemple MobileNetV3 → MNIST :
- Couche finale : 1024 × 10 + 10 (biais) = **10 250 paramètres** sur 1 528 106 au total

### Dégel partiel pour le Fine-tuning

```python
# Dégeler uniquement le classifier entier
for param in model.classifier.parameters():
    param.requires_grad = True

# Dégeler les derniers blocs de features
for param in model.features[-3:].parameters():
    param.requires_grad = True
```

---

## 5. DataLoader et transformations d'images

### Statistiques de normalisation ImageNet

Tous les modèles pré-entraînés sur ImageNet attendent des images normalisées avec :

$$\mu = [0.485,\ 0.456,\ 0.406] \qquad \sigma = [0.229,\ 0.224,\ 0.225]$$

La normalisation est appliquée canal par canal :

$$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$

### Pipeline de transformations standard

```python
from torchvision import transforms

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Pour des images déjà en couleur (ex. Dogs vs Cats)
imagenet_format = transforms.Compose([
    transforms.CenterCrop(224),  # Recadrage à 224×224
    transforms.ToTensor(),       # PIL Image → Tensor [0,1]
    normalize,                   # Normalisation ImageNet
])

# Pour des images en niveaux de gris (ex. MNIST)
mnist_transform = transforms.Compose([
    transforms.Resize((224, 224)),                  # Redimensionner 28×28 → 224×224
    transforms.Grayscale(num_output_channels=3),    # Gris → RGB (3 canaux)
    transforms.ToTensor(),
    normalize,
])
```

> **Attention :** Les modèles pré-entraînés attendent des images `224×224` en RGB (3 canaux). MNIST est en `28×28` niveaux de gris : il faut impérativement redimensionner ET convertir en 3 canaux.

### Chargement avec `ImageFolder` (arborescence de dossiers)

```python
import os
from torchvision import datasets

# Structure attendue :
# data_dir/train/cats/img1.jpg
# data_dir/train/dogs/img2.jpg
# data_dir/valid/cats/img3.jpg

data_dir = './dogscats'

dsets = {
    split: datasets.ImageFolder(
        root=os.path.join(data_dir, split),
        transform=imagenet_format
    )
    for split in ['train', 'valid']
}

dset_sizes = {x: len(dsets[x]) for x in ['train', 'valid']}
# Classes inférées automatiquement depuis les noms de dossiers
print(dsets['train'].classes)       # ['cats', 'dogs']
print(dsets['train'].class_to_idx)  # {'cats': 0, 'dogs': 1}
```

### Chargement d'un dataset standard (ex. MNIST)

```python
train_dataset = datasets.MNIST(
    root='./data', train=True,  download=True, transform=mnist_transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=mnist_transform
)
```

### Création des DataLoaders

```python
from torch.utils.data import DataLoader

# Dogs vs Cats
loader_train = DataLoader(dsets['train'], batch_size=64, shuffle=True,  num_workers=6)
loader_valid = DataLoader(dsets['valid'], batch_size=5,  shuffle=False, num_workers=6)

# MNIST
train_loader = DataLoader(train_dataset, batch_size=64,  shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=100, shuffle=False, num_workers=2)
```

> **Paramètres clés du DataLoader :**
> - `batch_size` : nombre d'images par batch
> - `shuffle=True` : mélanger les données à chaque epoch (entraînement uniquement, jamais sur le test)
> - `num_workers` : nombre de processus parallèles pour le chargement (0 = chargement dans le process principal)

### Inspecter un batch

```python
inputs, labels = next(iter(train_loader))
print(f"Shape du batch : {inputs.shape}")   # torch.Size([64, 3, 224, 224])
print(f"Labels         : {labels[:8]}")
```

### Afficher des images (dénormalisation)

```python
import numpy as np
import matplotlib.pyplot as plt

def imshow(inp, title=None):
    """Affiche un tensor image après dénormalisation."""
    inp = inp.numpy().transpose((1, 2, 0))  # (C, H, W) → (H, W, C)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    inp  = np.clip(std * inp + mean, 0, 1)  # dénormalisation
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.axis('off')

# Afficher une grille de 8 images
out = torchvision.utils.make_grid(inputs[:8])
imshow(out, title=[str(x.item()) for x in labels[:8]])
```

---

## 6. Boucle d'entraînement standard

### Définir la fonction de perte et l'optimiseur

```python
# CrossEntropyLoss pour la classification multi-classes
criterion = nn.CrossEntropyLoss()

# SGD uniquement sur les paramètres de la tête de classification
optimizer_vgg = torch.optim.SGD(model_vgg.classifier[6].parameters(), lr=0.001)
optimizer     = torch.optim.SGD(model.classifier[3].parameters(), lr=0.001, momentum=0.9)
```

> **Astuce :** Ne passer à l'optimiseur que les paramètres de la nouvelle tête. Pour le fine-tuning complet, utiliser un lr beaucoup plus faible (ex. `1e-4`) pour les couches pré-entraînées afin de ne pas détruire les représentations apprises.

### La fonction de perte : CrossEntropyLoss

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{z_{y_i}}}{\sum_j e^{z_j}}$$

où $z$ est le vecteur logit de sortie du réseau et $y_i$ la vraie classe de l'exemple $i$.

> `nn.CrossEntropyLoss` combine `LogSoftmax` + `NLLLoss` en une seule opération. Ne jamais appliquer de Softmax manuellement avant cette perte.

### Boucle d'entraînement complète

```python
from tqdm import tqdm

def train_model(model, dataloader, size, epochs=1, optimizer=None):
    model.train()  # Active Dropout, BatchNorm en mode train

    for epoch in range(epochs):
        running_loss     = 0.0
        running_corrects = 0

        for inputs, classes in tqdm(dataloader):
            inputs, classes = inputs.to(device), classes.to(device)

            outputs = model(inputs)              # 1. Forward pass
            loss    = criterion(outputs, classes) # 2. Calcul de la perte

            optimizer.zero_grad()  # 3. Remise à zéro des gradients
            loss.backward()        # 4. Backward pass
            optimizer.step()       # 5. Mise à jour des poids

            _, preds = torch.max(outputs.data, 1)
            running_loss     += loss.data.item()
            running_corrects += torch.sum(preds == classes.data)

        epoch_loss = running_loss / size
        epoch_acc  = running_corrects.data.item() / size
        print(f'Epoch {epoch+1}/{epochs} — Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
```

### Les 5 étapes de la boucle d'entraînement

```
Pour chaque batch :
  1. outputs = model(inputs)          ← forward pass
  2. loss = criterion(outputs, y)     ← calcul de la perte
  3. optimizer.zero_grad()            ← remise à zéro des gradients
  4. loss.backward()                  ← backward pass (calcul des gradients)
  5. optimizer.step()                 ← mise à jour des poids
```

---

## 7. Évaluation du modèle

### Fonction de test

```python
def test_model(model, dataloader, size, n_classes=10):
    model.eval()  # Désactive Dropout, BatchNorm en mode eval

    predictions = np.zeros(size)
    all_classes = np.zeros(size)
    all_proba   = np.zeros((size, n_classes))
    i = 0
    running_corrects = 0

    with torch.no_grad():  # Désactive le calcul de gradient
        for inputs, classes in dataloader:
            inputs, classes = inputs.to(device), classes.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, classes)

            _, preds = torch.max(outputs.data, 1)
            running_corrects += torch.sum(preds == classes.data)

            predictions[i:i+len(classes)] = preds.cpu().numpy()
            all_classes[i:i+len(classes)] = classes.cpu().numpy()
            all_proba[i:i+len(classes), :] = nn.functional.softmax(outputs, dim=1).cpu().numpy()
            i += len(classes)

    print(f'Accuracy : {running_corrects.item() / size:.4f}')
    return predictions, all_proba, all_classes
```

### Différences clés entraînement / évaluation

| Aspect | `model.train()` | `model.eval()` |
|---|---|---|
| Dropout | Actif | Désactivé |
| BatchNorm | Stats du batch courant | Stats mémorisées |
| `torch.no_grad()` | Non | Oui |
| `loss.backward()` | Oui | Non |
| `optimizer.step()` | Oui | Non |

### Matrice de confusion

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(all_classes, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.show()
```

---

## 8. Référence rapide

### Modèles utilisés dans le TP

| Modèle | Params totaux | Tâche | Couche à remplacer | Params entraînés |
|---|---|---|---|---|
| `vgg16` | ~138 M | Dogs vs Cats (2 cl.) | `classifier[6]` → `Linear(4096, 2)` | 8 194 |
| `mobilenet_v3_small` | ~2.5 M | MNIST (10 cl.) | `classifier[3]` → `Linear(1024, 10)` | 10 250 |

### Commandes essentielles

```python
# Chargement
model = models.vgg16(weights='DEFAULT').to(device)

# Gel total
for param in model.parameters(): param.requires_grad = False

# Remplacement de la tête
model.classifier._modules['6'] = nn.Linear(4096, N)

# Optimiseur sur la tête uniquement
optimizer = torch.optim.SGD(model.classifier[6].parameters(), lr=0.001)

# Modes
model.train() / model.eval()

# Inférence sans gradient
with torch.no_grad(): outputs = model(inputs)

# Prédiction
_, preds = torch.max(outputs, dim=1)

# Probabilités
probs = nn.functional.softmax(outputs, dim=1)

# Compter les paramètres entraînables
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

### Transformations `torchvision` fréquentes

| Transform | Utilité |
|---|---|
| `transforms.Resize((224, 224))` | Redimensionner |
| `transforms.CenterCrop(224)` | Recadrer au centre |
| `transforms.RandomHorizontalFlip()` | Augmentation |
| `transforms.Grayscale(num_output_channels=3)` | Gris → RGB |
| `transforms.ToTensor()` | PIL → Tensor `[0,1]` |
| `transforms.Normalize(mean, std)` | Normalisation |

### Checklist Transfer Learning

- [ ] Choisir le modèle pré-entraîné
- [ ] Pipeline de transformations avec normalisation ImageNet
- [ ] DataLoaders (train `shuffle=True`, test `shuffle=False`)
- [ ] Geler tous les paramètres
- [ ] Remplacer la tête avec `nn.Linear(in, N_classes)`
- [ ] Optimiseur sur la tête uniquement
- [ ] Entraîner : `train()` → forward → loss → zero_grad → backward → step
- [ ] Évaluer : `eval()` + `no_grad()`
- [ ] Analyser : matrice de confusion, accuracy par classe
