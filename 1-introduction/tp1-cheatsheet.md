# TP1 Cheatsheet — Transfer Learning on MNIST

---

## 1. Transforms

Les modèles pré-entraînés sur ImageNet attendent des images de taille **224×224** avec **3 canaux** (RGB), normalisées avec les statistiques d'ImageNet. Les images MNIST sont en niveaux de gris (1 canal) et font 28×28 pixels. Il faut donc les transformer avant de les passer au réseau.

- `Resize((224, 224))` : redimensionne chaque image à la taille attendue par MobileNet.
- `Grayscale(3)` : duplique le canal gris 3 fois pour simuler une image RGB.
- `ToTensor()` : convertit l'image PIL en tenseur PyTorch et ramène les valeurs entre 0 et 1.
- `Normalize(...)` : centre et réduit les valeurs selon les statistiques d'ImageNet (moyenne et écart-type par canal).

```python
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize((224, 224)),   # MobileNet expects 224x224
    transforms.Grayscale(3),         # MNIST is grayscale → convert to 3-channel RGB
    transforms.ToTensor(),
    normalize,                       # ImageNet stats
])
```

---

## 2. Dataset & DataLoader

`datasets.MNIST` télécharge et charge automatiquement les données. Le `DataLoader` découpe le dataset en **batchs** et gère le chargement en parallèle (`num_workers`).

- `shuffle=True` sur le train : mélanger les données à chaque époque évite que le modèle apprenne un ordre particulier.
- `shuffle=False` sur le test : l'ordre n'a pas d'importance pour l'évaluation, et le conserver facilite l'analyse des résultats.
- `batch_size` : nombre d'images traitées simultanément. Un batch plus grand est plus stable mais consomme plus de mémoire.

```python
from torchvision import datasets
from torch.utils.data import DataLoader

train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64,  shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=100, shuffle=False, num_workers=2)
```

---

## 3. Charger un modèle pré-entraîné (MobileNetV3-small)

MobileNetV3-small est un réseau de neurones convolutif léger (~1.5M paramètres), conçu pour être efficace sur des appareils mobiles. En spécifiant `weights='DEFAULT'`, on charge les poids pré-entraînés sur ImageNet (1000 classes).

L'architecture se décompose en deux parties :
- `features` : les couches convolutives qui extraient les caractéristiques visuelles de l'image.
- `classifier` : la tête de classification qui transforme ces caractéristiques en prédictions de classes.

```python
from torchvision import models

model = models.mobilenet_v3_small(weights='DEFAULT')
# Classifier: Linear(576→1024) → Hardswish → Dropout(0.2) → Linear(1024→1000)
```

---

## 4. Transfer Learning — Geler les poids et remplacer la tête

Le **transfer learning** consiste à réutiliser les poids d'un modèle pré-entraîné en ne réentraînant que la dernière couche, adaptée à la nouvelle tâche.

- **Geler** (`requires_grad = False`) : on indique à PyTorch de ne pas calculer les gradients pour ces paramètres. Ils ne seront donc pas mis à jour pendant l'entraînement. Cela préserve les représentations apprises sur ImageNet.
- **Remplacer la dernière couche** : la couche finale passe de 1000 sorties (classes ImageNet) à 10 sorties (chiffres MNIST). Cette nouvelle couche a `requires_grad=True` par défaut : c'est elle seule qui sera entraînée.

```python
import torch.nn as nn

# Freeze ALL parameters
for param in model.parameters():
    param.requires_grad = False

# Replace last layer: 1000 classes → 10 classes
model.classifier._modules['3'] = nn.Linear(1024, 10)
# New layer is trainable by default (requires_grad=True)
```

> Paramètres entraînables : 1024×10 + 10 = **10 250** (sur ~1.5M au total)

---

## 5. Fonction de perte & Optimiseur

- `CrossEntropyLoss` : la perte standard pour la classification multi-classes. Elle combine en interne un `LogSoftmax` et un `NLLLoss`. Elle prend en entrée les **logits bruts** (sorties du réseau avant softmax) et les labels entiers.
- `SGD` (Stochastic Gradient Descent) : met à jour les poids en suivant le gradient de la perte. Le `momentum` accélère la convergence en accumulant une partie du gradient précédent.
- On ne passe à l'optimiseur **que les paramètres de la nouvelle couche** : les couches gelées sont ainsi ignorées lors de la mise à jour.

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier[3].parameters(), lr=0.001, momentum=0.9)
# Only pass the NEW layer's parameters — frozen layers are ignored
```

---

## 6. Boucle d'entraînement

À chaque batch, on exécute les 4 étapes de la descente de gradient :

1. **Passe avant** (`model(inputs)`) : le modèle calcule ses prédictions.
2. **Calcul de la perte** : on compare les prédictions aux vraies étiquettes.
3. **Remise à zéro des gradients** (`zero_grad`) : obligatoire car PyTorch accumule les gradients par défaut.
4. **Rétropropagation** (`backward`) : calcul des gradients de la perte pour chaque paramètre entraînable.
5. **Mise à jour** (`step`) : l'optimiseur ajuste les poids dans la direction opposée au gradient.

```python
model.train()
for inputs, labels in dataloader:
    inputs, labels = inputs.to(device), labels.to(device)

    outputs = model(inputs)            # forward pass
    loss = criterion(outputs, labels)  # compute loss

    optimizer.zero_grad()              # clear gradients
    loss.backward()                    # backprop
    optimizer.step()                   # update weights

    _, preds = torch.max(outputs, 1)   # predicted class
```

---

## 7. Évaluation

Pendant l'évaluation, on désactive deux mécanismes coûteux et inutiles :

- `model.eval()` : passe le modèle en mode évaluation — désactive le Dropout et fixe les statistiques de BatchNorm.
- `torch.no_grad()` : désactive le calcul des gradients, ce qui économise de la mémoire et accélère les calculs.

On applique `softmax` uniquement pour obtenir des probabilités interprétables ; `CrossEntropyLoss` l'applique déjà en interne pendant l'entraînement.

```python
model.eval()
with torch.no_grad():
    for inputs, labels in dataloader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        proba = nn.functional.softmax(outputs, dim=1)
```

---

## 8. Analyse post-évaluation

Après avoir récupéré toutes les prédictions (`predictions`) et les vraies étiquettes (`all_classes`) sous forme de tableaux numpy, on peut calculer différentes métriques :

- **Indices corrects/incorrects** : `np.where` retourne les positions où la condition est vraie — utile pour visualiser des exemples spécifiques.
- **Confiance** : la probabilité maximale attribuée par le modèle à sa prédiction. Une confiance élevée sur une mauvaise prédiction révèle un modèle trop sûr de lui (*overconfident*).
- **Incertitude** : une faible confiance maximale signifie que le modèle hésite entre plusieurs classes.
- **Matrice de confusion** : montre pour chaque vraie classe, quelles classes le modèle a prédit — très utile pour identifier les confusions systématiques (ex. 4 confondu avec 9).

```python
import numpy as np

# Correct / incorrect indices
correct   = np.where(predictions == all_classes)[0]
incorrect = np.where(predictions != all_classes)[0]

# Accuracy
accuracy = len(correct) / len(all_classes)

# Confidence (max softmax probability per sample)
confidences    = np.max(all_proba, axis=1)
most_confident = np.argsort(-confidences)[:16]
most_uncertain = np.argsort(confidences)[:16]   # lowest max-proba = most uncertain

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(all_classes, predictions)
```

---

## Concepts clés

| Concept | Explication |
|---|---|
| **Transfer learning** | Réutiliser les poids d'un modèle pré-entraîné sur un grand dataset (ImageNet) pour une nouvelle tâche, en ne réentraînant qu'une partie du réseau. |
| **Geler les poids** | `param.requires_grad = False` — aucun gradient n'est calculé, les poids ne bougent pas. Préserve les représentations apprises. |
| **Fine-tuning** | Variante où l'on dégèle quelques couches profondes et on les réentraîne avec un très faible taux d'apprentissage. |
| **Normalisation ImageNet** | Tous les modèles pré-entraînés de torchvision attendent `mean=[0.485, 0.456, 0.406]` — ne pas oublier cette étape. |
| **`torch.max(outputs, 1)`** | Retourne `(valeurs, indices)` — les indices sont les classes prédites (argmax sur la dimension des classes). |
| **Softmax vs CrossEntropyLoss** | `CrossEntropyLoss` applique déjà le log-softmax en interne. N'appliquer softmax que pour afficher des probabilités. |

---

## Configuration du device

PyTorch permet d'utiliser le GPU (CUDA pour NVIDIA, MPS pour Apple Silicon) pour accélérer les calculs. Si aucun GPU n'est disponible, on replie sur le CPU. Il faut envoyer le modèle **et** les données sur le même device.

```python
device = (
    "cuda:0" if torch.cuda.is_available() else
    "mps"    if torch.backends.mps.is_available() else
    "cpu"
)
model = model.to(device)
inputs, labels = inputs.to(device), labels.to(device)
```
