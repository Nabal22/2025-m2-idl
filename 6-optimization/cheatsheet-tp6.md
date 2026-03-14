# TP-6 Cheatsheet — Optimisation

---

## Table des matières

1. [Rappel : objectif de l'optimisation](#1-rappel--objectif-de-loptimisation)
2. [Gradient Descent — SGD](#2-gradient-descent--sgd)
3. [Mini-batch Gradient Descent](#3-mini-batch-gradient-descent)
4. [Momentum](#4-momentum)
5. [Nesterov (NAG)](#5-nesterov-nag)
6. [Adagrad](#6-adagrad)
7. [RMSProp](#7-rmsprop)
8. [Adam](#8-adam)
9. [AMSGrad](#9-amsgrad)
10. [Learning Rate Scheduling](#10-learning-rate-scheduling)
11. [Problèmes courants : gradients](#11-problèmes-courants--gradients)
12. [Gradient Clipping](#12-gradient-clipping)
13. [Comparaison des optimiseurs](#13-comparaison-des-optimiseurs)
14. [Référence rapide](#14-référence-rapide)

---

## 1. Rappel : objectif de l'optimisation

| Objectif | Cible |
|---|---|
| Optimisation | Réduire l'**erreur d'entraînement** |
| Apprentissage | Réduire l'**erreur de généralisation** (test error) |

### Template de boucle d'entraînement

```python
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()                # 1. Remise à zéro des gradients
        y_pred = model(x_batch)              # 2. Forward pass
        loss   = criterion(y_pred, y_batch)  # 3. Calcul de la loss
        loss.backward()                      # 4. Backward pass
        optimizer.step()                     # 5. Mise à jour des paramètres
```

---

## 2. Gradient Descent — SGD

### Formules

| Variante | Formule |
|---|---|
| Batch GD | $\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)$ |
| SGD (stochastique) | $\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t; x_i, y_i)$ |
| Mini-batch GD | $\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t; x_{i:i+b}, y_{i:i+b})$ |

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

### Limitations

- Même `lr` pour tous les paramètres
- Sensible aux points selle ($\nabla J \approx 0$)
- Oscillations si `lr` trop grand

---

## 3. Mini-batch Gradient Descent

```python
from torch.utils.data import DataLoader, TensorDataset

dataset    = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
```

| Batch size | Avantages | Inconvénients |
|---|---|---|
| Petit (8-32) | Bonne généralisation, moins de mémoire | Plus bruité |
| Grand (256+) | Calculs parallèles efficaces | Peut sur-apprendre |

---

## 4. Momentum

### Concept

Accumule les gradients passés pour amortir les oscillations et accélérer la convergence.

### Formules

$$v_{t+1} = \gamma v_t + \eta \nabla J(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_{t+1}$$

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### Implémentation from scratch

```python
class Momentum:
    def __init__(self, params, eta=0.4, gamma=0.9):
        self.params = params
        self.eta, self.gamma = eta, gamma
        self.v = [0.0 for _ in params]

    def step(self):
        for i, param in enumerate(self.params):
            self.v[i] = self.gamma * self.v[i] + self.eta * param.grad
            param.value -= self.v[i]
```

> **Valeur typique** : `momentum=0.9`. Permet souvent d'utiliser un `lr` plus élevé qu'un SGD pur.

---

## 5. Nesterov (NAG)

Variante de Momentum : calcule le gradient **en avance** le long de la trajectoire du momentum.

$$v_{t+1} = \gamma v_t + \eta \nabla J(\theta_t - \gamma v_t)$$

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
```

---

## 6. Adagrad

Adapte le **learning rate par paramètre** via la somme cumulée des gradients au carré.

$$s_{t+1,i} = s_{t,i} + g_{t,i}^2 \qquad \theta_{t+1,i} = \theta_{t,i} - \frac{\eta}{\sqrt{s_{t+1,i} + \epsilon}}\, g_{t,i}$$

```python
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
```

> **Limitation** : le learning rate décroît de façon monotone et tend vers 0 (l'optimiseur n'oublie jamais les gradients passés).

---

## 7. RMSProp

Corrige Adagrad en remplaçant la somme cumulée par une **moyenne mobile exponentielle**.

$$s_{t+1,i} = \gamma s_{t,i} + (1-\gamma)g_{t,i}^2 \qquad \theta_{t+1,i} = \theta_{t,i} - \frac{\eta}{\sqrt{s_{t+1,i} + \epsilon}}\, g_{t,i}$$

```python
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-8)
```

---

## 8. Adam

**Adam = Adaptive Moment Estimation**. Combine Momentum (1er moment) et RMSProp (2ème moment). **L'optimiseur le plus utilisé en pratique.**

### Formules

$$m_{t+1} = \beta_1 m_t + (1-\beta_1)g_t \qquad \text{(moyenne des gradients)}$$
$$v_{t+1} = \beta_2 v_t + (1-\beta_2)g_t^2 \qquad \text{(variance des gradients)}$$

**Correction du biais :**

$$\hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^{t+1}} \qquad \hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{t+1}}$$

**Mise à jour :**

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon}\, \hat{m}_{t+1}$$

### Paramètres par défaut

| Paramètre | Symbole | Valeur typique | Rôle |
|---|---|---|---|
| `lr` | $\eta$ | `1e-3` | Learning rate global |
| `betas[0]` | $\beta_1$ | `0.9` | Décroissance 1er moment |
| `betas[1]` | $\beta_2$ | `0.999` | Décroissance 2ème moment |
| `eps` | $\epsilon$ | `1e-8` | Stabilité numérique |

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

# AdamW : Adam + weight decay découplé (recommandé pour Transformers)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

### Groupes de paramètres (lr différents)

```python
optimizer = torch.optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.head.parameters(),     'lr': 1e-3},
])
```

---

## 9. AMSGrad

Variante d'Adam utilisant le **maximum cumulé** du 2ème moment (convergence garantie théoriquement).

$$\hat{v}_{t+1} = \max(\hat{v}_t, v_{t+1})$$

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
```

---

## 10. Learning Rate Scheduling

### StepLR

```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
# Divise le lr par 10 tous les 30 epochs

for epoch in range(num_epochs):
    train_one_epoch(...)
    scheduler.step()   # après chaque epoch
```

### CosineAnnealingLR

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t\pi}{T_{\max}}\right)\right)$$

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
```

### ReduceLROnPlateau

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

val_loss = evaluate(...)
scheduler.step(val_loss)   # passe la métrique (pas l'epoch)
```

### Comparaison des schedulers

| Scheduler | Quand utiliser | Paramètres clés |
|---|---|---|
| `StepLR` | Protocoles classiques | `step_size`, `gamma` |
| `MultiStepLR` | Jalons connus à l'avance | `milestones`, `gamma` |
| `CosineAnnealingLR` | Entraînements longs | `T_max`, `eta_min` |
| `ReduceLROnPlateau` | Plateau inconnu | `patience`, `factor` |

> `scheduler.step()` doit être appelé **après** `optimizer.step()`.

---

## 11. Problèmes courants : gradients

### Vanishing Gradients

**Symptômes** : loss stagne, premières couches n'apprennent pas, gradients ≈ 0.

```python
# Solutions :
nn.ReLU()                                     # activation non-saturante
nn.BatchNorm2d(channels)                      # normalise les activations
nn.init.kaiming_uniform_(layer.weight)        # initialisation He (pour ReLU)
nn.init.xavier_uniform_(layer.weight)         # initialisation Xavier (pour tanh)
```

### Exploding Gradients

**Symptômes** : loss diverge (NaN/inf), poids très grands.

```python
# Solutions :
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
# Réduire le learning rate
# Batch Normalization
```

---

## 12. Gradient Clipping

Si $\|\mathbf{g}\| > g_{\max}$ → rescaler : $\mathbf{g} \leftarrow \frac{g_{\max}}{\|\mathbf{g}\|} \cdot \mathbf{g}$

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # après backward
optimizer.step()                                                     # avant step
```

> **Ordre critique** : toujours clipper **après** `backward()` et **avant** `optimizer.step()`.

| Domaine | `max_norm` recommandé |
|---|---|
| RNN / LSTM | `1.0` — `5.0` |
| Transformers | `1.0` |
| CNN | `1.0` (si instabilité) |

---

## 13. Comparaison des optimiseurs

| Optimiseur | LR adaptatif | Mémoire extra | Points forts | Limitations |
|---|---|---|---|---|
| **SGD** | Non | Aucune | Simple, bonne généralisation | Sensible au lr |
| **SGD + Momentum** | Non | $O(p)$ | Amortit les oscillations | $\gamma$ à tuner |
| **Nesterov** | Non | $O(p)$ | Légèrement plus rapide | Plus complexe |
| **Adagrad** | Oui | $O(p)$ | Bon pour données sparses | LR → 0 |
| **RMSProp** | Oui | $O(p)$ | Stable, corrige Adagrad | Pas de bias correction |
| **Adam** | Oui | $O(2p)$ | Polyvalent, robuste | Peut sous-performer SGD |
| **AdamW** | Oui | $O(2p)$ | Weight decay découplé | — |
| **AMSGrad** | Oui | $O(3p)$ | Convergence garantie | Parfois plus lent |

### Recommandations pratiques

- **Adam** (`lr=1e-3`) : point de départ universel
- **AdamW** : Transformers, fine-tuning de modèles pré-entraînés
- **SGD + Momentum** : vision (ResNet, VGG) pour la meilleure généralisation
- **RMSProp** : RNN, apprentissage par renforcement

---

## 14. Référence rapide

### Paramètres recommandés

| Optimiseur | `lr` | Autres | Cas typique |
|---|---|---|---|
| `SGD` | `0.01`–`0.1` | `momentum=0.9`, `weight_decay=1e-4` | Vision |
| `Adam` | `1e-3` | `betas=(0.9, 0.999)` | Cas général |
| `AdamW` | `5e-4` | `weight_decay=0.01` | Transformers |
| `RMSprop` | `1e-3` | `alpha=0.9` | RNN, RL |

### Formules résumées

| Algorithme | Mise à jour |
|---|---|
| SGD | $\theta \leftarrow \theta - \eta\, g$ |
| Momentum | $v \leftarrow \gamma v + \eta\, g$ ; $\theta \leftarrow \theta - v$ |
| RMSProp | $s \leftarrow \gamma s + (1-\gamma)g^2$ ; $\theta \leftarrow \theta - \frac{\eta}{\sqrt{s+\epsilon}} g$ |
| Adam | $\theta \leftarrow \theta - \frac{\eta}{\sqrt{\hat{v}}+\epsilon}\hat{m}$ |
| AMSGrad | Idem Adam avec $\hat{v} \leftarrow \max(\hat{v}_{\text{prev}}, v)$ |

### Checklist d'entraînement

- [ ] `optimizer.zero_grad()` avant chaque batch
- [ ] `loss.backward()` pour calculer les gradients
- [ ] `clip_grad_norm_(..., max_norm=1.0)` si RNN ou Transformer
- [ ] `optimizer.step()` pour mettre à jour les poids
- [ ] `scheduler.step()` pour mettre à jour le lr (après chaque epoch)
