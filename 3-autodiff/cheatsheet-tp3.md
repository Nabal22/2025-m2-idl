# TP-3 Cheatsheet — Autodiff & Rétropropagation

---

## 1. Différentiation automatique — Vue d'ensemble

La **différentiation automatique** (autodiff) permet de calculer exactement les dérivées de fonctions définies par du code, en appliquant la règle de la chaîne de manière systématique sur un graphe de calcul.

### Règle de la chaîne (Chain Rule)

Pour $\ell = (\exp(wx + b) - y)^2$, on décompose en opérations élémentaires :

$$\frac{\partial \ell}{\partial w} = \frac{\partial \ell}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial v} \cdot \frac{\partial v}{\partial u} \cdot \frac{\partial u}{\partial w}$$

### Forward mode vs Backward mode

| Critère | Forward mode | Backward mode (backprop) |
|---|---|---|
| Direction | Entrées → Sorties | Sorties → Entrées |
| Passes pour $n$ entrées, $1$ sortie | $n$ passes | **1 seule passe** |
| Passes pour $1$ entrée, $m$ sorties | **1 seule passe** | $m$ passes |
| Cas optimal | Peu d'entrées, beaucoup de sorties | Beaucoup d'entrées (paramètres), 1 sortie |
| Utilisation en deep learning | Rare | **Quasi-exclusif** (PyTorch, TensorFlow…) |
| Implémentation naturelle | Nombres duaux | Graphe de calcul + accumulation des gradients |

> **Pourquoi le backward mode domine en deep learning ?** Un réseau de neurones possède des millions de paramètres (entrées du calcul de gradient) et une seule loss scalaire (sortie). Le backward mode calcule tous les gradients en une seule passe.

---

## 2. Nombres duaux (Dual Numbers)

### Concept

Un **nombre dual** étend les réels en introduisant un élément $\varepsilon$ tel que :

$$\varepsilon^2 = 0$$

Un nombre dual a la forme $a + b\varepsilon$ où :
- $a$ est la **partie réelle** (valeur de la fonction)
- $b$ est la **partie duale** (valeur de la dérivée)

### Propriété fondamentale

En évaluant $f$ en $x + \varepsilon$ :

$$f(x + \varepsilon) = f(x) + f'(x)\varepsilon$$

La partie réelle donne $f(x)$, la partie duale donne $f'(x)$ — c'est le **forward mode AD**.

### Opérations algébriques

| Opération | Résultat |
|---|---|
| Addition | $(a + b\varepsilon) + (c + d\varepsilon) = (a+c) + (b+d)\varepsilon$ |
| Soustraction | $(a + b\varepsilon) - (c + d\varepsilon) = (a-c) + (b-d)\varepsilon$ |
| Multiplication | $(a + b\varepsilon)(c + d\varepsilon) = ac + (ad+bc)\varepsilon$ |
| Division | $\dfrac{a + b\varepsilon}{c + d\varepsilon} = \dfrac{a}{c} + \dfrac{bc - ad}{c^2}\varepsilon$ |
| Puissance | $(a + b\varepsilon)^n = a^n + na^{n-1}b\varepsilon$ |

> La multiplication reproduit la **règle du produit**, la division reproduit la **règle du quotient**, la puissance reproduit la **règle des puissances** — de façon automatique.

### Implémentation Python

```python
class Dual:
    def __init__(self, value, epsilon=0.0):
        self.value = float(value)
        self.epsilon = float(epsilon)

    def __repr__(self):
        return f"{self.value} + {self.epsilon}ε"

    def __add__(self, other):
        if not isinstance(other, Dual): other = Dual(other)
        return Dual(self.value + other.value, self.epsilon + other.epsilon)

    def __radd__(self, other): return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, Dual): other = Dual(other)
        return Dual(self.value - other.value, self.epsilon - other.epsilon)

    def __rsub__(self, other): return Dual(other - self.value, -self.epsilon)

    def __mul__(self, other):
        if not isinstance(other, Dual): other = Dual(other)
        return Dual(
            self.value * other.value,
            self.value * other.epsilon + self.epsilon * other.value,
        )

    def __rmul__(self, other): return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, Dual): other = Dual(other)
        return Dual(
            self.value / other.value,
            (other.value * self.epsilon - self.value * other.epsilon) / other.value**2,
        )

    def __pow__(self, n):
        return Dual(self.value**n, n * self.value ** (n - 1) * self.epsilon)


def derivative(f, x):
    """Calcule f'(x) via les nombres duaux (forward mode)."""
    d = f(Dual(x, 1))
    return d.epsilon if isinstance(d, Dual) else 0
```

### Exemples

```python
# Dérivée de P(x) = 1 + x + 3x² en x=1  →  P'(1) = 1 + 6 = 7
derivative(lambda x: 1 + x + 3 * x**2, 1)   # → 7.0

a = Dual(3, 1)
print(a ** 2)    # → 9.0 + 6.0ε   (dérivée de x² en x=3 = 6)
print(a ** 7)    # → 2187.0 + 5103.0ε
```

---

## 3. Graphe de calcul (Computational Graph)

### Décomposition de $\ell = (\exp(wx + b) - y)^2$

```
x, w  →  [u = w × x]  →  [v = u + b]  →  [ŷ = exp(v)]  →  [ℓ = (ŷ - y)²]
     b ──────────────────────────↗                   y ───────────────────↗
```

### Dérivées locales de chaque nœud

$$\frac{\partial u}{\partial w} = x \qquad \frac{\partial v}{\partial u} = 1 \qquad \frac{\partial \hat{y}}{\partial v} = \exp(v) = \hat{y} \qquad \frac{\partial \ell}{\partial \hat{y}} = 2(\hat{y} - y)$$

### Backward pass — étapes détaillées

$$
\begin{align}
&\text{1. Initialisation :} & \frac{\partial \ell}{\partial \ell} &= 1 \\[6pt]
&\text{2. Loss :} & \frac{\partial \ell}{\partial \hat{y}} &= 2(\hat{y}-y) \\[6pt]
&\text{3. Exponentielle :} & \frac{\partial \ell}{\partial v} &= 2(\hat{y}-y) \cdot \hat{y} \\[6pt]
&\text{4. Addition :} & \frac{\partial \ell}{\partial u} &= 2(\hat{y}-y) \cdot \hat{y} \\[6pt]
&\text{5. Multiplication :} & \frac{\partial \ell}{\partial w} &= 2(\hat{y}-y) \cdot \hat{y} \cdot x
\end{align}
$$

$$\boxed{\frac{\partial \ell}{\partial w} = 2(\hat{y} - y) \cdot \hat{y} \cdot x \qquad \frac{\partial \ell}{\partial b} = 2(\hat{y} - y) \cdot \hat{y}}$$

> Le backward mode obtient $\frac{\partial \ell}{\partial w}$ **et** $\frac{\partial \ell}{\partial b}$ en une seule passe.

---

## 4. PyTorch : `requires_grad`, `backward()`, `.grad`

```python
import torch

w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(1.0)   # pas de gradient nécessaire sur les données
y = torch.tensor(7.5)

# Forward — PyTorch construit le graphe dynamiquement
y_hat = torch.exp(w * x + b)
loss  = (y_hat - y) ** 2

# Backward — calcule tous les gradients en une passe
loss.backward()

print(w.grad)   # ∂loss/∂w
print(b.grad)   # ∂loss/∂b
```

> `backward()` ne peut être appelé **qu'une seule fois** par défaut (le graphe est libéré). Utiliser `retain_graph=True` si nécessaire.

---

## 5. Rétropropagation manuelle — Implémentation pas à pas

```python
import numpy as np

class add_bias:
    """v = u + b"""
    def __init__(self, b): self.b = b

    def forward(self, x): return x + self.b

    def backward(self, grad):
        self.grad = grad   # ∂ℓ/∂b = grad
        return grad        # ∂ℓ/∂u = grad

    def step(self, lr): self.b -= lr * self.grad


class multiplication_weight:
    """u = w * x"""
    def __init__(self, w): self.w = w

    def forward(self, x):
        self.x = x         # mémoriser x
        return self.w * x

    def backward(self, grad):
        self.grad = grad * self.x   # ∂ℓ/∂w = grad · x
        return grad * self.w        # ∂ℓ/∂x = grad · w

    def step(self, lr): self.w -= lr * self.grad


class my_exp:
    """ŷ = exp(v)"""
    def forward(self, x):
        self.output = np.exp(x)
        return self.output

    def backward(self, grad): return grad * self.output  # grad · exp(v)

    def step(self, lr): pass


class my_composition:
    def __init__(self, layers): self.layers = layers

    def forward(self, x):
        for layer in self.layers: x = layer.forward(x)
        return x

    def compute_loss(self, y_est, y_target):
        loss = (y_est - y_target) ** 2
        self.grad_loss = 2 * (y_est - y_target)
        return loss

    def backward(self):
        grad = self.grad_loss
        for layer in reversed(self.layers):   # parcours inverse !
            grad = layer.backward(grad)

    def step(self, lr):
        for layer in self.layers: layer.step(lr)
```

### Boucle SGD complète

```python
model = my_composition([
    multiplication_weight(1.0),   # w initial (vrai = 0.5)
    add_bias(1.0),                # b initial (vrai = 2.0)
    my_exp()
])

for i in range(5000):
    j      = np.random.randint(0, len(xx))
    y_pred = model.forward(xx[j])
    loss   = model.compute_loss(y_pred, yy[j])
    model.backward()
    model.step(lr=1e-4)
```

---

## 6. `torch.no_grad()` et `detach()`

```python
# no_grad : désactive la construction du graphe (inférence / validation)
with torch.no_grad():
    y = model(x)           # pas de grad_fn

# detach : détache un tenseur du graphe (partage les données, sans grad)
z = y.detach()
arr = y.detach().numpy()   # nécessaire pour convertir en numpy avec requires_grad
```

| | `torch.no_grad()` | `.detach()` |
|---|---|---|
| Portée | Bloc entier | Tenseur spécifique |
| Usage | Inférence, évaluation | Couper le gradient à un point précis |
| Mémoire | Économise le graphe | Partage les données (pas de copie) |

---

## 7. Accumulation des gradients et `zero_grad()`

```python
# PROBLÈME : gradients accumulés sans zero_grad
w = torch.tensor(1.0, requires_grad=True)
(w**2).backward()
print(w.grad)   # 2.0
(w**2).backward()
print(w.grad)   # 4.0  ← ACCUMULATION ! (2 + 2)

# SOLUTION : toujours appeler zero_grad() avant backward
optimizer.zero_grad()   # remet tous les .grad à 0
loss.backward()
optimizer.step()
```

### Pattern complet d'une boucle d'entraînement

```python
for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()               # 1. Remise à zéro
        y_pred = model(x_batch)             # 2. Forward
        loss   = criterion(y_pred, y_batch) # 3. Loss
        loss.backward()                     # 4. Backward
        optimizer.step()                    # 5. Mise à jour

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(x_val), y_val)
```

### Gradient accumulation (simulation de grands batchs)

```python
K = 4   # simuler un batch K fois plus grand
optimizer.zero_grad()
for i, (x, y) in enumerate(dataloader):
    loss = criterion(model(x), y) / K
    loss.backward()
    if (i + 1) % K == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 8. Référence rapide

### Tableau des concepts clés

| Concept | Syntaxe PyTorch |
|---|---|
| Activer le suivi | `torch.tensor(..., requires_grad=True)` |
| Forward pass | `y = model(x)` |
| Backward pass | `loss.backward()` |
| Accéder au gradient | `param.grad` |
| Remettre à zéro | `optimizer.zero_grad()` |
| Inférence sans graphe | `with torch.no_grad():` |
| Couper le gradient | `t.detach()` |
| Vers numpy (avec grad) | `t.detach().numpy()` |
| Backward multiple | `loss.backward(retain_graph=True)` |

### Dérivées locales des opérations communes

| Opération $f(x)$ | Backward : `grad_in = grad_out ×` |
|---|---|
| $f(x) = ax$ | `a` |
| $f(x) = x + c$ | `1` |
| $f(x) = \exp(x)$ | `output` (valeur mémorisée) |
| $f(x) = \ln(x)$ | `1 / x` |
| $f(x) = \sigma(x)$ | `output * (1 - output)` |
| $f(x) = \text{ReLU}(x)$ | `(x > 0).float()` |
| $f(x, y) = x + y$ | chaque entrée reçoit `grad_out` |
| $f(x, y) = x \cdot y$ | `grad_out * y` pour x, `grad_out * x` pour y |

### Checklist de débogage des gradients

```
[ ] optimizer.zero_grad() avant loss.backward() ?
[ ] loss est bien un scalaire ?
[ ] Les paramètres ont requires_grad=True ?
[ ] model.eval() + torch.no_grad() en validation ?
[ ] .detach().numpy() et non .numpy() directement ?
[ ] Pas d'opérations in-place (+=) sur des tenseurs dans le graphe ?
```

### Forward vs Backward mode — résumé

```
Forward mode (nombres duaux)
  Entrée    : x + 1·ε
  Direction : gauche → droite
  Résultat  : f(x) + f'(x)·ε en une passe
  Coût      : 1 passe PAR variable d'entrée
  Idéal     : peu d'entrées, beaucoup de sorties

Backward mode (PyTorch autograd)
  Phase 1   : forward pass — calcule et stocke les intermédiaires
  Phase 2   : backward pass — remonte le graphe depuis ∂ℓ/∂ℓ = 1
  Résultat  : ∂ℓ/∂θ pour TOUS les paramètres en une seule passe
  Idéal     : beaucoup de paramètres, une sortie scalaire
```
