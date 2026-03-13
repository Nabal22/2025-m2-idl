# TP-2 Cheatsheet — Tenseurs & Régression Linéaire

---

## Partie 1 — Tenseurs PyTorch

### Création

```python
torch.zeros(3, 4)          # zéros, forme (3,4)
torch.ones(5)              # uns
torch.empty(3)             # non initialisé (rapide, mais valeurs arbitraires !)
torch.rand(5)              # uniforme [0, 1)
torch.randn(5)             # normale N(0, 1)
torch.randint(0, 10, (5,)) # entiers dans [0, 10)
torch.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
torch.linspace(0, 1, 5)    # [0.0, 0.25, 0.5, 0.75, 1.0]
torch.tensor([1, 2, 3])    # depuis une liste Python
torch.from_numpy(arr)      # depuis un tableau NumPy (mémoire partagée !)
```

**Types disponibles :** `torch.float32` (défaut), `torch.float64`, `torch.int32`, `torch.long`

```python
torch.zeros(5, dtype=torch.int32)
x.float()   # convertit en float32
x.long()    # convertit en int64
```

---

### Indexation & Slicing

```python
x[0]          # premier élément
x[-1]         # dernier élément
x[2:5]        # éléments 2, 3, 4
x[::2]        # un élément sur deux
x[5].item()   # extraire la valeur scalaire Python

# 2D
matrix[1, 2]     # ligne 1, colonne 2
matrix[0, :]     # première ligne entière
matrix[:, 0]     # première colonne entière
matrix[1:, 2:]   # sous-matrice depuis ligne 1, col 2

# Avancé
x[[0, 2, 4]]              # indexation par liste
x[torch.tensor([1, 3])]   # indexation par tenseur
x[x > 5]                  # masque booléen

# Ellipsis (batch, canaux, H, W)
x[0, ...]   # == x[0, :, :, :]
x[..., -1]  # dernier élément sur la dernière dimension
```

> Les slices sont des **vues** — les modifier modifie le tenseur original.

---

### Broadcasting

Règles (appliquées de droite à gauche sur les formes) :
1. Compléter la forme avec des 1 à gauche si nécessaire
2. Si une dimension vaut 1, elle est étendue pour correspondre à l'autre
3. Si les dimensions diffèrent et qu'aucune ne vaut 1 → **erreur**

```python
# (3,) + scalaire → OK
torch.tensor([1, 2, 3]) * 2

# (4, 1) + (4,) → (4, 4)
a = torch.arange(0, 40, 10).unsqueeze(1)  # forme (4, 1)
b = torch.arange(0, 4)                    # forme (4,)
a + b  # → forme (4, 4)

# (4, 3) + (4,) → ERREUR (3 ≠ 4, aucune dimension ne vaut 1)
```

---

### Opérations utiles

```python
x.shape             # torch.Size([3, 4])
x.reshape(2, 6)     # changer la forme
x.unsqueeze(1)      # ajouter une dimension en index 1
x.permute(0,3,1,2)  # réordonner les dimensions (ex. BHWC → BCHW)
x.t()               # transposée 2D
x.numpy()           # vers NumPy (CPU uniquement)

# Opérations en place (suffixées par _)
x.add_(y)   # x += y, modifie x en place
x.t_()      # transposée en place
x.fill_(0)  # remplir avec une valeur
```

---

### GPU / Device

```python
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

x = x.to(device)                        # déplacer vers le GPU
y = torch.ones_like(x, device=device)   # créer directement sur le GPU
z = z.to("cpu")                         # retour sur CPU

x.item()        # tenseur scalaire → float Python (fonctionne sur tout device)
x.cpu().numpy() # tenseur → numpy (doit être sur CPU)
```

---

### Pont NumPy — Attention !

```python
a = np.ones(5)
b = torch.from_numpy(a)   # mémoire partagée !
a[2] = 0   # → b[2] vaut aussi 0
b[3] = 5   # → a[3] vaut aussi 5
```

Attention aux types : `float32 + float64 = float64`

---

## Partie 2 — Régression Linéaire

### Le modèle

$$\hat{y} = w \cdot x + b$$

```python
def predict(x, w, b):
    return w * x + b
```

---

### Fonction de perte : Erreur Quadratique Moyenne (MSE)

$$L(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

```python
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()
```

---

### Gradients (calcul manuel, règle de la chaîne)

$$\frac{\partial L}{\partial w} = -\frac{2}{n} \sum (y_i - \hat{y}_i) \cdot x_i$$

$$\frac{\partial L}{\partial b} = -\frac{2}{n} \sum (y_i - \hat{y}_i)$$

```python
def compute_gradients(x, y_true, y_pred):
    error = y_true - y_pred
    grad_w = -2 * (error * x).sum() / len(x)
    grad_b = -2 * error.sum() / len(x)
    return grad_w, grad_b
```

---

### Boucle de descente de gradient

```python
w = torch.tensor([0.5])
b = torch.tensor([0.5])
lr = 0.1

for _ in range(200):
    y_pred = predict(x, w, b)
    loss = mse_loss(y, y_pred)
    grad_w, grad_b = compute_gradients(x, y, y_pred)
    w = w - lr * grad_w
    b = b - lr * grad_b
```

---

### Standardisation — Pourquoi & Comment

Sans standardisation : $w$ et $b$ ont des gradients de magnitudes très différentes → convergence lente et déséquilibrée.

$$x_{\text{std}} = \frac{x - \mu_x}{\sigma_x}$$

```python
x_std = (x - x.mean()) / x.std()
y_std = (y - y.mean()) / y.std()
```

**Retour à l'échelle originale après entraînement :**
```python
w_orig = w * y.std() / x.std()
b_orig = b * y.std() + y.mean() - w_orig * x.mean()
```

Avantages : taux d'apprentissage plus grand possible, convergence plus rapide, surface de perte plus "circulaire".

---

### Autodiff (autograd PyTorch)

```python
w = torch.tensor([0.5], requires_grad=True)
b = torch.tensor([0.5], requires_grad=True)

for _ in range(200):
    y_pred = predict(x_std, w, b)
    loss = mse_loss(y_std, y_pred)

    loss.backward()                  # calcule les gradients automatiquement

    with torch.no_grad():            # mise à jour sans tracking
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()                   # IMPORTANT : remettre les gradients à zéro !
    b.grad.zero_()
```

> Toujours appeler `.grad.zero_()` — les gradients s'accumulent par défaut.

---

### PyTorch nn.Linear (pipeline complet)

```python
model = torch.nn.Linear(1, 1)           # 1 entrée, 1 sortie
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

X = x_std.unsqueeze(1)  # forme (n, 1)
y = y_std.unsqueeze(1)  # forme (n, 1)

for _ in range(200):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()   # remettre les gradients à zéro
    loss.backward()         # rétropropagation
    optimizer.step()        # mettre à jour les paramètres
```

---

### Moindres Carrés (solution analytique)

```python
X = torch.stack([surface, torch.ones_like(surface)], dim=1)  # (n, 2)
y = rent.unsqueeze(1)
solution = torch.linalg.lstsq(X, y).solution
# → [w, b]
```

---

## Référence rapide

| Opération | Code |
|---|---|
| Créer un tenseur | `torch.tensor([1,2,3])` |
| Déplacer vers GPU | `x.to(device)` |
| Scalaire → Python | `x.item()` |
| Ajouter une dimension | `x.unsqueeze(0)` |
| Réordonner les dims | `x.permute(0,3,1,2)` |
| Masque booléen | `x[x > 0]` |
| Perte MSE | `((y - y_hat)**2).mean()` |
| Activer le gradient | `requires_grad=True` |
| Rétropropagation | `loss.backward()` |
| Remettre gradients à zéro | `optimizer.zero_grad()` |
| Mettre à jour les paramètres | `optimizer.step()` |
| Standardiser | `(x - x.mean()) / x.std()` |
