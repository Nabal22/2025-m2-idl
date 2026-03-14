# Forward mode AD with dual numbers

_Adapted from [Dataflowr Module 2](https://github.com/dataflowr/notebooks/blob/master/Module2/AD_with_dual_numbers_Julia.ipynb) by Marc Lelarge (which is adapted from [Automatic Differentiation in 10 minutes with Julia](https://youtu.be/vAp6nUMrKYg))_


[Dual numbers](https://en.wikipedia.org/wiki/Automatic_differentiation#Automatic_differentiation_using_dual_numbers) extend the real numbers by introducing a new element $\varepsilon$ (epsilon) with the special property:
$$\varepsilon^2 = 0$$

A dual number has the form:
$$a + b\varepsilon$$
where $a$ is the **real part** (or value) and $b$ is the **dual part** (or epsilon coefficient).

The key insight: if we evaluate a function $f$ at the dual number $x + \varepsilon$, we get:
$$f(x + \varepsilon) = f(x) + f'(x)\varepsilon$$

The real part gives us the function value, and the dual part gives us the derivative! This is **forward-mode automatic differentiation**.

## Operations on dual numbers

**Addition:**
$$(a + b\varepsilon) + (c + d\varepsilon) = (a + c) + (b + d)\varepsilon$$

**Subtraction:**
$$(a + b\varepsilon) - (c + d\varepsilon) = (a - c) + (b - d)\varepsilon$$

**Multiplication:**
$$
\begin{align*}
(a + b\varepsilon)(c + d\varepsilon) &= ac + ad\varepsilon + bc\varepsilon + bd\varepsilon^2
\\
&= ac + (ad + bc)\varepsilon + bd \cdot 0 \quad (\text{using $\varepsilon^2 = 0$}) \\
&= ac + (ad + bc)\varepsilon
\end{align*}
$$

_Note_: This matches the product rule: if $f(x) = u(x) \cdot v(x)$, then $f'(x) = u'(x)v(x) + u(x)v'(x)$.

**Division:** 
$$
\begin{align*}
\frac{a + b\varepsilon}{c + d\varepsilon} &= \frac{a + b\varepsilon}{c + d\varepsilon} \cdot \frac{c - d\varepsilon}{c - d\varepsilon}\\\\
&= \frac{(a + b\varepsilon)(c - d\varepsilon)}{c^2}\\\\
&=\frac{ac - ad\varepsilon + bc\varepsilon - bd\varepsilon^2}{c^2}\\\\
&= \frac{a}{c} + \frac{bc - ad}{c^2}\varepsilon
\end{align*}
$$

Using: $(c + d\varepsilon)(c - d\varepsilon) = c^2 - d^2\varepsilon^2 = c^2$:


_Note_: This matches the quotient rule: if $f(x) = \frac{u(x)}{v(x)}$, then $f'(x) = \frac{u'(x)v(x) - u(x)v'(x)}{v(x)^2}$.

**Power:**
$$
\begin{align*}
(a + b\varepsilon)^n &= a^n \left(1 + \frac{b}{a}\varepsilon\right)^n \\\\
&= a^n \left(1 + n\frac{b}{a}\varepsilon + \binom{n}{2}\left(\frac{b}{a}\varepsilon\right)^2 + \ldots\right) \\\\
&= a^n \left(1 + n\frac{b}{a}\varepsilon\right) \quad \text{(higher order terms vanish)} \\\\
&= a^n + na^{n-1}b\varepsilon
\end{align*}
$$

This matches the power rule: if $f(x) = x^n$, then $f'(x) = nx^{n-1}$.

## Implementation in Python

Using Python, we can create a class for dual numbers as follows by redefining the meaning of arithmetic operators.


```python
class Dual:
    def __init__(self, value, epsilon=0.0):
        self.value = float(value)
        self.epsilon = float(epsilon)

    def __repr__(self):
        return f"{self.value} + {self.epsilon}ε"

    def __add__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return Dual(self.value + other.value, self.epsilon + other.epsilon)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return Dual(self.value - other.value, self.epsilon - other.epsilon)

    def __rsub__(self, other):
        return Dual(other - self.value, -self.epsilon)

    def __mul__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return Dual(
            self.value * other.value,
            self.value * other.epsilon + self.epsilon * other.value,
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return Dual(
            self.value / other.value,
            (other.value * self.epsilon - self.value * other.epsilon) / other.value**2,
        )

    def __rtruediv__(self, other):
        return Dual(other / self.value, -other * self.epsilon / self.value**2)

    def __pow__(self, n):
        return Dual(self.value**n, n * self.value ** (n - 1) * self.epsilon)
```

We can initialize a dual number with a value and $1$ for the $\varepsilon$ part.


```python
a = Dual(3, 1)
a
```




    3.0 + 1.0ε



We are now ready to start playing with dual numbers.


```python
a * a
```




    9.0 + 6.0ε




```python
a**2
```




    9.0 + 6.0ε




```python
a**7
```




    2187.0 + 5103.0ε




```python
2 * a
```




    6.0 + 2.0ε




```python
a - 1
```




    2.0 + 1.0ε




```python
b = Dual(1, 1)
3 * (a + b) ** 2
```




    48.0 + 48.0ε



## Automatic differentiation for polynomials

We can now get derivatives for polynomials. Consider $P(x) = p_0+p_1x+p_2x^2+\dots +p_n x^n$, note that since $\varepsilon^2 =0$ ($\varepsilon$ is nilpotent), we have $(x+\varepsilon y)^k = x^k + kx^{k-1}\varepsilon y$. Hence, we have
$$
P(x+\varepsilon y) = P(x) +  \left(p_1 + 2p_2 x+ \dots np_n x^{n-1}\right)y\varepsilon = P(x) + P'(x)y\varepsilon 
$$


```python
a = Dual(2, 1)
```


```python
3 * (1 + a) ** 2
```




    27.0 + 18.0ε




```python
def value(z):
    return z.value if isinstance(z, Dual) else z


def epsilon(z):
    return z.epsilon if isinstance(z, Dual) else 0
```


```python
epsilon(3 * (1 + a) ** 2)
```




    18.0



We can now define a simple function to compute the derivative of a polynomial at a given point as follows:


```python
def derivative(f, x):
    return epsilon(f(Dual(x, 1)))
```


```python
derivative(lambda x: 1 + x + 3 * x**2, 1)
```




    7.0



## Going further and dealing with $\sqrt{}$

To deal with this problem, we follow the Babylonian method as described in this nice [tutorial](https://github.com/JuliaAcademy/JuliaTutorials/blob/master/introductory-tutorials/intro-to-julia/AutoDiff.ipynb)

> Repeat $ t \leftarrow  \frac{1}{2}\left(t+\frac{x}{t}\right)$ until $t$ converges to $\sqrt{x}$.


```python
def babylonian(x, N=10):
    t = (1 + x) / 2
    for i in range(N):
        t = (t + x / t) / 2
    return t
```


```python
import math

babylonian(2), math.sqrt(2)
```




    (1.414213562373095, 1.4142135623730951)



The Babylonian algorithm uses only addition and division. Division for dual numbers is already implemented:
$$\frac{a+b\varepsilon}{c+d\varepsilon}= \frac{a}{c}\left(1+\frac{b}{a}\varepsilon\right)\left(1-\frac{d}{c}\varepsilon\right)= \frac{a}{c} +\frac{bc-ad}{c^2}\varepsilon.$$


```python
a
```




    2.0 + 1.0ε




```python
1 / (1 + a)
```




    0.3333333333333333 + -0.1111111111111111ε




```python
babylonian(a), math.sqrt(2), 0.5 / math.sqrt(2)
```




    (1.414213562373095 + 0.35355339059327373ε,
     1.4142135623730951,
     0.35355339059327373)




```python
derivative(lambda x: babylonian(x), 2)
```




    0.35355339059327373



We can also manually compute the derivative of the Babylonian algorithm:


```python
def d_babylonian(x, N=10):
    t = (1 + x) / 2
    dt = 0.5
    for i in range(N):
        t = (t + x / t) / 2
        dt = (dt + (t - x * dt) / t**2) / 2
    return dt
```


```python
x = 2
d_babylonian(x), 0.5 / math.sqrt(x)
```




    (0.35355339059327373, 0.35355339059327373)




```python
derivative(lambda x: babylonian(x) ** 4, 4)
```




    8.0




```python
derivative(lambda x: (1 + 3 * babylonian(x)) ** 3 / babylonian(x), 2)
```




    36.36916485072176



## Comparison with PyTorch's autograd

PyTorch provides automatic differentiation through its autograd system:


```python
import torch

# Using PyTorch autograd
x = torch.tensor(2.0, requires_grad=True)
y = torch.sqrt(x)
y.backward()
print(f"PyTorch gradient: {x.grad.item()}")
print(f"Our dual number: {derivative(lambda x: babylonian(x), 2)}")
```

    PyTorch gradient: 0.3535533845424652
    Our dual number: 0.35355339059327373



```python
# More complex example
x = torch.tensor(2.0, requires_grad=True)
y = (1 + 3 * torch.sqrt(x)) ** 3 / torch.sqrt(x)
y.backward()
print(f"PyTorch gradient: {x.grad.item()}")
print(
    f"Our dual number: {derivative(lambda x: (1 + 3*babylonian(x))**3 / babylonian(x), 2)}"
)
```

    PyTorch gradient: 36.369163513183594
    Our dual number: 36.36916485072176



```python

```
