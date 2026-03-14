# Quizz!

## Easy


```python
import torch
```


```python
torch.ones(10) * 2
```




    tensor([2., 2., 2., 2., 2., 2., 2., 2., 2., 2.])




```python
torch.arange(10)
```




    tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
torch.arange(10)[-1]
```




    tensor(9)




```python
torch.arange(10)[2:]
```




    tensor([2, 3, 4, 5, 6, 7, 8, 9])




```python
torch.arange(10)[:-2]
```




    tensor([0, 1, 2, 3, 4, 5, 6, 7])



## Medium


```python
torch.cat((torch.zeros(10), torch.ones(10)))
```




    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])




```python
# print(torch.zeros((10,3)))
# print(torch.ones((10, 2)))

torch.cat((torch.zeros((10,3)), torch.ones((10, 2))), dim=-1)
```




    tensor([[0., 0., 0., 1., 1.],
            [0., 0., 0., 1., 1.],
            [0., 0., 0., 1., 1.],
            [0., 0., 0., 1., 1.],
            [0., 0., 0., 1., 1.],
            [0., 0., 0., 1., 1.],
            [0., 0., 0., 1., 1.],
            [0., 0., 0., 1., 1.],
            [0., 0., 0., 1., 1.],
            [0., 0., 0., 1., 1.]])



## Hard


```python
print(torch.arange(10)[..., None])
print(torch.arange(10))
```

    tensor([[0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8],
            [9]])
    tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])



```python
torch.abs(torch.arange(10)[..., None] - torch.arange(10))
```




    tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            [2, 1, 0, 1, 2, 3, 4, 5, 6, 7],
            [3, 2, 1, 0, 1, 2, 3, 4, 5, 6],
            [4, 3, 2, 1, 0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0, 1, 2, 3, 4],
            [6, 5, 4, 3, 2, 1, 0, 1, 2, 3],
            [7, 6, 5, 4, 3, 2, 1, 0, 1, 2],
            [8, 7, 6, 5, 4, 3, 2, 1, 0, 1],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])




```python
torch.arange(10)[:, None] == torch.arange(10)
```




    tensor([[ True, False, False, False, False, False, False, False, False, False],
            [False,  True, False, False, False, False, False, False, False, False],
            [False, False,  True, False, False, False, False, False, False, False],
            [False, False, False,  True, False, False, False, False, False, False],
            [False, False, False, False,  True, False, False, False, False, False],
            [False, False, False, False, False,  True, False, False, False, False],
            [False, False, False, False, False, False,  True, False, False, False],
            [False, False, False, False, False, False, False,  True, False, False],
            [False, False, False, False, False, False, False, False,  True, False],
            [False, False, False, False, False, False, False, False, False,  True]])




```python

```
