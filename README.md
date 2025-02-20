# kerneloose

Python implementation of the kernel method for out-of-sample extension (OOSE) of dimensionality reduction techniques.

Based on ["Parametric nonlinear dimensionality reduction using kernel t-SNE" by Gisbrecht, Schulz, and Hammer](https://www.sciencedirect.com/science/article/pii/S0925231214007036).

The kernel method is particularly useful for projection techniques that are computationally expensive and/or have non-convex objective functions, such as [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).

## Installation

```
pip install kerneloose
```

## Usage example

The syntax follows scikit learn conventions.
Assume `hd_data` is a numpy array containing high-dimensional data, and an array `ld_data` of equal length but lower dimension was obtained by some projection technique.
An OOSE of that projection can be obtained by:

```python
from kerneloose import KernelMap

kernel_oose = KernelMap()
kernel_oose.fit(hd_data, ld_data)
```

The mapping can be applied to `new_data` (with same dimensionality as `ld_data`) simply by:

```python
kernel_oose.transform(new_data)
```

Parameters of the calculated OOSE mapping can be saved and loaded for later use:

```python
kernel_oose.save('some/file/name')

resume_later = KernelMap()
resume_later.load('some/file/name')
```
