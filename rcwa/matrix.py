import torch
import numpy as np

from rcwa.params import SYMMETRY


class Matrix(torch.Tensor):
    """
    A simple, dense, complex matrix inheriting from torch.Tensor.
    Supports basic operations and inversion.
    Default dtype: complex128, device: cuda if available else cpu.
    """
    DEFAULT_DTYPE = torch.complex128
    DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __new__(cls, data, *args, dtype=None, device=None, **kwargs):
        # If input is a torch.Tensor or Matrix, use its device unless explicitly overridden
        if isinstance(data, torch.Tensor):
            device = device if device is not None else data.device
            dtype = dtype if dtype is not None else data.dtype
        else:
            device = device if device is not None else cls.DEFAULT_DEVICE
            dtype = dtype if dtype is not None else cls.DEFAULT_DTYPE
            data = torch.as_tensor(data, dtype=dtype, device=device)
        obj = torch.Tensor._make_subclass(cls, data, require_grad=False)
        return obj

    def __matmul__(self, other):
        return Matrix(torch.matmul(self, other))

    def __add__(self, other):
        return Matrix(torch.add(self, other))

    def __sub__(self, other):
        return Matrix(torch.sub(self, other))

    def __mul__(self, other):
        """
        Point-wise (Hadamard) multiplication if other is a matrix/tensor of the same shape.
        Scalar multiplication if other is a number (int, float, complex).
        """
        if isinstance(other, (int, float, complex)):
            return Matrix(super().__mul__(other))
        return Matrix(torch.mul(self, other))

    def __rmul__(self, other):
        """
        Scalar multiplication if other is a number (int, float, complex).
        Point-wise (Hadamard) multiplication if other is a matrix/tensor of the same shape.
        """
        if isinstance(other, (int, float, complex)):
            return Matrix(super().__mul__(other))
        return Matrix(torch.mul(other, self))

    def inv(self):
        """Return the matrix inverse."""
        return Matrix(torch.inverse(self))

