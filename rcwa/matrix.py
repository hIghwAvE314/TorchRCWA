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

    def inv(self) -> 'Matrix':
        """Return the matrix inverse."""
        return Matrix(torch.inverse(self))

    def view(self) -> np.ndarray:
        """Return a view (numpy array) of the matrix."""
        return self.detach().cpu().numpy()  # pylint: disable=not-callable
