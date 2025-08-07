import torch
import numpy as np
import pytest
from rcwa.matrix import Matrix

def test_matrix_creation_from_numpy():
    arr = np.array([[1, 2], [3, 4]], dtype=np.complex128)
    m = Matrix(arr)
    assert isinstance(m, Matrix)
    assert m.shape == (2, 2)
    assert torch.allclose(m, torch.tensor([[1, 2], [3, 4]], dtype=torch.complex128, device=m.device))

def test_matrix_creation_from_tensor_cpu():
    t = torch.tensor([[1, 2], [3, 4]], dtype=torch.complex128, device='cpu')
    m = Matrix(t)
    assert isinstance(m, Matrix)
    assert m.shape == (2, 2)
    assert m.device == t.device
    assert torch.allclose(m, t)

def test_matrix_creation_from_tensor_cuda():
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    t = torch.tensor([[1, 2], [3, 4]], dtype=torch.complex128, device='cuda')
    m = Matrix(t)
    assert isinstance(m, Matrix)
    assert m.shape == (2, 2)
    assert m.device == t.device
    assert torch.allclose(m, t)

def test_addition():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])
    c = a + b
    assert torch.allclose(c, torch.tensor([[6, 8], [10, 12]], dtype=torch.complex128, device=a.device))

def test_subtraction():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])
    c = a - b
    assert torch.allclose(c, torch.tensor([[-4, -4], [-4, -4]], dtype=torch.complex128, device=a.device))

def test_matmul():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])
    c = a @ b
    assert torch.allclose(c, torch.tensor([[19, 22], [43, 50]], dtype=torch.complex128, device=a.device))

def test_pointwise_mul():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])
    c = a * b
    assert torch.allclose(c, torch.tensor([[5, 12], [21, 32]], dtype=torch.complex128, device=a.device))

def test_scalar_mul():
    a = Matrix([[1, 2], [3, 4]])
    c = a * 2
    assert torch.allclose(c, torch.tensor([[2, 4], [6, 8]], dtype=torch.complex128, device=a.device))
    d = 3 * a
    assert torch.allclose(d, torch.tensor([[3, 6], [9, 12]], dtype=torch.complex128, device=a.device))

def test_inv():
    a = Matrix([[1, 2], [3, 4]])
    inv_a = a.inv()
    expected = torch.tensor([[-2, 1], [1.5, -0.5]], dtype=torch.complex128, device=a.device)
    assert torch.allclose(inv_a, expected)

def test_numpy():
    a = Matrix([[1, 2], [3, 4]])
    arr = a.detach().cpu().numpy()
    assert isinstance(arr, np.ndarray)
    assert np.allclose(arr, np.array([[1, 2], [3, 4]], dtype=np.complex128))

def test_to_cpu():
    a = Matrix([[1, 2], [3, 4]])
    b = a.to(device='cpu')
    assert b.device.type == 'cpu'
    assert torch.allclose(b, a.cpu())
