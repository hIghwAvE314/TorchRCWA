import pytest

def test_imports():
    packages = [
        'torch',
        'rcwa',
        'rcwa.geometry',
        'rcwa.matrix',
        'rcwa.params',
    ]
    for package in packages:
        try:
            __import__(package)
        except ImportError as e:
            pytest.fail(f"Failed to import package: {package}. Error: {e}")