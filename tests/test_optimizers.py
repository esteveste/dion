import pytest
import torch
import torch.nn as nn

from dion.dion2 import Dion2
from dion.muon import Muon
from dion.normuon import NorMuon

# -----------------------------------------------------------------------------#
# General settings
# -----------------------------------------------------------------------------#

torch._dynamo.config.cache_size_limit = 100  # noqa: SLF001


class DummyModel(nn.Module):
    def __init__(self, in_features=16, hidden_features=32, out_features=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _get_optimizer_groups(model):
    """
    Split parameters into 2D+ tensors for the main algorithms (e.g. Dion2, Muon) 
    and element-wise params (like 1D biases and layer norms) for AdamW.
    """
    algorithm_params = []
    adamw_params = []
    for name, p in model.named_parameters():
        if p.ndim >= 2:
            algorithm_params.append(p)
        else:
            adamw_params.append(p)
    return algorithm_params, adamw_params


@pytest.mark.parametrize("opt_class, algo_name", [
    (Dion2, "dion2"),
    (Muon, "muon"),
    (NorMuon, "normuon")
])
def test_optimizer_on_cpu(opt_class, algo_name):
    """
    Test that the distributed optimizers (Dion2, Muon, NorMuon) can be 
    initialized and take steps on CPU without crashing.
    """
    model = DummyModel()
    algo_params, adamw_params = _get_optimizer_groups(model)
    
    # Configure parameter groups
    param_groups = [
        {"params": algo_params, "algorithm": algo_name},
        {"params": adamw_params, "algorithm": "adamw"}
    ]
    
    # Initialize optimizer
    # use_triton is False by default, so it'll use the PyTorch fallback (newton_schulz.py)
    # We pass use_triton=False explicitly for clarity
    optimizer = opt_class(param_groups, lr=0.01, use_triton=False)
    
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(3):  # Run a few steps to ensure stability
        # Dummy input and target
        device = next(model.parameters()).device
        x = torch.randn(8, 16, device=device)
        y = torch.randint(0, 10, (8,), device=device)
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        
        # Step
        optimizer.step()
        
    # If we made it here without error, the optimizers run fine on CPU.
