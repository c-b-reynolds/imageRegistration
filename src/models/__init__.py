from .base import BaseRegistrationModel
from .direct_hybrid_registration import DirectHybridRegistration
from .eulerian_hybrid_registration import EulerianHybridODERegistration
from .gate_flow import GateFlow
from .gate_flow_ms import GateFlowMS
from .hybrid_ode_registration import HybridODERegistration
from .neural_ode_registration import NeuralODERegistration
from .simple_cnn import SimpleCNN
from .unet_registration import UNetRegistration

# Registry — add new architectures here so scripts can resolve them by name
REGISTRY: dict = {
    "UNetRegistration": UNetRegistration,
    "SimpleCNN": SimpleCNN,
    "NeuralODERegistration": NeuralODERegistration,
    "HybridODERegistration": HybridODERegistration,
    "EulerianHybridODERegistration": EulerianHybridODERegistration,
    "DirectHybridRegistration": DirectHybridRegistration,
    "GateFlow": GateFlow,
    "GateFlowMS": GateFlowMS,
}


def build_model(name: str, **kwargs) -> BaseRegistrationModel:
    import inspect
    if name not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(REGISTRY)}")
    cls = REGISTRY[name]
    accepted = inspect.signature(cls.__init__).parameters
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(**filtered)


__all__ = ["BaseRegistrationModel", "UNetRegistration", "SimpleCNN",
           "NeuralODERegistration", "HybridODERegistration",
           "EulerianHybridODERegistration", "DirectHybridRegistration",
           "GateFlow", "GateFlowMS", "REGISTRY", "build_model"]
