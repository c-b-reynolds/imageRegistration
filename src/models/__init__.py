from .base import BaseRegistrationModel
from .simple_cnn import SimpleCNN
from .unet_registration import UNetRegistration

# Registry — add new architectures here so scripts can resolve them by name
REGISTRY: dict = {
    "UNetRegistration": UNetRegistration,
    "SimpleCNN": SimpleCNN,
}


def build_model(name: str, **kwargs) -> BaseRegistrationModel:
    import inspect
    if name not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(REGISTRY)}")
    cls = REGISTRY[name]
    accepted = inspect.signature(cls.__init__).parameters
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(**filtered)


__all__ = ["BaseRegistrationModel", "UNetRegistration", "SimpleCNN", "REGISTRY", "build_model"]
