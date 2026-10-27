def pick_device(preference: str) -> str:
    """'auto' → mps, ha elérhető (Apple GPU), különben cpu."""
    if preference != "auto":
        return preference
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"
