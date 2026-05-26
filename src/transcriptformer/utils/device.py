import logging

import torch


def _normalize_device_preference(device_preference: str | None) -> str:
    if device_preference is None:
        return "auto"
    value = str(device_preference).strip().lower()
    if value == "gpu":
        return "cuda"
    if value in {"auto", "cpu", "cuda", "mps"}:
        return value
    raise ValueError(f"Unsupported device preference '{device_preference}'. Use one of: auto, cpu, cuda, mps")


def _parse_num_gpus(num_gpus: int | str | None) -> int:
    if num_gpus is None:
        return 1
    if isinstance(num_gpus, int):
        return num_gpus

    value = str(num_gpus).strip().lower()
    if value in {"-1", "all", "auto"}:
        return -1
    return int(value)


def resolve_checkpoint_map_location(device_preference: str | None = "auto") -> str:
    """Resolve checkpoint map_location from a device preference."""
    device = _normalize_device_preference(device_preference)
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_accelerator_and_devices(
    device_preference: str | None = "auto",
    num_gpus: int | str | None = 1,
) -> tuple[str, int]:
    """Resolve Lightning accelerator/devices from the shared device policy."""
    device = _normalize_device_preference(device_preference)
    requested = _parse_num_gpus(num_gpus)

    if device == "cpu":
        return "cpu", 1

    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return "mps", 1

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        available = torch.cuda.device_count()
        if requested == -1:
            return "gpu", available
        if requested > 1:
            if available < requested:
                logging.warning(
                    "Requested %s CUDA devices but only %s available. Using %s.",
                    requested,
                    available,
                    available,
                )
            return "gpu", min(requested, available)
        return "gpu", 1

    # auto
    if requested == -1:
        if torch.cuda.is_available():
            return "gpu", torch.cuda.device_count()
        if torch.backends.mps.is_available():
            return "mps", 1
        return "cpu", 1

    if requested > 1:
        available = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if available < requested:
            logging.warning(
                "Requested %s GPUs but only %s available. Using %s GPU(s).",
                requested,
                available,
                available if available > 0 else 1,
            )
            if available > 0:
                return "gpu", available
            if torch.backends.mps.is_available():
                return "mps", 1
            return "cpu", 1
        return "gpu", requested

    if torch.cuda.is_available():
        return "gpu", 1
    if torch.backends.mps.is_available():
        return "mps", 1
    return "cpu", 1