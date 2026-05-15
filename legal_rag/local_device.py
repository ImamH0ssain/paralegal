"""Local CPU/CUDA selection for WSL and offline model execution.

The helper is used by shell scripts to avoid routing PyTorch workloads to GPUs
that are visible but unsupported by the installed torch build. It prints shell
exports so bash scripts can source the resolved policy directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import shlex
from typing import Callable, Mapping
import warnings


@dataclass(frozen=True)
class CudaInfo:
    """Result of probing PyTorch CUDA availability."""

    available: bool
    name: str = ""
    capability: int | None = None
    reason: str = ""


@dataclass(frozen=True)
class LocalDeviceConfig:
    """Resolved device policy for OCR, embeddings, reranking, and CUDA visibility."""

    accelerator: str
    ocr_device: str
    embedding_device: str
    rerank_device: str
    cuda_visible_devices: str | None
    unset_cuda_visible_devices: bool
    reason: str
    gpu_name: str = ""
    gpu_capability: int | None = None


def detect_torch_cuda() -> CudaInfo:
    """Probe the first CUDA device through PyTorch without surfacing warnings."""
    try:
        import torch
    except Exception as exc:
        return CudaInfo(available=False, reason=f"torch unavailable: {exc}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if not torch.cuda.is_available():
                return CudaInfo(available=False, reason="torch.cuda.is_available() is false")
            capability = torch.cuda.get_device_capability(0)
            return CudaInfo(
                available=True,
                name=torch.cuda.get_device_name(0),
                capability=capability[0] * 10 + capability[1],
            )
    except Exception as exc:
        return CudaInfo(available=False, reason=f"CUDA probe failed: {exc}")


def resolve_local_devices(
    env: Mapping[str, str] | None = None,
    *,
    detect_cuda: Callable[[], CudaInfo] = detect_torch_cuda,
) -> LocalDeviceConfig:
    """Resolve local device settings from environment variables.

    ``LOCAL_ACCELERATOR=auto`` uses CUDA only when PyTorch reports a device at
    or above ``LOCAL_CUDA_MIN_CAPABILITY``. Explicit ``cpu`` and ``cuda`` modes
    remain available for reproducibility and debugging.
    """

    env = env or os.environ
    requested = env.get("LOCAL_ACCELERATOR", "auto").strip().lower()
    min_capability = _int_env(env, "LOCAL_CUDA_MIN_CAPABILITY", 75)

    if requested in {"cuda", "gpu"}:
        accelerator = "cuda"
        info = detect_cuda()
        reason = "LOCAL_ACCELERATOR forces CUDA"
    elif requested == "cpu":
        accelerator = "cpu"
        info = CudaInfo(available=False)
        reason = "LOCAL_ACCELERATOR forces CPU"
    else:
        info = detect_cuda()
        if info.available and (info.capability or 0) >= min_capability:
            accelerator = "cuda"
            reason = f"CUDA device meets minimum capability sm_{min_capability}"
        elif info.available:
            accelerator = "cpu"
            reason = f"CUDA device capability sm_{info.capability} is below minimum sm_{min_capability}"
        else:
            accelerator = "cpu"
            reason = info.reason or "no CUDA device detected"

    default_device = "cuda" if accelerator == "cuda" else "cpu"
    cuda_visible = env.get("CUDA_VISIBLE_DEVICES")
    unset_cuda_visible = False
    if accelerator == "cpu":
        cuda_visible = ""
    elif cuda_visible == "":
        cuda_visible = None
        unset_cuda_visible = True

    return LocalDeviceConfig(
        accelerator=accelerator,
        ocr_device=_device_env(env, "LOCAL_OCR_DEVICE", default_device),
        embedding_device=_device_env(env, "LOCAL_EMBEDDING_DEVICE", default_device),
        rerank_device=_device_env(env, "LOCAL_RERANK_DEVICE", default_device),
        cuda_visible_devices=cuda_visible,
        unset_cuda_visible_devices=unset_cuda_visible,
        reason=reason,
        gpu_name=info.name,
        gpu_capability=info.capability,
    )


def shell_exports(config: LocalDeviceConfig) -> str:
    """Render a ``LocalDeviceConfig`` as bash export statements."""
    lines = [
        _export("LOCAL_ACCELERATOR_RESOLVED", config.accelerator),
        _export("LOCAL_OCR_DEVICE", config.ocr_device),
        _export("LOCAL_EMBEDDING_DEVICE", config.embedding_device),
        _export("LOCAL_RERANK_DEVICE", config.rerank_device),
    ]
    if config.unset_cuda_visible_devices:
        lines.append("unset CUDA_VISIBLE_DEVICES")
    elif config.cuda_visible_devices is not None:
        lines.append(_export("CUDA_VISIBLE_DEVICES", config.cuda_visible_devices))
    detail = config.reason
    if config.gpu_name:
        capability = f" sm_{config.gpu_capability}" if config.gpu_capability else ""
        detail = f"{detail}: {config.gpu_name}{capability}"
    lines.append(f"echo {_quote('[local-device] ' + detail)} >&2")
    return "\n".join(lines)


def _export(name: str, value: str) -> str:
    return f"export {name}={_quote(value)}"


def _quote(value: str) -> str:
    return shlex.quote(value)


def _device_env(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name, "").strip().lower()
    return value or default


def _int_env(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(env.get(name, str(default)))
    except ValueError:
        return default


def main() -> None:
    """Print bash exports for the resolved local device policy."""
    config = resolve_local_devices()
    print(shell_exports(config))


if __name__ == "__main__":
    main()
