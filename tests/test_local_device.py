from __future__ import annotations

from paralegal.local_device import CudaInfo, resolve_local_devices, shell_exports


def test_auto_uses_cpu_for_unsupported_older_cuda_capability() -> None:
    config = resolve_local_devices(
        {"LOCAL_ACCELERATOR": "auto", "LOCAL_CUDA_MIN_CAPABILITY": "75"},
        detect_cuda=lambda: CudaInfo(available=True, name="NVIDIA GeForce MX250", capability=61),
    )

    assert config.accelerator == "cpu"
    assert config.ocr_device == "cpu"
    assert config.embedding_device == "cpu"
    assert config.rerank_device == "cpu"
    assert config.cuda_visible_devices == ""


def test_auto_uses_cuda_for_supported_gpu() -> None:
    config = resolve_local_devices(
        {"LOCAL_ACCELERATOR": "auto", "LOCAL_CUDA_MIN_CAPABILITY": "75"},
        detect_cuda=lambda: CudaInfo(available=True, name="NVIDIA RTX 4070", capability=89),
    )

    assert config.accelerator == "cuda"
    assert config.ocr_device == "cuda"
    assert config.embedding_device == "cuda"
    assert config.rerank_device == "cuda"
    assert config.cuda_visible_devices is None
    assert config.unset_cuda_visible_devices is False


def test_explicit_device_env_overrides_auto_defaults() -> None:
    config = resolve_local_devices(
        {
            "LOCAL_ACCELERATOR": "auto",
            "LOCAL_EMBEDDING_DEVICE": "cpu",
            "LOCAL_RERANK_DEVICE": "cpu",
            "CUDA_VISIBLE_DEVICES": "0",
        },
        detect_cuda=lambda: CudaInfo(available=True, name="NVIDIA RTX 4090", capability=89),
    )

    assert config.accelerator == "cuda"
    assert config.ocr_device == "cuda"
    assert config.embedding_device == "cpu"
    assert config.rerank_device == "cpu"
    assert config.cuda_visible_devices == "0"
    assert config.unset_cuda_visible_devices is False


def test_shell_exports_include_safe_cuda_visibility_policy() -> None:
    cpu_config = resolve_local_devices(
        {"LOCAL_ACCELERATOR": "cpu"},
        detect_cuda=lambda: CudaInfo(available=True, name="NVIDIA RTX 4090", capability=89),
    )
    cuda_config = resolve_local_devices(
        {"LOCAL_ACCELERATOR": "cuda", "CUDA_VISIBLE_DEVICES": ""},
        detect_cuda=lambda: CudaInfo(available=False),
    )

    assert "export CUDA_VISIBLE_DEVICES=''" in shell_exports(cpu_config)
    assert "unset CUDA_VISIBLE_DEVICES" in shell_exports(cuda_config)
