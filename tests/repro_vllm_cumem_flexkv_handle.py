#!/usr/bin/env python3
"""Minimal vLLM CuMemAllocator to FlexKV handle-export reproducer.

This is a GPU integration test. It does not start a vLLM engine, load a model,
or depend on VERL. On the affected non-Fabric H100 setup it fails while
TensorSharedHandle exports the vLLM VMM allocation because the allocation has
requestedHandleTypes=0x0.

Run with both source trees on PYTHONPATH, for example:

    PYTHONPATH=/path/to/vllm:/path/to/FlexKV \
      python3 tests/repro_vllm_cumem_flexkv_handle.py
"""

from __future__ import annotations

import argparse

import torch

from flexkv.common.memory_handle import TensorSharedHandle, _is_vmm_pointer
from vllm.device_allocator import get_mem_allocator_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--size-mib", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size_mib <= 0:
        raise ValueError("--size-mib must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    allocator = get_mem_allocator_instance()

    print(f"device={torch.cuda.get_device_name(args.device)!r}")
    print(f"torch={torch.__version__}")
    print(f"allocator={type(allocator).__module__}.{type(allocator).__name__}")

    size_bytes = args.size_mib * 1024 * 1024
    with allocator.use_memory_pool(tag="flexkv-vmm-handle-repro"):
        tensor = torch.zeros(size_bytes, dtype=torch.uint8, device=device)
        torch.cuda.synchronize(device)

        is_vmm = _is_vmm_pointer(tensor.data_ptr())
        print(
            f"tensor_ptr=0x{tensor.data_ptr():x} size_bytes={tensor.numel()} "
            f"is_vmm={is_vmm}"
        )
        if not is_vmm:
            raise RuntimeError("vLLM CuMemAllocator did not produce a VMM allocation")

        print("exporting with FlexKV TensorSharedHandle...")
        handle = TensorSharedHandle(tensor, device_id=args.device)

        print(
            "PASS: "
            f"handle_type={handle.handle_type} "
            f"allocation_size={handle.vmm_allocation_size} "
            f"granularity={handle.vmm_granularity}"
        )


if __name__ == "__main__":
    main()
