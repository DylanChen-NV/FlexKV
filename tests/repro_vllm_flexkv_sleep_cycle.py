#!/usr/bin/env python3
"""Compact vLLM + FlexKV sleep/wake GPU integration gate."""

from __future__ import annotations

import argparse
import os

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.getenv("VLLM_CUMEM_ENABLE_SHAREABLE_HANDLE") != "1":
        raise RuntimeError(
            "Set VLLM_CUMEM_ENABLE_SHAREABLE_HANDLE=1 before running this gate"
        )

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        max_num_seqs=8,
        max_num_batched_tokens=args.max_model_len,
        gpu_memory_utilization=0.5,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        enable_sleep_mode=True,
        kv_transfer_config=KVTransferConfig(
            kv_connector="FlexKVConnectorV1",
            kv_role="kv_both",
        ),
    )
    prompt = "Explain CUDA virtual memory clearly. " * 300
    sampling_params = SamplingParams(max_tokens=8, temperature=0)

    llm.generate([prompt], sampling_params)
    print("PHASE_GENERATE_BEFORE_PASS", flush=True)
    llm.sleep(level=1)
    print("PHASE_SLEEP_PASS", flush=True)
    llm.wake_up()
    print("PHASE_WAKE_PASS", flush=True)
    llm.generate([prompt], sampling_params)
    print("PHASE_GENERATE_AFTER_PASS", flush=True)


if __name__ == "__main__":
    main()
