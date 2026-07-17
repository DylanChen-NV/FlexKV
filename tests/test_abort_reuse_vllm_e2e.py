"""End-to-end abort checkpoint and retry reuse test for vLLM + FlexKV."""

import asyncio
import os
from contextlib import ExitStack

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flexkv")
pytest.importorskip("vllm")

from vllm import SamplingParams  # noqa: E402
from vllm.config import KVTransferConfig  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.v1.engine.async_llm import AsyncLLM  # noqa: E402
from vllm.v1.metrics.reader import get_metrics_snapshot  # noqa: E402

MODEL = os.getenv("FLEXKV_TEST_MODEL", "/raid/model/Qwen3-8B")
_EXT_HITS = "vllm:external_prefix_cache_hits"
_EXT_QUERIES = "vllm:external_prefix_cache_queries"


def _counter(name: str) -> int:
    return sum(
        int(getattr(metric, "value", 0))
        for metric in get_metrics_snapshot()
        if metric.name == name
    )


@pytest.mark.asyncio
async def test_aborted_partial_trajectory_reuses_flexkv():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    os.environ.setdefault("FLEXKV_CPU_CACHE_GB", "2")
    engine_args = AsyncEngineArgs(
        model=MODEL,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=8192,
        max_num_seqs=32,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.5,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        enable_sleep_mode=False,
        disable_log_stats=False,
        kv_transfer_config=KVTransferConfig(
            kv_connector="FlexKVConnectorV1",
            kv_role="kv_both",
        ),
    )

    with ExitStack() as after:
        engine = AsyncLLM.from_engine_args(engine_args)
        after.callback(engine.shutdown)

        prompt = "[abort-reuse] " + ("Explain dynamic resource scheduling. " * 400)
        outputs = []

        async def generate_until_abort():
            async for output in engine.generate(
                prompt=prompt,
                sampling_params=SamplingParams(
                    max_tokens=512,
                    temperature=0,
                    ignore_eos=True,
                ),
                request_id="abort-source",
            ):
                outputs.append(output)
            return outputs[-1]

        generation = asyncio.create_task(generate_until_abort())
        while not outputs or len(outputs[-1].outputs[0].token_ids) < 32:
            await asyncio.sleep(0.01)

        await engine.pause_generation(
            mode="abort",
            clear_cache=False,
            offload_aborted_kv=True,
        )
        aborted = await asyncio.wait_for(generation, timeout=30)
        assert aborted.finished
        assert aborted.outputs[0].finish_reason == "abort"
        assert len(aborted.outputs[0].token_ids) >= 32

        assert await engine.reset_prefix_cache(reset_connector=False)
        await engine.resume_generation()

        retry_tokens = (
            list(aborted.prompt_token_ids) + list(aborted.outputs[0].token_ids)
        )
        hits_before = _counter(_EXT_HITS)
        queries_before = _counter(_EXT_QUERIES)

        retry = None
        async for retry in engine.generate(
            prompt={"prompt_token_ids": retry_tokens},
            sampling_params=SamplingParams(max_tokens=8, temperature=0),
            request_id="abort-retry",
        ):
            pass

        assert retry is not None and retry.finished
        hit_delta = _counter(_EXT_HITS) - hits_before
        query_delta = _counter(_EXT_QUERIES) - queries_before
        print(
            "[abort-reuse] "
            f"partial_tokens={len(aborted.outputs[0].token_ids)} "
            f"external_hits={hit_delta}/{query_delta}"
        )
        assert query_delta > 0
        assert hit_delta > 0
