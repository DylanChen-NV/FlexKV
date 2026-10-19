"""Targeted refresh uses the Redis worker, never the request's connection."""

import concurrent.futures
import shutil
import socket
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import redis
import torch

from flexkv.cache.cache_engine import GlobalCacheEngine
from flexkv.cache.hie_cache_engine import HierarchyLRCacheEngine
from flexkv.cache.radix_remote import DistributedRadixTree
from flexkv.cache.redis_meta import BlockMeta, RedisMetaChannel, dist_available


@pytest.fixture
def redis_port(tmp_path):
    if not dist_available() or shutil.which("redis-server") is None:
        pytest.skip("P2P extension and redis-server required")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen(
        [
            "redis-server",
            "--bind",
            "127.0.0.1",
            "--port",
            str(port),
            "--save",
            "",
            "--appendonly",
            "no",
            "--dir",
            str(tmp_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    client = redis.Redis(port=port)
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                if client.ping():
                    break
            except redis.ConnectionError:
                pass
            assert time.monotonic() < deadline
            time.sleep(0.01)
        client.hset("node:1", mapping={"ip": "127.0.0.1"})
        client.hset("node:2", mapping={"ip": "127.0.0.2"})
        yield port
    finally:
        client.close()
        process.terminate()
        process.wait(timeout=10)


def test_explicit_refresh_installs_published_prefix_and_survives_reset(redis_port):
    source = RedisMetaChannel("127.0.0.1", redis_port, 1, "127.0.0.1", "CPUB")
    target = RedisMetaChannel("127.0.0.1", redis_port, 2, "127.0.0.2", "CPUB")
    assert source.connect() and target.connect()
    tree = DistributedRadixTree(4, 32, 2, rebuild_interval_ms=3600000)
    hashes = torch.tensor([7, 11], dtype=torch.int64)
    assert not tree.refresh(timeout_ms=1)
    assert tree.start(target)
    try:
        for cycle in range(3):
            assert source.publish_batch(
                [
                    BlockMeta(ph=0, pb=0, nid=1, hash=7, lt=int(time.time() * 1000) + 30000),
                    BlockMeta(ph=7, pb=1, nid=1, hash=11, lt=int(time.time() * 1000) + 30000),
                ]
            )
            # The periodic rebuild is deliberately far away: ordinary matching
            # must not perform I/O or turn this stale miss into a hit.
            assert tree.match_prefix(hashes, 2, False).num_matched_blocks == 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                assert all(executor.map(lambda _: tree.refresh(), range(16)))
            assert tree.match_prefix(hashes, 2, False).num_matched_blocks == 2
            tree.stop()
            assert not tree.refresh(timeout_ms=1)
            tree.reset()
            if cycle < 2:
                assert tree.start(target)
    finally:
        tree.stop()


@pytest.mark.parametrize("coverage,expected", [(0, 1), (1, 1), (2, 0)])
def test_refresh_only_when_required_prefix_is_missing(coverage, expected):
    engine = GlobalCacheEngine.__new__(GlobalCacheEngine)
    engine._cache_tree_lock = threading.RLock()
    tier = HierarchyLRCacheEngine.__new__(HierarchyLRCacheEngine)
    tier.local_index = Mock()
    tier.remote_index = Mock()
    for index in (tier.local_index, tier.remote_index):
        index.match_prefix.return_value = SimpleNamespace(num_matched_blocks=coverage)
    tier.remote_index.refresh.return_value = True
    engine.cpu_cache_engine = tier
    meta = SimpleNamespace(gen_hashes=lambda: None, block_hashes=np.array([7, 11]), num_blocks=2)
    engine._prepare_request = Mock(return_value=SimpleNamespace(block_end_idx=2, sequence_meta=meta))
    engine.refresh_remote_if_needed(np.arange(8), np.ones(8, dtype=bool))
    assert tier.remote_index.refresh.call_count == expected


def test_non_p2p_engine_does_not_refresh():
    engine = GlobalCacheEngine.__new__(GlobalCacheEngine)
    engine._cache_tree_lock = threading.RLock()
    engine.cpu_cache_engine = object()
    engine.refresh_remote_if_needed(np.arange(8), np.ones(8, dtype=bool))


def test_partial_index_can_refresh_to_full_and_timeout_is_bounded(redis_port):
    channel = RedisMetaChannel("127.0.0.1", redis_port, 2, "127.0.0.2", "CPUB")
    publisher = RedisMetaChannel("127.0.0.1", redis_port, 1, "127.0.0.1", "CPUB")
    assert channel.connect() and publisher.connect()
    tree = DistributedRadixTree(4, 32, 2, rebuild_interval_ms=3600000)
    hashes = torch.tensor([7, 11], dtype=torch.int64)
    assert tree.start(channel)
    client = redis.Redis(port=redis_port)
    try:
        lease = int(time.time() * 1000) + 30000
        assert publisher.publish_one(BlockMeta(ph=0, pb=0, nid=1, hash=7, lt=lease))
        assert tree.refresh()
        assert tree.match_prefix(hashes, 2, False).num_matched_blocks == 1
        assert publisher.publish_one(BlockMeta(ph=7, pb=1, nid=1, hash=11, lt=lease))
        assert tree.match_prefix(hashes, 2, False).num_matched_blocks == 1
        # Pause is test-only fault injection, not a serving readiness delay.
        client.execute_command("CLIENT", "PAUSE", 200)
        start = time.monotonic()
        assert not tree.refresh(timeout_ms=5)
        assert time.monotonic() - start < 0.15
        assert tree.refresh()
        assert tree.match_prefix(hashes, 2, False).num_matched_blocks == 2
    finally:
        tree.stop()
        client.close()
