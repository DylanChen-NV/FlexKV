"""CPU-only contract tests; execute the actual method AST without importing CUDA."""

import ast
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

tree = ast.parse((Path(__file__).parents[1] / "flexkv/transfer/transfer_engine.py").read_text())
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TransferEngine")
method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "resume_gpu_mappings")
module = ast.parse("from __future__ import annotations")
module.body.append(method)
scope = {"ThreadPoolExecutor": ThreadPoolExecutor, "os": os, "time": time,
         "flexkv_logger": SimpleNamespace(info=lambda message: None)}
exec(compile(module, "<actual-resume-gpu-mappings>", "exec"), scope)
resume = scope["resume_gpu_mappings"]


class ParallelGpuRemapTest(unittest.TestCase):
    def fixture(self, first, second):
        engine = SimpleNamespace(_gpu_mappings_suspended=True, gpu_handle_groups={"group": []},
                                 h2d_workers={"group": first}, d2h_workers={"group": second})
        handles = {"group": [SimpleNamespace(get_tensor_handle_list=lambda: ["tensor"])]}
        return engine, handles

    def test_default_is_serial_and_preserves_payload(self):
        calls = []
        def worker(name):
            def control(command, payload):
                self.assertEqual((command, payload), ("resume_gpu", [["tensor"]]))
                calls.append(name)
                return 2
            return SimpleNamespace(control=control)
        engine, handles = self.fixture(worker("h2d"), worker("d2h"))
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resume(engine, handles), 4)
        self.assertEqual(calls, ["h2d", "d2h"])
        self.assertFalse(engine._gpu_mappings_suspended)
        self.assertIs(engine.gpu_handle_groups, handles)

    def test_opt_in_joins_independent_controls_before_resuming(self):
        barrier = threading.Barrier(2, timeout=2)
        def control(command, payload):
            barrier.wait()
            return 2
        engine, handles = self.fixture(SimpleNamespace(control=control), SimpleNamespace(control=control))
        with patch.dict(os.environ, {"FLEXKV_PARALLEL_GPU_REMAP": "1"}):
            self.assertEqual(resume(engine, handles), 4)
        self.assertFalse(engine._gpu_mappings_suspended)

    def test_aliases_fall_back_without_concurrent_pipe_readers(self):
        calls = []
        worker = SimpleNamespace(control=lambda *args: calls.append(args) or 2)
        engine, handles = self.fixture(worker, worker)
        with patch.dict(os.environ, {"FLEXKV_PARALLEL_GPU_REMAP": "1"}):
            self.assertEqual(resume(engine, handles), 4)
        self.assertEqual(len(calls), 2)

    def test_failure_cannot_mark_gpu_mappings_ready(self):
        def fail(*args):
            raise RuntimeError("import failed")
        engine, handles = self.fixture(SimpleNamespace(control=fail), SimpleNamespace(control=lambda *args: 2))
        with patch.dict(os.environ, {"FLEXKV_PARALLEL_GPU_REMAP": "1"}), \
                self.assertRaisesRegex(RuntimeError, "import failed"):
            resume(engine, handles)
        self.assertTrue(engine._gpu_mappings_suspended)
        self.assertIsNot(engine.gpu_handle_groups, handles)


if __name__ == "__main__":
    unittest.main()
