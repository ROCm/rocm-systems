# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PyTorch functional tests on simulated GPU via rocjitsu KMD interposer.

Run under LD_PRELOAD=librocjitsu_kmd.so with RJ_CONFIG/RJ_SCHEMA env vars
so that PyTorch's ROCm/HIP backend talks to the simulator instead of real
hardware.

Usage (standalone):
    LD_PRELOAD=<build>/lib/rocjitsu/librocjitsu_kmd.so \
    RJ_CONFIG=<src>/configs/amdgpu_cdna4_kmd.json \
    RJ_SCHEMA=<src>/schemas/simulation_config.fbs \
    python -m unittest tests.torch_test -v

Usage (via ctest):
    cd build && ctest -R TorchTest -V
"""

import math
import unittest

import torch

DEVICE = torch.device("cuda", 0) if torch.cuda.is_available() else None

def _skip_no_gpu(cls):
    """Class decorator: skip every test when no ROCm device is visible."""
    if DEVICE is None:
        return unittest.skip("torch.cuda (ROCm) not available — is LD_PRELOAD set?")(cls)
    return cls


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------

@_skip_no_gpu
class TestDeviceDiscovery(unittest.TestCase):
    """Verify that PyTorch detects the simulated GPU."""

    def test_cuda_available(self):
        self.assertTrue(torch.cuda.is_available())

    def test_device_count(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 1)

    def test_device_name(self):
        name = torch.cuda.get_device_name(0)
        self.assertIsInstance(name, str)
        self.assertGreater(len(name), 0)


# ---------------------------------------------------------------------------
# Tensor creation and host↔device transfers
# ---------------------------------------------------------------------------

@_skip_no_gpu
class TestTensorTransfer(unittest.TestCase):
    """Test tensor creation on device and data movement."""

    def test_create_on_device(self):
        t = torch.zeros(64, device=DEVICE)
        self.assertEqual(t.device.type, "cuda")
        self.assertEqual(t.shape, (64,))

    def test_host_to_device(self):
        h = torch.arange(32, dtype=torch.float32)
        d = h.to(DEVICE)
        self.assertEqual(d.device.type, "cuda")
        self.assertTrue(torch.equal(d.cpu(), h))

    def test_device_to_host(self):
        d = torch.ones(16, device=DEVICE)
        h = d.cpu()
        self.assertEqual(h.device.type, "cpu")
        self.assertTrue(torch.all(h == 1.0))

    def test_round_trip_float(self):
        src = torch.linspace(0, 1, 128)
        dst = src.to(DEVICE).cpu()
        self.assertTrue(torch.allclose(dst, src))

    def test_round_trip_int(self):
        src = torch.randint(0, 1000, (256,), dtype=torch.int32)
        dst = src.to(DEVICE).cpu()
        self.assertTrue(torch.equal(dst, src))

    def test_device_to_device(self):
        a = torch.randn(64, device=DEVICE)
        b = a.clone()
        self.assertTrue(torch.equal(a, b))


# ---------------------------------------------------------------------------
# Element-wise arithmetic
# ---------------------------------------------------------------------------

@_skip_no_gpu
class TestArithmetic(unittest.TestCase):
    """Basic element-wise operations on GPU tensors."""

    def test_add(self):
        a = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
        b = torch.tensor([4.0, 5.0, 6.0], device=DEVICE)
        c = a + b
        self.assertTrue(torch.allclose(c.cpu(), torch.tensor([5.0, 7.0, 9.0])))

    def test_sub(self):
        a = torch.tensor([10.0, 20.0, 30.0], device=DEVICE)
        b = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
        self.assertTrue(torch.allclose((a - b).cpu(), torch.tensor([9.0, 18.0, 27.0])))

    def test_mul(self):
        a = torch.tensor([2.0, 3.0, 4.0], device=DEVICE)
        b = torch.tensor([5.0, 6.0, 7.0], device=DEVICE)
        self.assertTrue(torch.allclose((a * b).cpu(), torch.tensor([10.0, 18.0, 28.0])))

    def test_div(self):
        a = torch.tensor([10.0, 21.0, 35.0], device=DEVICE)
        b = torch.tensor([2.0, 3.0, 5.0], device=DEVICE)
        self.assertTrue(torch.allclose((a / b).cpu(), torch.tensor([5.0, 7.0, 7.0])))

    def test_neg(self):
        a = torch.tensor([1.0, -2.0, 3.0], device=DEVICE)
        self.assertTrue(torch.allclose((-a).cpu(), torch.tensor([-1.0, 2.0, -3.0])))

    def test_scalar_broadcast(self):
        a = torch.tensor([2.0, 4.0, 6.0], device=DEVICE)
        self.assertTrue(torch.allclose((a * 3).cpu(), torch.tensor([6.0, 12.0, 18.0])))

    def test_fma(self):
        a = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
        b = torch.tensor([4.0, 5.0, 6.0], device=DEVICE)
        c = torch.tensor([7.0, 8.0, 9.0], device=DEVICE)
        result = torch.addcmul(c, a, b, value=1)
        self.assertTrue(torch.allclose(result.cpu(), torch.tensor([11.0, 18.0, 27.0])))


# ---------------------------------------------------------------------------
# Comparison and logical operations
# ---------------------------------------------------------------------------

@_skip_no_gpu
class TestComparison(unittest.TestCase):

    def test_eq(self):
        a = torch.tensor([1, 2, 3], device=DEVICE)
        b = torch.tensor([1, 0, 3], device=DEVICE)
        self.assertTrue(torch.equal((a == b).cpu(), torch.tensor([True, False, True])))

    def test_gt(self):
        a = torch.tensor([3.0, 1.0, 2.0], device=DEVICE)
        b = torch.tensor([1.0, 2.0, 2.0], device=DEVICE)
        self.assertTrue(torch.equal((a > b).cpu(), torch.tensor([True, False, False])))

    def test_clamp(self):
        a = torch.tensor([-1.0, 0.5, 2.0], device=DEVICE)
        clamped = torch.clamp(a, 0.0, 1.0)
        self.assertTrue(torch.allclose(clamped.cpu(), torch.tensor([0.0, 0.5, 1.0])))

    def test_where(self):
        cond = torch.tensor([True, False, True], device=DEVICE)
        a = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
        b = torch.tensor([4.0, 5.0, 6.0], device=DEVICE)
        result = torch.where(cond, a, b)
        self.assertTrue(torch.allclose(result.cpu(), torch.tensor([1.0, 5.0, 3.0])))


# ---------------------------------------------------------------------------
# Math / transcendental functions
# ---------------------------------------------------------------------------

@_skip_no_gpu
class TestMath(unittest.TestCase):

    def test_exp(self):
        a = torch.tensor([0.0, 1.0], device=DEVICE)
        result = torch.exp(a).cpu()
        self.assertTrue(torch.allclose(result, torch.tensor([1.0, 2.718281828]), atol=1e-5))

    def test_log(self):
        a = torch.tensor([1.0, 2.718281828], device=DEVICE)
        result = torch.log(a).cpu()
        self.assertTrue(torch.allclose(result, torch.tensor([0.0, 1.0]), atol=1e-5))

    def test_sqrt(self):
        a = torch.tensor([4.0, 9.0, 16.0], device=DEVICE)
        self.assertTrue(torch.allclose(torch.sqrt(a).cpu(), torch.tensor([2.0, 3.0, 4.0])))

    def test_abs(self):
        a = torch.tensor([-3.0, 0.0, 5.0], device=DEVICE)
        self.assertTrue(torch.allclose(torch.abs(a).cpu(), torch.tensor([3.0, 0.0, 5.0])))

    def test_sin_cos(self):
        a = torch.tensor([0.0], device=DEVICE)
        self.assertTrue(torch.allclose(torch.sin(a).cpu(), torch.tensor([0.0]), atol=1e-6))
        self.assertTrue(torch.allclose(torch.cos(a).cpu(), torch.tensor([1.0]), atol=1e-6))

    def test_pow(self):
        a = torch.tensor([2.0, 3.0], device=DEVICE)
        self.assertTrue(torch.allclose(torch.pow(a, 3).cpu(), torch.tensor([8.0, 27.0])))


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

@_skip_no_gpu
class TestReductions(unittest.TestCase):

    def test_sum(self):
        a = torch.ones(1024, device=DEVICE)
        self.assertTrue(torch.allclose(a.sum().cpu(), torch.tensor(1024.0)))

    def test_mean(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
        self.assertTrue(torch.allclose(a.mean().cpu(), torch.tensor(2.5)))

    def test_min_max(self):
        a = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0], device=DEVICE)
        self.assertAlmostEqual(a.min().item(), 1.0)
        self.assertAlmostEqual(a.max().item(), 5.0)

    def test_argmax(self):
        a = torch.tensor([0.1, 0.9, 0.3], device=DEVICE)
        self.assertEqual(a.argmax().item(), 1)

    def test_sum_along_dim(self):
        a = torch.ones(4, 8, device=DEVICE)
        row_sum = a.sum(dim=1)
        self.assertTrue(torch.allclose(row_sum.cpu(), torch.full((4,), 8.0)))


# ---------------------------------------------------------------------------
# Matrix operations
# ---------------------------------------------------------------------------

@_skip_no_gpu
class TestMatrixOps(unittest.TestCase):

    def test_matmul_small(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=DEVICE)
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=DEVICE)
        c = torch.matmul(a, b)
        expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
        self.assertTrue(torch.allclose(c.cpu(), expected))

    def test_matmul_identity(self):
        n = 32
        eye = torch.eye(n, device=DEVICE)
        a = torch.randn(n, n)
        result = torch.matmul(a.to(DEVICE), eye).cpu()
        self.assertTrue(torch.allclose(result, a, atol=1e-5))

    def test_mv(self):
        M = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=DEVICE)
        v = torch.tensor([3.0, 4.0], device=DEVICE)
        self.assertTrue(torch.allclose(torch.mv(M, v).cpu(), torch.tensor([3.0, 4.0])))

    def test_transpose(self):
        a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=DEVICE)
        t = a.t()
        self.assertEqual(t.shape, (3, 2))
        self.assertTrue(torch.equal(t.cpu(), torch.tensor([[1, 4], [2, 5], [3, 6]], dtype=torch.float32)))

    def test_bmm(self):
        batch = 4
        a = torch.randn(batch, 3, 4)
        b = torch.randn(batch, 4, 5)
        expected = torch.bmm(a, b)
        result = torch.bmm(a.to(DEVICE), b.to(DEVICE)).cpu()
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))


# ---------------------------------------------------------------------------
# Indexing and reshaping
# ---------------------------------------------------------------------------

@_skip_no_gpu
class TestIndexingReshape(unittest.TestCase):

    def test_reshape(self):
        a = torch.arange(24, dtype=torch.float32, device=DEVICE)
        b = a.reshape(2, 3, 4)
        self.assertEqual(b.shape, (2, 3, 4))
        self.assertTrue(torch.equal(b.reshape(-1).cpu(), torch.arange(24, dtype=torch.float32)))

    def test_slice(self):
        a = torch.arange(10, dtype=torch.float32, device=DEVICE)
        s = a[2:5]
        self.assertTrue(torch.equal(s.cpu(), torch.tensor([2.0, 3.0, 4.0])))

    def test_index_select(self):
        a = torch.tensor([10, 20, 30, 40, 50], dtype=torch.float32, device=DEVICE)
        idx = torch.tensor([0, 2, 4], device=DEVICE)
        result = a[idx]
        self.assertTrue(torch.equal(result.cpu(), torch.tensor([10.0, 30.0, 50.0])))

    def test_cat(self):
        a = torch.tensor([1.0, 2.0], device=DEVICE)
        b = torch.tensor([3.0, 4.0], device=DEVICE)
        c = torch.cat([a, b])
        self.assertTrue(torch.equal(c.cpu(), torch.tensor([1.0, 2.0, 3.0, 4.0])))

    def test_stack(self):
        a = torch.tensor([1.0, 2.0], device=DEVICE)
        b = torch.tensor([3.0, 4.0], device=DEVICE)
        s = torch.stack([a, b])
        self.assertEqual(s.shape, (2, 2))
        self.assertTrue(torch.equal(s.cpu(), torch.tensor([[1.0, 2.0], [3.0, 4.0]])))

    def test_contiguous(self):
        a = torch.randn(4, 4, device=DEVICE)
        t = a.t()
        self.assertFalse(t.is_contiguous())
        c = t.contiguous()
        self.assertTrue(c.is_contiguous())
        self.assertTrue(torch.equal(c.cpu(), t.cpu()))

    def test_expand(self):
        a = torch.tensor([[1.0], [2.0], [3.0]], device=DEVICE)
        b = a.expand(3, 4)
        self.assertEqual(b.shape, (3, 4))
        self.assertTrue(torch.allclose(b.sum(dim=1).cpu(), torch.tensor([4.0, 8.0, 12.0])))


# ---------------------------------------------------------------------------
# Type casting
# ---------------------------------------------------------------------------

@_skip_no_gpu
class TestDtypes(unittest.TestCase):

    def test_float_to_half(self):
        a = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
        h = a.half()
        self.assertEqual(h.dtype, torch.float16)
        self.assertTrue(torch.allclose(h.float().cpu(), torch.tensor([1.0, 2.0, 3.0])))

    def test_int_to_float(self):
        a = torch.tensor([1, 2, 3], dtype=torch.int32, device=DEVICE)
        f = a.float()
        self.assertEqual(f.dtype, torch.float32)
        self.assertTrue(torch.allclose(f.cpu(), torch.tensor([1.0, 2.0, 3.0])))

    def test_bool(self):
        a = torch.tensor([0.0, 1.0, -1.0], device=DEVICE)
        b = a.bool()
        self.assertTrue(torch.equal(b.cpu(), torch.tensor([False, True, True])))


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

@_skip_no_gpu
class TestActivations(unittest.TestCase):

    def test_relu(self):
        a = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=DEVICE)
        result = torch.relu(a)
        self.assertTrue(torch.equal(result.cpu(), torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])))

    def test_sigmoid(self):
        a = torch.tensor([0.0], device=DEVICE)
        result = torch.sigmoid(a)
        self.assertTrue(torch.allclose(result.cpu(), torch.tensor([0.5])))

    def test_tanh(self):
        a = torch.tensor([0.0], device=DEVICE)
        result = torch.tanh(a)
        self.assertTrue(torch.allclose(result.cpu(), torch.tensor([0.0])))

    def test_softmax(self):
        a = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
        s = torch.softmax(a, dim=0)
        self.assertTrue(torch.allclose(s.sum().cpu(), torch.tensor(1.0), atol=1e-5))
        sc = s.cpu()
        self.assertLess(sc[0], sc[1])
        self.assertLess(sc[1], sc[2])

    def test_gelu(self):
        a = torch.tensor([0.0], device=DEVICE)
        result = torch.nn.functional.gelu(a)
        self.assertTrue(torch.allclose(result.cpu(), torch.tensor([0.0]), atol=1e-5))


# ---------------------------------------------------------------------------
# Neural network layers (forward pass)
# ---------------------------------------------------------------------------

@_skip_no_gpu
class TestNNLayers(unittest.TestCase):

    def test_linear(self):
        torch.manual_seed(42)
        layer = torch.nn.Linear(8, 4).to(DEVICE)
        x = torch.randn(2, 8, device=DEVICE)
        y = layer(x)
        self.assertEqual(y.shape, (2, 4))

    def test_conv2d(self):
        torch.manual_seed(42)
        conv = torch.nn.Conv2d(1, 4, kernel_size=3, padding=1).to(DEVICE)
        x = torch.randn(1, 1, 8, 8, device=DEVICE)
        y = conv(x)
        self.assertEqual(y.shape, (1, 4, 8, 8))

    def test_batchnorm(self):
        bn = torch.nn.BatchNorm1d(4).to(DEVICE)
        bn.eval()
        x = torch.randn(8, 4, device=DEVICE)
        y = bn(x)
        self.assertEqual(y.shape, (8, 4))

    def test_layernorm(self):
        ln = torch.nn.LayerNorm(16).to(DEVICE)
        x = torch.randn(4, 16, device=DEVICE)
        y = ln(x)
        self.assertEqual(y.shape, (4, 16))

    def test_embedding(self):
        emb = torch.nn.Embedding(100, 16).to(DEVICE)
        idx = torch.tensor([0, 5, 99], device=DEVICE)
        y = emb(idx)
        self.assertEqual(y.shape, (3, 16))

    def test_dropout_eval(self):
        drop = torch.nn.Dropout(p=0.5).to(DEVICE)
        drop.eval()
        x = torch.ones(100, device=DEVICE)
        y = drop(x)
        self.assertTrue(torch.equal(y.cpu(), torch.ones(100)))


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------

@_skip_no_gpu
class TestAutograd(unittest.TestCase):

    def test_backward_simple(self):
        x = torch.tensor([2.0, 3.0], device=DEVICE, requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        self.assertTrue(torch.allclose(x.grad.cpu(), torch.tensor([4.0, 6.0])))

    def test_backward_chain(self):
        x = torch.tensor([1.0], device=DEVICE, requires_grad=True)
        y = 3 * x + 2
        z = y ** 2
        z.backward()
        self.assertTrue(torch.allclose(x.grad.cpu(), torch.tensor([30.0])))

    def test_no_grad_context(self):
        x = torch.tensor([1.0], device=DEVICE, requires_grad=True)
        with torch.no_grad():
            y = x * 2
        self.assertFalse(y.requires_grad)


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------

@_skip_no_gpu
class TestMemory(unittest.TestCase):

    def test_empty_cache(self):
        """torch.cuda.empty_cache() should not raise."""
        t = torch.randn(1024, device=DEVICE)
        del t
        torch.cuda.empty_cache()

    def test_memory_allocated(self):
        before = torch.cuda.memory_allocated(DEVICE)
        t = torch.randn(1024, device=DEVICE)
        after = torch.cuda.memory_allocated(DEVICE)
        self.assertGreaterEqual(after, before)

    def test_synchronize(self):
        """torch.cuda.synchronize() should not raise."""
        _ = torch.randn(64, device=DEVICE) + torch.randn(64, device=DEVICE)
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Random number generation
# ---------------------------------------------------------------------------

@_skip_no_gpu
class TestRNG(unittest.TestCase):

    def test_manual_seed_reproducible(self):
        torch.cuda.manual_seed(12345)
        a = torch.randn(64, device=DEVICE)
        torch.cuda.manual_seed(12345)
        b = torch.randn(64, device=DEVICE)
        self.assertTrue(torch.equal(a, b))

    def test_randn_stats(self):
        t = torch.randn(10000, device=DEVICE)
        self.assertAlmostEqual(t.mean().item(), 0.0, delta=0.1)
        self.assertAlmostEqual(t.std().item(), 1.0, delta=0.1)

    def test_randint(self):
        t = torch.randint(0, 10, (100,), device=DEVICE)
        self.assertGreaterEqual(t.min().item(), 0)
        self.assertLess(t.max().item(), 10)


if __name__ == "__main__":
    unittest.main()
