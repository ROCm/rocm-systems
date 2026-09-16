"""SymmetricBuffer allocation synchronization tests."""

import rocshmem4py


def test_symmetric_buffer_synchronizes_before_malloc(monkeypatch):
    events = []

    def fake_sync():
        events.append("sync")

    def fake_malloc(size):
        events.append(("malloc", size))
        return 0x1000

    def fake_free(ptr):
        events.append(("free", ptr))

    monkeypatch.setattr(rocshmem4py, "hip_device_synchronize", fake_sync)
    monkeypatch.setattr(rocshmem4py, "rocshmem_malloc", fake_malloc)
    monkeypatch.setattr(rocshmem4py, "rocshmem_free", fake_free)

    buf = rocshmem4py.SymmetricBuffer(4096)
    buf._hip = None
    try:
        assert events[:2] == ["sync", ("malloc", 4096)]
    finally:
        buf.free()
