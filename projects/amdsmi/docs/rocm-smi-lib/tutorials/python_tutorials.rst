====================
Python Tutorials
====================

This chapter is the rocm_smi Python api tutorials.

.. code-block:: python

    import sys
    sys.path.append("/opt/rocm/libexec/rocm_smi/")
    try:
        import rocm_smi
    except ImportError:
        raise ImportError("Could not import /opt/rocm/libexec/rocm_smi/rocm_smi.py")

    class prof_utils:
        def __init__(self, mode) -> None:
            rocm_smi.initializeRsmi()

        def getPower(self, device):
            return rocm_smi.getPower(device)

        def listDevices(self):
            return rocm_smi.listDevices()

        def getMemInfo(self, device):
            (memUsed, memTotal) = rocm_smi.getMemInfo(device, "vram")
            return round(float(memUsed)/float(memTotal) * 100, 2)

