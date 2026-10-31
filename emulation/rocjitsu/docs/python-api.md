# Python execution API

The `rocjitsu` package runs Python callables on simulated AMD GPUs from scripts,
tests, interactive shells, and notebooks. `run` starts a worker for one call;
`Session` retains imports and worker state across calls.

```python
import rocjitsu

def add_one():
    import torch
    return (torch.arange(8, device="cuda") + 1).cpu()

with rocjitsu.Session(rocjitsu.config.gfx1250_mi455x) as sim:
    result = sim.run(add_one)
```

Install the Python package and matching native runtime into your environment.
Workloads need their own dependencies, such as ROCm-enabled PyTorch. Bundled
wheels require follow-up TheRock packaging support.

See the [package guide](https://github.com/ROCm/rocm-systems/blob/develop/emulation/rocjitsu/python/README.md)
for installation, configuration presets, worker environments, error handling,
and Jupyter or Colab examples.
