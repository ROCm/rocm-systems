# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reuse the pinned upstream ToyLlama loader, inputs and cross-entropy oracle."""

import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

TOKENS = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('suite', 'vmfb', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--plugin', type=Path)
    args = parser.parse_args()
    os.environ['ASSET_PATH'] = str(args.suite.parent)
    spec = importlib.util.spec_from_file_location(
        'upstream_toy_llama', args.suite / 'test_llama.py'
    )
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    model = upstream.ToyLlama(
        compiled=args.vmfb.read_bytes(),
        device_id='hip',
        irpa=upstream.llama_irpa,
        sharding=1,
    )
    original = model.prefill

    def retain_logits(**kwargs):
        logits = original(**kwargs)
        np.save(args.output / 'logits.npy', logits)
        return logits

    model.prefill = retain_logits
    value = float(upstream.prefill_cross_entropy(model, TOKENS))
    logits = np.load(args.output / 'logits.npy', allow_pickle=False)
    result = {
        'cross_entropy': value,
        'passed': bool(np.isfinite(value) and value == pytest.approx(0.589, rel=0.1)),
        'shape': list(logits.shape),
        'finite': bool(np.isfinite(logits).all()),
        'dtype': str(logits.dtype),
    }
    if args.plugin:
        plugin = ctypes.CDLL(str(args.plugin), mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
        plugin.rj_consan_finish.restype = ctypes.c_int
        result['finish'] = plugin.rj_consan_finish()
        assert result['finish'] == 0, result
    (args.output / 'oracle.json').write_text(json.dumps(result, indent=2) + '\n')
    assert result['passed'] and result['finite'], result
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
