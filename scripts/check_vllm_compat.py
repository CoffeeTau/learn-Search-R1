#!/usr/bin/env python3
"""Read-only preflight for the Search-R1 hybrid FSDP/vLLM rollout."""

import importlib
import importlib.metadata
import os
import sys

from packaging.version import Version


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    names = ('torch', 'vllm', 'transformers', 'tensordict', 'ray')
    versions = {name: package_version(name) for name in names}
    for name in names:
        print(f'{name:12} {versions[name] or "MISSING"}')

    errors = []
    warnings = []
    if versions['torch'] is None:
        errors.append('torch is not installed')
    if versions['vllm'] is None:
        errors.append('vllm is not installed')
    else:
        parsed_vllm = Version(versions['vllm'])
        legacy_versions = {'0.3.1', '0.4.2', '0.5.4', '0.6.3'}
        if (versions['vllm'] not in legacy_versions
                and not Version('0.7.0') <= parsed_vllm < Version('0.9.0')):
            errors.append(f'vllm {versions["vllm"]} is not supported')

    if versions['tensordict'] and Version(versions['tensordict']) > Version('0.6.2'):
        warnings.append(
            f'tensordict {versions["tensordict"]} is newer than the upstream-tested 0.6.2')

    if versions['vllm'] and Version(versions['vllm']) >= Version('0.7.0'):
        if os.environ.get('VLLM_USE_V1') not in (None, '0'):
            errors.append('set VLLM_USE_V1=0; this backport synchronizes weights through vLLM V0')
        if os.environ.get('VLLM_ATTENTION_BACKEND'):
            warnings.append('unset VLLM_ATTENTION_BACKEND for the vLLM 0.8 path')
        try:
            importlib.import_module('vllm.distributed.parallel_state')
        except Exception as exc:
            errors.append(f'cannot import vllm.distributed.parallel_state: {exc}')

    try:
        import torch
        print(f'cuda_available {torch.cuda.is_available()}')
        print(f'cuda_devices   {torch.cuda.device_count()}')
        if not torch.cuda.is_available():
            errors.append('CUDA is not available')
    except Exception as exc:
        errors.append(f'cannot import torch: {exc}')

    for message in warnings:
        print(f'WARNING: {message}')
    for message in errors:
        print(f'ERROR: {message}')

    if errors:
        return 1
    print('OK: package imports and the vLLM compatibility surface look usable')
    return 0


if __name__ == '__main__':
    sys.exit(main())
