# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from importlib.metadata import version, PackageNotFoundError

from packaging.version import Version


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


package_name = 'vllm'
package_version = get_version(package_name)
vllm_version = package_version

if package_version == '0.3.1':
    vllm_version = '0.3.1'
    from .vllm_v_0_3_1.llm import LLM
    from .vllm_v_0_3_1.llm import LLMEngine
    from .vllm_v_0_3_1 import parallel_state
elif package_version == '0.4.2':
    vllm_version = '0.4.2'
    from .vllm_v_0_4_2.llm import LLM
    from .vllm_v_0_4_2.llm import LLMEngine
    from .vllm_v_0_4_2 import parallel_state
elif package_version == '0.5.4':
    vllm_version = '0.5.4'
    from .vllm_v_0_5_4.llm import LLM
    from .vllm_v_0_5_4.llm import LLMEngine
    from .vllm_v_0_5_4 import parallel_state
elif package_version == '0.6.3':
    vllm_version = '0.6.3'
    from .vllm_v_0_6_3.llm import LLM
    from .vllm_v_0_6_3.llm import LLMEngine
    from .vllm_v_0_6_3 import parallel_state
elif (package_version is not None
      and Version('0.7.0') <= Version(package_version) < Version('0.9.0')):
    # vLLM 0.7+ has the external-launcher SPMD support needed by veRL.  Do
    # not import the vendored 0.6.3 engine internals: their Worker/Engine
    # constructors are incompatible with vLLM 0.8.x.
    #
    # This backport uses the stable V0 engine layout when synchronizing FSDP
    # weights.  vLLM 0.8.4 still ships V0, while its V1 engine exposes a
    # different model-executor API.  Reject an incompatible explicit override
    # before vLLM is imported and allocates GPU memory.
    if os.environ.get('VLLM_USE_V1') not in (None, '0'):
        raise RuntimeError(
            'Search-R1 requires VLLM_USE_V1=0 with vLLM 0.7+. '
            'Unset VLLM_USE_V1 or export VLLM_USE_V1=0 before launching.'
        )
    os.environ['VLLM_USE_V1'] = '0'
    os.environ.pop('VLLM_ATTENTION_BACKEND', None)

    from vllm import LLM
    from vllm.distributed import parallel_state
else:
    raise ValueError(
        f'vllm version {package_version} not supported. Currently supported versions are '
        '0.3.1, 0.4.2, 0.5.4, 0.6.3 and the 0.7.x-0.8.x series.'
    )
