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
"""vLLM 0.7+ rollout based on its native external-launcher SPMD support."""

from contextlib import contextmanager
from typing import List

import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from vllm import LLM, SamplingParams

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    non_pad = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)
    if non_pad.numel() == 0:
        return []
    return prompt_token_ids[non_pad[0][0]:].tolist()


class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__()
        self.config = config
        tensor_parallel_size = config.get('tensor_model_parallel_size', 1)
        world_size = torch.distributed.get_world_size()
        assert tensor_parallel_size <= world_size
        assert world_size % tensor_parallel_size == 0

        max_model_len = int(config.get('max_model_len', None)
                            or config.prompt_length + config.response_length)
        max_num_batched_tokens = max(
            int(config.get('max_num_batched_tokens', 8192)), max_model_len)

        if model_hf_config.max_position_embeddings < max_model_len:
            raise ValueError(
                'model context length must cover prompt_length + response_length: '
                f'{model_hf_config.max_position_embeddings} < {max_model_len}')

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend='external_launcher',
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=int(config.get('max_num_seqs', 1024)),
            enable_chunked_prefill=bool(config.get('enable_chunked_prefill', False)),
            enable_prefix_caching=False,
            disable_log_stats=bool(config.get('disable_log_stats', True)),
        )
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            'n': 1,
            'logprobs': 0,
            'max_tokens': config.response_length,
            'detokenize': False,
        }
        sampling_defaults = SamplingParams()
        for key in config.keys():
            if hasattr(sampling_defaults, str(key)):
                sampling_kwargs[key] = config.get(key)
        self.sampling_params = SamplingParams(**sampling_kwargs)
        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        old_values = {}
        for key, value in kwargs.items():
            if hasattr(self.sampling_params, key):
                old_values[key] = getattr(self.sampling_params, key)
                setattr(self.sampling_params, key, value)
        try:
            yield
        finally:
            for key, value in old_values.items():
                setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        input_ids = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']
        batch_size = input_ids.size(0)

        prompt_token_ids = [
            _pre_process_inputs(self.pad_token_id, input_ids[i])
            for i in range(batch_size)
        ]
        vllm_inputs = [
            {'prompt_token_ids': ids}
            for ids in prompt_token_ids
        ]

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1,
            }

        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                vllm_inputs,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            response_ids = [
                sample.token_ids
                for request_output in outputs
                for sample in request_output.outputs
            ]
            responses = pad_2d_list_to_length(
                response_ids,
                self.pad_token_id,
                max_length=self.config.response_length,
            ).to(input_ids.device)

            num_samples = self.sampling_params.n if do_sample else 1
            if num_samples > 1:
                input_ids = input_ids.repeat_interleave(num_samples, dim=0)
                attention_mask = attention_mask.repeat_interleave(num_samples, dim=0)
                position_ids = position_ids.repeat_interleave(num_samples, dim=0)
                batch_size *= num_samples

        sequence = torch.cat([input_ids, responses], dim=-1)
        response_length = responses.size(1)
        delta_position_id = torch.arange(
            1, response_length + 1, device=position_ids.device).unsqueeze(0)
        delta_position_id = delta_position_id.expand(batch_size, -1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_eos_mask(
            response_id=responses,
            eos_token=eos_token_id,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat([attention_mask, response_attention_mask], dim=-1)

        batch = TensorDict(
            {
                'prompts': input_ids,
                'responses': responses,
                'input_ids': sequence,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch)
