"""vLLM engine settings shared by worker.py and engine_smoke.py.

- enable_chunked_prefill=False with a max_model_len (r7: 40960) token step budget: every prompt is
  prefilled as one segment, so FLA's 64-token GDN chunks start at the request's
  first token exactly as in training. vLLM 0.29 otherwise splits a prefill at
  whatever token budget remains (shared by up to four requests), which is
  usually not 64-aligned and changes GDN numerics.
- mm_processor_kwargs do_resize=False: images are resized by training_image.py
  exactly as in training (swift rescale_image, then qwen_vl_utils) and must not
  be resized again by the HF processor.
- MTP: num_speculative_tokens equals the trained MTP depth (training-args.json
  mtp_num_layers=1, unshared). v1 used 3, re-running the single MTP layer for
  draft positions it was never trained on.
- reproducibility (plan['decoding']['reproducibility']):
    standard  batch 4, the benchmark setting. Not reproducible run to run: the
              EngineCore schedules requests as they arrive, so batch composition
              and MTP verification shapes vary (2/3 pilot outputs changed between
              two identical runs).
    batch1    one request at a time; reproducible (concurrent runs on different
              GPUs gave identical outputs for all pilot cases with MTP 3 and 1). Use it for
              reproducibility checks.
  VLLM_BATCH_INVARIANT is not an option: vLLM 0.29 raises "batch_invariant mode
  is not supported for GDN_ATTN" for this Qwen3.5 hybrid model.
"""
import json
import os
from pathlib import Path

REPRODUCIBILITY = ('standard', 'batch1')


def mode(plan):
    value = plan['decoding'].get('reproducibility', 'standard')
    assert value in REPRODUCIBILITY, value
    return value


def prepare_environment(plan):
    """Call before importing vLLM; EngineCore processes inherit the variable."""
    mode(plan)
    assert os.environ.get('VLLM_BATCH_INVARIANT', '0') in ('', '0'), 'VLLM_BATCH_INVARIANT is unsupported for GDN'


def batch_size(plan):
    return 1 if mode(plan) == 'batch1' else plan['decoding']['batch_size']


def training_mtp_layers():
    args = json.loads((Path(__file__).resolve().parent / 'training-args.json').read_text())
    assert args['mtp_shared_weights'] is False
    return args['mtp_num_layers']


def engine_kwargs(plan, model, hf_overrides):
    decoding = plan['decoding']
    assert decoding['mtp_tokens'] == training_mtp_layers(), 'MTP draft depth differs from training'
    length = decoding['max_model_len']
    assert decoding['enable_chunked_prefill'] is False and decoding['max_num_batched_tokens'] >= length
    assert decoding['mm_processor_kwargs'] == {'do_resize': False}
    return dict(model=model, tokenizer=model, trust_remote_code=True, dtype='bfloat16',
                tensor_parallel_size=1, max_model_len=length, max_num_seqs=batch_size(plan),
                enable_chunked_prefill=False, max_num_batched_tokens=decoding['max_num_batched_tokens'],
                gpu_memory_utilization=0.8, limit_mm_per_prompt={'image': 12, 'video': 0},
                mm_processor_kwargs=dict(decoding['mm_processor_kwargs']), seed=decoding['seed'],
                attention_backend='FLASH_ATTN', enforce_eager=True, enable_prefix_caching=False,
                hf_overrides=hf_overrides, additional_config={'gdn_prefill_backend': 'triton'},
                speculative_config={'method': 'mtp', 'num_speculative_tokens': decoding['mtp_tokens'],
                                    'attention_backend': 'FLASH_ATTN'})


def check_engine(llm, plan):
    """Fail closed if vLLM silently changed the scheduling contract."""
    config = llm.llm_engine.vllm_config
    scheduler = config.scheduler_config
    assert scheduler.enable_chunked_prefill is False, 'chunked prefill is still enabled'
    assert scheduler.max_num_batched_tokens >= plan['decoding']['max_model_len']
    assert scheduler.max_num_seqs == batch_size(plan)
    assert config.cache_config.enable_prefix_caching is False
    assert config.speculative_config.num_speculative_tokens == training_mtp_layers()
    return dict(num_speculative_tokens=config.speculative_config.num_speculative_tokens, enable_chunked_prefill=False, max_num_batched_tokens=scheduler.max_num_batched_tokens,
                max_num_seqs=scheduler.max_num_seqs, reproducibility=mode(plan))
