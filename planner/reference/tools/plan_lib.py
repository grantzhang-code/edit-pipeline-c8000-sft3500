"""Build pe/plans/plan-<step>.json for one r8 checkpoint on one 4-node RTX group.

Shared by make_plans.py (c18000/c19000) and dispatch.py (later checkpoints).
Every checkpoint publishes to its own new, empty S3 prefix.
"""
import hashlib
import json
from pathlib import Path

PE = Path(__file__).resolve().parents[1]/'pe'
R8 = 's3://core-cn-oss.canva.com/usr/rabinxie/design-layout-megatron-swift/runs/rabinxie-editparser-frozen-360b300-1022-r8/mcore/'
R7 = 's3://core-cn-oss.canva.com/usr/rabinxie/design-layout-megatron-swift/runs/rabinxie-editparser-frozen-360b300-1020-r7/mcore/'
R3 = 's3://core-cn-oss.canva.com/usr/rabinxie/design-layout-megatron-swift/runs/rabinxie-editparser-9b-360b300-1023-r3/mcore/'
R8_9B = 's3://core-cn-oss.canva.com/usr/rabinxie/design-layout-megatron-swift/runs/rabinxie-editparser-9b-336b300-1024-r8/mcore/'
R10_9B = 's3://core-cn-oss.canva.com/usr/rabinxie/design-layout-megatron-swift/runs/rabinxie-editparser-9b-360b300-1024-r10/mcore/'
R12_9B = 's3://core-cn-oss.canva.com/usr/rabinxie/design-layout-megatron-swift/runs/rabinxie-editparser-9b-360b300-1025-r12/mcore/'
R16_9B = 's3://core-cn-oss.canva.com/usr/rabinxie/design-layout-megatron-swift/runs/rabinxie-editparser-9b-856b300-1025-r16/mcore/'
FZSFT_9B = 's3://core-cn-oss.canva.com/usr/fengzhida/design-layout-megatron-swift/runs/fengzhida-editing-planner-9b-sft-128g-1025c/mcore/'
# Base HF artifact each lineage trained from (export bootstrap: config/tokenizer + the expected tensor names).
BASE_4B = dict(s3='s3://core-cn-oss.canva.com/usr/joeyzhang/design-layout-megatron-swift/models/qwen35-4b-base-hf-mtp-851bf6e-v1/',
               manifest_sha256='d355542d67dbc5deb206034c0782daec202f57828bae79b79478417add2c750e',
               local_dir='/mnt/ephemeral/models/qwen35-4b-base-hf-mtp-851bf6e', weight_keys=738, mtp_keys=15, tie_word_embeddings=True)
BASE_9B = dict(s3='s3://core-cn-oss.canva.com/usr/grantzhang/design-layout-megatron-swift/models/qwen35-9b-base-hf-mtp-c202236-v1/',
               manifest_sha256='43b13c361cef398c56983b8674935f7692cc972311806552be159ae1bb46fe1c',
               local_dir='/mnt/ephemeral/models/qwen35-9b-base-hf-mtp-c202236', weight_keys=775, mtp_keys=15, tie_word_embeddings=False)
# Training identity per lineage; pe/lineage/<name>/ holds that run's overlay, args, identity and jobset yaml.
# r8 is a full-state resume of r7 c17000 (same image, prompt and numerics; launcher-only overlay change).
LINEAGES = {
    'r8': dict(ckpt_root=R8, run_id='4d74b5f1191d', training_jobset='rabinxie-editparser-frozen-360b300-1022-r8',
               initial_model=BASE_4B, ablation='given_ratio_inline_edits', tag='rtx32_20261022',
               note='r8 code identity is the pinned configmap overlay rabinxie-editparser-frozen-360b300-1022-r8-code (training_overlay_sha256); r8 = full-state resume of r7 227d90c8f6ed checkpoint-17000; runtime image ds-core-cn-omra-trainer@sha256:54d365f8...'),
    'r7': dict(ckpt_root=R7, run_id='227d90c8f6ed', training_jobset='rabinxie-editparser-frozen-360b300-1020-r7',
               initial_model=BASE_4B, ablation='given_ratio_inline_edits', tag='rtx32_20261022',
               note='r7 code identity is the pinned configmap overlay rabinxie-editparser-frozen-360b300-1020-r7-code (training_overlay_sha256); runtime image ds-core-cn-omra-trainer@sha256:54d365f8...'),
    # r3: Qwen3.5-9B from the base HF artifact (no parent), plan-only target, 32768 context.
    'r3': dict(ckpt_root=R3, run_id='5a86f6d858df', training_jobset='rabinxie-editparser-9b-360b300-1023-r3',
               initial_model=BASE_9B, ablation='given_ratio_plan_only_inline_edits', tag='rtx8_20261023',
               note='r3 code identity is the pinned configmap overlay rabinxie-editparser-9b-360b300-1023-r3-code (training_overlay_sha256 f616ea12..., byte-identical to git 5ed3723); runtime image ds-core-cn-omra-trainer@sha256:54d365f8...'),
    # r8-9b: full-state resume of r3 c9000 on 336 B300 (42 x 8); same prompt / numerics code as r3 (overlay 256bec2b).
    'r8-9b': dict(ckpt_root=R8_9B, run_id='f190b7efe908', training_jobset='rabinxie-editparser-9b-336b300-1024-r8',
               initial_model=BASE_9B, ablation='given_ratio_plan_only_inline_edits', tag='rtx64_bs64_20261024',
               topology=dict(world_size=336, replicas=42),
               note='r8-9b code identity is the pinned configmap overlay rabinxie-editparser-9b-336b300-1024-r8-code (training_overlay_sha256 256bec2b..., git c55fcc40); full-state resume of r3 5a86f6d858df checkpoint-9000; runtime image ds-core-cn-omra-trainer@sha256:54d365f8...'),
    # r10-9b: full-state resume of r8-9b c12000 on 360 B300 (45 x 8); same prompt / numerics code as r3 / r8-9b (overlay e9ae8a36).
    'r10-9b': dict(ckpt_root=R10_9B, run_id='73f37d4788c6', training_jobset='rabinxie-editparser-9b-360b300-1024-r10',
               initial_model=BASE_9B, ablation='given_ratio_plan_only_inline_edits', tag='rtx32_bs64_20261024',
               topology=dict(world_size=360, replicas=45),
               note='r10-9b code identity is the pinned configmap overlay rabinxie-editparser-9b-360b300-1024-r10-code (training_overlay_sha256 e9ae8a36..., git 9ff6f83a); full-state resume of r8-9b f190b7efe908 checkpoint-12000; runtime image ds-core-cn-omra-trainer@sha256:54d365f8...'),
    # r12-9b: full-state resume of r10-9b c16000 on 360 B300 (45 x 8); v4 training data, same prompt / numerics code as r10-9b (overlay 616d7e63).
    'r12-9b': dict(ckpt_root=R12_9B, run_id='55930c837a64', training_jobset='rabinxie-editparser-9b-360b300-1025-r12',
               initial_model=BASE_9B, ablation='given_ratio_plan_only_inline_edits', tag='rtx32_bs64_20261025',
               topology=dict(world_size=360, replicas=45),
               note='r12-9b code identity is the pinned configmap overlay rabinxie-editparser-9b-360b300-1025-r12-code (training_overlay_sha256 616d7e63..., git 953e5726); full-state resume of r10-9b 73f37d4788c6 checkpoint-16000; runtime image ds-core-cn-omra-trainer@sha256:54d365f8...'),
    # r16-9b: resume of r12-9b c21000 on 856 B300 (107 x 8, fresh-cursor consumed-row bitmap, shard-strided data partition); same
    # prompt / model code and v4 data as r12-9b, overlay 8c982a6a changes only topology / resume / data-partition code.
    'r16-9b': dict(ckpt_root=R16_9B, run_id='fb7e3ba506f6', training_jobset='rabinxie-editparser-9b-856b300-1025-r16',
               initial_model=BASE_9B, ablation='given_ratio_plan_only_inline_edits', tag='rtx32_bs64_20261025',
               topology=dict(world_size=856, replicas=107),
               note='r16-9b code identity is the pinned configmap overlay rabinxie-editparser-9b-856b300-1025-r16-code (training_overlay_sha256 8c982a6a..., git 166c82a8); resume of r12-9b 55930c837a64 checkpoint-21000; runtime image ds-core-cn-omra-trainer@sha256:54d365f8...'),
    # fzsft-9b: editing-planner SFT on 128 B300 (16 x 8, GA1), model-only warm start from r16-9b c25000; new corpus
    # (josephbai mie-parse-merge-orig-20261026), same plan-only prompt / model / numerics code as r16-9b (overlay cd532929).
    # Its steps (500, 1000, ...) overlap the r3 era's, so its checkpoint files are pe/checkpoint-fzsft-9b-<step>-*.json.
    'fzsft-9b': dict(ckpt_root=FZSFT_9B, run_id='4b3162283b0b', training_jobset='fengzhida-editing-planner-9b-sft-128g-1025c',
               initial_model=BASE_9B, ablation='given_ratio_plan_only_inline_edits', tag='rtx32_bs64_20261026',
               topology=dict(world_size=128, replicas=16), ckpt_file_prefix='checkpoint-fzsft-9b-',
               note='fzsft-9b code identity is the pinned configmap overlay fengzhida-editing-planner-9b-sft-128g-1025c-code (training_overlay_sha256 cd532929..., git 5242e3d8, branch fengzhida/editing-planner-9b-sft); model-only warm start from r16-9b fb7e3ba506f6 checkpoint-25000; runtime image ds-core-cn-omra-trainer@sha256:54d365f8...'),
}
LINEAGE_FILES = ('training-overlay.tar.gz', 'training-args.json', 'TRAINING_IDENTITY.json', 'training.yaml')
CODE = ['controller.py', 'frozen_training_mrope.py', 'common.py', 'worker.py', 'sourcejson_contract.py', 'export_model.py',
        'training_vision_rope_patch.py', 'sitecustomize.py', 'hsdp_model_only_export.py', 'frozen_training_vision.py',
        'publish.py', 'cache_model.py', 'precision_probe.py', 'vision_reference_probe.py', 'gdn_reference_probe.py',
        'engine_smoke.py', 'engine_capture_probe.py', 'training_alignment_compare.py', 'pe_training_alignment.py',
        'engine_config.py', 'training_fla_guard.py', 'r7_chunk_o_pin.py', 'training_image.py', 'template_probe.py',
        'contract_utils.py', 'r3_prompt_contract.json', 'context_guard.py', 'aligned_recurrent.py', 'frozen_sdpa.py', 'frozen_vision_utils.py',
        'training_variants.py', 'preflight.py', 'preflight_dataset.py', 's5_store.py', 'vision_gap_probe.py', 'rerun_subset.py', 'bin/s5cmd', 'sm120_fla_policy.py', 'training-overlay.tar.gz', 'training-args.json', 'TRAINING_IDENTITY.json']



def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_id(step, attempt=1, lineage='r8', dataset=None, tag=None):
    # a re-run after a lost allocation never shares a directory with the earlier attempt
    return (f'infer_editparser_{lineage}_{LINEAGES[lineage]["run_id"]}_c{step}_{tag or LINEAGES[lineage]["tag"]}'
            + (f'_{dataset}' if dataset else '') + ('' if attempt == 1 else f'_a{attempt}'))


def output_s3(step, attempt=1, lineage='r8', dataset=None, tag=None):
    # a re-run publishes to its own new prefix: no stale LEADER_READY / DONE markers from the lost attempt
    return (f's3://core-cn-oss.canva.com/usr/grantzhang/{dataset or "balanced-1500-r6"}-editparser-{lineage}-{LINEAGES[lineage]["run_id"]}-c{step}-trainprompt-greedy-{(tag or LINEAGES[lineage]["tag"]).replace("_", "-")}'
            + ('' if attempt == 1 else f'-attempt{attempt}') + '/')


def make_plan(step, jobset, jobset_uid, cluster, hold_nodes, attempt=1, lineage='r8', dataset=None, tag=None, reuse_export_s3=None):
    line = LINEAGES[lineage]
    source = lambda name: PE/'lineage'/lineage/name if name in LINEAGE_FILES else PE/name
    base = json.loads((PE/'plan.json').read_text())
    assert base['training_ablation'] == line['ablation'], 'base plan prompt contract belongs to another lineage'
    missing = set(base['execution_code_sha256']) - set(CODE)
    assert not missing, missing
    ck = PE/f'{line.get("ckpt_file_prefix", "checkpoint-")}{step}'
    success = json.loads(Path(f'{ck}-SUCCESS.json').read_text())
    assert success['step'] == step and success['status'] == 'complete' and success['manifest_sha256'] == sha(f'{ck}-manifest.json')
    task, out = task_id(step, attempt, lineage, dataset, tag), output_s3(step, attempt, lineage, dataset, tag)
    if dataset and lineage not in ('r3', 'r8-9b', 'r10-9b', 'r12-9b', 'r16-9b', 'fzsft-9b'):  # a dataset may name its own result prefix (a 4B-r8-era template; the 9B lineages use their own)
        custom = json.loads((PE/'datasets'/dataset/'DATASET.json').read_text()).get('output_s3')
        if custom:
            out = custom.format(step=step).rstrip('/') + ('' if attempt == 1 else f'-attempt{attempt}') + '/'
    plan = dict(base)
    plan.update(task_id=task, runtime_task_id=task, shared=f'/mnt/ephemeral/{task}',
        training_runtime=f'/mnt/ephemeral/{task}/training-runtime', run_id=line['run_id'], lineage=lineage,
        training_jobset=line['training_jobset'], initial_model=line['initial_model'], training_topology=line.get('topology', dict(world_size=360, replicas=45)),
        training_overlay_sha256=sha(source('training-overlay.tar.gz')),
        training_git_commit_note=line['note'],
        jobset=jobset, jobset_uid=jobset_uid, cluster=cluster, hold_nodes=list(hold_nodes),
        gpu='NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120)', nodes=len(hold_nodes), gpus=8*len(hold_nodes),
        steps=[step], attempt=attempt, output_s3=out, overlength_output=f'checkpoint-{step}/overlength.jsonl',
        models=[dict(step=step, checkpoint_s3=line['ckpt_root']+f'checkpoint-{step}/', checkpoint_sha256=success['manifest_sha256'],
                     commit=success, artifact_s3=out+f'models/checkpoint-{step}/', arm='sourcejson',
                     nodes=list(range(len(hold_nodes))))],
        execution_code_sha256={name: sha(source(name)) for name in CODE})
    plan.pop('user_authorized_interruption_replay', None)
    if reuse_export_s3:  # leader restores this run's published HF export of the same checkpoint (cache_model) instead of exporting
        plan['reuse_export_s3'] = reuse_export_s3.rstrip('/') + '/'
        plan['models'][0]['artifact_s3'] = plan['reuse_export_s3'] + f'models/checkpoint-{step}/'
    if dataset:
        # Non-benchmark dataset: its own cases / regions / targets / exclusions (pe/datasets/<name>/, copied to the
        # task root by node_run.sh) and the training-path prompt gate preflight_dataset.py.
        d = PE/'datasets'/dataset
        meta = json.loads((d/'DATASET.json').read_text())
        for name, digest in meta['files'].items():
            assert sha(d/name) == digest, name
        for key in ('source_caption_pool_s3', 'source_caption_file_sha256', 'source_caption_arm', 'source_regions_note'):
            plan.pop(key, None)
        plan.update(dataset=dataset, dataset_meta=meta, benchmark_s3=meta['source_uri'],
            benchmark_cases_sha256=meta['files']['cases.jsonl'], source_regions_sha256=meta['files']['source-regions.json'],
            eligible_benchmark_cases=meta['eligible'], instruction_source_s3=meta['source_uri'],
            instruction_policy=meta['instruction_policy'], instruction_variant=meta.get('instruction_variant', 'frame-v8-original-instructions'),
            target_ratio_policy=meta['target_size_policy'], preflight='preflight_dataset.py',
            expected_prompt_hashes_sha256=sha(d/'expected-prompt-hashes.json'),
            engine_probe=dict(cases=meta['engine_probe_cases'], gdn_layers=[0, 14, 30],
                              rationale='longest-context multi-image eligible cases of the dataset (12 and 8 sources)'))
    (PE/'plans').mkdir(exist_ok=True)
    path = PE/'plans'/f'plan-{step}.json'
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False)+'\n')
    return path
