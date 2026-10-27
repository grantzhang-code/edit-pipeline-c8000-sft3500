"""Targeted re-run of selected benchmark cases for one checkpoint on one 8-GPU node, then a patched publish.

Uses the controller's runtime restore and image prefetch, the exported HF model of the original run
(restored and validated by cache_model.py against its published manifest), the vision gates
(vision_reference_probe on the training venv; vision_gap_probe compares the engine's vision module
with the training planner operators), then 8 workers with --only. Publishes to plan['output_s3']:
  checkpoint-<step>/rerun.jsonl          the re-run records (same schema as captions rows)
  checkpoint-<step>/captions.jsonl       the original run's captions with these cases replaced
  checkpoint-<step>/RERUN_REPORT.json    old vs new per case

usage: rerun_subset.py <original_output_s3> <comma-separated input indices> [export_output_s3]
  export_output_s3: run whose HF export to restore (an in-domain run reuses the balanced run's export); default the original.
  plan['decoding_override'] (optional): sampling for the re-run instead of the plan's greedy decoding.
  plan['decoding_rounds'] (optional, {"<round>": sampling, ...}): every round's sampling of all the cases in this one run,
    sharing the restore and the gates; the 8 GPUs are split over the rounds (GPU g -> round g mod #rounds), each round's
    workers run in their own task dir round-<n>/ (plan.json with that round's decoding_override) and its three files
    are published under plan['output_s3'] + 'round-<n>/'.
"""
import json, os, subprocess, sys, time
from pathlib import Path

import controller as c
from common import digest, get, upload, write

ORIGINAL, ONLY = sys.argv[1].rstrip('/') + '/', sys.argv[2]
EXPORT = (sys.argv[3].rstrip('/') + '/') if len(sys.argv) > 3 else ORIGINAL
INDICES = [int(x) for x in ONLY.split(',')]
STEP, CP, OUT = c.STEP, c.CP, c.OUT
SHARED_FILES = ('sourcejson_contract.py', 'cases.jsonl', 'target-dimensions.json', 'source-regions.json', 'input-exclusions.json',
                'expected-prompt-hashes.json')  # worker.py's own reads from its --shared dir besides plan.json / checkpoint-<step>/


def round_dir(n, sampling):
    d = c.ROOT / f'round-{n}'
    (d / f'checkpoint-{STEP}').mkdir(parents=True, exist_ok=True)
    plan = {k: v for k, v in c.PLAN.items() if k != 'decoding_rounds'}
    plan['decoding_override'] = sampling
    (d / 'plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n')
    for src in c.ROOT.iterdir():  # contract / dataset files (load_contract also reads its siblings)
        if src.name not in ('plan.json', f'checkpoint-{STEP}') and not src.name.startswith('round-') and not (d / src.name).exists():
            (d / src.name).symlink_to(src)
    assert all((d / name).exists() for name in SHARED_FILES)
    return d


def report_files(rows, summary, sampling, prefix):
    import gzip
    old = {}
    for line in gzip.decompress(get(c.S3, ORIGINAL + f'checkpoint-{STEP}/responses.jsonl.gz')).decode().splitlines():
        row = json.loads(line); old[row['input_index']] = row
    new = {r['input_index']: r for r in rows}
    report = []
    for i in sorted(INDICES):
        o, n = old.get(i, {}), new[i]
        report.append(dict(input_index=i, response_changed=o.get('response_sha256') != n['response_sha256'],
                           old={k: o.get(k) for k in ('json_valid', 'schema_valid', 'source_references_valid', 'finish_reason', 'output_tokens', 'validation_error')},
                           new={k: n.get(k) for k in ('json_valid', 'schema_valid', 'source_references_valid', 'finish_reason', 'output_tokens', 'validation_error')}))
    patched = []
    for line in get(c.S3, ORIGINAL + f'checkpoint-{STEP}/captions.jsonl').decode().splitlines():
        row = json.loads(line)
        if row['input_index'] in new:
            fresh = new[row['input_index']]
            row.update(caption=fresh['prediction'], caption_grouped=fresh.get('prediction_grouped'), json_valid=fresh['json_valid'],
                       schema_valid=fresh['schema_valid'], source_references_valid=fresh['source_references_valid'],
                       vision_attention_policy='training_vision_native_forward_batch', rerun_sampling=sampling,
                       replaced_from_rerun=OUT + prefix + f'checkpoint-{STEP}/rerun.jsonl')
        patched.append(json.dumps(row, ensure_ascii=False))
    return {prefix + f'checkpoint-{STEP}/rerun.jsonl': ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode(),
            prefix + f'checkpoint-{STEP}/captions.jsonl': ('\n'.join(patched) + '\n').encode(),
            prefix + f'checkpoint-{STEP}/RERUN_REPORT.json': json.dumps(dict(original=ORIGINAL, export=EXPORT, sampling=sampling, indices=INDICES, cases=report,
                new_valid=dict(json=sum(r['json_valid'] for r in rows), schema=sum(r['schema_valid'] for r in rows),
                               refs=sum(bool(r['source_references_valid']) for r in rows)),
                changed_responses=sum(r['response_changed'] for r in report), vision_gap=summary), indent=1).encode()}


def main():
    c.NODE = 0
    for name, d in c.PLAN['execution_code_sha256'].items():
        assert c.sha(c.ROOT / name) == d, name
    assert json.loads((c.ROOT / 'TRAINING_PROMPT_PREFLIGHT.json').read_text())['status'] == 'PASS'
    CP.mkdir(exist_ok=True)
    c.state('rerun_restoring_runtime', indices=INDICES)
    c.runtime(); c.training_runtime(); c.prefetch_images()
    for name in ('EXPORTED.json', 'HSDP_EXPORT_PROOF.json', 'CHECKPOINT_VERIFIED.json'):
        (CP / name).write_bytes(get(c.S3, EXPORT + f'checkpoint-{STEP}/' + name))
    exported = json.loads((CP / 'EXPORTED.json').read_text())
    c.PLAN['models'][0]['artifact_s3'] = EXPORT + f'models/checkpoint-{STEP}/'
    (c.ROOT / 'plan.json').write_text(json.dumps(c.PLAN, indent=2, ensure_ascii=False) + '\n')
    c.run([c.NATIVE, '-u', str(c.ROOT / 'cache_model.py'), str(c.ROOT), str(STEP)], 'cache_model.log', dict(os.environ, PYTHONPATH=str(c.ROOT)))
    env, model, evidence = c.engine_env()
    c.state('rerun_vision_gates')
    c.run([c.NATIVE, '-u', str(c.ROOT / 'vision_reference_probe.py')], 'vision_reference_probe.py.log',
          dict(env, CUDA_VISIBLE_DEVICES='0', PE_TRAINING_ALIGNMENT='0'))
    assert json.loads((evidence / 'VISION_REFERENCE_PASS.json').read_text())['status'] == 'PASS'
    gap = evidence / 'VISION_GAP.json'
    c.run([c.NATIVE, '-u', str(c.ROOT / 'vision_gap_probe.py'), str(c.ROOT), str(model), str(c.ROOT / 'pe-images'), str(gap), *map(str, INDICES)],
          'vision_gap_probe.log', dict(env, CUDA_VISIBLE_DEVICES='0', PE_TRAINING_ALIGNMENT='0',
                                       PYTHONPATH=f'{c.PLAN["training_runtime"]}:{c.ROOT}', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'))
    summary = json.loads(gap.read_text())['summary']
    assert summary['cases'] == len(INDICES) and summary['varlen_cases_A_eq_B'] == summary['varlen_cases'] \
        and summary['other_cases_A_eq_B'] == summary['cases'] - summary['varlen_cases'], summary
    for name in ('VISION_REFERENCE_PASS.json', 'VISION_GAP.json'):
        upload(c.S3, OUT + 'provenance/' + name, (evidence / name).read_bytes())
    rounds = c.PLAN.get('decoding_rounds')
    # (shared dir, rank, world size, sampling, publish prefix) per GPU
    if rounds:
        names = sorted(rounds, key=int)
        dirs = {n: round_dir(n, rounds[n]) for n in names}
        groups = {n: [g for g in range(8) if names[g % len(names)] == n] for n in names}
        slots = {g: (dirs[n], groups[n].index(g), len(groups[n]), rounds[n], f'round-{n}/') for n in names for g in groups[n]}
    else:
        slots = {g: (c.ROOT, g, 8, c.PLAN.get('decoding_override'), '') for g in range(8)}
    c.state('rerun_generating', indices=INDICES, **({'rounds': names} if rounds else {}))
    children = []
    for gpu in range(8):
        shared, rank, world, _, _ = slots[gpu]
        genv = dict(env, CUDA_VISIBLE_DEVICES=str(gpu), TORCHINDUCTOR_CACHE_DIR=str(c.ROOT / f'cache/torch-{gpu}'),
                    TRITON_CACHE_DIR=str(c.ROOT / f'cache/triton-{gpu}'), VLLM_CACHE_ROOT=str(c.ROOT / f'cache/vllm-{gpu}'))
        log = (c.ROOT / f'caption-{gpu:02d}.log').open('ab', buffering=0)
        children.append(subprocess.Popen([str(c.RUNTIME), '-u', str(c.ROOT / 'worker.py'), '--shared', str(shared), '--model', str(model),
                                          '--step', str(STEP), '--world-size', str(world), '--rank', str(rank), '--only', ONLY],
                                         env=genv, cwd=c.ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in children]
    assert codes == [0] * 8, codes
    hashes = json.loads((c.ROOT / 'expected-prompt-hashes.json').read_text())
    files = {}
    for shared, world, sampling, prefix in sorted({(s[0], s[2], json.dumps(s[3], sort_keys=True), s[4]) for s in slots.values()}, key=lambda s: s[3]):
        rows = [json.loads(l) for r in range(world) for l in (shared / f'checkpoint-{STEP}' / f'rank-{r:02d}.jsonl').read_text().splitlines()]
        assert sorted(r['input_index'] for r in rows) == sorted(INDICES)
        assert all(r['status'] == 'inferred' and r['prompt_sha256'] == hashes[str(r['input_index'])] for r in rows)
        files.update(report_files(rows, summary, json.loads(sampling), prefix))
    c.state('rerun_publishing')
    published = []
    for key, raw in files.items():
        upload(c.S3, OUT + key, raw)
        assert digest(get(c.S3, OUT + key)) == digest(raw)
        published.append(dict(uri=OUT + key, sha256=digest(raw)))
    write(c.ROOT / 'RERUN_COMPLETE.json', dict(files=published, epoch=time.time()))
    c.state('rerun_complete_published_verified', files=[p['uri'] for p in published])


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        c.state('failed_held', error=traceback.format_exc())
        raise
