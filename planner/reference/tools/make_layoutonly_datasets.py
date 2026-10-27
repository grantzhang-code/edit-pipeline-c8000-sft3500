"""Build the out-domain datasets with layout-only Source regions (user, 2026-10-26): pe/datasets/balanced-1500-layoutonly/ and
pe/datasets/balanced-1500-qwenpe-multi-layoutonly/, the same cases / targets / instructions as balanced-1500-r6 and
balanced-1500-qwenpe-multi, with each Source image's regions taken from the layout_only arm of the 10-14 source-JSON pool
(ericdu layout-only-shortstart-c10000, output = compositional_deconstruction.elements with label / parent / bbox) instead
of the layoutfirst full-JSON arm. A case is eligible only where the full-JSON run infers it too (pe/input-exclusions.json /
regions) and every Source has a schema-valid layout_only output, so both variants compare on the same cases; other cases
are listed in input-exclusions.json. expected-prompt-hashes.json / TRAINING_PROMPT_PREFLIGHT.json come from
pe/preflight_dataset.py (training frame contract.row + editparser transform from the given training overlay).
  python3 make_layoutonly_datasets.py <layout_only source-captions.jsonl> <training overlay dir>
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PE = Path(__file__).resolve().parents[1] / 'pe'
POOL = 's3://core-cn-oss.canva.com/usr/grantzhang/balanced-1500-r6-source-json-dual-greedy-20261014/'
captions, overlay = Path(sys.argv[1]), sys.argv[2]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
complete = json.loads(subprocess.run(['aws', '--profile', 'corp_core-cn-readwrite', 's3', 'cp', POOL + 'COMPLETE.json', '-'], capture_output=True, check=True).stdout)
assert sha(captions) == next(o['sha256'] for o in complete['objects'] if o['uri'] == POOL + 'layout_only/source-captions.jsonl')

layout = {}
for line in captions.read_text().splitlines():
    r = json.loads(line)
    if not r['schema_valid'] or r['finish_reason'] == 'length':
        continue
    elements = json.loads(r['response'])['compositional_deconstruction']['elements']
    layout[r['input_metadata']['sha256']] = {'compositional_deconstruction': {'elements': [
        {k: e[k] for k in ('label', 'parent', 'bbox')} for e in elements]}}
full = json.loads((PE / 'source-regions.json').read_text())
base_plan = json.loads((PE / 'plan.json').read_text())

SETS = [
    dict(name='balanced-1500-layoutonly', src=PE, variant=base_plan['instruction_variant'], probe=base_plan['engine_probe']['cases'],
         base='balanced-1500-r6 (pe/cases.jsonl, Luna instructions)'),
    dict(name='balanced-1500-qwenpe-multi-layoutonly', src=PE / 'datasets' / 'balanced-1500-qwenpe-multi', variant='qwen-image21-pe-i2i-72927bc0',
         probe=[303, 306], base='balanced-1500-qwenpe-multi (728 multi-image cases, Qwen-Image-2.1-PE-I2I instructions)'),
]
for d in SETS:
    out = PE / 'datasets' / d['name']
    out.mkdir(parents=True, exist_ok=True)
    cases = [json.loads(l) for l in (d['src'] / 'cases.jsonl').read_text().splitlines()]
    base_excl = json.loads((d['src'] / 'input-exclusions.json').read_text())
    targets = json.loads((d['src'] / 'target-dimensions.json').read_text())
    regions, excl = {}, dict(base_excl)
    for i, c in enumerate(cases):
        keys = [x.get('source_key', x['sha256']) for x in c['inputs']]
        if str(i) in excl:
            continue
        missing = [k for k in keys if k not in full or k not in layout]
        if c['id'] not in targets or missing:
            excl[str(i)] = dict(input_index=i, case_id=c['id'], status='failed_source_annotation', missing_source_images=missing,
                                note='no schema-valid layout_only (or full-JSON) Source regions', inference_attempts=0, json_valid=None, schema_valid=None)
            continue
        for k in keys:
            regions[k] = {'regions': layout[k]}
    shutil.copy(d['src'] / 'cases.jsonl', out / 'cases.jsonl')
    shutil.copy(d['src'] / 'target-dimensions.json', out / 'target-dimensions.json')
    (out / 'source-regions.json').write_text(json.dumps(regions, ensure_ascii=False) + '\n')
    (out / 'input-exclusions.json').write_text(json.dumps(excl, indent=1) + '\n')
    eligible = len(cases) - len(excl)
    multi = sum(1 for i, c in enumerate(cases) if str(i) not in excl and c['input_count'] > 1)
    probe = [i for i in d['probe'] if str(i) not in excl]
    assert len(probe) == len(d['probe']), (d['name'], d['probe'])
    files = {n: sha(out / n) for n in ('cases.jsonl', 'source-regions.json', 'target-dimensions.json', 'input-exclusions.json')}
    meta = dict(dataset=d['name'], source_uri=base_plan['benchmark_s3'] if d['src'] == PE else
                json.loads((d['src'] / 'DATASET.json').read_text())['source_uri'],
                rows=len(cases), eligible=eligible, excluded=len(excl), single=eligible - multi, multi=multi, instruction_variant=d['variant'],
                instruction_policy=base_plan['instruction_policy'] if d['src'] == PE else json.loads((d['src'] / 'DATASET.json').read_text())['instruction_policy'],
                case_policy=f'{d["base"]}: the cases the full-JSON variant infers whose every Source has a schema-valid layout_only output; '
                            'the rest in input-exclusions.json',
                region_policy=f'layout-only Source regions: {POOL}layout_only/source-captions.jsonl (ericdu layout-only-shortstart-c10000, '
                              'task layout_only, greedy), compositional_deconstruction.elements with label / parent / bbox exactly as generated; '
                              'the full-JSON variant uses the layoutfirst arm of the same pool (pe/source-regions.json)',
                target_size_policy='pe/target-dimensions.json (balanced-1500-r6 GPT Image 2.5 target size)',
                image_policy='balanced-1500-r6 source images (inputs[].s3_uri, sha256-checked by the controller prefetch)',
                engine_probe_cases=probe, output_s3=None, files=files)
    (out / 'DATASET.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')
    # training-path prompt gate on the devbox: a scratch task root with the inference renderer and this dataset
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for n in ('sourcejson_contract.py', 'contract_utils.py', 'training_image.py', 'r3_prompt_contract.json', 'training_variants.py', 'preflight_dataset.py'):
            shutil.copy(PE / n, root / n)
        for n in files:
            shutil.copy(out / n, root / n)
        plan = dict(base_plan, dataset=d['name'], benchmark_cases_sha256=files['cases.jsonl'], source_regions_sha256=files['source-regions.json'],
                    instruction_variant=d['variant'], eligible_benchmark_cases=eligible, training_runtime_override=overlay)
        (root / 'plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n')
        r = subprocess.run([sys.executable, '-I', str(root / 'preflight_dataset.py'), str(root)], capture_output=True, text=True, cwd=root)
        print(d['name'], r.stdout[-400:], r.stderr[-1500:])
        assert r.returncode == 0
        for n in ('expected-prompt-hashes.json', 'TRAINING_PROMPT_PREFLIGHT.json'):
            shutil.copy(root / n, out / n)
    print(d['name'], 'eligible', eligible, 'excluded', len(excl), 'sources', len(regions))
