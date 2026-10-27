"""r8 editparser checkpoint PE inference on one 4-node x 8 RTX PRO 6000 group of the 64-GPU hold.

Node 0 (leader) exports the HSDP checkpoint, runs every training-alignment gate
and the 3-case pilot, then publishes LEADER_READY.json to S3. Other nodes wait
for it, restore the exported HF artifact by manifest and start their workers.
Global rank = node * 8 + gpu over plan['gpus'] shards; the leader collects all
shard receipts from S3 and publishes once.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import threading
import time
import traceback

from common import client, digest, get, sha, split, upload, write
import s5_store

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT/'plan.json').read_text())
S3 = client()
NATIVE = '/opt/mcore-venv/bin/python'
RUNTIME = Path('/mnt/ephemeral/craft-caption-accel029/runtime/bin/python')
STEP = PLAN['steps'][0]
CP = ROOT/f'checkpoint-{STEP}'
OUT = PLAN['output_s3']
WORLD = PLAN['gpus']
NODE = None


def state(phase, **kw):
    value = dict(phase=phase, node=NODE, epoch=time.time(), allocation_retained=True, **kw)
    write(ROOT/'STATE.json', value)
    upload(S3, OUT+f'control/STATE-node{NODE}.json', json.dumps(value).encode())
    print(json.dumps(value), flush=True)


def run(command, log, env):
    with (ROOT/log).open('ab', buffering=0) as stream:
        subprocess.run(command, env=env, cwd=ROOT, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def exists(uri):
    bucket, key = split(uri)
    try:
        S3.head_object(Bucket=bucket, Key=key)
        return True
    except S3.exceptions.ClientError:
        return False


def runtime():
    marker = Path('/mnt/ephemeral/craft-caption-accel029/runtime.ready')
    marker.parent.mkdir(exist_ok=True)
    with (marker.parent/'restore.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not marker.exists():
            archive = marker.parent/'runtime.tar.gz'
            s5_store.cp(PLAN['runtime_s3'], archive)
            assert sha(archive) == PLAN['runtime_sha256']
            with tarfile.open(archive) as tar:
                tar.extractall(marker.parent, filter='fully_trusted')
            assert RUNTIME.is_file()
            marker.write_text(PLAN['runtime_sha256'])
    assert marker.read_text().strip() == PLAN['runtime_sha256'] and RUNTIME.is_file()
    repo = ROOT/'source-repo'
    if not repo.exists():
        assert sha(ROOT/'source-repo.tar.gz') == PLAN['source_repo_sha256']
        repo.mkdir()
        with tarfile.open(ROOT/'source-repo.tar.gz') as tar:
            tar.extractall(repo, filter='data')
    vendor = ROOT/'fla052'
    vendor.mkdir(exist_ok=True)
    for name in ('fla', 'fla_core-0.5.2.dist-info', 'flash_linear_attention-0.5.2.dist-info'):
        shutil.copytree(Path('/opt/mcore-venv/lib/python3.12/site-packages')/name,
                        vendor/name, dirs_exist_ok=True)
    write(ROOT/'RUNTIME_READY.json', dict(status='PASS', sha256=PLAN['runtime_sha256']))


def training_runtime():
    training = Path(PLAN['training_runtime'])
    assert sha(ROOT/'training-overlay.tar.gz') == PLAN['training_overlay_sha256']
    training.mkdir(exist_ok=True)
    with tarfile.open(ROOT/'training-overlay.tar.gz') as tar:
        tar.extractall(training, filter='data')
    return training


def export():
    training = training_runtime()
    reuse = PLAN.get('reuse_export_s3')
    if reuse:
        # Same checkpoint already exported by an earlier run: restore that HF artifact (cache_model.py validates every
        # file against its published manifest) and keep its export proofs, instead of downloading the mcore checkpoint.
        CP.mkdir(exist_ok=True)
        for name in ('EXPORTED.json', 'HSDP_EXPORT_PROOF.json', 'CHECKPOINT_VERIFIED.json', 'GDN_EXPORT_ADAPTER.json'):
            (CP/name).write_bytes(get(S3, reuse + f'checkpoint-{STEP}/' + name))
        proof = json.loads((CP/'EXPORTED.json').read_text())
        assert proof['step'] == STEP and proof['source_checkpoint']['manifest_sha256'] == PLAN['models'][0]['checkpoint_sha256']
        assert proof['hsdp_export_proof']['all_parameters_verified']
        run([NATIVE, '-u', str(ROOT/'cache_model.py'), str(ROOT), str(STEP)], 'cache_model.log',
            dict(os.environ, PYTHONPATH=str(ROOT)))
        write(CP/'EXPORT_REUSED.json', dict(from_s3=reuse, hf_manifest_sha256=proof['hf_manifest_sha256']))
        for name in ('EXPORTED.json', 'HSDP_EXPORT_PROOF.json', 'CHECKPOINT_VERIFIED.json', 'GDN_EXPORT_ADAPTER.json', 'EXPORT_REUSED.json'):
            upload(S3, OUT+f'checkpoint-{STEP}/'+name, (CP/name).read_bytes())
        return
    run([NATIVE, '-u', str(ROOT/'export_model.py'), str(ROOT), str(STEP)], 'export.log',
        dict(os.environ, PE_TRAINING_ALIGNMENT='0', PYTHONPATH=str(training), CUDA_VISIBLE_DEVICES='0'))
    proof = json.loads((CP/'EXPORTED.json').read_text())
    assert proof['source_checkpoint']['manifest_sha256'] == PLAN['models'][0]['checkpoint_sha256']
    assert proof['hsdp_export_proof']['all_parameters_verified']
    for name in ('EXPORTED.json', 'HSDP_EXPORT_PROOF.json', 'CHECKPOINT_VERIFIED.json', 'GDN_EXPORT_ADAPTER.json'):
        upload(S3, OUT+f'checkpoint-{STEP}/'+name, (CP/name).read_bytes())


def engine_env():
    model = CP/'hf'
    evidence = ROOT/'precision-evidence'
    evidence.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT), PE_TRAINING_ALIGNMENT='1',
        PE_ALIGNMENT_DIR=str(evidence), PE_ALIGNMENT_MODEL=str(model),
        DLMS_HOPPER_GDN_SAFE_PATH='1', FLA_CACHE_MODE='disabled', FLA_USE_FAST_OPS='0',
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2',
        TOKENIZERS_PARALLELISM='false', VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_USE_DEEP_GEMM='0', VLLM_MOE_USE_DEEP_GEMM='0', VLLM_USE_FLASHINFER_SAMPLER='0',
        LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64', JOBSET_REPLICA_INDEX='0')
    env.pop('NCCL_NET_PLUGIN', None)
    return env, model, evidence


def leader_gates(env, model, evidence):
    for name in ('plan.json', 'TRAINING_PROMPT_PREFLIGHT.json', 'TRAINING_IDENTITY.json',
                 'sourcejson_contract.py', PLAN['prompt_contract_file'], 'training.yaml',
                 'training-args.json', 'expected-prompt-hashes.json'):
        upload(S3, OUT+'provenance/'+name, (ROOT/name).read_bytes())
    state('restoring_runtime_and_exporting_checkpoint', step=STEP)
    with ThreadPoolExecutor(2) as pool:
        for future in [pool.submit(runtime), pool.submit(export)]:
            future.result()
    state('checking_training_token_template')
    run([NATIVE, '-u', str(ROOT/'template_probe.py')], 'template_probe.log',
        dict(env, CUDA_VISIBLE_DEVICES='', PE_TRAINING_ALIGNMENT='0'))
    upload(S3, OUT+'provenance/TOKEN_TEMPLATE_PASS.json', (ROOT/'TOKEN_TEMPLATE_PASS.json').read_bytes())
    # engine_capture_probe records real engine inputs/GDN activations (runtime venv);
    # training_alignment_compare re-derives them with the training implementation on the same GPU.
    for script in ('precision_probe.py', 'vision_reference_probe.py', 'gdn_reference_probe.py', 'engine_smoke.py',
                   'engine_capture_probe.py', 'training_alignment_compare.py'):
        state('validating_'+script)
        native = script in ('vision_reference_probe.py', 'training_alignment_compare.py')
        run([NATIVE if native else str(RUNTIME), '-u', str(ROOT/script)], script+'.log',
            dict(env, CUDA_VISIBLE_DEVICES='0', PE_TRAINING_ALIGNMENT='0' if native else '1'))
    for name in ('GPU_PRECISION_PASS.json', 'VISION_REFERENCE_PASS.json', 'GDN_REFERENCE_PASS.json',
                 'ENGINE_SMOKE_PASS.json', 'TRAINING_ALIGNMENT_COMPARE.json'):
        assert json.loads((evidence/name).read_text())['status'] == 'PASS', name
        upload(S3, OUT+'provenance/'+name, (evidence/name).read_bytes())
    upload(S3, OUT+'provenance/ENGINE_CAPTURE.json', (evidence/'ENGINE_CAPTURE.json').read_bytes())
    state('training_prompt_pilot', cases=3)
    run(worker_command(model)+['--pilot'], 'pilot.log', dict(env, CUDA_VISIBLE_DEVICES='0'))
    rows = [json.loads(line) for line in (CP/'pilot.jsonl').read_text().splitlines()]
    hashes = json.loads((ROOT/'expected-prompt-hashes.json').read_text())
    assert len(rows) == 3 and any(r['status'] == 'inferred' for r in rows)
    assert all(r['status'] in ('inferred', 'exceeds_training_context') and
               r['prompt_sha256'] == hashes[str(r['input_index'])] and
               r['training_prompt_spec_sha256'] == PLAN['training_prompt_spec_sha256'] for r in rows)
    pilot = dict(status='PASS', records=3, input_indices=[r['input_index'] for r in rows],
                 prompt_hashes_verified=True, json_valid=sum(r['json_valid'] for r in rows),
                 schema_valid=sum(r['schema_valid'] for r in rows))
    write(CP/'PILOT_PASS.json', pilot)
    # Raw pilot responses survive a lost node (preemption deletes emptyDir); followers reuse them.
    for name in ('pilot.jsonl', 'pilot-attempts.jsonl', 'PILOT_DONE.json', 'PILOT_PASS.json'):
        if (CP/name).exists():
            upload(S3, OUT+f'checkpoint-{STEP}/pilot/'+name, (CP/name).read_bytes(), 'application/x-ndjson')
    exported = json.loads((CP/'EXPORTED.json').read_text())
    upload(S3, OUT+f'checkpoint-{STEP}/control/LEADER_READY.json', json.dumps(dict(
        step=STEP, hf_manifest_sha256=exported['hf_manifest_sha256'], pilot=pilot,
        gates=['TOKEN_TEMPLATE_PASS', 'GPU_PRECISION_PASS', 'VISION_REFERENCE_PASS', 'GDN_REFERENCE_PASS',
               'ENGINE_SMOKE_PASS', 'TRAINING_ALIGNMENT_COMPARE'], epoch=time.time())).encode())


def follower_setup():
    state('waiting_for_leader_gates', step=STEP)
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(runtime), pool.submit(training_runtime)]
        for future in futures:
            future.result()
    ready = OUT+f'checkpoint-{STEP}/control/LEADER_READY.json'
    started = time.time()
    while not exists(ready):
        # Only a leader failure from this attempt stops the follower (a relaunch leaves the old state behind).
        if exists(OUT+'control/STATE-node0.json'):
            leader = json.loads(get(S3, OUT+'control/STATE-node0.json'))
            if leader['phase'] == 'failed_held' and leader['epoch'] > started - 300:
                raise RuntimeError('leader failed; followers do not generate')
        time.sleep(30)
    leader = json.loads(get(S3, ready))
    CP.mkdir(exist_ok=True)
    for name in ('EXPORTED.json', 'HSDP_EXPORT_PROOF.json', 'CHECKPOINT_VERIFIED.json'):
        (CP/name).write_bytes(get(S3, OUT+f'checkpoint-{STEP}/'+name))
    for name in ('pilot.jsonl', 'PILOT_PASS.json'):
        (CP/name).write_bytes(get(S3, OUT+f'checkpoint-{STEP}/pilot/'+name))
    assert json.loads((CP/'EXPORTED.json').read_text())['hf_manifest_sha256'] == leader['hf_manifest_sha256']
    state('restoring_exported_model', hf_manifest_sha256=leader['hf_manifest_sha256'])
    run([NATIVE, '-u', str(ROOT/'cache_model.py'), str(ROOT), str(STEP)], 'cache_model.log',
        dict(os.environ, PYTHONPATH=str(ROOT)))


def prefetch_images():
    """One s5cmd batch for every benchmark source image; SHA-256 and size checked before workers use them."""
    cache, staging = ROOT/'pe-images', ROOT/'pe-images-staging'
    cache.mkdir(exist_ok=True)
    staging.mkdir(exist_ok=True)
    items = {}
    for line in (ROOT/'cases.jsonl').read_bytes().splitlines():
        for item in json.loads(line)['inputs']:
            items[item['sha256']] = item
    todo = [i for s, i in items.items() if not (cache/s).exists()]
    if todo:
        s5_store.run([f'cp --source-region us-east-1 "{i["s3_uri"]}" "{staging/i["sha256"]}"' for i in todo])
        for item in todo:
            raw = (staging/item['sha256']).read_bytes()
            assert digest(raw) == item['sha256'] and len(raw) == item['bytes'], item['s3_uri']
            (staging/item['sha256']).replace(cache/item['sha256'])
    write(ROOT/'IMAGES_PREFETCHED.json', dict(images=len(items), downloaded=len(todo), tool='s5cmd 2.3.0'))


def worker_command(model):
    return [str(RUNTIME), '-u', str(ROOT/'worker.py'), '--shared', str(ROOT),
            '--model', str(model), '--step', str(STEP), '--world-size', str(WORLD)]


def main(node):
    global NODE
    NODE = node
    # A group node may run on the main hold or on a listed substitute hold (plan['substitute_jobset_uids']).
    assert os.environ['JOBSET_UID'] in [PLAN['jobset_uid']] + PLAN.get('substitute_jobset_uids', []) and 0 <= node < PLAN['nodes']
    # Only the yielding GPU keepalive (~1.1 GiB/GPU, pauses when another process allocates) may be resident.
    used = [int(x) for x in subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).split()]
    assert len(used) == 8 and max(used) < 4096, used
    for name, digest in PLAN['execution_code_sha256'].items():
        assert sha(ROOT/name) == digest, name
    assert json.loads((ROOT/'TRAINING_PROMPT_PREFLIGHT.json').read_text())['status'] == 'PASS'
    CP.mkdir(exist_ok=True)
    state('prefetching_source_images_s5cmd')
    prefetch_images()
    env, model, evidence = engine_env()
    if node == 0:
        leader_gates(env, model, evidence)
    else:
        follower_setup()
    ranks = [node*8+gpu for gpu in range(8)]
    children = []
    for gpu, rank in enumerate(ranks):
        gpuenv = dict(env, CUDA_VISIBLE_DEVICES=str(gpu),
                      TORCHINDUCTOR_CACHE_DIR=str(ROOT/f'cache/torch-{gpu}'),
                      TRITON_CACHE_DIR=str(ROOT/f'cache/triton-{gpu}'),
                      VLLM_CACHE_ROOT=str(ROOT/f'cache/vllm-{gpu}'))
        with (ROOT/f'caption-{rank:02d}.log').open('ab', buffering=0) as log:
            proc = subprocess.Popen(worker_command(model)+['--rank', str(rank)], env=gpuenv, cwd=ROOT,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        children.append((rank, proc))
    state('generating', ranks=ranks, world_size=WORLD, worker_pids=[p.pid for _, p in children])

    def live_sync():
        # Partial shards to S3 every 5 min; finished shards are published again below.
        while any(p.poll() is None for _, p in children):
            for rank in ranks:
                for name in (f'rank-{rank:02d}.jsonl', f'attempts-{rank:02d}.jsonl'):
                    if (CP/name).exists():
                        try:
                            upload(S3, OUT+f'checkpoint-{STEP}/shards-live/'+name, (CP/name).read_bytes(), 'application/x-ndjson')
                        except Exception as error:
                            print('[live-sync] '+repr(error), flush=True)
            time.sleep(300)
    threading.Thread(target=live_sync, daemon=True).start()

    def collect(item):
        rank, proc = item
        rc = proc.wait()
        upload(S3, OUT+f'logs/caption-{rank:02d}.log', (ROOT/f'caption-{rank:02d}.log').read_bytes(), 'text/plain')
        if rc == 0:
            # DONE last: its presence on S3 means the shard it hashes is already there.
            for name in (f'rank-{rank:02d}.jsonl', f'attempts-{rank:02d}.jsonl', f'DONE-{rank:02d}.json'):
                upload(S3, OUT+f'checkpoint-{STEP}/shards/'+name, (CP/name).read_bytes())
        return rank, rc
    with ThreadPoolExecutor(8) as pool:
        codes = list(pool.map(collect, children))
    assert all(rc == 0 for _, rc in codes), codes
    if node != 0:
        state('node_shards_published', ranks=ranks)
        return
    state('waiting_for_all_shards', world_size=WORLD)
    for rank in range(WORLD):
        receipt = OUT+f'checkpoint-{STEP}/shards/DONE-{rank:02d}.json'
        while not exists(receipt):
            time.sleep(30)
        if rank not in ranks:
            for name in (f'rank-{rank:02d}.jsonl', f'attempts-{rank:02d}.jsonl', f'DONE-{rank:02d}.json'):
                (CP/name).write_bytes(get(S3, OUT+f'checkpoint-{STEP}/shards/'+name))
    write(CP/'SHARDS.json', dict(count=WORLD))
    state('verifying_and_publishing')
    from publish import publish
    result = publish(ROOT, STEP)
    assert result['inferred']+result['excluded_training_context'] == PLAN['eligible_benchmark_cases']
    assert result['prompt_alignment_verified_records'] == PLAN['eligible_benchmark_cases'] and result['worker_failed_cases'] == 0
    for item in result['files']:
        assert hashlib.sha256(get(S3, item['uri'])).hexdigest() == item['sha256']
    result.update(all_published_objects_read_back_verified=True, allocation_retained=True,
                  training_prompt_alignment_verified=True, jobset=PLAN['jobset'], world_size=WORLD,
                  gpu='NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120)')
    write(ROOT/'COMPLETE.json', result)
    upload(S3, OUT+'SUMMARY.json', json.dumps(result, indent=2).encode())
    state('complete_published_verified', inferred=result['inferred'],
          excluded_training_context=result['excluded_training_context'], captions_s3=result['captions_s3'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', type=int, required=True)
    node = parser.parse_args().node
    with (ROOT/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main(node)
        except BaseException:
            NODE = node
            state('failed_held', error=traceback.format_exc())
            raise
