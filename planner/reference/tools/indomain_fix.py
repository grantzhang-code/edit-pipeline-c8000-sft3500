"""Recover the failed / overlong in-domain planner outputs of the 9B checkpoints c10k-c16k (indomain500, multiref428).

Per case of the greedy run (planner-9b/c<N>k/in-domain/<dataset>/):
  * JSON syntax errors of an untruncated response (key-less region value, missing / trailing comma, unescaped control
    character, missing closing brackets; repair_json()) and schema violations of a parsable JSON plan (duplicate label,
    degenerate / reversed bbox, unknown parent, inline edit naming a frame without a reference_edits group, duplicated
    group) are repaired by fixed rules (repair()), then the full worker validation (schema + source references) runs again;
  * everything else (length truncation, unknown source reference_label, a plan the DiT renderer rejected or would
    reject as too long, a rule repair that still fails) is re-run with sampling, at most 3 rounds (ROUNDS), each round
    only for the cases still not accepted. Every retry output gets the same rule repair before the next round.
Accepted = JSON + schema + source references valid, not truncated, and at most MAX_TOKENS caption tokens (the DiT
renderer's text cap, estimated from tayo's ledgers: every rejected plan had > 15.8k tokens without boxes).
Retries run with pe/rerun_subset.py on workers FIX_WORKERS (default 0-3) of the Eagle hold FIX_HOLD (default 1024a), one run per checkpoint/dataset at a time
per worker. Results: planner-9b/c<N>k/in-domain/<dataset>-repair-retry3/ (patched captions, REPAIR_REPORT.json,
SOURCE.json). State in tools/indomain_fix/state.json, so the script resumes after a restart.
  python3 indomain_fix.py inventory | run | publish | reevaluate (re-judge every output after a rule change)
Other checkpoints get their own state directory: FIX_STEPS=17000 FIX_WORK=indomain_fix-c17k python3 indomain_fix.py run
FIX_DATASETS=balanced-1500,indomain500,multiref428 adds the out-domain run (planner-9b/c<N>k/balanced-1500-repair-retry3/).
FIX_YIELD_STATE=<dispatcher state file>: no new retry run starts while a checkpoint there waits for its greedy runs
(last phase planned / balanced), so the greedy inference of a new checkpoint gets the hold first.
FIX_MULTI=1 (default from c26k on): one node run per checkpoint/dataset samples all 3 rounds of every case needing a
retry at once (plan['decoding_rounds'], GPUs split over the rounds, one model restore); the rounds are then judged in
order exactly as sequential rounds would be (a case accepted in round n ignores its later rounds' outputs). A failed
multi run falls back to the sequential rounds for its cases.
"""
import collections
import copy
import fcntl
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
PE = TOOLS.parent / 'pe'
sys.path.insert(0, str(PE))
from contract_utils import plan_only_to_grouped, strict_json, validate_prediction  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_lib import LINEAGE_FILES, LINEAGES  # noqa: E402

G = 's3://core-cn-oss.canva.com/usr/grantzhang/'
P9 = os.environ.get('FIX_DEST', G + 'planner-9b/')  # results root (planner-9b-1025/ from c16k on, 2026-10-25)
TAYO = 's3://core-cn-oss.canva.com/usr/tayo/planner9b-allcohorts-scale-20261024/v1-ema26000-cfg2/versions/'
# c12000 balanced1500 was rendered in tayo's earlier fixed-EMA run (versions named 9b-r8-<run>-c<step>-...)
TAYO_FIXED = 's3://core-cn-oss.canva.com/usr/tayo/planner9b-fixed-ema-tracking-20261024/v1-ema26000-cfg2/versions/'
PFX = G + 'runbooks/editparser-r3-9b-pe-rtx8-20261023/packages'
HOLD = os.environ.get('FIX_HOLD', 'grantzhang-rtx32-r10pe-eagle-1024a')
WORKERS = [int(x) for x in os.environ.get('FIX_WORKERS', '0,1,2,3').split(',')]  # workers of HOLD this run may use
STEPS = [int(x) for x in os.environ['FIX_STEPS'].split(',')] if os.environ.get('FIX_STEPS') else range(10000, 17000, 1000)
DATASETS = tuple(os.environ.get('FIX_DATASETS', 'indomain500,multiref428').split(','))
BAL = 'balanced-1500'
YIELD = os.environ.get('FIX_YIELD_STATE')
MAX_TOKENS = 15800
ROUNDS = {  # round -> sampling (decoding_override); round 1 equals the earlier t0.7 retry of the truncated c10k-c14k cases
    1: dict(temperature=0.7, top_p=0.9, top_k=50, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.05, seed=42),
    2: dict(temperature=0.8, top_p=0.95, top_k=100, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.05, seed=1234),
    3: dict(temperature=0.6, top_p=0.9, top_k=50, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.1, seed=7),
}
TAG = {n: 'r{}-t{}p{}k{}rp{}s{}'.format(n, *(str(s[k]).replace('.', '') for k in ('temperature', 'top_p', 'top_k', 'repetition_penalty', 'seed')))
       for n, s in ROUNDS.items()}
TAG['multi'] = 'r123-multi'  # all rounds in one node run (FIX_MULTI)
MULTI = os.environ.get('FIX_MULTI', '1' if min(STEPS) >= 26000 else '0') == '1'
RETRY_DATE = os.environ.get('FIX_RETRY_DATE', '20261025')  # date tag of the retry run names (task dirs / output prefixes)
WORK = TOOLS / os.environ.get('FIX_WORK', 'indomain_fix')
STATE = WORK / 'state.json'
AWS = ['aws', '--profile', 'corp_core-cn-readwrite', 's3']
K = ['kubectl', '--context', os.environ.get('FIX_CONTEXT', 'p-iad-eagle'), '-n', 'b-core-cn-prod']  # FIX_CONTEXT: hold on USW2
ORDER = json.loads((PE / 'r3_prompt_contract.json').read_text())['SPEC']['target_elements_order']


def say(*a):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), *a, flush=True)


def get(uri):
    r = subprocess.run(AWS + ['cp', uri, '-'], capture_output=True, timeout=900)
    if r.returncode == 0:
        return r.stdout
    if b'(404)' in r.stderr or b'does not exist' in r.stderr:
        return None
    # credentials / network (e.g. the Roo tunnel down): must not read as "not published", or live runs get failed
    raise RuntimeError(f'S3 read failed for {uri}: {r.stderr.decode(errors="replace")[-300:]}')


def put(uri, raw):
    subprocess.run(AWS + ['cp', '-', uri], input=raw, check=True, timeout=900)
    assert hashlib.sha256(get(uri)).hexdigest() == hashlib.sha256(raw).hexdigest(), uri


LIN = os.environ.get('FIX_LINEAGE')  # one lineage for every step (e.g. fzsft-9b, whose steps overlap the r3 era's)


def ckdir(step):  # result directory of a checkpoint: c<N>k (step multiple of 1000) or, with FIX_LINEAGE, c<step>
    return f'c{step}' if LIN else f'c{step // 1000}k'


def lineage(step):
    if LIN:
        return LIN, LINEAGES[LIN]['run_id']
    return ('r8-9b', 'f190b7efe908') if step <= 12000 else ('r10-9b', '73f37d4788c6') if step <= 16000 else \
        ('r12-9b', '55930c837a64') if step <= 21000 else ('r16-9b', 'fb7e3ba506f6')


def run_date(step):  # date tag of the greedy runs (r12 runs started 2026-10-25)
    if LIN:
        return LINEAGES[LIN]['tag'].rsplit('_', 1)[1]
    return '20261024' if step <= 16000 else '20261025'


def original(step, ds):
    if ds == BAL:
        return export(step)
    lin, run = lineage(step)
    return G + f'{ds}-editparser-{lin}-{run}-c{step}-trainprompt-greedy-rtx8-bs64-{run_date(step)}/'


def export(step):
    lin, run = lineage(step)
    if os.environ.get('FIX_EXPORT_DATASET'):  # the run that exported the checkpoint, e.g. balanced-1500-layoutonly (fzsft c4000 on)
        return G + f'{os.environ["FIX_EXPORT_DATASET"]}-editparser-{lin}-{run}-c{step}-trainprompt-greedy-rtx8-bs64-{run_date(step)}/'
    return G + f'balanced-1500-r6-editparser-{lin}-{run}-c{step}-trainprompt-greedy-rtx32-bs64-{run_date(step)}/'


def retry_out(step, ds, n):
    lin, run = lineage(step)
    return G + f'{ds}-editparser-{lin}-{run}-c{step}-trainprompt-retry-{TAG[n]}-rtx8-{RETRY_DATE}/'


# ---- validation (same checks and order as pe/worker.py) ----
def _worker_function(name):  # pe/worker.py imports the engine stack; compile just this function from its source
    import ast
    tree = ast.parse((PE / 'worker.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(PE / 'worker.py'), 'exec'), ns)
    return ns[name]


validate_references = _worker_function('validate_references')


def check(plan, count, regions):
    """-> (json_valid, schema_valid, refs_valid, error, grouped) for a parsed plan-only prediction."""
    try:
        grouped = plan_only_to_grouped(plan, count)
        validate_prediction(grouped, 'sourcejson', count)
        assert list(grouped['elements']) == ORDER, 'Target elements order differs from training'
    except Exception as e:
        return True, False, False, str(e) or type(e).__name__, None
    try:
        validate_references(grouped, regions)
    except Exception as e:
        return True, True, False, str(e) or type(e).__name__, grouped
    return True, True, True, None, grouped


_TOK = None


def ntok(plan):
    """Caption tokens as the DiT text encoder sees them (boxes are rendered spatially, not as text)."""
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file('/home/coder/work/zhangguiwei/Qwen/Qwen3.5-4B/tokenizer.json')
    c = copy.deepcopy(plan)
    for e in c['elements'].get('layout') or []:
        e.pop('bbox', None)
    for x in c['elements'].get('reference_edits') or []:
        for r in x.get('remove') or []:
            r.pop('reference_bbox', None)
    return len(_TOK.encode(json.dumps(c, ensure_ascii=False)).ids)


# ---- rule repair of schema violations ----
def fix_box(box):
    if not (isinstance(box, list) and len(box) == 4 and all(type(x) is int for x in box)):
        return box, False
    x1, y1, x2, y2 = (min(max(v, 0), 1000) for v in box)
    x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
    if x1 == x2:
        x1, x2 = (x1, x2 + 1) if x2 < 1000 else (x1 - 1, x2)
    if y1 == y2:
        y1, y2 = (y1, y2 + 1) if y2 < 1000 else (y1 - 1, y2)
    new = [x1, y1, x2, y2]
    return new, new != box


def repair(plan, count):
    """Fixed-rule repair of a parsable plan-only prediction. -> (repaired plan, rules applied); never touches text."""
    p = copy.deepcopy(plan)
    rules = []
    el = p.get('elements') if isinstance(p, dict) else None
    if not isinstance(el, dict) or not isinstance(el.get('layout'), list) or not isinstance(el.get('reference_edits'), list):
        return p, rules
    layout = el['layout']
    for r in layout:  # degenerate / reversed / out-of-range boxes -> ordered, clipped, at least 1 unit wide
        if isinstance(r, dict) and 'bbox' in r:
            r['bbox'], changed = fix_box(r['bbox'])
            if changed:
                rules.append(f'bbox:{r.get("label")}')
    for g in el['reference_edits']:
        for rm in (g.get('remove') or []) if isinstance(g, dict) else []:
            if isinstance(rm, dict) and isinstance(rm.get('reference_bbox'), list) and len(rm['reference_bbox']) == 4:
                rm['reference_bbox'], changed = fix_box(rm['reference_bbox'])
                if changed:
                    rules.append(f'reference_bbox:{g.get("reference_id")}')
    kept, seen = [], {}
    for r in layout:  # duplicate labels: drop an exact copy, otherwise give the later region the next free index
        label = r.get('label') if isinstance(r, dict) else None
        if label in seen:
            if r == seen[label]:
                rules.append(f'drop_duplicate_region:{label}'); continue
            m = re.fullmatch(r'<(.+?) (\d+)>', label)
            stem = m.group(1) if m else label.strip('<>')
            used = {int(x.group(1)) for x in (re.fullmatch(rf'<{re.escape(stem)} (\d+)>', str(q.get('label', ''))) for q in layout if isinstance(q, dict)) if x}
            new = f'<{stem} {max(used | {1}) + 1}>'
            rules.append(f'rename_duplicate_label:{label}->{new}')
            r['label'] = label = new
        seen[label] = r
        kept.append(r)
    el['layout'] = layout = kept
    labels = {r.get('label') for r in layout if isinstance(r, dict)}
    for r in layout:  # parent naming no region of the plan -> ROOT
        if isinstance(r, dict) and r.get('parent') not in (None, 'ROOT') and r.get('parent') not in labels:
            rules.append(f'unknown_parent_to_root:{r.get("label")}<-{r["parent"]}'); r['parent'] = 'ROOT'
    groups = el['reference_edits']
    merged, index = [], {}
    for g in groups:  # duplicated reference_edits group -> one group, removals concatenated
        rid = g.get('reference_id') if isinstance(g, dict) else None
        if rid in index and isinstance(g.get('remove'), list):
            index[rid]['remove'] += g['remove']; rules.append(f'merge_duplicate_group:{rid}'); continue
        if rid is not None:
            index[rid] = g
        merged.append(g)
    frames = {f'frame_{i}' for i in range(count)}
    named = [a.get('reference_id') for r in layout if isinstance(r, dict) for a in (r.get('edit') or []) if isinstance(a, dict)]
    for rid in sorted({x for x in named if x in frames and x not in index}, key=lambda x: int(x.split('_')[1])):
        merged.append(dict(reference_id=rid, remove=[])); index[rid] = merged[-1]; rules.append(f'add_empty_group:{rid}')
    if any(r.startswith('add_empty_group') for r in rules):
        merged.sort(key=lambda g: int(str(g.get('reference_id', 'frame_999')).split('_')[1]) if str(g.get('reference_id', '')).startswith('frame_') else 999)
    el['reference_edits'] = merged
    return p, rules


# ---- rule repair of JSON syntax errors (an untruncated response that strict_json() rejects) ----
LAYOUT_TEXT_KEYS = ('type', 'label', 'parent', 'desc')  # required string fields of a layout region (validate_prediction)


def _scan(text, end):
    """Open brackets before end (string-aware): [(char, index)], or None if a closer does not match."""
    stack, i = [], 0
    while i < end:
        ch = text[i]
        if ch == '"':
            i += 1
            while i < end and text[i] != '"':
                i += 2 if text[i] == '\\' else 1
        elif ch in '{[':
            stack.append((ch, i))
        elif ch in '}]':
            if not stack or {'{': '}', '[': ']'}[stack.pop()[0]] != ch:
                return None
        i += 1
    return stack


def repair_json(text):
    """Deterministic fixes of JSON syntax, each followed by strict_json() again: -> (value or None, rules).
    Rules: a layout region value written without its key when exactly one required text key is missing; a missing comma
    between two values; a trailing comma; unescaped control characters inside strings; closing brackets missing at the end."""
    body = re.sub(r'^\s*<think>.*?</think>\s*', '', text, count=1, flags=re.S)
    body = re.sub(r'\s*```\s*$', '', re.sub(r'^\s*```(?:json)?\s*', '', body, count=1, flags=re.I), count=1)
    rules = []
    for _ in range(20):
        try:
            return strict_json(body), rules
        except json.JSONDecodeError as e:
            err, pos = e.msg, e.pos
        except ValueError:
            return None, rules
        if err == "Expecting ':' delimiter":  # ...,"<value>"} : a key-less string inside a layout region
            k1 = body.rfind('"', 0, body.rfind('"', 0, pos))  # opening quote of the string just parsed as a key
            stack = _scan(body, k1) if k1 > 0 else None
            if stack and stack[-1][0] == '{' and body[:k1].rstrip().endswith(','):
                keys = set(re.findall(r'"([a-z_]+)"\s*:', body[stack[-1][1]:k1]))
                missing = [k for k in LAYOUT_TEXT_KEYS if k not in keys]
                if 'label' in keys and len(missing) == 1:
                    body = body[:k1] + f'"{missing[0]}":' + body[k1:]
                    rules.append(f'json_missing_key:{missing[0]}'); continue
            return None, rules
        if err == "Expecting ',' delimiter" and pos < len(body) and body[pos] in '"{[-0123456789tfn':
            body = body[:pos] + ',' + body[pos:]; rules.append('json_missing_comma'); continue
        if err.startswith('Illegal trailing comma') and body[pos] == ',':  # Python >= 3.13 names it
            body = body[:pos] + body[pos + 1:]; rules.append('json_trailing_comma'); continue
        if err in ('Expecting property name enclosed in double quotes', 'Expecting value') and pos < len(body) and body[pos] in '}]':
            q = body.rfind(',', 0, pos)
            if q >= 0 and not body[q + 1:pos].strip():
                body = body[:q] + body[q + 1:]; rules.append('json_trailing_comma'); continue
        if err.startswith('Invalid control character'):
            ch = body[pos]
            body = body[:pos] + {'\n': '\\n', '\t': '\\t', '\r': '\\r'}.get(ch, '\\u%04x' % ord(ch)) + body[pos + 1:]
            rules.append('json_escape_control_char'); continue
        if pos >= len(body.rstrip()):
            stack = _scan(body, len(body))
            if stack:
                body = body.rstrip().rstrip(',') + ''.join({'{': '}', '[': ']'}[c] for c, _ in reversed(stack))
                rules.append(f'json_close_brackets:{len(stack)}'); continue
        return None, rules
    return None, rules


# ---- inventory ----
def load_run(step, ds):
    d = WORK / 'runs' / f'{ckdir(step)}-{ds}'
    d.mkdir(parents=True, exist_ok=True)
    for name in ('responses.jsonl.gz', 'captions.jsonl'):
        if not (d / name).exists():
            (d / name).write_bytes(get(original(step, ds) + f'checkpoint-{step}/{name}'))  # the files planner-9b/c<N>k/in-domain/<ds>/ copies
    responses = {r['input_index']: r for r in map(json.loads, gzip.open(d / 'responses.jsonl.gz'))}
    return responses


def ledger(step, ds):
    lin, run = lineage(step)
    d = WORK / 'ledgers'; d.mkdir(parents=True, exist_ok=True)
    f = d / f'{ckdir(step)}-{ds}.json'
    # tayo names versions <cohort>-9b-[r8-]<run>-c<step>-<hash>/, cohort balanced1500 for balanced-1500
    cohort = 'balanced1500' if ds == BAL else ds
    for root, pre in ((TAYO, f'{cohort}-9b-'), *(((TAYO_FIXED, '9b-'),) if ds == BAL else ())):
        if f.exists():
            break
        for n in subprocess.run(AWS + ['ls', root], capture_output=True, text=True).stdout.split():
            if n.startswith(pre) and f'{run}-c{step}-' in n:
                raw = get(root + n + 'ledger.json')
                if raw:
                    f.write_bytes(raw)
    return {r['input_index']: r for r in json.loads(f.read_text())} if f.exists() else {}


def accept(info):
    return info['json_valid'] and info['schema_valid'] and info['refs_valid'] and info['finish_reason'] != 'length' and info['tokens'] <= MAX_TOKENS


def evaluate(row):
    """Validity of a worker row (greedy or retry), with rule repair of a schema-invalid parsable plan."""
    info = dict(json_valid=bool(row['json_valid']), schema_valid=bool(row['schema_valid']), refs_valid=bool(row['source_references_valid']),
                finish_reason=row['finish_reason'], output_tokens=row.get('output_tokens'), error=row.get('validation_error'),
                rules=[], prediction=row.get('prediction'), grouped=row.get('prediction_grouped'))
    if not info['json_valid'] and row['finish_reason'] != 'length' and row.get('response'):
        value, jrules = repair_json(row['response'])
        info.update(repair_attempted=jrules)
        if value is not None:  # parsable now: the schema rules and the worker checks, as for a parsable output
            fixed, rules = repair(value, row['input_count'])
            j, s, r, err, grouped = check(fixed, row['input_count'], row['source_regions'])
            info.update(repair_attempted=jrules + rules, repair_error=err)
            if s:
                info.update(json_valid=True, schema_valid=True, refs_valid=r, rules=jrules + rules, prediction=fixed, grouped=grouped, error=err)
    elif info['json_valid'] and row['finish_reason'] != 'length' and not info['schema_valid']:
        fixed, rules = repair(row['prediction'], row['input_count'])
        j, s, r, err, grouped = check(fixed, row['input_count'], row['source_regions'])
        info.update(repair_attempted=rules, repair_error=err)
        if s:
            info.update(schema_valid=True, refs_valid=r, rules=rules, prediction=fixed, grouped=grouped, error=err)
    info['tokens'] = ntok(info['prediction']) if info['schema_valid'] else None
    if info['tokens'] is None:
        info['tokens'] = 10 ** 9
    return info


def inventory():
    st = dict(cases={}, rounds={}, created=time.time())
    status = json.loads(get(TAYO.rsplit('versions/', 1)[0] + 'STATUS.json'))
    rendered = {c['planner_step'] for c in status['checkpoints'] if c['complete']} if not LIN else set()  # DiT already ran every eligible case
    st['tayo_rendered_steps'] = sorted(rendered)
    for step in STEPS:
        for ds in DATASETS:
            responses, led = load_run(step, ds), ledger(step, ds)
            key = f'{step}:{ds}'
            for i, row in responses.items():
                if row.get('status') != 'inferred':
                    continue  # input-level exclusion (multiref sources without annotation): nothing to infer
                info = evaluate(row)
                native = (led.get(i) or {}).get('stage') == 'native'
                if row['finish_reason'] == 'length':
                    cat = 'truncated'
                elif not row['json_valid']:
                    cat = 'json_syntax'
                elif not row['schema_valid']:
                    cat = 'schema'
                elif not row['source_references_valid']:
                    cat = 'refs'
                elif native:
                    cat = 'dit_rejected'
                elif info['tokens'] > MAX_TOKENS and step not in rendered:
                    cat = 'overlong'
                else:
                    continue
                ok = accept(info) and not native
                st['cases'].setdefault(key, {})[str(i)] = dict(category=cat, dit_rejected=native, greedy_error=row.get('validation_error'),
                    greedy_tokens=info['tokens'] if info['tokens'] < 10 ** 9 else None, greedy_response_sha256=row['response_sha256'],
                    attempts=[dict(kind='rule_repair', rules=info['rules'] or info.get('repair_attempted'), accepted=ok,
                                   error=info.get('repair_error'), tokens=info['tokens'] if info['tokens'] < 10 ** 9 else None)] if cat in ('schema', 'json_syntax') else [],
                    accepted=dict(method='rule_repair', rules=info['rules']) if ok else None)
    save(st)
    return st


def save(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix('.tmp'); tmp.write_text(json.dumps(st, indent=1)); tmp.replace(STATE)


def summary(st):
    c = collections.Counter()
    for key, cases in st['cases'].items():
        for i, x in cases.items():
            c[(x['category'], x['accepted']['method'] if x['accepted'] else 'pending')] += 1
    return dict(sorted((f'{a}/{b}', n) for (a, b), n in c.items()))


# ---- retry rounds ----
def kexec(pod, cmd, timeout=120):
    r = subprocess.run(K + ['exec', pod, '-c', 'hold', '--', '/bin/bash', '-c', cmd], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    return r.returncode, r.stdout + r.stderr


def pods():
    out = subprocess.run(K + ['get', 'pods', '-o', 'name'], capture_output=True, text=True, timeout=120).stdout.split()
    return {w: next((p.split('/')[1] for p in out if f'{HOLD}-workers-{w}-0-' in p), None) for w in WORKERS}


def waiting_greedy():
    """A checkpoint of the dispatcher state still waits for (or runs) its balanced greedy run: leave the hold to it."""
    if not YIELD or not Path(YIELD).exists():
        return False
    last = {}
    for line in Path(YIELD).read_text().split('\n'):
        if line.strip():
            k, v = line.split()[:2]; last[k] = v
    # only checkpoints whose greedy runs use this hold (r12_9b_slots.txt "<step> <hold> <first worker>"; no line = HOLD)
    slots = Path(YIELD.replace('_state.txt', '_slots.txt'))  # r12_9b_state.txt -> r12_9b_slots.txt (r16_9b_... likewise)
    # and whose 4 greedy workers overlap this run's workers
    slot = {l.split()[0]: (l.split()[1], int(l.split()[2])) for l in (slots.read_text().split('\n') if slots.exists() else []) if l.strip()}
    default = ('grantzhang-rtx32-r10pe-eagle-1024a', 0)
    return any(v in ('planned', 'balanced', 'datasets') and slot.get(k, default)[0] == HOLD and set(range(slot.get(k, default)[1], slot.get(k, default)[1] + 4)) & set(WORKERS)
               for k, v in last.items())


def busy(pod):
    code, out = kexec(pod, 'pgrep -fc "[c]ontroller.py --node|[r]erun_subset.py|[n]ode_rerun.sh" || true')
    return code != 0 or out.strip() not in ('0', '')


def make_plan(step, ds, n, uid):
    lin, _ = lineage(step)
    base = json.loads((PE / 'plans' / f'plan-{step}-{"balanced" if ds == BAL else ds}-{lin}.json').read_text())
    tag = f'rtx8_retry_{TAG[n].replace("-", "_")}_{RETRY_DATE}'
    old = base['task_id']
    task = old.replace(f'_rtx8_bs64_{run_date(step)}_', f'_{tag}_') if ds != BAL else old.replace(f'_rtx32_bs64_{run_date(step)}', f'_{tag}_{BAL}')
    assert task != old
    sampling = dict(decoding_rounds={str(k): s for k, s in ROUNDS.items()}) if n == 'multi' else dict(decoding_override=ROUNDS[n])
    plan = dict(base, task_id=task, runtime_task_id=task, shared=f'/mnt/ephemeral/{task}', training_runtime=f'/mnt/ephemeral/{task}/training-runtime',
                jobset=HOLD, jobset_uid=uid, hold_nodes=[0], nodes=1, gpus=8, output_s3=retry_out(step, ds, n), **sampling,
                retry_of=dict(original=original(step, ds), export=export(step), round=n, policy='indomain_fix.py'),
                execution_code_sha256={k: hashlib.sha256(((PE / 'lineage' / lin / k) if k in LINEAGE_FILES else (PE / k)).read_bytes()).hexdigest()
                                           for k in base['execution_code_sha256']})
    name = f'plan-{step}-{ds}-{lin}-retry-{TAG[n]}'
    (PE / 'plans' / f'{name}.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n')
    return name


def package(n, names):
    with open(TOOLS / '.fix_git.lock', 'w') as lk:  # several fix runs commit to the same repo
        fcntl.flock(lk, fcntl.LOCK_EX)
        return _package(n, names)


def _package(n, names):
    root = TOOLS.parent
    git = ['git', '-C', str(root), '-c', 'core.fsmonitor=false']
    subprocess.run(git + ['add'] + [f'pe/plans/{x}.json' for x in names] + ['pe/worker.py'], check=True)
    if subprocess.run(git + ['diff', '--cached', '--quiet']).returncode:
        subprocess.run(git + ['commit', '-q', '-m', f'in-domain fix round {n}: retry plans ({TAG[n]}) for {",".join(f"c{x // 1000}k" for x in STEPS)}, Eagle hold {HOLD}\n\n'
                              'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'], check=True)
    commit = subprocess.run(git + ['rev-parse', '--short', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    tar = WORK / f'pe-{commit}.tar.gz'
    subprocess.run(git + ['archive', '--format=tar.gz', '--prefix=pe/', 'HEAD:pe', '-o', str(tar)], check=True)
    listing = subprocess.run(['tar', 'tzf', str(tar)], capture_output=True, text=True, check=True).stdout.split()
    assert all(f'pe/plans/{x}.json' in listing for x in names) and 'pe/rerun_subset.py' in listing
    digest = hashlib.sha256(tar.read_bytes()).hexdigest()
    uri = f'{PFX}/pe-{commit}-{digest}.tar.gz'
    subprocess.run(AWS + ['cp', str(tar), uri, '--only-show-errors'], check=True)
    assert hashlib.sha256(get(uri)).hexdigest() == digest
    return uri, digest


def launch(pod, job, pkg, digest, uid):
    s5 = str(PE / 'bin' / 's5cmd')
    subprocess.run(K + ['cp', s5, f'{pod}:/mnt/ephemeral/s5cmd', '-c', 'hold'], check=True, timeout=120)
    # fetch in the foreground; only the detached node_rerun.sh goes to the background (a backgrounded && list keeps
    # kubectl exec's stream open)
    cmd = (f'/mnt/ephemeral/s5cmd cp --source-region us-east-1 {PFX}/node_rerun.sh /mnt/ephemeral/node_rerun.sh >/dev/null && chmod +x /mnt/ephemeral/node_rerun.sh || exit 1; '
           f'setsid nohup /mnt/ephemeral/node_rerun.sh {job["step"]} {pkg} {digest} {HOLD} {uid} {job["plan"]} {original(job["step"], job["ds"])} '
           f'{",".join(map(str, job["indices"]))} {export(job["step"])} > /mnt/ephemeral/node_rerun_{job["plan"]}.log 2>&1 < /dev/null & sleep 2; echo started')
    try:
        code, out = kexec(pod, cmd)
    except subprocess.TimeoutExpired:
        out = ''
    return 'started' in out or running_on(pod, job)


def running_on(pod, job):
    try:
        return kexec(pod, f'pgrep -f "[n]ode_rerun.sh .*{job["plan"]}|[r]erun_subset.py {original(job["step"], job["ds"])}" >/dev/null')[0] == 0
    except subprocess.TimeoutExpired:
        return False


def phase(job):
    raw = get(retry_out(job['step'], job['ds'], job['round']) + 'control/STATE-node0.json')
    return json.loads(raw).get('phase') if raw else None


def collect(st, job):
    n, step, ds = job['round'], job['step'], job['ds']
    if n == 'multi':  # rounds in order; a case accepted in an earlier round ignores the later outputs. Failed run: nothing
        # recorded, the sequential rounds take its cases
        for k in ROUNDS:
            src = retry_out(step, ds, n) + f'round-{k}/checkpoint-{step}/rerun.jsonl'
            raw = get(src) if job['status'] == 'complete' else None
            cases = st['cases'][f'{step}:{ds}']
            rows = [r for r in map(json.loads, raw.decode().splitlines()) if not cases[str(r['input_index'])]['accepted']] if raw else []
            _apply(st, step, ds, k, rows, src)
        return
    src = retry_out(step, ds, n) + f'checkpoint-{step}/rerun.jsonl'
    raw = get(src)
    rows = [json.loads(l) for l in raw.decode().splitlines()] if raw else []
    _apply(st, step, ds, n, rows, src)
    cases = st['cases'][f'{step}:{ds}']
    missing = set(map(str, job['indices'])) - {str(r['input_index']) for r in rows}
    for i in missing:
        cases[i]['attempts'].append(dict(kind='retry', round=n, sampling=ROUNDS[n], accepted=False, error=f'retry run {job["status"]}'))


def _apply(st, step, ds, n, rows, source):
    cases = st['cases'][f'{step}:{ds}']
    for row in rows:
        x = cases[str(row['input_index'])]
        info = evaluate(row)
        changed = row['response_sha256'] != x['greedy_response_sha256']
        ok = accept(info) and (changed or not x['dit_rejected'])
        x['attempts'].append(dict(kind='retry', round=n, sampling=ROUNDS[n], accepted=ok, finish_reason=row['finish_reason'],
                                  output_tokens=row.get('output_tokens'), tokens=info['tokens'] if info['tokens'] < 10 ** 9 else None,
                                  json_valid=info['json_valid'], schema_valid=info['schema_valid'], refs_valid=info['refs_valid'],
                                  error=info['error'], rules=info['rules'], source=source))
        if ok and not x['accepted']:
            x['accepted'] = dict(method=f'retry_r{n}' + ('+rule_repair' if info['rules'] else ''), round=n, rules=info['rules'])


def reuse_round1(st):
    """The earlier t0.7 retry (same sampling, seed 42) of the truncated c10k-c14k cases counts as their round 1."""
    for j in json.loads((TOOLS / 'retry_jobs.json').read_text()):
        cases = st['cases'].get(f'{j["step"]}:{j["ds"]}', {})
        if not cases:
            continue
        raw = get(j['out'] + f'checkpoint-{j["step"]}/rerun.jsonl')
        for row in map(json.loads, raw.decode().splitlines()):
            x = cases.get(str(row['input_index']))
            if x is None or any(a.get('round') == 1 for a in x['attempts']):
                continue
            info = evaluate(row)
            ok = accept(info) and (row['response_sha256'] != x['greedy_response_sha256'] or not x['dit_rejected'])
            x['attempts'].append(dict(kind='retry', round=1, sampling=ROUNDS[1], accepted=ok, finish_reason=row['finish_reason'],
                                      output_tokens=row.get('output_tokens'), tokens=info['tokens'] if info['tokens'] < 10 ** 9 else None,
                                      json_valid=info['json_valid'], schema_valid=info['schema_valid'], refs_valid=info['refs_valid'],
                                      error=info['error'], rules=info['rules'], source=j['out'] + f'checkpoint-{j["step"]}/rerun.jsonl', reused=True))
            if ok and not x['accepted']:
                x['accepted'] = dict(method='retry_r1' + ('+rule_repair' if info['rules'] else ''), round=1, rules=info['rules'])


def run():
    st = json.loads(STATE.read_text()) if STATE.exists() else inventory()
    if not st.get('round1_reused'):
        reuse_round1(st); st['round1_reused'] = True; save(st)
    say('start', summary(st))
    uid = subprocess.run(K + ['get', 'jobset', HOLD, '-o', 'jsonpath={.metadata.uid}'], capture_output=True, text=True, check=True).stdout.strip()
    assert uid
    for n in (['multi'] if MULTI else []) + list(ROUNDS):
        rs = st['rounds'].setdefault(str(n), {})
        if rs.get('done'):
            continue
        if 'jobs' not in rs:
            jobs = []
            for key, cases in st['cases'].items():
                step, ds = int(key.split(':')[0]), key.split(':')[1]
                idx = sorted(int(i) for i, x in cases.items() if not x['accepted'] and not any(a.get('round') == n or (n == 'multi' and a['kind'] == 'retry')
                                                                                                for a in x['attempts']))
                if idx:
                    jobs.append(dict(round=n, step=step, ds=ds, indices=idx, plan=make_plan(step, ds, n, uid), status='queued'))
            rs['jobs'] = jobs
            if jobs:
                with open(TOOLS / '.fix_launch.lock', 'w') as lk:  # git commit of the plans: one fix run at a time
                    fcntl.flock(lk, fcntl.LOCK_EX)
                    rs['package'], rs['package_sha256'] = package(n, [j['plan'] for j in jobs])
            save(st)
            say(f'round {n}: {len(jobs)} runs, {sum(len(j["indices"]) for j in jobs)} cases')
        jobs = rs['jobs']
        adopt(n, jobs); save(st)
        while any(j['status'] in ('queued', 'running') for j in jobs):
            try:
                step_round(st, n, rs, jobs, uid)
            except Exception as e:  # kubectl / S3 hiccup: log and try again next tick
                say(f'round {n}: tick failed: {type(e).__name__}: {str(e)[:300]}')
            time.sleep(120)
        rs['done'] = True; save(st)
        say(f'round {n} done', summary(st))
    publish(st)


def adopt(n, jobs):
    """After a restart: a queued job whose launch went through (run alive on some worker, or its state published) is running."""
    ps = pods()
    for j in jobs:
        if j['status'] != 'queued':
            continue
        w = next((w for w, pod in ps.items() if pod and running_on(pod, j)), None)
        if w is not None:
            j.update(status='running', worker=w, pod=ps[w], started=time.time()); say(f'round {n}: adopted c{j["step"]} {j["ds"]} on worker {w}')
        elif phase(j) is not None:
            j.update(status='running', worker=None, pod=None, started=time.time() - 3600)  # started earlier, not alive: next tick fails it


def step_round(st, n, rs, jobs, uid):
    ps = pods()
    for j in jobs:
        if j['status'] != 'running':
            continue
        ph = phase(j)
        alive = ps.get(j['worker']) and running_on(ps[j['worker']], j)
        if not alive and ph != 'rerun_complete_published_verified':
            ph = phase(j)  # the run may have published and exited between the two reads
        if ph == 'rerun_complete_published_verified':
            j['status'] = 'complete'
        elif ph == 'failed_held' or (not alive and time.time() - j['started'] > 600) or time.time() - j['started'] > 4 * 3600:
            j['status'] = f'failed ({ph})'
        if j['status'] != 'running':
            collect(st, j); say(f'round {n} c{j["step"]} {j["ds"]}: {j["status"]}', summary(st)); save(st)
    used = {j['worker'] for j in jobs if j['status'] == 'running'}
    if waiting_greedy():
        return
    with open(TOOLS / '.fix_launch.lock', 'w') as lk:  # several fix runs share the hold: busy check + launch one at a time
        fcntl.flock(lk, fcntl.LOCK_EX)
        for w, pod in ps.items():
            queued = [j for j in jobs if j['status'] == 'queued']
            if not queued or w in used or not pod or busy(pod):
                continue
            j = queued[0]
            if launch(pod, j, rs['package'], rs['package_sha256'], uid):
                j.update(status='running', worker=w, pod=pod, started=time.time()); used.add(w)
                say(f'round {n}: launched c{j["step"]} {j["ds"]} ({len(j["indices"])} cases) on worker {w}')
                save(st)


# ---- publish ----
def publish(st=None):
    st = st or json.loads(STATE.read_text())
    index = []
    for key, cases in st['cases'].items():
        step, ds = int(key.split(':')[0]), key.split(':')[1]
        d = WORK / 'runs' / f'{ckdir(step)}-{ds}'
        responses = {r['input_index']: r for r in map(json.loads, gzip.open(d / 'responses.jsonl.gz'))}
        retried = collections.defaultdict(dict)
        for x in cases.values():
            for a in x['attempts']:
                if a.get('source') and a['source'] not in retried:
                    raw = get(a['source'])
                    retried[a['source']] = {r['input_index']: r for r in map(json.loads, raw.decode().splitlines())} if raw else {}
        out, report = [], []
        for line in (d / 'captions.jsonl').read_text().splitlines():
            row = json.loads(line)
            x = cases.get(str(row['input_index']))
            if x and x['accepted']:
                acc = x['accepted']
                if acc['method'] == 'rule_repair':
                    src = responses[row['input_index']]; sampling = None; origin = 'greedy'
                else:
                    a = next(a for a in x['attempts'] if a.get('round') == acc['round'] and a['accepted'])
                    src = retried[a['source']][row['input_index']]; sampling = ROUNDS[acc['round']]; origin = a['source']
                info = evaluate(src)
                assert accept(info)
                row.update(caption=info['prediction'], caption_grouped=info['grouped'], json_valid=True, schema_valid=True,
                           source_references_valid=True, repair_method=acc['method'], repair_rules=info['rules'],
                           rerun_sampling=sampling, replaced_from=origin)
            out.append(json.dumps(row, ensure_ascii=False))
        for i, x in sorted(cases.items(), key=lambda kv: int(kv[0])):
            report.append(dict(input_index=int(i), **x))
        rows = [json.loads(l) for l in out]
        sub = f'{ckdir(step)}/' + (f'{ds}-repair-retry3/' if ds.startswith(BAL) else f'in-domain/{ds}-repair-retry3/')  # balanced-1500(-qwenpe-multi): out-domain
        lin, run = lineage(step)
        counts = collections.Counter(x['accepted']['method'].split('+')[0] if x['accepted'] else 'unrecovered' for x in cases.values())
        meta = dict(step=step, dataset=ds, variant=f'{ds}-repair-retry3', run_id=run, lineage=lin, base_run=original(step, ds),
                    base_decoding='greedy (temperature 0, top_p 1, top_k -1, repetition_penalty 1, seed 42)',
                    policy=('greedy output, then every retry output: fixed-rule repair of JSON syntax of an untruncated response (key-less region '
                            'value, missing / trailing comma, unescaped control character, missing closing brackets) and of schema violations of a '
                            'parsable plan (bbox order/degenerate/clip, duplicate label, unknown parent -> ROOT, missing / duplicated reference_edits '
                            'group), then full worker validation; cases still failing: next sampled retry round, 3 rounds in total'),
                    acceptance=f'json + schema + source references valid, finish_reason != length, caption tokens without boxes <= {MAX_TOKENS} '
                               '(DiT text cap estimated from tayo ledgers); a case the DiT rejected needs a changed response',
                    retry_sampling={f'round{n}': s for n, s in ROUNDS.items()},
                    retry_mode='rounds 1-3 sampled concurrently in one node run per dataset (FIX_MULTI); judged in round order, first accepted wins'
                               if any('/round-' in a.get('source', '') for x in cases.values() for a in x['attempts']) else 'one node run per round',
                    retry_runs={f'round{n}': sorted({a['source'].rsplit('/checkpoint-', 1)[0] + '/' for x in cases.values() for a in x['attempts']
                                                     if a.get('round') == n and a.get('source')}) for n in ROUNDS},
                    cases_considered=len(cases), outcome=dict(counts), categories=dict(collections.Counter(x['category'] for x in cases.values())),
                    inferred=len(rows), json_valid=sum(bool(r['json_valid']) for r in rows), schema_valid=sum(bool(r['schema_valid']) for r in rows),
                    source_references_valid=sum(bool(r['source_references_valid']) for r in rows),
                    note='unrecovered cases keep the greedy output; every other row is the unchanged greedy output of base_run; '
                         'replaced rows carry repair_method / repair_rules / rerun_sampling / replaced_from')
        put(P9 + sub + 'captions.jsonl', ('\n'.join(out) + '\n').encode())
        put(P9 + sub + 'REPAIR_REPORT.json', json.dumps(report, indent=1, ensure_ascii=False).encode())
        put(P9 + sub + 'SOURCE.json', json.dumps(meta, indent=1).encode())
        index.append(dict(path=sub, root=P9, **{k: meta[k] for k in ('step', 'dataset', 'variant', 'run_id', 'lineage', 'base_run', 'cases_considered', 'outcome', 'inferred', 'json_valid', 'schema_valid', 'source_references_valid')}))
        say(sub, dict(counts), 'refs', meta['source_references_valid'])
    old = json.loads((WORK / 'published.json').read_text()) if (WORK / 'published.json').exists() else []
    index = [x for x in old if x.get('root', G + 'planner-9b/') != P9] + index  # entries published under another root stay listed
    (WORK / 'published.json').write_text(json.dumps(index, indent=1))


def reevaluate():
    """After a rule change: re-judge the greedy output and every finished retry output of each case with the current
    evaluate(); requeue jobs of unfinished rounds whose run was stopped or failed (their partial output prefix and failure
    attempts are removed, so adopt()
    does not take a stale STATE for a live run) and keep in them only the cases still not accepted."""
    st = json.loads(STATE.read_text())
    rows = {}
    for key, cases in st['cases'].items():
        step, ds = int(key.split(':')[0]), key.split(':')[1]
        responses = load_run(step, ds)
        for i, x in cases.items():
            info = evaluate(responses[int(i)])
            ok = accept(info) and not x['dit_rejected']
            x['attempts'] = [a for a in x['attempts'] if a['kind'] != 'rule_repair']
            if info['rules'] or info.get('repair_attempted') is not None:
                x['attempts'].insert(0, dict(kind='rule_repair', rules=info['rules'] or info.get('repair_attempted'), accepted=ok,
                                             error=info.get('repair_error'), tokens=info['tokens'] if info['tokens'] < 10 ** 9 else None))
            x['accepted'] = dict(method='rule_repair', rules=info['rules']) if ok else None
            for a in sorted((a for a in x['attempts'] if a['kind'] == 'retry' and a.get('source')), key=lambda a: a['round']):
                if a['source'] not in rows:
                    raw = get(a['source'])
                    rows[a['source']] = {r['input_index']: r for r in map(json.loads, raw.decode().splitlines())} if raw else {}
                row = rows[a['source']].get(int(i))
                if row is None:
                    continue
                info = evaluate(row)
                a.update(accepted=accept(info) and (row['response_sha256'] != x['greedy_response_sha256'] or not x['dit_rejected']),
                         json_valid=info['json_valid'], schema_valid=info['schema_valid'], refs_valid=info['refs_valid'],
                         error=info['error'], rules=info['rules'], tokens=info['tokens'] if info['tokens'] < 10 ** 9 else None)
                if a['accepted'] and not x['accepted']:
                    x['accepted'] = dict(method=f'retry_r{a["round"]}' + ('+rule_repair' if info['rules'] else ''), round=a['round'], rules=info['rules'])
    ps = pods()
    for n, rs in st['rounds'].items():
        if rs.get('done'):
            continue
        keep = []
        for j in rs.get('jobs', []):
            if j['status'] == 'running' and ps.get(j.get('worker')) and running_on(ps[j['worker']], j):
                keep.append(j); continue  # still running: run() picks it up
            if j['status'] == 'running' and phase(j) == 'rerun_complete_published_verified':
                j['status'] = 'complete'; collect(st, j)
            elif j['status'] == 'running' or j['status'].startswith('failed'):  # stopped by hand: run the round again
                subprocess.run(AWS + ['rm', '--recursive', retry_out(j['step'], j['ds'], j['round']), '--only-show-errors'], check=True)
                for i in j['indices']:
                    x = st['cases'][f'{j["step"]}:{j["ds"]}'][str(i)]
                    x['attempts'] = [a for a in x['attempts'] if not (a.get('round') == j['round'] and str(a.get('error', '')).startswith('retry run '))]
                for k in ('worker', 'pod', 'started'):
                    j.pop(k, None)
                j['status'] = 'queued'
            if j['status'] == 'queued':
                cases = st['cases'][f'{j["step"]}:{j["ds"]}']
                j['indices'] = [i for i in j['indices'] if not cases[str(i)]['accepted']]
                if not j['indices']:
                    continue
            keep.append(j)
        rs['jobs'] = keep
    save(st)
    say('reevaluated', summary(st))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'reevaluate':
        reevaluate()
    elif cmd == 'inventory':
        st = inventory(); print(json.dumps(summary(st), indent=1))
    elif cmd == 'run':
        run()
    elif cmd == 'publish':
        publish()
