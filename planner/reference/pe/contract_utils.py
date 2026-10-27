import argparse
import ast
import concurrent.futures
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import boto3
from botocore.config import Config
from PIL import Image

def strict_json(text):
    text = re.sub('^\\s*<think>.*?</think>\\s*', '', text, count=1, flags=re.S)
    text = re.sub('^\\s*```(?:json)?\\s*', '', text, count=1, flags=re.I)
    text = re.sub('\\s*```\\s*$', '', text, count=1)
    return json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def validate_regions(value):
    assert isinstance(value, dict) and set(value) == {'compositional_deconstruction'}
    composition = value['compositional_deconstruction']
    assert isinstance(composition, dict) and set(composition) == {'elements'}
    regions = composition['elements']
    assert isinstance(regions, list)
    labels = set()
    projected = []
    for region in regions:
        assert set(region) == {'label', 'parent', 'bbox'}
        label, parent, box = (region['label'], region['parent'], region['bbox'])
        assert isinstance(label, str) and label and (label not in labels)
        assert parent is None or isinstance(parent, str)
        assert isinstance(box, list) and len(box) == 4 and all((type(x) is int for x in box))
        assert 0 <= box[0] <= box[2] <= 1000 and 0 <= box[1] <= box[3] <= 1000
        labels.add(label)
        projected.append(dict(label=label, parent='ROOT' if parent is None else parent, bbox=box))
    assert all((r['parent'] == 'ROOT' or r['parent'] in labels for r in projected))
    return {'compositional_deconstruction': {'elements': projected}}

def validate_prediction(value, arm, count):
    fields = ['normalized_edit_instruction', 'high_level_description', 'style_description', 'elements']
    if arm == 'refearly':
        fields.insert(1, 'reference_edits')
    assert isinstance(value, dict) and list(value) == fields, 'Root fields/order differ from training'
    assert isinstance(value['normalized_edit_instruction'], str) and value['normalized_edit_instruction'].strip()
    assert isinstance(value['high_level_description'], str) and value['high_level_description'].strip()
    assert isinstance(value['style_description'], dict)
    assert set(value['style_description']) == {'aesthetics', 'lighting', 'medium', 'color_palette'}
    elements = value['elements']
    assert set(elements) == ({'background', 'layout'} if arm == 'refearly' else {'background', 'layout', 'reference_edits'})
    references = value['reference_edits'] if arm == 'refearly' else elements['reference_edits']
    assert isinstance(references, list) and isinstance(elements['layout'], list)
    seen, labels = (set(), set())
    for region in elements['layout']:
        assert {'type', 'label', 'parent', 'bbox', 'desc'} <= set(region)
        assert isinstance(region['label'], str) and region['label'] not in labels
        assert region['parent'] in (None, 'ROOT') or region['parent'] in labels
        box = region['bbox']
        assert isinstance(box, list) and len(box) == 4 and all((type(x) is int for x in box))
        assert 0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000
        labels.add(region['label'])
    for group in references:
        assert set(group) == {'reference_id', 'edit', 'remove'}
        assert group['reference_id'] in {f'frame_{i}' for i in range(count)}
        assert group['reference_id'] not in seen
        seen.add(group['reference_id'])
        assert isinstance(group['edit'], list) and isinstance(group['remove'], list)
        for edit in group['edit']:
            assert edit['label'] in labels and isinstance(edit['instruction'], str)
        for remove in group['remove']:
            assert isinstance(remove['instruction'], str)
            # Training targets keep removals that only have a Source reference_bbox
            # (no ID-based annotation); frame_v8.boxes() is the training check.
            if remove.get('reference_label') is None:
                reference_boxes(remove.get('reference_bbox'))
    return value

def reference_boxes(value):
    """One XYXY/1000 box or a nonempty list of them, as accepted by frame_v8.boxes()."""
    group = [value] if isinstance(value, list) and len(value) == 4 and all(type(x) is int for x in value) else value
    assert isinstance(group, list) and group, 'Removal needs reference_label or reference_bbox'
    for box in group:
        assert isinstance(box, list) and len(box) == 4 and all(type(x) is int for x in box)
        assert 0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000, 'Invalid Source reference_bbox'
    return group

def load_base_contract(path, digest):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    tree = ast.parse(raw)
    keep = []
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom) and statement.module == 'decimal':
            keep.append(statement)
        elif isinstance(statement, ast.Import):
            if all((a.name in ('hashlib', 'json') for a in statement.names)):
                keep.append(statement)
        elif isinstance(statement, ast.Assign):
            names = {x.id for x in statement.targets if isinstance(x, ast.Name)}
            if names & {'MAX_SOURCE_IMAGES', 'SYSTEM', 'PROMPT_SHA'}:
                keep.append(statement)
        elif isinstance(statement, ast.FunctionDef) and statement.name == 'user_prompt':
            keep.append(statement)
        elif isinstance(statement, ast.ClassDef) and statement.name in ('RejectedSample', 'RejectedInstruction'):
            keep.append(statement)
    namespace = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


R7_ARM = 'given_ratio_inline_edits'
# r3 (Qwen3.5-9B): the assistant emits the plan itself; the ratio is user input only.
PLAN_ONLY_ARM = 'given_ratio_plan_only_inline_edits'
FROZEN_ARMS = {R7_ARM: 'r7_prompt_contract.json', PLAN_ONLY_ARM: 'r3_prompt_contract.json'}
PLAN_FIELDS = ['normalized_edit_instruction', 'high_level_description', 'style_description', 'elements']


def target_ratio(size):
    """auto_ratio.ratio_string(): width/height, Decimal ROUND_HALF_UP to 4 places."""
    from decimal import Decimal, ROUND_HALF_UP
    width, height = size
    return float((Decimal(int(width)) / Decimal(int(height))).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))


def inline_to_grouped(value, count):
    """Invert the r7 training projection (inline_caption.project) without repairing anything.

    r7 output: {"aspect_ratio", "plan"}; each edited layout region carries an
    `edit` list immediately before `desc`; reference_edits groups keep only
    reference_id/remove. Returns the grouped (baseline-schema) plan so the
    existing schema/source-reference validators apply unchanged.
    """
    assert isinstance(value, dict) and list(value) == ['aspect_ratio', 'plan'], 'Root fields/order differ from training'
    assert type(value['aspect_ratio']) in (int, float), 'aspect_ratio must be a number'
    return inline_plan_to_grouped(value['plan'], count)


def plan_only_to_grouped(value, count):
    """r3 output is the inline plan object itself (no aspect_ratio wrapper)."""
    return inline_plan_to_grouped(value, count)


def inline_plan_to_grouped(plan, count):
    plan = json.loads(json.dumps(plan))
    assert isinstance(plan, dict) and list(plan) == PLAN_FIELDS, 'Plan fields/order differ from training'
    elements = plan['elements']
    assert isinstance(elements, dict) and list(elements) == ['background', 'layout', 'reference_edits']
    groups, layout = elements['reference_edits'], elements['layout']
    assert isinstance(groups, list) and isinstance(layout, list)
    edits = {}
    for group in groups:
        assert isinstance(group, dict) and list(group) == ['reference_id', 'remove'], 'Group fields differ from training'
        assert group['reference_id'] in {f'frame_{i}' for i in range(count)} and group['reference_id'] not in edits
        edits[group['reference_id']] = []
    for region in layout:
        assert isinstance(region, dict) and isinstance(region.get('label'), str)
        if 'edit' not in region:
            continue
        keys = list(region)
        assert 'desc' in region and keys.index('edit') + 1 == keys.index('desc'), 'edit must immediately precede desc'
        actions = region.pop('edit')
        assert isinstance(actions, list) and actions, 'edit must be a nonempty list'
        for action in actions:
            assert isinstance(action, dict) and list(action) in (
                ['reference_id', 'instruction'], ['reference_id', 'reference_label', 'instruction']), 'Edit action fields differ'
            assert action['reference_id'] in edits, 'Inline edit names a frame without a reference_edits group'
            edit = dict(label=region['label'])
            if 'reference_label' in action:
                edit['reference_label'] = action['reference_label']
            edit['instruction'] = action['instruction']
            edits[action['reference_id']].append(edit)
    elements['reference_edits'] = [dict(reference_id=g['reference_id'], edit=edits[g['reference_id']], remove=g['remove'])
                                   for g in groups]
    return plan


def load_contract(path, digest):
    """Bind the exact training variant to the exact training base contract."""
    import importlib.util
    import types
    path = Path(path)
    plan = json.loads((path.parent / 'plan.json').read_text())
    base = load_base_contract(path, digest)
    if plan['training_ablation'] in FROZEN_ARMS:
        name = FROZEN_ARMS[plan['training_ablation']]
        raw = (path.parent / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == plan['prompt_contract_sha256'], name
        frozen = json.loads(raw)
        spec = frozen['SPEC']
        assert spec['name'] == plan['training_ablation'] and spec['prompt_sha256'] == plan['training_prompt_spec_sha256']
        assert hashlib.sha256(frozen['SYSTEM'].encode()).hexdigest() == spec['system_sha256']
        rules = frozen['USER_OUTPUT_RULES']
        def frozen_user_prompt(instruction, count, size, source_regions=None):
            return base['user_prompt'](instruction, count, size, source_regions) + rules
        return dict(base, SYSTEM=frozen['SYSTEM'], user_prompt=frozen_user_prompt, SPEC=spec,
                    training_transform=None, base_contract=base, frozen=frozen)
    variant_path = path.parent / 'training_variants.py'
    assert hashlib.sha256(variant_path.read_bytes()).hexdigest() == plan['training_variants_sha256']
    spec = importlib.util.spec_from_file_location('frozen_training_variants', variant_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.base_contract = lambda: types.SimpleNamespace(**base)
    arm = plan['training_ablation']
    contract_spec = module.specification(arm)
    assert contract_spec['prompt_sha256'] == plan['training_prompt_spec_sha256']
    def user_prompt(instruction, count, size, source_regions=None):
        user = base['user_prompt'](instruction, count, size, source_regions)
        return module.interleave(user, count) if arm == 'p1' else user
    system = base['SYSTEM']
    if arm == 'o1':
        system = system.replace(module.SYSTEM_ORDER, module.O1_SYSTEM_ORDER)
    return dict(base, SYSTEM=system, user_prompt=user_prompt, SPEC=contract_spec,
                training_transform=module.transform, base_contract=base)
