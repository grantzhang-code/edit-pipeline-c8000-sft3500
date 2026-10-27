"""Exactly one sampled completion per benchmark case, with raw-response provenance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import argparse
import os
from pathlib import Path
import sys

from PIL import Image
from common import client, digest, get, write
from contract_utils import PLAN_ONLY_ARM, R7_ARM, inline_to_grouped, load_contract, plan_only_to_grouped, strict_json, target_ratio, validate_prediction
from training_image import IMAGE_COLOR_POLICY, IMAGE_RESIZE_POLICY, training_image
import engine_config


def validate_references(value,regions):
    labels = {r['label'] for r in value['elements']['layout']}
    referenced = set()
    for group in value['elements']['reference_edits']:
        source_labels = {r['label'] for r in regions[group['reference_id']]['compositional_deconstruction']['elements']}
        for edit in group['edit']:
            referenced.add(edit['label'])
            reference = edit.get('reference_label')
            if reference is not None:
                names = [reference] if isinstance(reference,str) else reference
                assert isinstance(names,list) and names and all(isinstance(k,str) and k in source_labels for k in names), 'Unknown Source reference_label'
                referenced.update(names)
        for remove in group['remove']:
            reference = remove.get('reference_label')
            if reference is None:
                continue  # bbox-only removal; reference_bbox checked by validate_prediction
            names = [reference] if isinstance(reference,str) else reference
            assert isinstance(names,list) and names and all(isinstance(k,str) and k in source_labels for k in names), 'Unknown removal Source reference_label'
            referenced.update(names)
    assert all('color_palette' not in region or region['label'] in referenced
               for region in value['elements']['layout']), 'Unreferenced region color_palette'


def main(args):
    shared=Path(args.shared)
    plan=json.loads((shared/'plan.json').read_text())
    model=next(m for m in plan['models'] if m['step']==args.step)
    sampling={k:plan['decoding'][k] for k in ('temperature','top_p','top_k','min_p','presence_penalty','repetition_penalty')}
    assert sampling == {'temperature': 0.0, 'top_p': 1.0, 'top_k': -1, 'min_p': 0.0, 'presence_penalty': 0.0, 'repetition_penalty': 1.0}
    seed=42
    if plan.get('decoding_override'):  # targeted retry (rerun_subset.py) of failed cases with other sampling; main runs stay greedy
        sampling={k:plan['decoding_override'][k] for k in sampling}
        seed=plan['decoding_override'].get('seed',42)
    contract=load_contract(shared/'sourcejson_contract.py',plan['contract_sha256'])
    cases=[json.loads(x) for x in (shared/'cases.jsonl').read_bytes().splitlines()]
    targets=json.loads((shared/'target-dimensions.json').read_text())
    source=json.loads((shared/'source-regions.json').read_text())
    destination=shared/f'checkpoint-{args.step}'/('pilot.jsonl' if args.pilot else f'rank-{args.rank:02d}.jsonl')
    destination.parent.mkdir(exist_ok=True)
    attempts_path=destination.parent/('pilot-attempts.jsonl' if args.pilot else f'attempts-{args.rank:02d}.jsonl')
    world_size=getattr(args,'world_size',8)
    assert 0<=args.rank<world_size
    assigned=list(enumerate(cases))[args.rank::world_size]
    only=getattr(args,'only',None)
    if only:  # targeted re-run: only these benchmark indices, sharded over the same world size
        keep={int(x) for x in only.split(',')}
        assigned=[(i,c) for i,c in enumerate(cases) if i in keep][args.rank::world_size]
    if args.pilot:
        eligible=[(i,c) for i,c in enumerate(cases) if c['id'] in targets and all(x.get('source_key',x['sha256']) in source for x in c['inputs'])]
        first=next((i,c) for i,c in eligible if c['input_count']==min(x['input_count'] for _,x in eligible))  # single-image unless the dataset is multi-image only
        assigned=[first,
                  next((i,c) for i,c in eligible if 1<c['input_count']<max(x['input_count'] for _,x in eligible) and i!=first[0]),
                  # most sources available (12 on balanced-1500 / multiref428; smaller datasets have fewer)
                  next((i,c) for i,c in eligible if c['input_count']==max(x['input_count'] for _,x in eligible))]
    completed={}
    if destination.exists():
        for line in destination.read_text().splitlines():
            row=json.loads(line)
            assert row['input_index'] not in completed,'Duplicate completion'
            completed[row['input_index']]=row
    if attempts_path.exists():
        attempts=[json.loads(line) for line in attempts_path.read_text().splitlines()]
        assert len({r['input_index'] for r in attempts})==len(attempts)
        assert all(r['input_index'] in completed for r in attempts), 'Uncommitted attempt: no retry allowed'
    # Canary records are the actual first and only completion for these cases.
    if not args.pilot and (shared/f'checkpoint-{args.step}/PILOT_PASS.json').exists():
        with destination.open('a',buffering=1) as stream:
            for line in (shared/f'checkpoint-{args.step}/pilot.jsonl').read_text().splitlines():
                row=json.loads(line)
                if row['input_index'] % world_size == args.rank and row['input_index'] not in completed:
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                    completed[row['input_index']]=row
    exclusions=json.loads((shared/'input-exclusions.json').read_text())
    expected_hashes=json.loads((shared/'expected-prompt-hashes.json').read_text())
    with destination.open('a',buffering=1) as stream:
        for i,c in assigned:
            if i not in completed and str(i) in exclusions:
                row=dict(exclusions[str(i)],training_ablation=plan['training_ablation'],training_prompt_spec_sha256=contract['SPEC']['prompt_sha256'],run_id=plan['run_id'],checkpoint_step=args.step,checkpoint_manifest_sha256=model['checkpoint_sha256'])
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');completed[i]=row
    pending=[(i,c) for i,c in assigned if i not in completed and c['id'] in targets and
             all(x.get('source_key',x['sha256']) in source for x in c['inputs'])]
    with destination.open('a',buffering=1) as stream:
        for i,c in assigned:
            if i in completed:
                continue
            missing=[x['sha256'] for x in c['inputs'] if x.get('source_key',x['sha256']) not in source]
            if c['id'] not in targets or missing:
                stream.write(json.dumps(dict(input_index=i,case_id=c['id'],run_id=plan['run_id'],
                    checkpoint_step=args.step,checkpoint_manifest_sha256=model['checkpoint_sha256'],
                    status='awaiting_gpt_image_target' if c['id'] not in targets else 'failed_source_annotation',
                    missing_source_images=missing,inference_attempts=0,json_valid=None,schema_valid=None))+'\n')
    llm=getattr(args,'llm',None)
    if pending:
        root=Path('/mnt/ephemeral')/plan['runtime_task_id']
        sys.path.insert(0,str(root/'source-repo'))
        from scripts.mtp_eval.run_test1070_inference import dynamic_max_new_tokens
        from context_guard import untruncated_prompt_counts
        engine_config.prepare_environment(plan)
        from pe_training_alignment import install, training_hf_overrides
        install()
        from vllm import LLM,SamplingParams
        if llm is None:
            llm=LLM(**engine_config.engine_kwargs(plan,args.model,training_hf_overrides))
        engine=engine_config.check_engine(llm,plan)
        batch_size=engine_config.batch_size(plan)
        tokenizer=llm.get_tokenizer()
        s3=client()
        cache=root/'pe-images'
        cache.mkdir(exist_ok=True)

        def image(item):
            path=cache/item['sha256']
            assert path.exists(), 'source image not prefetched by s5cmd: '+item['s3_uri']
            raw=path.read_bytes()
            assert digest(raw)==item['sha256']
            with Image.open(io.BytesIO(raw)) as im:
                assert im.size==(item['width'],item['height'])
                return training_image(im)

        def request(index,case):
            regions={f'frame_{i}':source[item.get('source_key',item['sha256'])]['regions'] for i,item in enumerate(case['inputs'])}
            target=targets[case['id']]
            size=[target['width'],target['height']]
            assert case['instruction']==case['user_instruction']==case['rewritten_instruction'] and case['instruction_variant']==plan['instruction_variant'], 'PE must use the Luna-rewritten instruction'
            user=contract['user_prompt'](case['instruction'],case['input_count'],size,regions)
            chunks=user.split('<image>');assert len(chunks)-1==case['input_count']
            content=([{'type':'text','text':chunks[0]}] if chunks[0] else [])
            for chunk in chunks[1:]:
                content.extend([{'type':'image','image':'<image>'},{'type':'text','text':chunk}])
            prompt=tokenizer.apply_chat_template([{'role':'system','content':contract['SYSTEM']},
                {'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            expected=('<|im_start|>system\n'+contract['SYSTEM']+'<|im_end|>\n<|im_start|>user\n'+
                user.replace('<image>','<|vision_start|><|image_pad|><|vision_end|>')+
                '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
            assert prompt==expected
            assert digest(prompt.encode())==expected_hashes[str(index)], 'Prompt differs from exact training variant'
            with ThreadPoolExecutor(max_workers=12) as pool:
                images=list(pool.map(image,case['inputs']))
            row=dict(input_index=index,case_id=case['id'],input_count=case['input_count'],
                original_instruction=case['instruction'],instruction=case['instruction'],
                user_instruction=case['user_instruction'],instruction_variant=case['instruction_variant'],
                benchmark_cases_sha256=plan['benchmark_cases_sha256'],
                instruction_source_s3=plan['instruction_source_s3'],
                pre_rewrite_instruction=case.get('original_user_instruction'),source_image_sha256=[x['sha256'] for x in case['inputs']],
                source_image_uris=[x['s3_uri'] for x in case['inputs']],source_image_color_policy=IMAGE_COLOR_POLICY,source_image_resize_policy=IMAGE_RESIZE_POLICY,
                model_image_sizes=[list(im.size) for im in images],engine=engine,source_image_roles=[x['role'] for x in case['inputs']],
                source_regions=regions,target_size=size,target_size_policy=plan['target_ratio_policy'],
                target_image_provenance=target,system_prompt=contract['SYSTEM'],user_prompt=user,prompt_sha256=digest(prompt.encode()),
                training_ablation=plan['training_ablation'],training_prompt_spec_sha256=contract['SPEC']['prompt_sha256'],run_id=plan['run_id'],checkpoint_step=args.step,checkpoint_manifest_sha256=model['checkpoint_sha256'],
                source_model_manifest_sha256=plan['source_manifest_sha256'],decoding=plan['decoding'],inference_attempts=1)
            return dict(prompt=prompt,multi_modal_data={'image':images},mm_processor_kwargs=dict(plan['decoding']['mm_processor_kwargs'])),row,images

        with destination.open('a',buffering=1) as stream:
            for offset in range(0,len(pending),batch_size):
                batch=[request(i,c) for i,c in pending[offset:offset+batch_size]]
                requests=[x[0] for x in batch]
                counts=untruncated_prompt_counts(llm,requests)
                kept=[];kept_counts=[]
                for item,count in zip(batch,counts,strict=True):
                    if count >= plan['decoding']['max_model_len']-1:
                        _,row,images=item
                        row.update(status='exceeds_training_context',prompt_tokens=count,
                            context_limit=plan['decoding']['max_model_len'],inference_attempts=0,
                            output_tokens=0,prompt_truncated=False,json_valid=False,schema_valid=False,
                            source_references_valid=False,prediction=None,
                            exclusion_reason='Complete prompt leaves no output budget within training length %d'%plan['decoding']['max_model_len'])
                        stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                        for im in images:im.close()
                    else:kept.append(item);kept_counts.append(count)
                batch=kept;counts=kept_counts;requests=[x[0] for x in batch]
                if not batch:continue
                budgets=[dynamic_max_new_tokens(prompt_tokens=n,max_model_len=plan['decoding']['max_model_len'],
                    cli_max_new_tokens=plan['decoding']['max_new_tokens'],eos_reserve_tokens=1) for n in counts]
                # Persist intent before generation: interrupted attempts cannot be resampled.
                with attempts_path.open('a',buffering=1) as journal:
                    for (_,row,_),budget in zip(batch,budgets,strict=True):
                        journal.write(json.dumps(dict(input_index=row['input_index'],case_id=row['case_id'],
                            prompt_sha256=row['prompt_sha256'],inference_attempts=1,temperature=sampling['temperature'],sampling_params=dict(sampling,n=1,max_tokens=budget,seed=seed),retries=0))+'\n')
                    journal.flush()
                    os.fsync(journal.fileno())
                outputs=llm.generate(requests,[SamplingParams(**sampling,n=1,max_tokens=b,seed=seed)
                    for b in budgets],use_tqdm=False)
                assert len(outputs)==len(batch)
                for (_,row,images),output,count,budget in zip(batch,outputs,counts,budgets,strict=True):
                    assert len(output.outputs)==1
                    response=output.outputs[0]
                    row.update(response=response.text,response_sha256=digest(response.text.encode()),
                        finish_reason=response.finish_reason,output_tokens=len(response.token_ids),prompt_tokens=count,
                        max_new_tokens_budget=budget,json_valid=False,schema_valid=False,source_references_valid=False,prediction=None,status='inferred')
                    try:
                        value=strict_json(response.text)
                        row.update(json_valid=True,prediction=value)
                        if plan['training_ablation']==R7_ARM:
                            row['aspect_ratio_matches_target']=(type(value.get('aspect_ratio')) in (int,float)
                                and float(value['aspect_ratio'])==target_ratio(row['target_size'])) if isinstance(value,dict) else False
                            grouped=inline_to_grouped(value,row['input_count'])
                            row['prediction_grouped']=grouped
                        elif plan['training_ablation']==PLAN_ONLY_ARM:
                            grouped=plan_only_to_grouped(value,row['input_count'])
                            row['prediction_grouped']=grouped
                        else:
                            grouped=value
                        validate_prediction(grouped,'sourcejson',row['input_count'])
                        assert list(grouped['elements']) == contract['SPEC']['target_elements_order'], 'Target elements order differs from training'
                        assert response.finish_reason!='length','Truncated generation'
                        row['schema_valid']=True
                        validate_references(grouped,row['source_regions'])
                        row['source_references_valid']=True
                    except Exception as error:
                        row['validation_error']=str(error) or type(error).__name__
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                    for im in images:im.close()
                print(json.dumps(dict(event='sampling_progress',step=args.step,rank=args.rank,
                    completed=min(offset+batch_size,len(pending)),assigned=len(pending))),flush=True)
    rows=[json.loads(x) for x in destination.read_text().splitlines()]
    assert len(rows)==len(assigned) and len({x['input_index'] for x in rows})==len(assigned)
    write(shared/f'checkpoint-{args.step}'/('PILOT_DONE.json' if args.pilot else f'DONE-{args.rank:02d}.json'),
        dict(step=args.step,rank=args.rank,records=len(rows),sha256=digest(destination.read_bytes()),
             inferred=sum(x['status']=='inferred' for x in rows),world_size=world_size))
    return llm


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--shared',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--step',required=True,type=int)
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--world-size',type=int,default=8)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--only',default=None,help='comma-separated benchmark input indices (targeted re-run)')
    main(parser.parse_args())
