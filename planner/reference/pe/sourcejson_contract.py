"""The parent planner prompt with source-image editing inputs."""
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import io
import json
from PIL import Image
from .prompt_iev6_ordered import ordered_target
from .multi_projection import pe_ordered_target
from .multi_transport import unpack_images
from .register_ocr_mds import make_row as validate_single_row

class RejectedSample(ValueError):
    """An intentionally excluded row; retry the same sampling channel."""

class RejectedInstruction(RejectedSample):
    """An instruction contains reserved template tokens and cannot be encoded."""

class RejectedSourceImages(RejectedSample):
    """The complete row exceeds the user-selected source image limit."""

MAX_SOURCE_IMAGES = 12

SYSTEM = (
    "You are an image editing layout planner. You receive ordered Source images, "
    "the original edit instruction, a Target aspect ratio (width/height), and "
    "Source region annotations in <source_regions>. The image marked frame_N "
    "corresponds to the frame_N annotation. Each Source annotation lists every "
    "region's label, parent, and bbox; ROOT denotes a top-level region. Labels "
    "and parent links are local to their Source frame. All bboxes are integer "
    "[x1, y1, x2, y2] coordinates normalized to 0-1000. Source bboxes describe "
    "locations in their corresponding Source image, while Target bboxes must "
    "describe the intended Target canvas. Use the images and Source region "
    "hierarchies together to ground references and understand the requested edit. "
    "Plan the complete intended Target, including unchanged content. Return "
    "one JSON object with fields in this order: normalized_edit_instruction, "
    "high_level_description, style_description, elements. "
    "style_description contains aesthetics, lighting, medium, and color_palette. "
    "elements contains background, layout, and reference_edits. List Target "
    "layout regions parent before child, with type, label, parent, bbox, optional "
    "text, desc, optional entity, and color_palette only for region labels "
    "that appear as label or reference_label in reference_edits. Omit color_palette from every other layout "
    "region. Always retain the overall style_description.color_palette. "
    "Group reference_edits by "
    "Source reference_id frame_N, with edit and remove arrays. Each edit names "
    "a Target label and describes the change with instruction. Its optional "
    "reference_label names the corresponding Source region in that frame; "
    "it may be one Source label or a list of Source labels. For new Target "
    "content with no Source region, omit reference_label. Each remove describes "
    "Source content absent from the Target using reference_label and instruction; "
    "include Source reference_bbox when the ID-based removal annotation provides it. "
    "Use Source label identifiers for cross-image grounding. "
    "Include changed regions in reference_edits; represent unchanged content "
    "in the complete Target layout without inventing extra edit actions. "
    "Follow the original edit instruction and the supplied Target aspect ratio. "
    "Output only the JSON object."
)

def user_prompt(instruction, count, size, source_regions=None):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('empty original instruction')
    if not 1 <= count <= MAX_SOURCE_IMAGES or len(size) != 2 or any(type(x) is not int or x <= 0 for x in size):
        raise ValueError('invalid sources or target geometry')
    # Preserve the original instruction exactly. Reserved media tokens are
    # rejected, never rewritten into a different instruction.
    if any(x in instruction for x in ('<image>', '<video>', '<audio>', '<|im_start|>', '<|im_end|>')):
        raise RejectedInstruction('reserved model token in original instruction')
    ratio = (Decimal(size[0]) / Decimal(size[1])).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
    regions = ''
    if source_regions is not None:
        if list(source_regions) != [f'frame_{i}' for i in range(count)]:
            raise ValueError('source region/image frame order mismatch')
        regions = '<source_regions>\n' + json.dumps(source_regions, ensure_ascii=False,
                    separators=(',', ':'), allow_nan=False) + '\n</source_regions>\n'
    return (''.join(f'<image>\nSource: frame_{i}\n' for i in range(count)) + regions +
            f'<user_prompt>\n{instruction}\n</user_prompt>\n<aspect_ratio>\n{ratio}\n</aspect_ratio>')

PROMPT_SHA = hashlib.sha256((SYSTEM + user_prompt('__ORIGINAL__', 1, (123,456),
    {'frame_0': {'compositional_deconstruction': {'elements': [{'label': '<region 1>', 'parent': 'ROOT', 'bbox': [18,22,982,978]}]}}})).encode()).hexdigest()

def row(instruction, images, target_bytes, target, channel, identifier, source_regions=None):
    with Image.open(io.BytesIO(target_bytes)) as im:
        size = im.size
    target = {key: target[key] for key in ('normalized_edit_instruction', 'high_level_description',
                                         'style_description', 'elements')}
    try:
        prompt=user_prompt(instruction,len(images),size,source_regions)
    except RejectedInstruction as error:
        error.sample_id=identifier
        error.channel=channel
        raise
    return dict(messages=[dict(role='system', content=SYSTEM),
                          dict(role='user', content=prompt),
                          dict(role='assistant',content=json.dumps(target,ensure_ascii=False,
                               separators=(',',':'),allow_nan=False))],
                images=[dict(bytes=x,path=None) for x in images],channel=channel,
                meta=dict(id=identifier,source_count=len(images),target_size=list(size)))

def single(sample, sidecar):
    record = json.loads(sample['row_json'])
    validate_single_row(record, sample, sidecar)
    return row(record['instruction'],[sample['source_image']],sample['target_image'],
               ordered_target(json.loads(sidecar['final_json_after'])),'single',record['sample_id'])

def multi(sample):
    label=json.loads(sample['row_json']); images=unpack_images(sample['reference_images'])
    if not 2 <= len(images) <= MAX_SOURCE_IMAGES or len(images) != label['num_references'] or label['bbox_order']!='xyxy_1000':
        raise ValueError('multi source cardinality/bbox contract changed')
    if [hashlib.sha256(x).hexdigest() for x in images] != json.loads(sample['reference_sha256']):
        raise ValueError('ordered multi-source SHA mismatch')
    if hashlib.sha256(sample['target_image']).hexdigest()!=sample['target_sha256']:
        raise ValueError('multi target SHA mismatch')
    return row(label['instruction'],images,sample['target_image'],
               pe_ordered_target(label['target_json'],len(images)),'multi',label['source_sample_id'])
