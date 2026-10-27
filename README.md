# Planner c8000 + layout-only Source JSON + Luna 指令 → DiT SFT-3500 编辑生图 → 10分-prism-like

本仓库整理 balanced-1500 编辑评测的完整流水线，覆盖 2026-10-26 已跑完的这一组配置：

```
balanced-1500-r6 benchmark ─┬─ Luna 改写指令 (cases.luna.jsonl)
                            ├─ Source 图（有序，1–12 张）
                            └─ Source layout-only JSON（复用 10-14 预生成池，ericdu layout-only-shortstart-c10000）
                                        │
        [Stage 1] 构造 Planner prompt   ▼   pipeline/01_build_planner_inputs.py      (CPU, 可复跑, 已逐字节验证)
        [Stage 2] Planner 推理          ▼   fzsft-9b c8000, greedy + 规则修复 + 3 轮采样重试 (repair-retry3)   (GPU, RTX PRO 6000)
        [Stage 3] 收集 Planner JSON     ▼   pipeline/03_collect_planner_json.py      (CPU)
        [Stage 4] Planner JSON → DiT 输入 ▼  pipeline/04_planner_to_dit.py           (CPU, 已逐行验证)
        [Stage 5] DiT 生图              ▼   tayo SFT-v2 step=3500 EMA, euler50 · CFG 2 · seed 42   pipeline/05_render_dit.py (GPU, 8×H100/H200)
        [Stage 6] 打分                  ▼   10分-prism-like (jimmyqiancode @5a9db85)   pipeline/06_eval_prism.py
```

所有 S3 位置、ckpt、sha256 统一记录在 [pins.json](pins.json)；脚本只从这里取值。

## 1. 模型与 ckpt

| 环节 | 模型 | ckpt / 制品 | 校验 |
|---|---|---|---|
| Source layout-only JSON（已复用，不重跑） | ericdu `layout-only-shortstart-c10000`（task `layout_only`，greedy） | HF `s3://core-cn-training.canva.com/pixelcraft/usr/ericdu/models/layout-only-shortstart-c10000-hf-trained-mtp3-v1/`，git `fbc18bf3` | HF manifest `4e5155a9…`，ckpt `062292e9…`，prompt `404d4acb…` |
| Planner | fzsft-9b（wandb `4b3162283b0b`，jobset `fengzhida-editing-planner-9b-sft-128g-1025c`），Qwen3.5-9B，r16-9b c25000 model-only 热启动 | mcore `s3://core-cn-oss.canva.com/usr/fengzhida/design-layout-megatron-swift/runs/fengzhida-editing-planner-9b-sft-128g-1025c/mcore/checkpoint-8000/`（130.6 GB） | manifest `744289f1…` |
| Planner（推理用 HF 导出） | 同上 c8000，trained MTP | `s3://core-cn-oss.canva.com/usr/grantzhang/balanced-1500-layoutonly-editparser-fzsft-9b-4b3162283b0b-c8000-trainprompt-greedy-rtx8-bs64-20261026/models/checkpoint-8000/`（19.3 GB） | HF manifest `fafca621…`；HSDP 导出 841 键全部回读校验 |
| DiT | tayo `tayo-edit1k-sft-v2-6m-360-20261026` step=3500，**EMA** 权重（从 pr377-nk62-r54k step=57000 热启动） | `s3://core-cn-training.canva.com/pixelcraft/usr/tayo/runs/tayo-edit1k-sft-v2-6m-360-20261026/checkpoints/step=3500.ckpt/`（154.7 GB） | 渲染包启动时校验 ETag / 分段 / 932 keys / source step |
| DiT 条件编码 | Qwen3.8-27B（文本/参考编码）+ FLUX.2-klein-4B VAE | `s3://core-cn-oss.canva.com/checkpoints/qwen/Qwen3.8-27B/`、`s3://core-cn-oss.canva.com/checkpoints/flux2/FLUX.2-klein-4B-vae/` | 渲染包冻结路径 |
| 指令改写（已复用） | gpt-6-luna（变体 `luna-20261013`，skill `a7c144dc`） | `…/balanced-1500-r6-20261012/cases.luna.jsonl` | sha `1f8466f6…` |
| 评委 | gpt-6.1-sol，reasoning effort medium，每轴 1 票 | OpenAI Responses API | 协议 `prism-edit1500-v7-single` |

冻结代码包（`pipeline/00_fetch.sh` 拉到 `vendor/` 并校验 sha）：

| 包 | 位置 | sha256 |
|---|---|---|
| Planner 推理包（r3-9b-pe-1023 @ `30c6515`，含 plan `plan-8000-balanced-1500-layoutonly-fzsft-9b.json`） | `s3://…/usr/grantzhang/runbooks/editparser-r3-9b-pe-rtx8-20261023/packages/pe-30c6515-….tar.gz` | `d408ce87…` |
| Planner 重试包（@ `5fe6b98`，plan `…-retry-r123-multi.json`） | `…/packages/pe-5fe6b98-….tar.gz` | `dcea0833…` |
| DiT 渲染包 `dit-render-cfg2-v3-tayo58k`（tayo cfg2-v3 冻结包；只放宽 `checkpoints.py` 白名单到 `tayo-edit1k-sft-*`，step ≥ 500） | `s3://…/usr/grantzhang/dit58k-planner9b-1025/ema58000-cfg2/code/27a5fcc2….tar.gz` | `27a5fcc2…` |
| 评测 skill `10分-prism-like` | https://github.com/canvanauts/jimmyqiancode/tree/main/10%E5%88%86-prism-like @ `5a9db85` | git |

## 2. 输入数据

### 2.1 Benchmark 与 Luna 指令

- `s3://core-cn-oss.canva.com/usr/tayo/testset-studio/shared/v1/benchmarks/balanced-1500-r6-20261012/`：1500 题（单图 750 / 多图 750）。
- Planner 用 `cases.luna.jsonl`：`instruction == user_instruction == rewritten_instruction`（Luna 改写，原文保留），原始用户话在 `original_user_instruction`。
- 目标尺寸：每题 GPT Image 2.5 目标图的 PNG 宽高，见 [data/balanced-1500-layoutonly/target-dimensions.json](data/balanced-1500-layoutonly/target-dimensions.json)。只用作 `<aspect_ratio>` 和 DiT 画布尺寸，不把目标图给模型。
- 可推理 1465 题（单图 743 / 多图 722）。35 题排除，原因见 [input-exclusions.json](data/balanced-1500-layoutonly/input-exclusions.json)：27 题有 Source 没有合格的 layout-only（或 full-JSON）标注，8 题缺目标图。

### 2.2 复用的 Source layout-only JSON

Planner 的 `<source_regions>` 不是现场生成，而是复用 10-14 在 Eagle H200 上为 balanced-1500 全部 3443 张 Source 图预先跑好的结果：

- 池：`s3://core-cn-oss.canva.com/usr/grantzhang/balanced-1500-r6-source-json-dual-greedy-20261014/`（task `infer_layout_dual_balanced1500_h200_20261014`，两路：`layoutfirst` = 完整 JSON，`layout_only` = 只出版面）。
- 本流水线用 **`layout_only/source-captions.jsonl`**（sha `c0a6c08f…`），模型 ericdu `layout-only-shortstart-c10000`，greedy（temperature 0、seed 42、max_new_tokens 32000、max_pixels 4 194 304）。3443 张中 schema 合格 3430、invalid 2、截断 11。完整运行参数见 [data/source-json-pool-plan.json](data/source-json-pool-plan.json)，prompt 见 [prompts/source_layout_only_prompt.json](prompts/source_layout_only_prompt.json)。
- 取法（Stage 1 的 `layout_regions()`）：跳过 `schema_valid=false` 或 `finish_reason=length` 的行，按 `input_metadata.sha256`（原图 sha）索引，每个元素只保留 `label / parent / bbox`。bbox 为 XYXY、0–1000 整数，`parent` 为 `ROOT` 或父元素 label，label 只在本 frame 内有效。
- 与 full-JSON 变体的区别：full-JSON 变体用同一个池的 `layoutfirst` 路。两者限在同一批题上（两种 JSON 都合格的 1465 题），可以同题配对比较。

## 3. Planner 输入 / 输出格式

### 3.1 输入（Stage 1 产出 `planner_requests.jsonl`）

训练合约 `given_ratio_plan_only_inline_edits` v1（prompt spec `eecfcbe5…`）。对话是 system + user 两条消息，`enable_thinking=False`，chat template `qwen3_5`：

- **system**：[prompts/planner_system.txt](prompts/planner_system.txt)（sha `6a4ef8df…`，与训练 yaml 钉死的值一致）。
- **user**（[样例](examples/case0000/1_planner_user_prompt.txt)），按以下顺序拼接：

```text
<image>\nSource: frame_0\n          ← 每张 Source 图一段，按 benchmark inputs 顺序；frame_i ↔ 第 i 张图
<image>\nSource: frame_1\n ...
<source_regions>\n{"frame_0":{"compositional_deconstruction":{"elements":[{"label":"<poster canvas 1>","parent":"ROOT","bbox":[0,0,1000,1000]},…]}},"frame_1":…}\n</source_regions>\n
<user_prompt>\n{Luna 改写后的指令，原样}\n</user_prompt>\n
<aspect_ratio>\n{目标宽/高，4 位小数 ROUND_HALF_UP}\n</aspect_ratio>
<output_format>…固定输出规则（USER_OUTPUT_RULES）…</output_format>
```

- `<source_regions>` 是紧凑 JSON（`separators=(',', ':')`，`ensure_ascii=False`）。
- 图像预处理（推理包 `training_image.py`，与训练相同）：透明图先合成到白底；swift `rescale_image`（bilinear）缩到 ≤ 1 048 576 px；再用 qwen_vl_utils 0.0.14 `smart_resize`（factor 32，3072–1 048 576 px，bicubic）；vLLM `do_resize=False`。
- `prompt_sha256` = sha256(完整 chat 模板文本)。Stage 1 对 1465 题全部逐字节复现：system、user、source_regions、prompt_sha256 都与已发布的 c8000 推理一致，也与 `expected-prompt-hashes.json` 一致。

`planner_requests.jsonl` 每行的字段：`input_index, case_id, input_count, instruction, instruction_variant, original_user_instruction, source_images[{uri, sha256, width, height}], source_regions, target_size[w,h], target_image_uri, system_prompt, user_prompt, prompt_sha256`。

### 3.2 推理设置

greedy：temperature 0、top_p 1、top_k −1、repetition_penalty 1、seed 42；max_model_len 32768；max_new_tokens = 32768 − prompt − 1（超长样本直接排除，不截断 prompt）；MTP 1；batch 64；关闭 chunked prefill。vLLM 0.29 runtime `76e2cdea…`，镜像是训练同一 digest（`ds-core-cn-omra-trainer@sha256:54d365f8…`）。数值对齐补丁（FP32 head、FP32 视觉 RoPE、GDN FP32 conv、训练视觉 varlen 等）都在推理包里，启动前会过 gate。硬件：Eagle RTX PRO 6000（sm_120），hold `grantzhang-rtx64-r12pe-eagle-1025a`，4 节点 × 8 卡。

### 3.3 repair-retry3

greedy 输出先做固定规则修复，再按顺序做最多 3 轮采样重试。参考实现是 [planner/reference/tools/indomain_fix.py](planner/reference/tools/indomain_fix.py)：

1. **规则修复**（只针对未截断的输出）：JSON 语法问题（region 值缺 key、多/少逗号、未转义控制字符、缺闭括号）；可解析计划的 schema 问题（bbox 顺序/退化/越界、label 重复、未知 parent 改为 ROOT、reference_edits 组缺失或重复）。修完重新跑完整校验。
2. **仍不合格就重采样**：round1 t0.7/p0.9/k50/rp1.05/seed42 → round2 t0.8/p0.95/k100/rp1.05/seed1234 → round3 t0.6/p0.9/k50/rp1.1/seed7。三轮并发采样，按轮次顺序判，第一轮合格的生效。每轮输出同样先做规则修复。
3. **合格标准**：JSON + schema + Source 引用都有效；`finish_reason != length`；去掉 bbox 后 caption ≤ 15800 token（DiT 文本上限的估计）。三轮都失败的题保留 greedy 输出，标为无效。

c8000 结果：greedy 有效（JSON / schema / 引用）1453 / 1452 / 1447 → 最终 1460 / 1460 / 1460。共 23 题进入修复：retry_r1 12（其中 1 题还做了规则修复）、retry_r2 4、retry_r3 1、rule_repair 1，unrecovered 5。

### 3.4 输出 = Planner JSON（Stage 3 产出 `planner_json.jsonl`）

最终目录 `s3://core-cn-oss.canva.com/usr/grantzhang/planner-9b-1025/fzsft-9b-4b3162283b0b/c8000/balanced-1500-layoutonly-repair-retry3/`：`captions.jsonl`（sha `b39abcc3…`，1465 行）、`SOURCE.json`、`REPAIR_REPORT.json`。被替换的行带 `repair_method / repair_rules / rerun_sampling / replaced_from`。

模型原始输出（`captions.caption`，[样例](examples/case0000/2_planner_raw_response.txt)）是 **plan-only inline-edit** 形式。edit 写在 `elements.layout[*].edit` 里（紧挨 `desc` 之前），每项是 `{reference_id, reference_label?, instruction}`；`reference_edits` 组里只剩 `reference_id + remove`。

送 DiT 的是反投影后的 **grouped** 形式（`captions.caption_grouped` = `planner_json`，[样例](examples/case0000/3_planner_json.json)），bbox 全部是 **XYXY、0–1000**：

```jsonc
{
  "normalized_edit_instruction": "Use the poster in <frame>0</frame> as the base. …",
  "high_level_description": "… <title block 1> … <text_unit 1> …",
  "style_description": {"aesthetics": "…", "lighting": "…", "medium": "…", "color_palette": ["#06060E", …]},
  "elements": {
    "background": "…",
    "layout": [   // Target 画布上的完整版面，父在子前
      {"type": "coarse|medium|fine|text|face|hand", "label": "<woman 1>", "parent": "ROOT", "bbox": [x1,y1,x2,y2],
       "text": "…(仅 text)", "desc": "…", "entity": "…(可选)", "color_palette": ["…"]  /* 仅被编辑的 region */}
    ],
    "reference_edits": [   // 按 Source frame 分组
      {"reference_id": "frame_0",
       "edit":   [{"label": "<woman 1>", "reference_label": "<woman photo 1>", "instruction": "Zoom into …"}],
       "remove": [{"reference_label": "<…>", "reference_bbox": [x1,y1,x2,y2], "instruction": "Remove …"}]}
    ]
  }
}
```

`edit.label` 指 Target layout 里的 region。`reference_label` 指该 frame 的 Source region label，可以是单个或列表；新增内容不写 `reference_label`。Stage 3 每行还带 `source_images、source_regions、target_size、repair_method、valid`。

## 4. DiT 输入 / 输出格式

### 4.1 Planner JSON → DiT 原生输入（Stage 4）

转换直接调用 DiT 渲染包里的函数（与 dit58k controller `prep1025.prepare` 是同一条路径）：

1. `planner_to_native(planner_json, Source region label→bbox, n)`：layout bbox 从 XYXY 换成 **YXYX**；`reference_label` 按该 frame 的 Source regions 解析成 `reference_bbox`（YXYX）；edit 移到对应 Target 元素上。
2. `adapt_native_edit_annotation`：转成训练原生 annotation。
3. `drop_color_palettes`：去掉所有 color_palette。

几何：目标画布 = GPT 目标尺寸按面积 ≤ 1024² 缩放，宽高都对齐到 32 的倍数（例：1047×1501 → 832×1216）。参考图先合成白底，再按同一规则 BICUBIC 缩放存成 PNG，文件名用 PNG 的 sha256。Planner 仍无效或转换失败的行写进 `ledger.json`。

`inputs/manifest.json`（JSON 数组，[样例行](examples/case0000/4_dit_manifest_row.json)）每行：

```jsonc
{
  "sample_id": "fzsft-c8000ball-<case_id 或其 sha256>",
  "original_sample_id": "<benchmark case_id>", "input_index": 0, "seed": 42,
  "instruction": {               // 原生 annotation（dict）
    "normalized_edit_instruction": "…", "high_level_description": "…",
    "style_description": {"aesthetics": "…", "lighting": "…", "medium": "…"},   // 无 color_palette
    "compositional_deconstruction": {
      "background": "…",
      "elements": [{"type": "coarse", "label": "<woman 1>", "parent": "ROOT", "bbox": [y1,x1,y2,x2], "desc": "…",
                    "edit": {"instruction": "…", "references": [{"reference_id": "frame_0", "reference_bbox": [y1,x1,y2,x2]}]}}],
      "remove": [{"instruction": "…", "references": [{"reference_id": "frame_0", "reference_bbox": [y1,x1,y2,x2]}]}]   // 有删除时才有
    }
  },
  "ref_relpaths": ["assets/<png sha256>.png"], "source_sha256": ["<原图 sha256>"],
  "target_width": 832, "target_height": 1216, "gpt_target_width": 1047, "gpt_target_height": 1501,
  "planner_checkpoint": 8000, "repair_method": "greedy"
}
```

### 4.2 渲染协议

协议 `euler50-cfg2-allsteps-v2`：Euler 50 步，**每一步** CFG 2（区间 [0, 1]，下界包含），seed 42，time shift 4.6，noise scale 1，EMA 权重。单机 8 rank（FSDP shard 8）；H100 用 ffa attention，`MAGI_ATTENTION_FA4_BACKEND=0`。Qwen / VAE 条件编码为 reference_only + reference KV cache。CPU 准入：超出原生 token 预算的样本先剔除（本次 3 题），其余进 GPU。

### 4.3 输出

`<work>/results/step=000003500/ema/images/<sample_id>.png`，以及同名的 `records/<sample_id>.json`（[样例](examples/case0000/5_dit_record.json)）。record 含 `checkpoint_step、weights、width/height、png_sha256、rgb_sha256、seed、guidance、steps、timeshift、sampling_protocol`（每步 CFG 序列、51 个时间点、sampler sha）、`input_manifest_sha256`、`sampling_seconds`。另有 `VERIFIED.json`：全部生成、且 8 rank 协议一致后才写。Stage 5 再写 `INDEX.json`，把 sample_id 映射回 case_id 和图片路径。

已发布：version `fzsft-balanced1500layoutonly-retry3-c8000-ditv2s3500-b39abcc3196c`（`s3://…/dit58k-planner9b-1025/ema58000-cfg2/versions/…/`，可渲 1460）。图片 1457 张在 `s3://core-cn-oss.canva.com/usr/grantzhang/fzsft-top4-candidates-20261026/c8000-sft3500/{images,records,INDEX.json}`；3 题因 caption 超过 DiT 文本预算未生成。

## 5. 评测：10分-prism-like

方案来源：[canvanauts/jimmyqiancode/10分-prism-like](https://github.com/canvanauts/jimmyqiancode/tree/main/10%E5%88%86-prism-like) @ `5a9db85`，协议 `prism-edit1500-v7-single`。

- 两个指标：**编辑指令遵循**（alignment）、**审美与画面质量**（aesthetic）。每个指标只评 1 次，给 1–10 分加一句依据；不复评、不投票、不做程序扣分或封顶。
- alignment 评委看：**原始用户指令**（skill `assets/cases.jsonl`，不是 Luna 改写）、按顺序的全部输入图、候选图、该任务类别的评审重点。aesthetic 评委只看原始指令和候选图。两边互不可见对方分数和模型身份，都不看 GT。
- 汇总：先在 20 个任务类别内各自平均，再对有分的类别等权平均；overall = 两个指标的均值。未评或缺图的题不补零。
- Tier：skill 官方是 **Flex**（background 轮询，`--mode official`）。下表数字是按用户要求用 **standard 同步**（`service_tier=default`、`background=false`，隔离版本 `prism10-v7-default-sync`，`--mode fast`）跑出来的，请求、校验和汇总都用 skill 自己的代码。两种 tier 的分数不要混在一起算。

| 系统 | 图片 | 类别覆盖 | 遵循 | 审美 | 总分 |
|---|---|---|---|---|---|
| Planner c8000 + layout-only + Luna → DiT SFT-3500 | 1457 | 20/20 | **7.933** | **8.492** | **8.213** |

结果：`s3://core-cn-oss.canva.com/usr/grantzhang/dit58k-planner9b-1025/ema58000-cfg2/prism10-v7-default-sync/{REPORT.json,results.jsonl,judgments.jsonl,raw.jsonl}`。用本仓库 vendored skill 的 `summarize()` 重算 results.jsonl 中这 1457 行，结果与上表一致。同一 Planner 输出换 DiT SFT-2500，配对 1457 题：总分 Δ −0.011，95% CI [−0.033, +0.014]，两者打平。

## 6. 复现

```bash
pipeline/00_fetch.sh
```

```bash
python3 pipeline/01_build_planner_inputs.py --out work/stage1 --verify
```

Stage 2（GPU）见 [planner/README.md](planner/README.md)。下面三步复用已发布的 c8000 结果：

```bash
python3 pipeline/03_collect_planner_json.py --out work/stage3
```

```bash
python3 pipeline/04_planner_to_dit.py --planner-json work/stage3/planner_json.jsonl --out work/stage4 --verify
```

Stage 5 要在 DiT 镜像（`pins.dit.image`）里、8 卡节点上跑：

```bash
python3 pipeline/05_render_dit.py --inputs work/stage4/inputs --run /mnt/local/dit-c8000-sft3500 --cache-dir /mnt/local/dit-cache
```

Stage 6 默认只做 dry run（准备请求、不调用 API）；加 `--execute` 才真正发出付费请求：

```bash
python3 pipeline/06_eval_prism.py --work work/prism --model-id "c8000 SFT-3500" --index /mnt/local/dit-c8000-sft3500/INDEX.json --mode fast --execute --concurrency 128
```

不传 `--index` 时，默认对已发布的 1457 张图打分。`--mode official` 走 skill 原生的 Flex 流程。

依赖：devbox 用 Python 3.11+、Pillow、AWS CLI（profile `corp_core-cn-readwrite`，可用 `AWS_PROFILE` 覆盖）；`PIPELINE_CACHE` 指定 S3 下载缓存目录（默认 `~/.cache/edit-pipeline-c8000-sft3500`）。

### 验证状态

| Stage | 本仓库内的验证 |
|---|---|
| 1 | 1465/1465 prompt 与已发布推理逐字节一致；`source-regions.json` 的 sha 与 DATASET.json 一致 |
| 3 | 1465 行，有效 1460，修复方式计数与 SOURCE.json 一致 |
| 4 | 全部 1460 行与已发布 version manifest 逐字段一致（annotation、尺寸、参考图 PNG sha256） |
| 5 | 渲染包 `render.py --dry-run` 接受 Stage 4 的输入，并解析出 SFT-3500 ckpt（154.7 GB）。脚本没有在本仓库里重新跑 GPU，逻辑照搬 dit58k pod_worker 的 CPU 准入加单次 GPU 启动；已发布的 1457 张图就是这个包、这个 ckpt 渲出来的 |
| 6 | dry run 两种 mode 都通过（请求已构造、未发出）；重算已发布结果与报告一致 |
| 2 | GPU 推理没有重跑；推理包、plan 和输出均已钉住，见 planner/README.md |

## 7. 目录

```text
pins.json                      全部 S3 位置 / ckpt / sha256 / 解码与渲染参数（单一来源）
prompts/                       Planner system prompt、训练 prompt 合约（含 USER_OUTPUT_RULES）、layout-only Source prompt
data/balanced-1500-layoutonly/ 目标尺寸、排除清单、期望 prompt hash、DATASET.json、训练 prompt preflight
data/source-json-pool-plan.json  10-14 Source JSON 池的运行计划（两路模型、解码参数）
pipeline/                      00–06 各阶段脚本 + common.py
planner/                       Stage 2 GPU 推理 / 修复的说明与参考代码快照（来自 r3-9b-pe-1023）
examples/case0000/             一题端到端样例：Source → prompt → 原始输出 → Planner JSON → DiT 输入 → record / 生成图 → 打分
vendor/                        00_fetch.sh 拉取的冻结包（gitignore）
```
