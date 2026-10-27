# Stage 2：Planner GPU 推理与 repair-retry3

这一阶段需要 Eagle（或 USW2）RTX PRO 6000 hold，由 r3-9b-pe-1023 的冻结推理包执行。本目录不重写这套代码，只记录两件事：本次实际跑的内容，以及重跑时要用的参考代码快照。`reference/` 下的文件逐字节取自 r3-9b-pe-1023 的 git 提交（`pe/` 取自 `30c6515`，与 S3 推理包里的同名文件逐字节一致；`tools/` 取自 `5fe6b98`），node_run_v2.sh 取自 S3。这些文件只供阅读，不要在本目录直接运行，原因见文末。

## 本次实际运行（2026-10-26）

| 项目 | 值 |
|---|---|
| 推理包 | `s3://core-cn-oss.canva.com/usr/grantzhang/runbooks/editparser-r3-9b-pe-rtx8-20261023/packages/pe-30c6515-d408ce87….tar.gz`（`pins.planner.inference_package`） |
| plan | [reference/pe/plans/plan-8000-balanced-1500-layoutonly-fzsft-9b.json](reference/pe/plans/plan-8000-balanced-1500-layoutonly-fzsft-9b.json)：dataset `balanced-1500-layoutonly`，lineage `fzsft-9b`，hold `grantzhang-rtx64-r12pe-eagle-1025a` 4 节点 32 卡，batch 64，max_model_len 32768 |
| 启动 | 每个节点执行 `tools/launch.sh <ctx> <pod> 8000 <group node> <package> <sha> 1 plan-8000-balanced-1500-layoutonly-fzsft-9b`。它在 pod 内拉 `node_run_v2.sh`，再依次做：校验包 → 放 lineage 文件和数据集 → 解 training overlay → `preflight_dataset.py` → `controller.py --node N` |
| 导出 | leader 把 mcore c8000（HSDP 128/16）导出为 HF trained-MTP 模型，841 个模型键全部回读校验，发布到 `<output>/models/checkpoint-8000/`。之后同一 ckpt 的其他数据集都用 `reuse_export_s3` 直接复用这份导出 |
| gate | TOKEN_TEMPLATE / GPU_PRECISION / VISION_REFERENCE / GDN_REFERENCE / ENGINE_SMOKE / TRAINING_ALIGNMENT_COMPARE 全部 PASS，见 `<output>/provenance/`；1465/1465 条 prompt 都通过对齐校验（`prompt_alignment_verified_records`） |
| greedy 输出 | `s3://core-cn-oss.canva.com/usr/grantzhang/balanced-1500-layoutonly-editparser-fzsft-9b-4b3162283b0b-c8000-trainprompt-greedy-rtx8-bs64-20261026/checkpoint-8000/{captions.jsonl, responses.jsonl.gz, failed.jsonl, overlength.jsonl, pending.jsonl, SUMMARY.json}` |
| repair-retry3 | `tools/indomain_fix.py run`，环境变量 `FIX_LINEAGE=fzsft-9b FIX_MULTI=1 FIX_STEPS=8000 FIX_DATASETS=balanced-1500-layoutonly,…`；重试包 `pe-5fe6b98-dcea0833….tar.gz`，plan `reference/pe/plans/…-retry-r123-multi.json`；三轮采样合在一次节点运行里，输出在 `…-c8000-trainprompt-retry-r123-multi-rtx8-20261026/round-{1,2,3}/` |
| 最终发布 | `s3://core-cn-oss.canva.com/usr/grantzhang/planner-9b-1025/fzsft-9b-4b3162283b0b/c8000/balanced-1500-layoutonly-repair-retry3/{captions.jsonl, SOURCE.json, REPAIR_REPORT.json}` |

`responses.jsonl.gz` 每行保留了完整的 `system_prompt / user_prompt / prompt_sha256 / source_regions / model_image_sizes / decoding / response / finish_reason / output_tokens`。所以 Stage 1 可以离线逐字节核对 prompt，Stage 3 / 4 也不需要重新推理。

## 参考代码

| 文件 | 作用 |
|---|---|
| `reference/pe/sourcejson_contract.py` | `user_prompt()`：Source 帧、`<source_regions>`、`<user_prompt>`、`<aspect_ratio>` 的拼接（Stage 1 按它复刻） |
| `reference/pe/r3_prompt_contract.json` | 从训练代码冻结出的 SYSTEM / USER_OUTPUT_RULES / SPEC（prompt `eecfcbe5…`，system `6a4ef8df…`） |
| `reference/pe/training_image.py` | 图像预处理：白底合成、swift rescale、qwen_vl_utils smart_resize |
| `reference/pe/worker.py`、`controller.py`、`engine_config.py` | 单 rank 推理（构造 prompt 并断言其 hash、vLLM 生成、校验）；多节点编排、导出、gate、发布；引擎参数 |
| `reference/pe/contract_utils.py` | `plan_only_to_grouped()`：inline 输出反投影为 grouped；`validate_prediction()`：schema 与 Source 引用校验 |
| `reference/pe/rerun_subset.py` | 复用 HF 导出、只对指定子集重跑（repair-retry3 的重试轮用它） |
| `reference/tools/indomain_fix.py` | repair-retry3：规则修复 `repair_json()` / `repair()`，3 轮采样参数 `ROUNDS`，合格判定，发布 `<dataset>-repair-retry3/` |
| `reference/tools/plan_lib.py` | `make_plan()`：按 hold / 节点 / lineage / dataset 生成 plan，输出写到新的空前缀 |
| `reference/tools/launch.sh`、`node_run_v2.sh` | 启动脚本：在 pod 内拉包、做 preflight、启动 controller |
| `reference/tools/fzsft_lo_run.sh` | 当时 layout-only 两个数据集「greedy → repair-retry3 → organize」的自动化 |
| `reference/tools/make_layoutonly_datasets.py` | 用 layout-only Source JSON 池生成 `pe/datasets/balanced-1500-layoutonly/`（本仓库 `data/` 中各文件的来源） |
| `reference/tools/hold64_fzsft_lo.yaml` | RTX hold 模板（high-priority，GPU keepalive） |

## 重跑 / 换 ckpt

在 r3-9b-pe-1023 工作区（devbox `/home/coder/work/zhangguiwei/r3-9b-pe-1023`，`tools/` 与 `pe/` 是同一个 git 仓库）里操作：

1. 申请 hold：`arnold submit -f tools/hold64_fzsft_lo.yaml --cluster p-iad-eagle --application-id core-cn --namespace b-core-cn-prod --priority high --job-name <name> --project-id craft --no-watch`。
2. 新 ckpt：把 `checkpoint-<step>/_SUCCESS` 存为 `pe/checkpoint-fzsft-9b-<step>-SUCCESS.json`，它指向的 manifest 存为 `-manifest.json`。
3. 生成 plan：`make_plan(step, hold, uid, ctx, workers, lineage='fzsft-9b', dataset='balanced-1500-layoutonly', tag='<新 tag>')`。**tag 必须是新的**，保证输出写到一个空前缀。同一 ckpt 已有 HF 导出时，传 `reuse_export_s3=<那次运行的前缀>` 跳过导出。
4. 提交 plan，`git archive --prefix=pe/ HEAD:pe` 打包，上传后回读校验 sha256，再在每个节点上执行 `tools/launch.sh …`。
5. 发布后跑 repair-retry3：`FIX_LINEAGE=fzsft-9b FIX_MULTI=1 FIX_STEPS=<step> FIX_DATASETS=balanced-1500-layoutonly FIX_HOLD=<hold> FIX_WORKERS=<a,b> FIX_DEST=<新的结果根> FIX_WORK=<新状态目录> python3 tools/indomain_fix.py run`。
6. 本仓库 `pipeline/03_collect_planner_json.py --planner-dir <FIX_DEST>c<step>/balanced-1500-layoutonly-repair-retry3/` 接后续阶段。

注意：

- 不要拿已有 c8000 的 tag 再跑一遍。`tools/fzsft_ds_run.sh 8000 balanced-1500-layoutonly …` 算出来的输出前缀正是已发布的那个，会覆盖它的拷贝。
- `indomain_fix.py` 会在它所在的 git 仓库里提交 plan（`TOOLS.parent`），所以只能在 r3-9b-pe-1023 里运行，不要从本仓库的 `reference/` 运行。
