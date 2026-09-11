# 上游合并冲突记录（2026-09-11）

本次把上游 12 个提交（`HEAD..MERGE_HEAD`，merge-base `7de4a95`）合入本地分支
`merge-upstream-20260911`。总原则：**功能相似的冲突以本地代码为准**，只在明确决策处吸收上游。

本文档逐条记录**所有与本地代码冲突的取舍点**，用于后续测试出问题时**按条回退到本地模式**。
每条给出：位置、上游做法、本地做法、当前采用、回退办法。

上游提交清单：

| commit | 说明 |
| --- | --- |
| `ed3cdb0` | r2v 参考图最长边 1024/1280/1536 |
| `68d4ee6` | 一采接缝光色对齐上一段一采 |
| `14fdcd6` | Refine 改画幅后引导钉同尺寸 latent |
| `075ef83` | 引导+重绘接缝跳变，重绘幅度允许 0、默认 0.1 |
| `5fede41` | Refine 二采可选空间分块 |
| `d19f338` | H3 latent 放大可选时间分块 |
| `b312f08` | 实时预览改循环 WebP，去掉成片整段 JPEG 回放 |
| `f1f68bc` | 参考音频保持 lazy，标签只对齐解码成功的槽 |
| `531adba` | Merge PR #128 参考音频音色 |
| `d0c3713` | 段间引导增加「保完整」开关 |
| `3cea821` | Linux 分段导出 stdin flush 空目录修复 |
| `aa5df2b` | 参考音频音色丢失/错位（`<Audio N>` 与槽位对齐） |

---

## 一、四项路线决策（用户拍板）

### 决策 1：引导+重绘 —— 保留本地结构，只吸收 `select_continuity_pin_latent`

- **上游（`075ef83`）**：认为「引导+重绘」接缝跳变，把重绘幅度下限放开到 `0.0`、默认降到 `0.10`，
  并在二采里用 `_relock_continue_refine` 做 continue re-lock。
- **本地**：结构是「先注入官方 Guide 关键帧 → 再叠加重绘掩码」，重绘幅度默认 `0.65`、下限 `0.40`；
  二采丢弃首遍重绘掩码、只做 Guide re-pin。
- **当前采用**：本地结构 + 本地数值；吸收上游 `14fdcd6` 的 `select_continuity_pin_latent`。

| 文件 | 现状 | 上游值 | 回退到上游的改法 |
| --- | --- | --- | --- |
| [director/h3_latent_continue.py:45-47](../director/h3_latent_continue.py#L45-L47) | `SEAM_MIN_MASK = 0.65` / `SEAM_FLOOR_MIN = 0.40` / `SEAM_FLOOR_MAX = 0.95` | `0.10` / `0.0` / `0.95` | 改这三个常量 |
| [director/segment_continuity.py:152](../director/segment_continuity.py#L152) | `DEFAULT_CONTINUITY_REDRAW = 0.65` | `0.10` | 改常量 |
| [web/js/minimax_timeline.js](../web/js/minimax_timeline.js) | `DEFAULT_CONTINUITY_REDRAW = 0.65`、`MIN_CONTINUITY_REDRAW = 0.40`，滑杆 `min="0.40" max="0.95" step="0.05" value="0.65"` | `0.10` / `0.0`，滑杆 `min="0"` | 同步改 JS 常量与 input 属性 |
| [web/js/minimax_i18n.js](../web/js/minimax_i18n.js) | `continuityRedraw` 文案写「0.40–0.95，默认 0.65」 | 上游文案「0–0.95，默认 0.1」 | 换回上游文案 |

**吸收的上游改动**（`14fdcd6`）——[director/executor_core.py](../director/executor_core.py)：

```python
# 【合并取舍·上游吸收】Refine 改过画幅时，用上一段「一采」AV 钉同尺寸 latent，
# 避免像素回编糊头、官方引导二采崩溃（上游 14fdcd6）。
prev_av = select_continuity_pin_latent(latent, prev_first_pass_av, prev_av)
```

> 回退：删掉这一行，直接用 `prev_av`。

**未启用的上游代码**——[director/refine_sampling.py](../director/refine_sampling.py)：

```python
# 【合并取舍】本地路线：引导+重绘的连续性由官方 Guide 关键帧提供，二采丢弃首遍
# 重绘掩码、只做 Guide re-pin，因此不启用上游的 continue re-lock（_relock_continue_refine
# 保留未调用，回退到上游路线时把下面这行改回上游的 re-lock 分支即可）。
continue_after_shift = None
```

> `_relock_continue_refine()` 函数体已完整保留但未被调用。回退时把 `continue_after_shift`
> 改回上游的 `_relock_continue_refine(...)` 调用即可。

另：上游在二采里用变量名 `guide_pin`，本地路线用 `pin_frames`；合并后统一为 `pin_frames`
（上游自动合入的下游代码里 4 处 `guide_pin` 已全部改名，否则 NameError）。

### 决策 2：实时预览 —— 上游 WebP + 本地开关/音频预览/成片回放

- **上游（`b312f08`）**：多帧循环 WebP 编码，**删除**了成片整段 JPEG 回放和
  `segment_runtime.tensor_frame_to_jpeg_b64`。
- **本地**：单帧 JPEG 预览，但有运行时开关、音频预览、节点内成片整段回放。
- **当前采用**：融合。

[director/executor_core.py `_report_step_preview`](../director/executor_core.py)：

```python
pv = get_preview(node_id)
# 【合并融合】上游多帧循环 WebP 编码 + 本地运行时开关；不要在这里 return，
# 否则下面的音频预览会被跳过。
if pv["tae"]:
    ...  # x0_to_preview_frames + encode_preview_payload（上游）
# （本地音频预览块紧随其后，未改动）
```

- [director/segment_runtime.py](../director/segment_runtime.py)：**恢复**被上游删掉的
  `tensor_frame_to_jpeg_b64()` 以及 `base64` / `io` / `PIL.Image` 导入，供本地成片回放使用。
- [director/executor_core.py](../director/executor_core.py) 导入块重新加回 `tensor_frame_to_jpeg_b64`。

> 回退到上游纯 WebP：删掉 `_report_step_preview` 里的音频预览块与成片整段回放块，
> 并把 `tensor_frame_to_jpeg_b64` 及其导入一并移除。

### 决策 3：段间引导「保完整」—— 保留本地无条件保完整，不要上游开关

- **上游（`d0c3713`）**：新增 `continuity_keep_tail` 开关（默认 on），UI + plan 字段 + 缓存指纹。
- **本地**：无条件保留对齐余帧。
- **当前采用**：本地。已删除的上游内容：
  - `DirectorPlan.continuity_keep_tail` 字段（[director/plan.py](../director/plan.py)）
  - `resolve_continuity_keep_tail` 的**调用**（[director/external_groups.py](../director/external_groups.py)、
    [director/fl2v_timeline.py](../director/fl2v_timeline.py)、[director/gen_timeline.py](../director/gen_timeline.py)）
  - [director/segment_cache.py](../director/segment_cache.py) 指纹里的 keep_tail 键
  - [web/js/minimax_timeline.js](../web/js/minimax_timeline.js) 里 `isContinuityKeepTail()` 及约 25 处 UI 引用
  - [web/js/minimax_i18n.js](../web/js/minimax_i18n.js) 中英文 keep-tail 文案

- **保留但已失效的残留**（无害，回退时可复用）：
  - [director/segment_continuity.py:192](../director/segment_continuity.py#L192) `resolve_continuity_keep_tail()` 函数体仍在，但无人调用
  - [director/h3_motion_context.py:772](../director/h3_motion_context.py#L772) `continuity_export_len(keep_tail=...)` 形参仍在
  - 调用处一律 `keep_tail=bool(getattr(plan, "continuity_keep_tail", True))`，因 plan 无该字段而恒为 `True` = 本地行为
  - [web/js/minimax_refine.js:327](../web/js/minimax_refine.js#L327) 只剩一条 `continuity_keep_tail: "保完整"` 词条

> 回退到上游：给 `DirectorPlan` 加回 `continuity_keep_tail: bool = True`，三处 timeline 解析加回
> `resolve_continuity_keep_tail(timeline)`，缓存指纹加回该键，JS 恢复开关 UI。

另：`export_len` 两处均保留本地算法
`int(sample_len) - int(trim_frames) if trim_frames > 0 else int(target_len)`（上游改用 keep_tail 分支）。

### 决策 4：参考音频 —— 采用上游 lazy + 槽位对齐 ⚠️ 重点观察项

> 用户明确要求：**先用上游方案，测试出问题就改回本地模式**。这是本次风险最高的一条。

- **上游（`aa5df2b` / `f1f68bc` / `531adba`）**：文件型参考音频**懒解码**，`<Audio N>` 标签只对齐
  「解码成功」的槽位，段跑完释放该段 PCM。
- **本地**：`_load_ref_audios` 时**立即解码**全部音频并常驻。

采用的上游实现：

| 文件 | 内容 |
| --- | --- |
| [director/plan.py](../director/plan.py) | `SegmentRefAudio.audio: dict \| None = None`（上传文件走 lazy） |
| [director/plan.py](../director/plan.py) | `_load_ref_audios` 只填 `audio=None` + `audio_path`，不解码 |
| [director/plan.py](../director/plan.py) | 新增 `ensure_ref_audio_pcm(item, *, cache=None)`、`ref_audios_to_dict(audios, *, cache=None)` |
| [director/plan.py](../director/plan.py) | `usable_ref_audio_indices(audios, *, cache=None)`、`drop_unusable_audio_prompt_tags(...)` |
| [director/plan.py:314](../director/plan.py#L314) | `DirectorPlan.audio_decode_cache: dict`（执行级解码缓存） |
| [director/executor_core.py:290](../director/executor_core.py#L290) | `_release_segment_file_ref_audios(plan, seg)` |
| [director/executor_core.py:247,273](../director/executor_core.py#L247) | 两处 `ref_audios_to_dict(..., cache=getattr(plan, "audio_decode_cache", None))` |
| [director/executor_core.py:824,838](../director/executor_core.py#L824) | r2v / rv2v 两处 `usable_ref_audio_indices(..., cache=...)` + `drop_unusable_audio_prompt_tags` |
| [director/executor_core.py](../director/executor_core.py) | `_run_one_segment` 调用包 `try/finally: _release_segment_file_ref_audios(plan, seg)` |

**回退到本地「全量预解码」的做法**（测试若出现音色丢失、音频缺失、`<Audio N>` 错位、
重复解码卡顿等问题）：

1. `_load_ref_audios` 改回本地版：构造 `SegmentRefAudio` 时就 `load_audio_file(path)` 填 `audio=`；
2. `ensure_ref_audio_pcm` 退化为 `return item.audio`（或直接删掉，`ref_audios_to_dict` 不再懒解码）；
3. 删除 `_release_segment_file_ref_audios` 的 `try/finally` 包裹（否则第 2 段起 PCM 被清空且不再重解码）；
4. `usable_ref_audio_indices` / `drop_unusable_audio_prompt_tags` 可保留（槽位对齐本身是修 bug，
   与懒解码正交），若怀疑标签被误删再改回「按 `seg.ref_audios` 原顺序全量出标签」。

> 注意：第 3 步与第 1 步必须**成对**改。只回退解码时机而保留释放逻辑，会导致后续段音频为空。

---

## 二、其余冲突取舍（非四项决策）

| 位置 | 上游 | 本地 | 采用 | 回退办法 |
| --- | --- | --- | --- | --- |
| [director/core_sampling.py](../director/core_sampling.py) 采样入参 | 新增 `enable_tiling` / `tile_count` / `tile_overlap`（`5fede41`） | 有 `shifted_model` / `on_loaded` | **并集**：两边参数都留 | 删除 tiling 三参及 `wrap_sampler_spatial_tiles` 包装 |
| [director/refine_sampling.py](../director/refine_sampling.py) 二采签名 | `prev_refine_av` / `prev_end_frame` / `prev_tail`（`68d4ee6` 光色对齐） | `on_loaded` | **并集** | 删除上游三参及其在 sample 调用中的传递 |
| [director/executor_core.py](../director/executor_core.py) 段缓存条件 | `if (will_refine or continuity_active) and not skip_first_sample:` | 本地旧条件 | **上游** | 换回本地条件 |
| [director/executor_core.py](../director/executor_core.py) 工作集裁剪 | 无 | `_prune_continuity_working_set(...)` | **本地保留** | — |
| [director/segment_cache.py](../director/segment_cache.py) | 纳秒级 mtime | 秒级 | **上游** | 换回秒级 mtime |
| [director/segment_cache.py](../director/segment_cache.py) 指纹 | 含 keep_tail | 含 `stable_seg_id(seg)` | **本地 seg_id + 去掉 keep_tail 键** | 见决策 3 |
| [lib/video_export.py](../lib/video_export.py) | ffmpeg stdin 捕获 `BrokenPipeError`/`ValueError`，`proc.returncode not in (0, None)`（`3cea821`） | 旧写法 | **上游** | 换回本地 stdin 写法 |
| [nodes/director_common.py](../nodes/director_common.py) | 上游判断 | `if getattr(plan, "continuity_enabled", False):` | **本地** | — |
| [director/external_groups.py](../director/external_groups.py) 导入 | `usable_ref_audio_indices` | `resolve_seg_id` | **并集** | — |
| [director/external_groups.py](../director/external_groups.py) / [fl2v_timeline.py](../director/fl2v_timeline.py) / [gen_timeline.py](../director/gen_timeline.py) | keep_tail 解析 | `resolve_audio_continuity_enabled` + `exposure_anchor_*` | **本地** | 见决策 3 |
| [director/plan.py](../director/plan.py) `DirectorPlan` 字段 | 无 | `continuity_redraw=0.65`、`audio_continuity_enabled=True`、`exposure_anchor_enabled=True`、`exposure_anchor_strength=0.40` | **本地全保留** | — |
| [director/h3_motion_context.py](../director/h3_motion_context.py) 管线 ID | — | `CONTINUITY_PIPELINE_ID = "minimax_h3_motion_context_v9"` 两侧注释均保留 | **并集注释** | — |
| [web/js/minimax_i18n.js](../web/js/minimax_i18n.js) `refImageSize` | 新 1024/1280/1536 文案（`ed3cdb0`） | 旧文案 | **上游** | 换回旧文案 |
| [web/js/minimax_i18n.js](../web/js/minimax_i18n.js) `exportMode` | 上游文案 | 「选择导出」 | **本地** | — |
| [web/js/minimax_i18n.js](../web/js/minimax_i18n.js) `liveTaePreview` | WebP 描述 | 开关描述 | **融合** | — |

上游**纯新增、无冲突**（直接并入，默认关闭，风险低）：
[director/spatial_tiled_sampling.py](../director/spatial_tiled_sampling.py)（新文件）、
[director/h3_latent_upscale.py](../director/h3_latent_upscale.py) 时间分块、
[nodes/director_refine.py](../nodes/director_refine.py) 分块参数、
[lib/image_prep.py](../lib/image_prep.py) 长边工具。

---

## 三、合并过程中修掉的「半自动合并」坏点

git 自动合并留下的不一致，已修正，回退时注意别再踩：

1. **`gen_timeline.py` 重复关键字**：本地已在第 710 行传 `global_ref_audios=shared_ref_audios`，
   上游同名参数再加一次 → duplicate keyword argument。已删除新增行。
2. **`refine_sampling.py` `guide_pin` 未定义**：自动合入的上游下游代码引用 `guide_pin`，
   而本地冲突侧只定义 `pin_frames`。4 处已改名。
3. **`minimax_timeline.js` `isContinuityKeepTail` 未定义**：函数定义按决策 3 删除，
   但约 25 处自动合入的调用点残留。已全部清理，`node --check` 通过。
4. **`tensor_frame_to_jpeg_b64` 未定义**：上游删函数、本地保留调用。已在
   `segment_runtime.py` 恢复函数与导入。
5. **`plan.py` 参考音频半新半旧**：上游 `usable_ref_audio_indices` 调用了不存在的
   `ensure_ref_audio_pcm`（因为 git 保留了本地的即时解码 `_load_ref_audios`）。
   按决策 4 整块采用上游 lazy 实现。
6. **行尾被脚本改坏**：批量改写脚本把 6 个原本 CRLF 的文件（`h3_latent_continue.py`、
   `h3_motion_context.py`、`plan.py`、`refine_sampling.py`、`segment_continuity.py`、
   `segment_runtime.py`）转成了 LF，导致整文件 diff。已全部还原为 CRLF，与本地一致。
7. **`_prune_continuity_working_set` 参数不匹配**（真机验证 2026-09-11 第一轮暴露）：
   合并给调用点加了 `completed_first_pass_av`（上游 `14fdcd6` 的一采 AV 工作集）成为 4 参调用，
   但函数定义仍是本地 3 参 → `TypeError: takes 3 positional arguments but 4 were given`。
   已把函数扩为 4 参并将 first_pass_av 一并按「只留 N-1」规则裁剪。

---

## 四、验证状态

- 44 处冲突标记全部消除，`grep '^<<<<<<<'` 无命中
- 全部 `.py` 通过 `ast.parse`
- 全部 `web/js/*.js` 通过 `node --check`
- `pyflakes`（装在项目内 `./tmp/pylibs`）无未定义名，仅剩合并前就有的未使用导入/变量告警
- **尚未做任何运行时测试**——需先确认测试环境

## 五、测试重点（按风险排序）

1. **参考音频**（决策 4）：多段 + 多音频槽，检查音色是否串、`<Audio N>` 是否对齐、
   第 2 段起音频是否还在、是否反复重解码卡顿。出问题按决策 4 的四步回退。
2. **引导+重绘**：默认 0.65 下接缝是否跳变；Refine 改画幅后走 `select_continuity_pin_latent`
   是否正常（本地 0.65 + 上游钉图逻辑是首次组合）。
3. **实时预览**：WebP 循环预览 + 音频预览 + 成片整段回放三者是否同时工作、开关能否运行中切换。
4. **段缓存**：nanosecond mtime + `stable_seg_id` 指纹换代后，旧缓存应整体失效而非误命中。
5. **分段导出**（`3cea821`）：Linux 下不再出空目录。
