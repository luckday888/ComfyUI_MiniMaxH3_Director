"""段间 / 运行结束的模型与显存回收（MiniMax H3 Director）。

关键约束：不能在 ModelPatcher 已被 Python GC 之后再做模型清理。ComfyUI 的
``current_loaded_models`` 用弱引用登记 ModelPatcher；一旦临时 clone patcher
（SigmaShift / remask）被 GC、而底层 MiniMaxH3 仍被节点输入的主 MODEL 持有，
该条目就变成 ``is_dead()``——``free_memory`` 跳过它、无法卸载，
``cleanup_models_gc`` 还会在每次加载时刷 "memory leak with model MiniMaxH3"
警告并逐段堆积。

因此这里：
- 卸载必须在 patcher 仍被调用方强引用（存活）时，走官方
  ``unload_model_and_clones`` / ``unload_all_models`` 正常摘除登记条目；
- 不主动调用 ``cleanup_models_gc``（它只负责把死条目刷成警告）；
- ``gc.collect()`` 放到官方摘除条目之后，避免先回收 patcher 造出 is_dead 死条目；
- ``_evict_dead_loaded_models`` 仅作历史遗留死条目的兜底摘除。

模型 patcher 的跨段 / 跨运行保活由 ``core_sampling.ShiftedModelCache``（进程级）
与执行器里的滚动 ``on_loaded`` 槽负责，本模块只负责“存活期卸载”与收尾。
"""

from __future__ import annotations

import gc
import logging

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.vram")


def _dedupe_models(models) -> list:
    """按对象身份去重，过滤 None，保持顺序。"""
    seen: set[int] = set()
    out: list = []
    for model in models or ():
        if model is None:
            continue
        key = id(model)
        if key in seen:
            continue
        seen.add(key)
        out.append(model)
    return out


def _evict_dead_loaded_models() -> int:
    """兜底摘除 ``free_memory`` 永远跳过的 is_dead LoadedModel 条目。

    ``is_dead()`` 意味着 ModelPatcher 弱引用已死、而共享的 MiniMaxH3 权重仍活
    （图 MODEL / Sage 循环持有）。这种条目无法卸载，还会触发 memory leak 警告。
    正常路径下本函数应一无所获——它只是历史遗留死条目的清道夫。
    """
    try:
        import comfy.model_management as mm
    except Exception:
        return 0
    models = getattr(mm, "current_loaded_models", None)
    if not models:
        return 0
    evicted = 0
    for i in range(len(models) - 1, -1, -1):
        cur = models[i]
        try:
            if not cur.is_dead():
                continue
            name = "?"
            try:
                real = cur.real_model()
                name = type(real).__name__ if real is not None else "?"
            except Exception:
                pass
            models.pop(i)
            evicted += 1
            log.info("MiniMax H3 Director: evicted dead LoadedModel slot (%s)", name)
        except Exception:
            continue
    return evicted


def cleanup_segment_vram(
    *,
    enabled: bool = True,
    unload_models: bool = True,
    models=(),
) -> None:
    """段间回收：在 patcher 存活时经官方路径卸载模型并清空 CUDA 缓存。

    ``models`` 传入“当前仍被调用方强引用”的 patcher（持久 SigmaShift clone 与
    最近一次采样的加载 patcher）。它们存活时，``unload_model_and_clones`` 会按
    ``clone_base_uuid`` 把同一权重族（含 shift / remask clones）的登记条目正常
    摘除并卸载，不留 is_dead 死条目。旧版 ComfyUI 无该 API 时回退
    ``unload_all_models``。
    """
    if not enabled:
        return
    targets = _dedupe_models(models)
    try:
        import comfy.model_management as mm

        if unload_models:
            unload_fn = getattr(mm, "unload_model_and_clones", None)
            if targets and callable(unload_fn):
                for model in targets:
                    try:
                        unload_fn(
                            model,
                            unload_additional_models=True,
                            all_devices=False,
                        )
                    except Exception as exc:
                        log.debug("unload_model_and_clones skipped for %r: %s", model, exc)
            else:
                mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception as exc:
        log.warning("Segment VRAM cleanup failed: %s", exc)
        return
    # 历史遗留死条目兜底摘除（正常应为 0），再在条目摘除后回收垃圾，
    # 避免先 GC 出 is_dead 死条目。
    _evict_dead_loaded_models()
    gc.collect()
    if unload_models:
        log.debug("MiniMax H3 Director: segment VRAM cleanup (models unloaded, cache cleared)")
    else:
        log.debug("MiniMax H3 Director: segment VRAM cleanup (cache cleared, models kept loaded)")


def restore_persistent_model_registration(persistent, last_loaded) -> None:
    """运行收尾：把 H3 的“当前已加载模型”登记交还给跨运行存活的持久 shift clone。

    末段若是段间引导+重绘，最后加载的是一次性 remask clone（被执行器滚动槽强引用、
    此刻仍存活）。这里 ``load_models_gpu([persistent])`` 会在该 remask clone 仍存活
    时经官方 ``is_clone`` 清扫正常摘除其登记，并复用同一份已在显存的权重（不采样、
    不读盘、近乎空转）。这样本函数返回、滚动槽随栈帧销毁后，
    ``current_loaded_models`` 里只剩跨运行存活的持久 patcher，零 is_dead 残留，
    下一次运行不会再刷 MiniMaxH3 memory leak 警告。
    """
    if persistent is None or last_loaded is None or last_loaded is persistent:
        return
    try:
        import comfy.model_management as mm

        mm.load_models_gpu([persistent])
        log.debug("MiniMax H3 Director: restored loaded-model registration to persistent shift clone")
    except Exception as exc:
        log.debug("Persistent model registration restore skipped: %s", exc)
