"""Release GPU memory between MiniMax H3 Director segment runs."""

from __future__ import annotations

import gc
import logging

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.vram")

# 守卫是否已安装，避免重复包装 mm.cleanup_models_gc
_guard_installed = False


def _evict_dead_loaded_models() -> int:
    """Pop Comfy LoadedModel slots that ``free_memory`` will skip forever.

    ``is_dead()`` means the ModelPatcher weakref is gone while the shared
    MiniMaxH3 module is still alive (graph MODEL). Those slots would otherwise
    log ``WARNING, memory leak with model MiniMaxH3`` and then sit in
    ``current_loaded_models``, so later unloads cannot touch them.
    Evicting the slot does not copy weights; it restores unload bookkeeping.
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


def install_dead_slot_guard() -> bool:
    """在 ``mm.cleanup_models_gc`` 执行前先驱逐死槽。

    ComfyUI 核心在 ``free_memory`` / ``load_models_gpu`` 开头都会调用
    ``cleanup_models_gc``，频率很高。只要死槽存在，核心扫描时就会打印
    memory leak WARNING。包装后让死槽在被检测到之前就被 pop 掉，
    从而抑制刷屏；只动槽位簿记，不碰模型权重。
    """
    global _guard_installed
    if _guard_installed:
        return True
    try:
        import comfy.model_management as mm
    except Exception:
        return False

    original = getattr(mm, "cleanup_models_gc", None)
    if original is None or getattr(original, "_director_dead_slot_guard", False):
        _guard_installed = True
        return True

    def guarded_cleanup_models_gc():
        # 先回收 + 驱逐死槽，使原函数扫描时 current_loaded_models 中不存在死槽
        _evict_dead_loaded_models()
        return original()

    guarded_cleanup_models_gc._director_dead_slot_guard = True
    mm.cleanup_models_gc = guarded_cleanup_models_gc
    _guard_installed = True
    log.info("MiniMax H3 Director: dead LoadedModel guard installed")
    return True


def cleanup_segment_vram(*, enabled: bool = True, unload_models: bool = True) -> None:
    """Release segment GPU memory: gc, optional unload of ComfyUI models, empty CUDA cache."""
    if not enabled:
        return
    gc.collect()
    try:
        import comfy.model_management as mm

        # 关键顺序：先驱逐死槽，再调用任何会触发 cleanup_models_gc 的流程，
        # 否则死槽在被清理前就已先打印 WARNING。
        _evict_dead_loaded_models()
        if unload_models:
            mm.unload_all_models()
            mm.cleanup_models()
        _evict_dead_loaded_models()
        gc.collect()
        mm.soft_empty_cache()
    except Exception as exc:
        log.warning("Segment VRAM cleanup failed: %s", exc)
        return
    if unload_models:
        log.debug("MiniMax H3 Director: segment VRAM cleanup (models unloaded, cache cleared)")
    else:
        log.debug("MiniMax H3 Director: segment VRAM cleanup (cache cleared, models kept loaded)")
