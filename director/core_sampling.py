"""Single-stage sampling via official MiniMax H3 custom-sampler nodes.

Matches ``video_minimax_h3_r2v.json``:
MiniMaxH3SigmaShift → BasicScheduler → BasicGuider (or CFGGuider) →
KSamplerSelect → RandomNoise → SamplerCustomAdvanced.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Callable

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.core_sampling")

PhaseCallback = Callable[[str, float], None]
StepPreviewCallback = Callable[[int, int, Any], None]
StepStateCallback = Callable[[int, int, Any, Any], None]
LoadedModelCallback = Callable[[Any], None]


class _FixedNoise:
    """Resume helper: return a precomputed NestedTensor / tensor as sampler noise."""

    def __init__(self, noise, seed: int = 0) -> None:
        self.seed = int(seed or 0)
        self._noise = noise

    def generate_noise(self, input_latent):
        del input_latent
        return self._noise


class _ZeroNoise:
    """Resume helper: SamplerCustomAdvanced still calls generate_noise()."""

    def __init__(self) -> None:
        self.seed = 0

    def generate_noise(self, input_latent):
        import torch

        samples = input_latent["samples"] if isinstance(input_latent, dict) else input_latent
        # torch.Tensor.unbind splits the batch axis — only NestedTensor is AV streams.
        if torch.is_tensor(samples):
            return torch.zeros_like(samples)
        if getattr(samples, "is_nested", False) and hasattr(samples, "unbind"):
            parts = tuple(torch.zeros_like(p) for p in samples.unbind())
            try:
                import comfy.nested_tensor

                return comfy.nested_tensor.NestedTensor(parts)
            except Exception:
                try:
                    return type(samples)(parts)
                except Exception:
                    return parts
        if isinstance(samples, (tuple, list)):
            return type(samples)(torch.zeros_like(p) for p in samples)
        raise TypeError(f"Cannot build zero noise for {type(samples)!r}")


def _unpack_node_output(out):
    if hasattr(out, "args"):
        args = out.args
        if args:
            return args
    if isinstance(out, (tuple, list)):
        return out
    raise RuntimeError(f"Unexpected node output type: {type(out)!r}")


def _use_basic_guider(cfg: float, negative) -> bool:
    """Official r2v template uses BasicGuider (no CFG)."""
    if negative:
        return False
    return abs(float(cfg) - 1.0) < 1e-6


# ── 进程级 SigmaShift clone 复用 ─────────────────────────────────────────────
# 官方 MiniMaxH3SigmaShift.execute 每次都 model.clone()（共享底层 MiniMaxH3 权重，
# 仅叠加 model_sampling 补丁）。clone 一旦失去全部强引用被 GC，而底层权重仍被
# 节点主 MODEL 持有，ComfyUI 的登记条目即变 is_dead：free_memory 跳过、无法卸载，
# cleanup_models_gc 还会在每次加载刷 "memory leak with model MiniMaxH3"。
# 因此缓存为进程级、跨运行存活（不复制权重、不重读盘），LRU 上限防无限堆积。
_SHIFT_CLONE_STORE: "OrderedDict[tuple, Any]" = OrderedDict()
_SHIFT_CLONE_MAX = 16


class ShiftedModelCache:
    """按主 MODEL + shift 参数复用同一个 SigmaShift clone（进程级，跨运行存活）。

    实例自身不持有条目——所有实例共享模块级 ``_SHIFT_CLONE_STORE``，因此一次
    执行结束**不得**清空缓存：清空即让 clone 失去最后强引用，退化为 is_dead
    死条目（"memory leak with model MiniMaxH3" 假告警的来源）。
    """

    def __init__(self) -> None:
        self._items = _SHIFT_CLONE_STORE

    def get(self, model, shift_video: float, shift_audio: float):
        key = (id(model), round(float(shift_video), 6), round(float(shift_audio), 6))
        hit = _SHIFT_CLONE_STORE.get(key)
        if hit is not None:
            _SHIFT_CLONE_STORE.move_to_end(key)
            return hit
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

        shifted = MiniMaxH3SigmaShift.execute(model, float(shift_video), float(shift_audio))
        model_use = _unpack_node_output(shifted)[0]
        _SHIFT_CLONE_STORE[key] = model_use
        _SHIFT_CLONE_STORE.move_to_end(key)
        while len(_SHIFT_CLONE_STORE) > _SHIFT_CLONE_MAX:
            _SHIFT_CLONE_STORE.popitem(last=False)
        return model_use

    def holds(self, model) -> bool:
        return any(item is model for item in _SHIFT_CLONE_STORE.values())

    def peek(self, model, shift_video: float, shift_audio: float):
        """已缓存则返回该 clone（不创建）；供清理路径拿到存活 patcher。"""
        key = (id(model), round(float(shift_video), 6), round(float(shift_audio), 6))
        return _SHIFT_CLONE_STORE.get(key)

    def clear(self) -> None:
        """清空进程缓存。仅供显式调试用，运行收尾不得调用（见类 docstring）。"""
        _SHIFT_CLONE_STORE.clear()


def sample_single_stage(
    *,
    model,
    positive,
    negative,
    latent,
    seed: int,
    cfg: float,
    steps: int,
    sampler_name: str,
    scheduler: str,
    shift_video: float = 12.0,
    shift_audio: float = 3.0,
    on_phase: PhaseCallback | None = None,
    on_step_preview: StepPreviewCallback | None = None,
    preview_every: int = 1,
    denoise: float = 1.0,
    phase_name: str = "sample",
    sigmas=None,
    apply_shift: bool = True,
    after_shift=None,
    enable_tiling: bool = False,
    tile_count: int = 2,
    tile_overlap: int = 128,
    shift_cache: ShiftedModelCache | None = None,
    on_step_state: StepStateCallback | None = None,
    on_loaded: LoadedModelCallback | None = None,
    zero_noise: bool = False,
    noise_override=None,
):
    import torch
    from comfy_extras.nodes_custom_sampler import (
        BasicGuider,
        BasicScheduler,
        CFGGuider,
        KSamplerSelect,
        RandomNoise,
        SamplerCustomAdvanced,
    )
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

    def notify(phase: str, value: float) -> None:
        if on_phase:
            on_phase(phase, value)

    notify(phase_name, 0)
    model_use = model
    if apply_shift:
        if shift_cache is not None:
            model_use = shift_cache.get(model, shift_video, shift_audio)
        else:
            shifted = MiniMaxH3SigmaShift.execute(model, float(shift_video), float(shift_audio))
            model_use = _unpack_node_output(shifted)[0]

    if sigmas is not None:
        if torch.is_tensor(sigmas):
            sigma_t = sigmas.detach().float().cpu().reshape(-1)
        else:
            sigma_t = torch.tensor([float(x) for x in sigmas], dtype=torch.float32)
    else:
        denoise_use = float(max(0.0, min(1.0, denoise)))
        sigma_out = BasicScheduler.execute(
            model_use, str(scheduler), int(steps), denoise_use
        )
        sigma_t = _unpack_node_output(sigma_out)[0]

    if callable(after_shift):
        remasked = after_shift(model_use, latent, sigma_t)
        if remasked is not None:
            model_use = remasked

    # 上报本次真正参与采样（将被 load_models_gpu 登记）的最终 patcher——
    # after_shift 产生的一次性 remask clone 也在此列。执行器用滚动槽强引用
    # 保活它到下次加载 / 运行收尾，避免 clone 被 GC 成 is_dead 死条目。
    if on_loaded is not None:
        try:
            on_loaded(model_use)
        except Exception as exc:
            log.debug("on_loaded model pin callback skipped: %s", exc)

    sampler_obj = _unpack_node_output(KSamplerSelect.execute(str(sampler_name)))[0]
    if noise_override is not None:
        noise_obj = _FixedNoise(noise_override, seed=int(seed or 0))
    elif zero_noise:
        noise_obj = _ZeroNoise()
    else:
        noise_obj = _unpack_node_output(RandomNoise.execute(int(seed)))[0]
    restore_tiles = None
    if enable_tiling:
        from .spatial_tiled_sampling import wrap_sampler_spatial_tiles

        restore_tiles = wrap_sampler_spatial_tiles(
            sampler_obj,
            n_tiles=tile_count,
            overlap_pixels=tile_overlap,
        )

    neg = negative if negative else []
    if _use_basic_guider(cfg, neg):
        guider = _unpack_node_output(BasicGuider.execute(model_use, positive))[0]
    else:
        guider = _unpack_node_output(
            CFGGuider.execute(model_use, positive, neg, float(cfg))
        )[0]

    def _run_official() -> dict:
        sampled = SamplerCustomAdvanced.execute(
            noise_obj, guider, sampler_obj, sigma_t, latent
        )
        return _unpack_node_output(sampled)[0]

    orig_sample = (
        guider.sample if (on_step_preview is not None or on_step_state is not None) else None
    )
    if orig_sample is not None:
        every = max(1, int(preview_every))

        def sample_wrapped(noise, latent_image, sampler, sigmas_in, **kwargs):
            inner_cb = kwargs.get("callback")

            def callback(step, x0, x, total_steps):
                if on_step_state is not None:
                    try:
                        on_step_state(int(step), int(total_steps), x0, x)
                    except Exception as exc:
                        log.debug("Step state callback skipped: %s", exc)
                try:
                    if on_step_preview is not None:
                        last = max(0, int(total_steps) - 1)
                        if int(preview_every) < 0:
                            show = step >= last
                        else:
                            show = step % every == 0 or step >= last
                        if show:
                            on_step_preview(int(step), int(total_steps), x0)
                except Exception as exc:
                    log.debug("Step preview callback skipped: %s", exc)
                if inner_cb is not None:
                    inner_cb(step, x0, x, total_steps)

            kwargs["callback"] = callback
            return orig_sample(noise, latent_image, sampler, sigmas_in, **kwargs)

        guider.sample = sample_wrapped
    try:
        out = _run_official()
    finally:
        if orig_sample is not None:
            guider.sample = orig_sample
        if restore_tiles is not None:
            restore_tiles()
        if callable(after_shift):
            try:
                from .h3_latent_continue import uninstall_continue_prefix_remask

                uninstall_continue_prefix_remask(model_use)
            except Exception as exc:
                log.debug("Prefix remask uninstall skipped: %s", exc)
        # Drop sampler-graph refs. Keep a cached SigmaShift clone alive so
        # segment cleanup can unload it instead of leaving a dead LoadedModel.
        guider = None
        noise_obj = None
        sampler_obj = None
        if model_use is not model and (shift_cache is None or not shift_cache.holds(model_use)):
            model_use = None

    notify(phase_name, 1)
    return out
