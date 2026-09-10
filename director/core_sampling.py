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
LoadedModelCallback = Callable[[Any], None]


# ── 进程级 SigmaShift clone 复用 ────────────────────────────────────────────
# shift_video / shift_audio 是运行级常量。官方 MiniMaxH3SigmaShift.execute 每次都
# 返回一个 m = model.clone()（共享同一份底层 MiniMaxH3 权重，仅叠加 model_sampling
# 补丁）。本插件在单个节点内用 Python 直接编排多个采样段，这些 clone 不是图节点
# 输出、不被 ComfyUI 执行器缓存钉住，函数返回即成为垃圾；段间被 Python GC 后，
# 底层模型仍被节点输入的主 MODEL 持有 → ComfyUI 判定 is_dead()（patcher 弱引用
# 已死、nn.Module 仍活），每次 load_models_gpu 都刷 "memory leak with model
# MiniMaxH3" 警告，死条目还逐段堆积。按“主 patcher 身份 + shift 参数”复用同一个
# clone，使其跨段、跨运行始终存活，从根上消除 is_dead（不复制权重、不重读磁盘）。
_SHIFT_CLONE_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()
_SHIFT_CLONE_MAX = 16


def get_persistent_shifted_model(base, shift_video: float, shift_audio: float):
    """返回（必要时创建并缓存）主 MODEL 的持久 SigmaShift clone。"""
    key = (id(base), round(float(shift_video), 6), round(float(shift_audio), 6))
    cached = _SHIFT_CLONE_CACHE.get(key)
    if cached is not None:
        _SHIFT_CLONE_CACHE.move_to_end(key)
        return cached
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

    shifted = MiniMaxH3SigmaShift.execute(base, float(shift_video), float(shift_audio))
    model_use = _unpack_node_output(shifted)[0]
    _SHIFT_CLONE_CACHE[key] = model_use
    _SHIFT_CLONE_CACHE.move_to_end(key)
    while len(_SHIFT_CLONE_CACHE) > _SHIFT_CLONE_MAX:
        _SHIFT_CLONE_CACHE.popitem(last=False)
    return model_use


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
    shifted_model=None,
    on_loaded: LoadedModelCallback | None = None,
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

    def notify(phase: str, value: float) -> None:
        if on_phase:
            on_phase(phase, value)

    notify(phase_name, 0)
    if shifted_model is not None:
        # 调用方复用进程级持久 shift clone（跨段同一 patcher），不再每段新建即弃 clone。
        model_use = shifted_model
    elif apply_shift:
        # 默认也走持久缓存：多段共享一个 SigmaShift clone，避免 is_dead 假泄漏。
        model_use = get_persistent_shifted_model(
            model, float(shift_video), float(shift_audio)
        )
    else:
        model_use = model

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

    sampler_obj = _unpack_node_output(KSamplerSelect.execute(str(sampler_name)))[0]
    noise_obj = _unpack_node_output(RandomNoise.execute(int(seed)))[0]

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

    if on_step_preview is None:
        out = _run_official()
    else:
        orig_sample = guider.sample
        every = max(1, int(preview_every))

        def sample_wrapped(noise, latent_image, sampler, sigmas_in, **kwargs):
            inner_cb = kwargs.get("callback")

            def callback(step, x0, x, total_steps):
                try:
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
            guider.sample = orig_sample

    # 上报本次真正被 load_models_gpu 登记的最终 patcher（after_shift 产生的一次性
    # remask clone，或持久 shift clone）。调用方据回调把它强引用到“下一次模型加载
    # 之后”，使 ComfyUI 的 is_clone 清扫能在旧 patcher 仍存活时正常摘除其登记条目，
    # 杜绝“patcher 已被 GC、底层 MiniMaxH3 仍存活”的 is_dead 假泄漏堆积。
    if on_loaded is not None:
        try:
            on_loaded(model_use)
        except Exception as exc:
            log.debug("on_loaded model pin callback skipped: %s", exc)

    notify(phase_name, 1)
    return out
