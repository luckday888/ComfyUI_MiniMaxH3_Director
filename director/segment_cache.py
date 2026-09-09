"""Disk cache for MiniMax H3 Director segment decode outputs (partial re-run + merge).

Cache is best-effort: write failures (cloud RO mounts, same-name overwrite
blocks, full disks) must never abort the main generation run.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Callable

import torch

import folder_paths

from .h3_latent_continue import CONTINUE_PIPELINE_ID
from .h3_motion_context import CONTINUITY_PIPELINE_ID, trim_context_prefix, trim_export_tail
from .plan import DirectorPlan, SegmentPlan, resolve_ref_image_size

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.cache")

SOURCE_VIDEO_FP_KEY = "source_video"

# ── 稳定段身份 -> 磁盘存储前缀 ─────────────────────────────────────────────
# 旧版本缓存按“位置下标”命名（seg_0009.*）。删除前导段后保留段会被重新编号，
# 位置键随之改变，导致段间引导找不到上一段产物。现改为按 timeline 持久段 id
# 命名（segid_<id>.*），旧的位置命名保留为读取回退；meta 内额外记录 seg_id，
# 以便 prune 能把“移位后的旧位置文件”认领到当前段，而不是误删。
_SEG_ID_SANITIZE_RE = re.compile(r"[^0-9A-Za-z_-]+")

# 最终组 / 一采组磁盘后缀
SFX_FINAL_FRAMES = ".pt"
SFX_FINAL_META = ".meta.json"
SFX_FINAL_AV = ".av.pt"
SFX_FINAL_HANDOFF = ".handoff.json"
SFX_FINAL_AUDIO = ".audio.pt"
SFX_PRE_FRAMES = ".pre.pt"
SFX_PRE_META = ".pre.meta.json"
SFX_PRE_AV = ".pre.av.pt"
SFX_PRE_HANDOFF = ".pre.handoff.json"


def stable_seg_id(seg: SegmentPlan) -> str:
    """段的持久身份；无身份来源时为空串（调用方据此回退旧位置键）。"""
    return str(getattr(seg, "seg_id", "") or "").strip()


def _sanitize_seg_id(sid: str) -> str:
    return _SEG_ID_SANITIZE_RE.sub("_", str(sid).strip())[:80] or "seg"


def _legacy_stem(index: int) -> str:
    """旧版本位置命名前缀。"""
    return f"seg_{int(index):04d}"


def _stable_stem(seg: SegmentPlan) -> str:
    return f"segid_{_sanitize_seg_id(stable_seg_id(seg))}"


def _candidate_stems(seg: SegmentPlan) -> list[str]:
    """读取候选前缀：稳定 id 命名优先，旧的位置数字命名作为回退，去重保序。"""
    stems = [_stable_stem(seg), _legacy_stem(seg.index)]
    out: list[str] = []
    for stem in stems:
        if stem and stem not in out:
            out.append(stem)
    return out


def _write_stem(seg: SegmentPlan) -> str:
    """保存目标前缀：有稳定 id 用稳定命名，否则回退旧位置命名。"""
    return _candidate_stems(seg)[0]


def _resolve_stem(
    root: Path, seg: SegmentPlan, meta_sfx: str, primary_sfx: str
) -> str:
    """定位该段在磁盘上实际使用的前缀。

    以 meta 文件为锚（指纹文件），其次主资源；都不存在时返回首选前缀（供
    保存/拼路径）。这样同一段的 frames/av/handoff/audio 始终取自同一组文件。
    """
    candidates = _candidate_stems(seg)
    for suffix in (meta_sfx, primary_sfx):
        for stem in candidates:
            if (root / f"{stem}{suffix}").is_file():
                return stem
    return candidates[0]


def _reject_owner_mismatch(stored: Any, seg: SegmentPlan) -> bool:
    """旧位置命名文件可能因段移位而属于别的段——按 meta 内 seg_id 硬拦截。

    与“源视频变更”同级，优先于 allow_stale：即使允许 stale，也绝不拿别的段的
    产物充当本段。历史缓存 meta 无 seg_id 时不拦截，保持向后兼容。
    """
    sid = stable_seg_id(seg)
    if not sid or not isinstance(stored, dict):
        return False
    stored_sid = str(stored.get("seg_id") or "").strip()
    return bool(stored_sid) and stored_sid != sid


def source_video_identity(plan: DirectorPlan) -> list[str]:
    """Stable source-clip identity: relative path + size + mtime (overwrite-safe)."""
    from ..lib.video_io import resolve_video_path, video_clips_from_timeline

    clips = video_clips_from_timeline((plan.raw or {}) if plan is not None else {})
    tokens: list[str] = []
    for clip in clips:
        if not isinstance(clip, dict):
            continue
        rel = str(clip.get("videoFile") or clip.get("fileName") or "").strip().replace("\\", "/")
        if not rel:
            continue
        try:
            path = resolve_video_path(clip)
            st = os.stat(path)
            mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
            tokens.append(f"{rel}:{st.st_size}:{mtime_ns}")
        except Exception:
            tokens.append(f"{rel}:missing")
    return tokens


def source_identity_changed(stored: Any, expected: dict[str, Any]) -> bool:
    """True when the current plan has a source video that does not match cache meta.

    Gen timelines (no source clips) never count as a source change, so stale
    fill/continuity still work after pipeline-only fingerprint churn.
    """
    exp = expected.get(SOURCE_VIDEO_FP_KEY) or []
    if not exp:
        return False
    if not isinstance(stored, dict) or SOURCE_VIDEO_FP_KEY not in stored:
        return True
    return stored.get(SOURCE_VIDEO_FP_KEY) != exp


def _reject_source_stale(
    stored: Any,
    expected: dict[str, Any],
    *,
    seg_index: int,
    quiet: bool = False,
) -> bool:
    if not source_identity_changed(stored, expected):
        return False
    if not quiet:
        log.info(
            "Segment %d cache is from a different source video; ignoring stale render.",
            seg_index + 1,
        )
    return True


def _cache_root(node_id: str) -> Path | None:
    try:
        root = Path(folder_paths.get_output_directory()) / "minimax_seg_cache" / str(node_id)
        root.mkdir(parents=True, exist_ok=True)
        return root
    except OSError as exc:
        log.warning("Segment cache dir unavailable (%s); cache disabled for this run.", exc)
        return None


def _ref_audio_file_stamp(audio: Any, fallback_index: int) -> str:
    """Fingerprint fragment for one reference audio.

    Keying on the uploaded filename alone misses the case where the user keeps
    the same slot/filename but replaces the audio content. Stamp the source
    file's mtime+size so a content swap invalidates the first-pass cache.
    """
    index = getattr(audio, "index", fallback_index)
    name = getattr(audio, "audio_file", "") or ""
    path = getattr(audio, "audio_path", "") or ""
    stamp = ""
    if path:
        try:
            st = os.stat(path)
            stamp = f"{int(st.st_mtime)}:{int(st.st_size)}"
        except OSError:
            stamp = ""
    return f"aud{index}:{name}:{stamp}"


def _segment_identity_fingerprint(seg: SegmentPlan, plan: DirectorPlan) -> dict[str, Any]:
    """Identity that affects first-pass sampling (no Refine settings)."""
    ref_files = sorted(
        f"img{ref.index}:{(getattr(ref, 'image_file', '') or '')}"
        for ref in seg.refs
    )
    ref_audio_files = sorted(
        _ref_audio_file_stamp(a, i)
        for i, a in enumerate(getattr(seg, "ref_audios", None) or [])
    )
    ref_video_files = sorted(
        f"vid{getattr(v, 'index', i)}:{(getattr(v, 'video_file', '') or '')}"
        for i, v in enumerate(getattr(seg, "ref_videos", None) or [])
    )
    ref_video_file = (
        seg.reference_video_meta.get("videoFile")
        or seg.reference_video_meta.get("fileName")
        or ""
    ).strip()
    return {
        "seg_id": stable_seg_id(seg),
        "index": seg.index,
        "start": seg.start_frame,
        "end": seg.end_frame,
        "prompt": seg.prompt,
        "negative": seg.negative_prompt,
        "task_key": seg.task_key,
        "width": plan.width,
        "height": plan.height,
        "frame_rate": float(getattr(plan, "frame_rate", 24) or 24),
        "output_mode": plan.output_mode,
        "ref_max": plan.ref_max_size,
        "ref_image_size": resolve_ref_image_size(seg, plan),
        "refs": ref_files,
        "ref_audios": ref_audio_files,
        "ref_videos": ref_video_files,
        "ref_video": ref_video_file,
        "ref_video_start": seg.reference_video_start_frame,
        SOURCE_VIDEO_FP_KEY: source_video_identity(plan),
        "continuity": plan.continuity_enabled,
        "continuity_overlap": plan.continuity_overlap_frames if plan.continuity_enabled else 0,
        "continuity_from_prev": bool(getattr(seg, "continuity_from_prev", True)),
        "continuity_mode": (
            str(getattr(plan, "continuity_mode", "guide") or "guide")
            if plan.continuity_enabled
            else "off"
        ),
        "continuity_redraw": (
            round(float(getattr(plan, "continuity_redraw", 0.65) or 0.65), 2)
            if plan.continuity_enabled
            and str(getattr(plan, "continuity_mode", "guide") or "guide") == "continue"
            else 0
        ),
        "continuity_pipeline": (
            CONTINUE_PIPELINE_ID
            if plan.continuity_enabled
            and str(getattr(plan, "continuity_mode", "guide") or "guide") == "continue"
            else CONTINUITY_PIPELINE_ID
        ),
    }


def first_pass_cache_fingerprint(seg: SegmentPlan, plan: DirectorPlan) -> dict[str, Any]:
    """Exact-match key for first-pass AV latent. Refine knobs are excluded."""
    fp = _segment_identity_fingerprint(seg, plan)
    sigmas = getattr(plan, "sample_sigmas", None)
    linked = bool(sigmas) or bool(getattr(plan, "sample_sigmas_linked", False))
    fp.update({
        "kind": "first_pass",
        "seed": int(getattr(plan, "sample_seed", 0) or 0),
        "cfg": round(float(getattr(plan, "sample_cfg", 1.0) or 1.0), 6),
        "sampler": str(getattr(plan, "sample_sampler", "") or ""),
        "shift_video": round(float(getattr(plan, "sample_shift_video", 12.0) or 12.0), 6),
        "shift_audio": round(float(getattr(plan, "sample_shift_audio", 3.0) or 3.0), 6),
    })
    if linked:
        fp["steps"] = 0
        fp["scheduler"] = "external_sigmas"
        fp["sigmas_source"] = "linked"
        if sigmas:
            fp["sigmas"] = [round(float(x), 6) for x in sigmas]
    else:
        fp["steps"] = int(getattr(plan, "sample_steps", 25) or 25)
        fp["scheduler"] = str(getattr(plan, "sample_scheduler", "") or "")
    return fp


def segment_cache_fingerprint(seg: SegmentPlan, plan: DirectorPlan) -> dict[str, Any]:
    """Stable identity for a segment — cache invalidates when edit params change."""
    fp = _segment_identity_fingerprint(seg, plan)
    from .refine_pack import refine_fingerprint

    fp.update(refine_fingerprint(plan))
    return fp


def _safe_unlink(path: Path) -> bool:
    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
        return True
    except OSError:
        return False


def _atomic_publish(tmp: Path, dest: Path) -> None:
    """Move ``tmp`` 鈫?``dest``, tolerating clouds that block same-name overwrite."""
    try:
        os.replace(tmp, dest)
        return
    except OSError:
        pass
    # Some cloud mounts reject overwrite of an existing name 鈥?remove then rename.
    _safe_unlink(dest)
    try:
        os.replace(tmp, dest)
        return
    except OSError:
        pass
    try:
        tmp.rename(dest)
        return
    except OSError:
        # Last resort: keep the unique temp as the published file name is blocked.
        # Caller may still fail if even create-new is denied.
        raise


def _write_via_temp(dest: Path, write_fn: Callable[[Path], None]) -> None:
    """Write to a unique temp name in the same folder, then publish to ``dest``."""
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")
    try:
        write_fn(tmp)
        _atomic_publish(tmp, dest)
    finally:
        _safe_unlink(tmp)


def _audio_payload_to_cpu(audio: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize export AUDIO dict for disk cache (waveform on CPU)."""
    if not isinstance(audio, dict):
        return None
    wave = audio.get("waveform")
    if not isinstance(wave, torch.Tensor) or wave.numel() <= 0:
        return None
    sr = int(audio.get("sample_rate") or 0) or 32000
    return {
        "waveform": wave.detach().cpu().contiguous(),
        "sample_rate": sr,
    }


def _frames_to_disk(tensor: torch.Tensor) -> torch.Tensor:
    """Store pixel frames as uint8 [0,255]. Export is 8-bit anyway; float32 is 4× larger."""
    x = tensor.detach().cpu()
    if x.dtype == torch.uint8:
        return x.contiguous()
    return x.float().clamp(0, 1).mul(255).round().clamp(0, 255).to(torch.uint8).contiguous()


def _frames_from_disk(loaded: Any) -> torch.Tensor | None:
    """Restore uint8 cache to float32 [0,1]; pass through legacy float caches."""
    if not isinstance(loaded, torch.Tensor):
        return None
    if loaded.dtype == torch.uint8:
        return loaded.float().div(255.0)
    return loaded.float()


def _retire_legacy_copies(
    root: Path, seg: SegmentPlan, suffixes, meta_sfx: str = SFX_FINAL_META
) -> None:
    """段已写入稳定命名后，删除属于同一段的旧位置命名副本，避免双份占用。

    仅当旧位置文件的 meta 明确记录 seg_id == 当前段时才删除；身份不明的历史
    缓存保留（保守，绝不误删）。无稳定 id（保存前缀本身即旧位置命名）时跳过。
    """
    sid = stable_seg_id(seg)
    if not sid or _write_stem(seg) == _legacy_stem(seg.index):
        return
    legacy = _legacy_stem(seg.index)
    meta_path = root / f"{legacy}{meta_sfx}"
    try:
        if not meta_path.is_file():
            return
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        if str((stored or {}).get("seg_id") or "").strip() != sid:
            return
    except Exception:
        return
    for suffix in suffixes:
        _safe_unlink(root / f"{legacy}{suffix}")


_FINAL_ALL_SUFFIXES = (
    SFX_FINAL_FRAMES,
    SFX_FINAL_META,
    SFX_FINAL_AV,
    SFX_FINAL_HANDOFF,
    SFX_FINAL_AUDIO,
)
_PRE_ALL_SUFFIXES = (
    SFX_PRE_FRAMES,
    SFX_PRE_META,
    SFX_PRE_AV,
    SFX_PRE_HANDOFF,
)


def save_segment_cache(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    tensor: torch.Tensor,
    *,
    av_latent: dict | None = None,
    handoff: dict[str, Any] | None = None,
    audio: dict[str, Any] | None = None,
    replace_audio: bool = True,
) -> None:
    """Persist a segment tensor (+ optional AV latent / export audio). Never raises.

    ``replace_audio``:
      - True (default): write ``audio`` when present, otherwise delete stale audio.pt
        (fresh sample with mute/empty decode).
      - False: write ``audio`` when present, otherwise **keep** existing audio.pt
        (phase-align trim re-save must not wipe a prior audio cache).
    """
    if not node_id:
        return
    root = _cache_root(node_id)
    if root is None:
        return
    fp = segment_cache_fingerprint(seg, plan)
    idx = seg.index
    stem = _write_stem(seg)
    pt_path = root / f"{stem}{SFX_FINAL_FRAMES}"
    meta_path = root / f"{stem}{SFX_FINAL_META}"
    latent_path = root / f"{stem}{SFX_FINAL_AV}"
    handoff_path = root / f"{stem}{SFX_FINAL_HANDOFF}"
    audio_path = root / f"{stem}{SFX_FINAL_AUDIO}"
    try:
        payload = _frames_to_disk(tensor)
        _write_via_temp(pt_path, lambda p: torch.save(payload, p))
        text = json.dumps(fp, ensure_ascii=False, sort_keys=True)
        _write_via_temp(
            meta_path,
            lambda p: p.write_text(text, encoding="utf-8"),
        )
        if av_latent is not None and isinstance(av_latent, dict) and "samples" in av_latent:
            cpu_latent = _av_latent_to_cpu(av_latent)
            _write_via_temp(latent_path, lambda p: torch.save(cpu_latent, p))
        if handoff:
            _write_via_temp(
                handoff_path,
                lambda p: p.write_text(
                    json.dumps(handoff, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8",
                ),
            )
        audio_cpu = _audio_payload_to_cpu(audio)
        if audio_cpu is not None:
            _write_via_temp(audio_path, lambda p: torch.save(audio_cpu, p))
        elif replace_audio:
            # Fresh sample with no waveform — drop stale audio from an older run.
            _safe_unlink(audio_path)
        log.debug(
            "Cached segment %d for node %s (%d frames%s%s)",
            idx + 1,
            node_id,
            int(tensor.shape[0]),
            ", +av_latent" if av_latent is not None else "",
            ", +audio" if audio_cpu is not None else (
                ", keep-audio" if not replace_audio else ""
            ),
        )
        # 段移位后用稳定 id 重新落盘，清掉同段的旧位置命名文件，避免双份。
        _retire_legacy_copies(root, seg, _FINAL_ALL_SUFFIXES)
    except Exception as exc:
        # Xiangong / similar: RO mount or same-name write → skip cache, keep run alive.
        log.warning(
            "Segment %d cache write skipped (%s). Generation continues without disk cache.",
            idx + 1,
            exc,
        )
        for stray in root.glob(f".{stem}.*"):
            _safe_unlink(stray)


def _fingerprint_diff_keys(stored: Any, expected: dict[str, Any]) -> list[str]:
    if not isinstance(stored, dict):
        return ["<invalid-meta>"]
    keys = sorted(set(stored) | set(expected))
    return [k for k in keys if stored.get(k) != expected.get(k)]


def load_segment_handoff_meta(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
) -> dict[str, Any] | None:
    """Load trim/export handoff metadata (fingerprint must match unless ``allow_stale``)."""
    if not node_id:
        return None
    root = _cache_root(node_id)
    if root is None:
        return None
    idx = seg.index
    stem = _resolve_stem(root, seg, SFX_FINAL_META, SFX_FINAL_FRAMES)
    meta_path = root / f"{stem}{SFX_FINAL_META}"
    handoff_path = root / f"{stem}{SFX_FINAL_HANDOFF}"
    if not meta_path.is_file() or not handoff_path.is_file():
        return None
    try:
        expected = segment_cache_fingerprint(seg, plan)
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        if _reject_owner_mismatch(stored, seg):
            return None
        if stored != expected:
            if _reject_source_stale(stored, expected, seg_index=idx, quiet=True) or not allow_stale:
                return None
        data = json.loads(handoff_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _av_latent_to_cpu(av_latent: dict) -> dict:
    samples = av_latent["samples"]
    if hasattr(samples, "unbind"):
        parts = [p.detach().cpu().contiguous() for p in samples.unbind()]
        try:
            import comfy.nested_tensor

            samples_cpu = comfy.nested_tensor.NestedTensor(tuple(parts))
        except Exception:
            samples_cpu = tuple(parts)
    elif isinstance(samples, (tuple, list)):
        samples_cpu = tuple(p.detach().cpu().contiguous() for p in samples)
    elif torch.is_tensor(samples):
        samples_cpu = samples.detach().cpu().contiguous()
    else:
        samples_cpu = samples
    out = {"samples": samples_cpu}
    for key, value in av_latent.items():
        if key == "samples":
            continue
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().contiguous()
        else:
            out[key] = value
    return out


def load_segment_av_latent(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
) -> dict | None:
    """Load cached AV latent for continuity handoff (fingerprint must match unless stale-ok)."""
    if not node_id:
        return None
    root = _cache_root(node_id)
    if root is None:
        return None
    idx = seg.index
    stem = _resolve_stem(root, seg, SFX_FINAL_META, SFX_FINAL_FRAMES)
    meta_path = root / f"{stem}{SFX_FINAL_META}"
    latent_path = root / f"{stem}{SFX_FINAL_AV}"
    if not meta_path.is_file() or not latent_path.is_file():
        return None
    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = segment_cache_fingerprint(seg, plan)
        if _reject_owner_mismatch(stored, seg):
            return None
        if stored != expected:
            if _reject_source_stale(stored, expected, seg_index=idx, quiet=True) or not allow_stale:
                return None
        payload = torch.load(latent_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "samples" not in payload:
            return None
        return payload
    except Exception as exc:
        log.warning("Failed to load segment %d AV latent cache: %s", idx + 1, exc)
        return None


def _fingerprint_matches(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
) -> bool:
    if not node_id:
        return False
    root = _cache_root(node_id)
    if root is None:
        return False
    stem = _resolve_stem(root, seg, SFX_FINAL_META, SFX_FINAL_FRAMES)
    meta_path = root / f"{stem}{SFX_FINAL_META}"
    tensor_path = root / f"{stem}{SFX_FINAL_FRAMES}"
    if not meta_path.is_file():
        return False
    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = segment_cache_fingerprint(seg, plan)
        if _reject_owner_mismatch(stored, seg):
            return False
        if stored == expected:
            return True
        if _reject_source_stale(stored, expected, seg_index=seg.index, quiet=True):
            return False
        return bool(allow_stale and tensor_path.is_file())
    except Exception:
        return False


def load_segment_cache(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
) -> torch.Tensor | None:
    """Load cached segment frames.

    ``allow_stale=True``: used for「选择运行」+「全部导出」fill of unselected
    segments. Prefer the last render on disk over blank/gray source placeholders
    when the fingerprint drifted (pipeline bump, minor plan churn). A different
    source video is never treated as usable stale — callers then passthrough
    the current clip (v2v/rv2v) or skip (gen timelines).
    """
    if not node_id:
        return None
    root = _cache_root(node_id)
    if root is None:
        return None
    idx = seg.index
    stem = _resolve_stem(root, seg, SFX_FINAL_META, SFX_FINAL_FRAMES)
    meta_path = root / f"{stem}{SFX_FINAL_META}"
    tensor_path = root / f"{stem}{SFX_FINAL_FRAMES}"
    if not tensor_path.is_file():
        return None
    try:
        expected = segment_cache_fingerprint(seg, plan)
        if meta_path.is_file():
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            if _reject_owner_mismatch(stored, seg):
                return None
            if stored != expected:
                if _reject_source_stale(stored, expected, seg_index=idx):
                    return None
                diff = _fingerprint_diff_keys(stored, expected)
                if not allow_stale:
                    log.info(
                        "Segment %d cache stale (diff=%s); re-run this segment to refresh.",
                        idx + 1,
                        diff[:8],
                    )
                    return None
                log.warning(
                    "Segment %d: using stale cache for export fill (diff=%s).",
                    idx + 1,
                    diff[:8],
                )
        elif not allow_stale:
            log.info(
                "Segment %d cache missing meta; re-run this segment to refresh.",
                idx + 1,
            )
            return None
        else:
            log.warning(
                "Segment %d: using cache without meta for export fill.",
                idx + 1,
            )
        return _frames_from_disk(
            torch.load(tensor_path, map_location="cpu", weights_only=True)
        )
    except Exception as exc:
        log.warning("Failed to load segment %d cache: %s", idx + 1, exc)
        return None


def load_segment_audio(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    allow_stale: bool = False,
) -> dict[str, Any] | None:
    """Load cached export audio for a segment (same fingerprint policy as video)."""
    if not node_id or not _fingerprint_matches(
        node_id, seg, plan, allow_stale=allow_stale
    ):
        return None
    root = _cache_root(node_id)
    if root is None:
        return None
    stem = _resolve_stem(root, seg, SFX_FINAL_META, SFX_FINAL_FRAMES)
    audio_path = root / f"{stem}{SFX_FINAL_AUDIO}"
    if not audio_path.is_file():
        return None
    try:
        payload = torch.load(audio_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            return None
        wave = payload.get("waveform")
        if not isinstance(wave, torch.Tensor) or wave.numel() <= 0:
            return None
        sr = int(payload.get("sample_rate") or 0) or 32000
        return {"waveform": wave.contiguous(), "sample_rate": sr}
    except Exception as exc:
        log.warning("Failed to load segment %d audio cache: %s", seg.index + 1, exc)
        return None


def save_first_pass_cache(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    av_latent: dict | None = None,
    frames: torch.Tensor | None = None,
    handoff: dict[str, Any] | None = None,
) -> None:
    """Persist first-pass AV latent for confirm-then-refine. Never raises."""
    if not node_id:
        return
    if av_latent is None or not isinstance(av_latent, dict) or "samples" not in av_latent:
        return
    root = _cache_root(node_id)
    if root is None:
        return
    fp = first_pass_cache_fingerprint(seg, plan)
    idx = seg.index
    stem = _write_stem(seg)
    meta_path = root / f"{stem}{SFX_PRE_META}"
    latent_path = root / f"{stem}{SFX_PRE_AV}"
    frames_path = root / f"{stem}{SFX_PRE_FRAMES}"
    handoff_path = root / f"{stem}{SFX_PRE_HANDOFF}"
    try:
        cpu_latent = _av_latent_to_cpu(av_latent)
        _write_via_temp(latent_path, lambda p: torch.save(cpu_latent, p))
        text = json.dumps(fp, ensure_ascii=False, sort_keys=True)
        _write_via_temp(meta_path, lambda p: p.write_text(text, encoding="utf-8"))
        if handoff:
            _write_via_temp(
                handoff_path,
                lambda p: p.write_text(
                    json.dumps(handoff, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8",
                ),
            )
        if isinstance(frames, torch.Tensor) and frames.numel() > 0:
            payload = _frames_to_disk(frames)
            _write_via_temp(frames_path, lambda p: torch.save(payload, p))
        log.debug(
            "Cached first-pass segment %d for node %s (seed=%s)",
            idx + 1,
            node_id,
            fp.get("seed"),
        )
        # 段移位后用稳定 id 重新落盘，清掉同段的旧位置命名一采副本。
        _retire_legacy_copies(
            root, seg, _PRE_ALL_SUFFIXES, meta_sfx=SFX_PRE_META
        )
    except Exception as exc:
        log.warning(
            "Segment %d first-pass cache write skipped (%s).",
            idx + 1,
            exc,
        )
        for stray in root.glob(f".{stem}.*"):
            _safe_unlink(stray)


def _trim_stale_first_pass_frames(
    frames: torch.Tensor,
    *,
    plan: DirectorPlan,
    handoff: dict[str, Any] | None,
    match_len: int | None,
) -> torch.Tensor | None:
    """Match in-memory first-pass export: drop context prefix, then crop length."""
    fps = float(getattr(plan, "frame_rate", 24) or 24)
    trim_frames = int((handoff or {}).get("trim_frames") or 0)
    export_len = int((handoff or {}).get("export_frames") or 0)
    if trim_frames > 0:
        if int(frames.shape[0]) <= trim_frames:
            return None
        frames, _ = trim_context_prefix(
            frames, None, trim_frames, fps=fps, match_tail=True
        )
    if export_len > 0 and int(frames.shape[0]) > export_len:
        frames = frames[:export_len]
    want = int(match_len or 0)
    extra = int(frames.shape[0]) - want if want > 0 else 0
    if extra > 0:
        frames, _ = trim_export_tail(frames, None, extra, fps=fps)
    return frames


def load_first_pass_frames_stale(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
    *,
    match_len: int | None = None,
) -> torch.Tensor | None:
    """Load ``.pre.pt`` frames for unselected-segment pre-refine fill.

    Stale-tolerant counterpart of :func:`load_first_pass_cache`: fingerprint
    drift (different seed, sampling-knob churn) does NOT invalidate the fill,
    so「选择运行」re-roll previews merge all-first-pass frames instead of
    mixing a fresh first pass with cached refined renders. A different source
    video still rejects (same rule as the final-cache fill). Never raises.

    Disk ``.pre.pt`` is written before export trim; this reapplies
    ``.pre.handoff.json`` (context prefix + export length) and optionally
    matches the final-cache frame count after later phase-align tail trims.
    """
    if not node_id:
        return None
    root = _cache_root(node_id)
    if root is None:
        return None
    idx = seg.index
    stem = _resolve_stem(root, seg, SFX_PRE_META, SFX_PRE_AV)
    frames_path = root / f"{stem}{SFX_PRE_FRAMES}"
    meta_path = root / f"{stem}{SFX_PRE_META}"
    handoff_path = root / f"{stem}{SFX_PRE_HANDOFF}"
    if not frames_path.is_file():
        return None
    try:
        if meta_path.is_file():
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            expected = first_pass_cache_fingerprint(seg, plan)
            if _reject_owner_mismatch(stored, seg):
                return None
            if _reject_source_stale(stored, expected, seg_index=idx, quiet=True):
                return None
        loaded = torch.load(frames_path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, torch.Tensor) or loaded.numel() <= 0:
            return None
        frames = _frames_from_disk(loaded)
        if frames is None:
            return None
        handoff = None
        if handoff_path.is_file():
            try:
                data = json.loads(handoff_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    handoff = data
            except Exception:
                handoff = None
        return _trim_stale_first_pass_frames(
            frames, plan=plan, handoff=handoff, match_len=match_len
        )
    except Exception as exc:
        log.debug("Segment %d first-pass stale frames skipped: %s", idx + 1, exc)
    return None


def load_first_pass_cache(
    node_id: str | None,
    seg: SegmentPlan,
    plan: DirectorPlan,
) -> dict[str, Any] | None:
    """Load first-pass cache only on exact fingerprint match. Never stale."""
    if not node_id:
        return None
    root = _cache_root(node_id)
    if root is None:
        return None
    idx = seg.index
    stem = _resolve_stem(root, seg, SFX_PRE_META, SFX_PRE_AV)
    meta_path = root / f"{stem}{SFX_PRE_META}"
    latent_path = root / f"{stem}{SFX_PRE_AV}"
    frames_path = root / f"{stem}{SFX_PRE_FRAMES}"
    handoff_path = root / f"{stem}{SFX_PRE_HANDOFF}"
    if not meta_path.is_file() or not latent_path.is_file():
        return None
    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = first_pass_cache_fingerprint(seg, plan)
        if isinstance(stored, dict) and _reject_owner_mismatch(stored, seg):
            return None
        if not isinstance(stored, dict) or stored != expected:
            if isinstance(stored, dict) and _reject_source_stale(
                stored, expected, seg_index=idx, quiet=True,
            ):
                return None
            diff = _fingerprint_diff_keys(stored, expected) if isinstance(stored, dict) else ["<invalid-meta>"]
            log.info(
                "Segment %d first-pass cache miss (diff=%s); will sample first pass.",
                idx + 1,
                diff[:8],
            )
            return None
        payload = torch.load(latent_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "samples" not in payload:
            return None
        frames = None
        if frames_path.is_file():
            try:
                loaded = torch.load(frames_path, map_location="cpu", weights_only=True)
                if isinstance(loaded, torch.Tensor) and loaded.numel() > 0:
                    frames = _frames_from_disk(loaded)
            except Exception as exc:
                log.debug("Segment %d first-pass frames skipped: %s", idx + 1, exc)
        handoff: dict[str, Any] = {}
        if handoff_path.is_file():
            try:
                data = json.loads(handoff_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    handoff = data
            except Exception:
                handoff = {}
        return {"av_latent": payload, "frames": frames, "handoff": handoff}
    except Exception as exc:
        log.warning("Failed to load segment %d first-pass cache: %s", idx + 1, exc)
        return None


# 文件名后缀（特异性长的在前），用于把缓存文件名剥离出存储前缀
_CACHE_SUFFIXES_FOR_PARSE = (
    SFX_PRE_META,
    SFX_PRE_AV,
    SFX_PRE_HANDOFF,
    SFX_PRE_FRAMES,
    SFX_FINAL_META,
    SFX_FINAL_AV,
    SFX_FINAL_HANDOFF,
    SFX_FINAL_AUDIO,
    SFX_FINAL_FRAMES,
)
_STABLE_PREFIX_RE = re.compile(r"^segid_(.+)$")
_LEGACY_PREFIX_RE = re.compile(r"^seg_(\d+)$")


def _split_cache_stem(name: str) -> str | None:
    """从缓存文件名剥离已知后缀，返回存储前缀；无法识别返回 None。"""
    if name.startswith("."):
        return None
    for suffix in _CACHE_SUFFIXES_FOR_PARSE:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def prune_segment_cache(node_id: str | None, valid_segments) -> None:
    """删除时间轴上已不存在的段所拥有的缓存文件。

    ``valid_segments`` 接收当前段对象（含 ``seg_id/index``）；也兼容旧的 int
    下标列表。判定规则（绝不在身份不明时误删）：

    - ``segid_<id>.*``（新稳定命名）：属主 id 不在当前段集合 → 删除。
    - ``seg_XXXX.*``（旧位置命名）：读取其 meta 内 seg_id——
        * 明确属主且该属主已不在 → 删除；
        * 明确属主且仍在（段删除前导后移位，旧位置文件仍是它的回退）→ 保留；
        * 无 seg_id（升级前遗留缓存，无法判定归属）→ 保守保留。
    - 其它无法识别的文件：保留。

    使用全部当前段（而非「选择运行」子集），未选中的段仍保留导出填充缓存。
    Never raises.
    """
    if not node_id:
        return
    try:
        root = Path(folder_paths.get_output_directory()) / "minimax_seg_cache" / str(node_id)
        if not root.is_dir():
            return
        valid_ids: set[str] = set()
        valid_safe_ids: set[str] = set()
        for item in valid_segments:
            if hasattr(item, "seg_id"):
                sid = str(getattr(item, "seg_id", "") or "").strip()
                if sid:
                    valid_ids.add(sid)
                    valid_safe_ids.add(_sanitize_seg_id(sid))
            # 旧的 int 下标入参无法表达稳定身份：只保留不识别，交由下方
            # “无属主历史文件保守保留”逻辑处理。

        def legacy_owner(stem: str) -> str | None:
            """读取旧位置命名 meta 里记录的属主 seg_id（无则 None）。"""
            for meta_sfx in (SFX_FINAL_META, SFX_PRE_META):
                meta_path = root / f"{stem}{meta_sfx}"
                if not meta_path.is_file():
                    continue
                try:
                    data = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                owner = str((data or {}).get("seg_id") or "").strip()
                if owner:
                    return owner
            return None

        removed = 0
        for path in sorted(root.iterdir()):
            if not path.is_file():
                continue
            stem = _split_cache_stem(path.name)
            if stem is None:
                continue
            keep = True
            stable = _STABLE_PREFIX_RE.match(stem)
            if stable:
                keep = stable.group(1) in valid_safe_ids
            else:
                legacy = _LEGACY_PREFIX_RE.match(stem)
                if legacy is None:
                    continue  # 不识别的文件，保留
                owner = legacy_owner(stem)
                if owner is not None:
                    keep = owner in valid_ids
                else:
                    # 升级前遗留缓存，meta 无属主：无法安全判定，保守保留。
                    keep = True
            if not keep and _safe_unlink(path):
                removed += 1
        if removed:
            log.info(
                "Segment cache pruned %d stale file(s) for node %s.", removed, node_id
            )
    except Exception as exc:
        log.debug("Segment cache prune skipped (%s).", exc)


def first_pass_cache_disk_signature(node_id: str | None) -> str:
    """Fingerprint confirm-first-pass ``*.pre.*`` files without creating the cache dir.

    Director ``IS_CHANGED`` cannot see the linked Refine pack (ComfyUI only
    forwards widgets). These ``.pre`` files are written only by the confirmation
    hold, so a second Queue observes a new signature and continues into refine.
    """
    if not node_id:
        return ""
    root = Path(folder_paths.get_output_directory()) / "minimax_seg_cache" / str(node_id)
    if not root.is_dir():
        return ""
    parts: list[str] = []
    try:
        paths: list[Path] = []
        for pattern in ("segid_*.pre.*", "seg_*.pre.*"):
            paths.extend(root.glob(pattern))
        for path in sorted(paths):
            try:
                st = path.stat()
            except OSError:
                continue
            parts.append(f"{path.name}:{int(st.st_mtime_ns)}:{int(st.st_size)}")
    except OSError:
        return ""
    return "|".join(parts)


def inspect_first_pass_cache(
    node_id: str | None,
    plan: DirectorPlan,
) -> dict[str, Any]:
    """Inspect first-pass cache files without loading their tensor payloads.

    Always walks the whole timeline so「选择运行」unselected slots stay visible.
    ``final_cached_count`` is file presence only (``seg_XXXX.pt``), not a
    fingerprint match — Refine knobs are not on this status request.
    """
    current_seed = int(getattr(plan, "sample_seed", 0) or 0)
    result: dict[str, Any] = {
        "exists": False,
        "matches": False,
        "current_seed": current_seed,
        "cached_seeds": [],
        "segment_total": 0,
        "cached_count": 0,
        "matched_count": 0,
        "selected_total": 0,
        "selected_cached": 0,
        "selected_matched": 0,
        "final_cached_count": 0,
        "diff_keys": [],
        "segments": [],
    }
    if not node_id:
        return result

    root = Path(folder_paths.get_output_directory()) / "minimax_seg_cache" / str(node_id)
    all_segments = list(getattr(plan, "segments", None) or [])
    run_indices = getattr(plan, "run_indices", None)
    selected_set = frozenset(run_indices) if run_indices is not None else None
    result["segment_total"] = len(all_segments)

    cached_seeds: set[int] = set()
    all_diffs: set[str] = set()
    rows: list[dict[str, Any]] = []
    final_cached = 0
    for seg in all_segments:
        is_selected = selected_set is None or int(seg.index) in selected_set
        idx = int(seg.index)
        pre_stem = _resolve_stem(root, seg, SFX_PRE_META, SFX_PRE_AV)
        meta_path = root / f"{pre_stem}{SFX_PRE_META}"
        latent_path = root / f"{pre_stem}{SFX_PRE_AV}"
        meta_exists = meta_path.is_file()
        latent_exists = latent_path.is_file()
        cache_exists = meta_exists and latent_exists
        final_stem = _resolve_stem(root, seg, SFX_FINAL_META, SFX_FINAL_FRAMES)
        if (root / f"{final_stem}{SFX_FINAL_FRAMES}").is_file():
            final_cached += 1
        stored: Any = None
        read_error = ""
        if meta_exists:
            try:
                stored = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                read_error = str(exc)

        expected = first_pass_cache_fingerprint(seg, plan)
        stored_cmp = stored
        expected_cmp = expected
        if getattr(plan, "sample_sigmas_linked", False) and isinstance(stored, dict):
            stored_cmp = {k: v for k, v in stored.items() if k != "sigmas"}
            expected_cmp = {k: v for k, v in expected.items() if k != "sigmas"}
        matches = bool(cache_exists and isinstance(stored, dict) and stored_cmp == expected_cmp)
        diff = (
            _fingerprint_diff_keys(stored_cmp, expected_cmp)
            if isinstance(stored, dict)
            else (["<invalid-meta>"] if meta_exists else ["<missing-cache>"])
        )
        if not cache_exists:
            status = "missing"
        elif matches:
            status = "valid"
        else:
            status = "mismatch"
        cached_seed = stored.get("seed") if isinstance(stored, dict) else None
        try:
            if cached_seed is not None:
                cached_seed = int(cached_seed)
                cached_seeds.add(cached_seed)
        except (TypeError, ValueError):
            cached_seed = None
        all_diffs.update(diff)
        rows.append(
            {
                "segment": idx + 1,
                "exists": cache_exists,
                "matches": matches,
                "status": status,
                "selected": is_selected,
                "cached_seed": cached_seed,
                "diff_keys": diff,
                "error": read_error,
            }
        )

    cached_count = sum(1 for row in rows if row["exists"])
    matched_count = sum(1 for row in rows if row["matches"])
    selected_rows = [row for row in rows if row["selected"]]
    selected_total = len(selected_rows) if selected_set is not None else len(rows)
    total = len(rows)
    result.update(
        {
            "exists": cached_count > 0,
            "matches": total > 0 and matched_count == total,
            "cached_seeds": sorted(cached_seeds),
            "cached_count": cached_count,
            "matched_count": matched_count,
            "selected_total": selected_total,
            "selected_cached": sum(1 for row in selected_rows if row["exists"]),
            "selected_matched": sum(1 for row in selected_rows if row["matches"]),
            "final_cached_count": final_cached,
            "diff_keys": sorted(all_diffs),
            "segments": rows,
        }
    )
    return result


def clear_segment_cache(node_id: str | None, kind: str = "final") -> int:
    """Delete cached segment files for this Director node.

    ``kind``:
      - ``first_pass``: only ``seg_XXXX.pre.*`` (一采)
      - ``final``: everything except ``.pre.*`` (成片 / 二采，含 ``.audio.pt``)
      - ``all``: both

    Never creates the cache dir. Returns the number of files removed.
    """
    if not node_id:
        return 0
    if kind not in {"first_pass", "final", "all"}:
        raise ValueError("kind must be first_pass, final or all")
    root = Path(folder_paths.get_output_directory()) / "minimax_seg_cache" / str(node_id)
    if not root.is_dir():
        return 0
    try:
        entries = list(root.iterdir())
    except OSError:
        return 0
    removed = 0
    for path in entries:
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        is_pre = ".pre." in path.name
        if kind == "first_pass" and not is_pre:
            continue
        if kind == "final" and is_pre:
            continue
        if _safe_unlink(path):
            removed += 1
    if removed:
        log.info("Cleared %s cache for node %s (%d file(s)).", kind, node_id, removed)
    return removed
