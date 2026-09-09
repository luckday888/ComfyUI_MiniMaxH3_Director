"""段间引导缓存「稳定段 id」回归测试。

复现并固化修复：删除前导段后，保留段被重新编号（位置 index 改变），
其磁盘缓存必须仍能被找回（段间引导接续），且 prune 不得误删；旧的位置
命名缓存若属于别的段，绝不允许串用。

运行方式（在装有 ComfyUI / torch 的环境）：
    cd <ComfyUI>/custom_nodes/ComfyUI_MiniMaxH3_Director
    PYTHONPATH=<ComfyUI 根目录> python -m pytest tests/test_segment_stable_cache.py -q
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

# 以真实顶层包名导入（插件目录名），使 director 内部的 ``from ..lib`` 相对
# 导入成立。parents[1]=插件根，parents[2]=其所在的 custom_nodes 目录。
# folder_paths / comfy 由 PYTHONPATH 指向 ComfyUI 根提供。
_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PLUGIN_ROOT.parent))

import torch

import folder_paths

_PKG = _PLUGIN_ROOT.name
sc = importlib.import_module(f"{_PKG}.director.segment_cache")
resolve_seg_id = importlib.import_module(f"{_PKG}.director.plan").resolve_seg_id


def _seg(seg_id: str, index: int):
    """缓存函数只用到 seg_id / index（fingerprint 已在测试中打桩）。"""
    return SimpleNamespace(seg_id=seg_id, index=index)


def _patch_fp(monkeypatch):
    """受控指纹：仅含属主、位置下标与源身份，避免依赖完整 DirectorPlan 字段。

    保存与读取都走这个桩：删除前导后 index 由 9 变为 0，指纹因此不同（stale），
    但源身份不变、属主相同——allow_stale 的接续路径必须仍命中。
    """

    def fp(seg, plan):
        return {
            "seg_id": seg.seg_id,
            "index": int(seg.index),
            sc.SOURCE_VIDEO_FP_KEY: [],
        }

    monkeypatch.setattr(sc, "segment_cache_fingerprint", fp)


def _frames():
    return torch.zeros(6, 8, 8, 3, dtype=torch.float32)


def test_resolve_seg_id_prefers_persistent_id():
    assert resolve_seg_id({"id": "abc"}, 3) == "abc"
    assert resolve_seg_id({"seg_id": "xyz"}, 2) == "xyz"
    # 无身份来源时回退到确定性的位置键（跨运行稳定，不能是随机值）
    assert resolve_seg_id({}, 3) == "p0003"
    assert resolve_seg_id(None, 1) == "p0001"


def test_cache_survives_delete_predecessors_and_prune(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    _patch_fp(monkeypatch)
    node = "nodeA"
    plan = SimpleNamespace(raw={})

    # ── 删除前：时间轴上有 S1..S10 十段，S10 在位置下标 9 ──
    for i in range(10):
        seg = _seg(f"S{i + 1}", i)
        sc.save_segment_cache(
            node,
            seg,
            plan,
            _frames(),
            av_latent={"samples": torch.zeros(1)},
            handoff={"trim_frames": 4, "export_frames": 120},
            audio={"waveform": torch.zeros(1, 320), "sample_rate": 32000},
        )

    # ── 删除前导 1-9：保留段 S10 被重排到下标 0；新增 S11 在下标 1 ──
    kept = _seg("S10", 0)
    new_seg = _seg("S11", 1)

    # 每次运行开始的清理：只认识当前两段
    sc.prune_segment_cache(node, [kept, new_seg])

    root = tmp_path / "minimax_seg_cache" / node
    # 已删段 S1..S9 的稳定命名缓存被清掉
    for i in range(1, 10):
        assert not (root / f"segid_S{i}.pt").exists(), f"S{i} 应被 prune 删除"
    # 保留段 S10 的缓存必须存活（修复前会因下标 9 不在 {0,1} 被误删）
    assert (root / "segid_S10.pt").is_file()

    # 位置漂移导致精确指纹 miss（符合预期：帧范围变了）
    assert sc.load_segment_cache(node, kept, plan) is None
    # 段间引导走 allow_stale：源未变 + 属主相同 → 必须命中
    frames = sc.load_segment_cache(node, kept, plan, allow_stale=True)
    assert frames is not None and int(frames.shape[0]) == 6

    # 下一段 S11 接续所需的 AV latent / handoff / audio 也都要能取到
    assert sc.load_segment_av_latent(node, kept, plan, allow_stale=True) is not None
    handoff = sc.load_segment_handoff_meta(node, kept, plan, allow_stale=True)
    assert handoff == {"trim_frames": 4, "export_frames": 120}
    audio = sc.load_segment_audio(node, kept, plan, allow_stale=True)
    assert audio is not None and audio["sample_rate"] == 32000


def test_legacy_position_cache_owned_by_other_segment_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    _patch_fp(monkeypatch)
    node = "nodeB"
    root = tmp_path / "minimax_seg_cache" / node
    root.mkdir(parents=True, exist_ok=True)

    # 手工构造一个旧位置命名 seg_0000 缓存，其 meta 明确属于别的段 OTHER
    torch.save(_frames(), root / "seg_0000.pt")
    (root / "seg_0000.meta.json").write_text(
        json.dumps({"seg_id": "OTHER", "index": 0, sc.SOURCE_VIDEO_FP_KEY: []}),
        encoding="utf-8",
    )

    # 当前下标 0 的段是 S10（删除前导后重排而来）：不允许拿 OTHER 的产物接续
    cur = _seg("S10", 0)
    plan = SimpleNamespace(raw={})
    assert sc.load_segment_cache(node, cur, plan, allow_stale=True) is None

    # 同样防 AV latent 串段
    torch.save({"samples": torch.zeros(1)}, root / "seg_0000.av.pt")
    assert sc.load_segment_av_latent(node, cur, plan, allow_stale=True) is None


def test_legacy_cache_without_owner_is_backward_compatible(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    _patch_fp(monkeypatch)
    node = "nodeC"
    root = tmp_path / "minimax_seg_cache" / node
    root.mkdir(parents=True, exist_ok=True)

    # 升级前遗留缓存：meta 没有 seg_id 字段
    torch.save(_frames(), root / "seg_0009.pt")
    (root / "seg_0009.meta.json").write_text(
        json.dumps({"index": 9, sc.SOURCE_VIDEO_FP_KEY: []}),
        encoding="utf-8",
    )
    seg = _seg("S10", 9)  # 位置仍对齐（未删除）的普通场景
    plan = SimpleNamespace(raw={})
    # 历史缓存向后兼容：allow_stale 下可读取，不应被属主校验误伤
    assert sc.load_segment_cache(node, seg, plan, allow_stale=True) is not None

    # prune 对无属主的历史文件保守保留（绝不误删升级前缓存）
    sc.prune_segment_cache(node, [seg])
    assert (root / "seg_0009.pt").is_file()
