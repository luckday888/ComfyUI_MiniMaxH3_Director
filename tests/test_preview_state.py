"""f0f8f47「采样运行中即时开关 实时预览/音频预览」后端逻辑回归测试。

覆盖两块可在无 GPU/无真实采样下确定性验证的逻辑：

1. ``director.preview_state`` 进程内状态机：
   - 任务启动 init 用提交值重置、清掉上次运行残留（含一次性 audio_arm）；
   - 采样进行中 set：音频 关→开 置 audio_arm，开→开不重复置位；
   - get 的 audio_arm 读后即焚（只强推下一采样步一帧）；
   - 多节点隔离、空 node_id 安全。
2. HTTP 路由 ``POST /minimax/director/set_preview``（``minimax_set_preview``）：
   - 合法切换落状态，下一步 get 立即读到（含 arm）；
   - node_id 非数字 / 非法 JSON → 400；
   - 缺省的开关项保持不变（None=不改）。

采样回调「真的解码/推送画面与音频」属于运行时交互，需真实采样，不在本测试范围。

运行：
    cd /workspace/ComfyUI
    PYTHONPATH=/workspace/ComfyUI:<custom_nodes 父目录> \\
        /root/.pyenv/versions/3.11.1/bin/python -m pytest \\
        <插件根>/tests/test_preview_state.py -q
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

# 以真实顶层包名导入（使 director 内相对导入成立）；folder_paths/server 由
# PYTHONPATH 指向 ComfyUI 根提供。
_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PLUGIN_ROOT.parent))

_PKG = _PLUGIN_ROOT.name
preview_state = importlib.import_module(f"{_PKG}.director.preview_state")
http_routes = importlib.import_module(f"{_PKG}.director.http_routes")


class _FakeRequest:
    def __init__(self, payload=None, *, raise_json: bool = False):
        self._payload = payload if payload is not None else {}
        self._raise = raise_json

    async def json(self):
        if self._raise:
            raise ValueError("bad json")
        return self._payload


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def setup_function(_):
    # 每个用例独立节点，避免模块级状态串扰。
    pass


def test_init_resets_submitted_values_and_clears_arm():
    nid = "node-init"
    preview_state.init_preview_state(nid, tae=True, audio=False)
    # 运行中把音频打开（置 arm）
    preview_state.set_preview(nid, audio=True)
    assert preview_state.get_preview(nid)["audio_arm"] is True

    # 新一轮任务启动：提交值 audio=False，且残留 arm 必须被清掉
    preview_state.init_preview_state(nid, tae=True, audio=False)
    pv = preview_state.get_preview(nid)
    assert pv == {"tae": True, "audio": False, "audio_arm": False}


def test_audio_off_to_on_arms_once_then_consumed():
    nid = "node-arm"
    preview_state.init_preview_state(nid, tae=False, audio=False)

    # 关→开：置 arm，下一步立即推一帧
    preview_state.set_preview(nid, audio=True)
    first = preview_state.get_preview(nid)
    assert first["audio"] is True and first["audio_arm"] is True

    # arm 读后即焚：再读不再强制
    second = preview_state.get_preview(nid)
    assert second["audio"] is True and second["audio_arm"] is False

    # 开→开：不重复置 arm
    preview_state.set_preview(nid, audio=True)
    assert preview_state.get_preview(nid)["audio_arm"] is False

    # 开→关→开：再次置 arm
    preview_state.set_preview(nid, audio=False)
    preview_state.set_preview(nid, audio=True)
    assert preview_state.get_preview(nid)["audio_arm"] is True


def test_tae_independent_and_node_isolation():
    a, b = "node-a", "node-b"
    preview_state.init_preview_state(a, tae=False, audio=False)
    preview_state.init_preview_state(b, tae=True, audio=True)

    preview_state.set_preview(a, tae=True)  # 只动 A 的画面
    pa = preview_state.get_preview(a)
    pb = preview_state.get_preview(b)
    assert pa["tae"] is True and pa["audio"] is False
    assert pb["tae"] is True and pb["audio"] is True


def test_none_node_id_is_safe():
    preview_state.init_preview_state(None, tae=True, audio=True)
    preview_state.set_preview(None, tae=False)
    assert preview_state.get_preview(None) == {
        "tae": False,
        "audio": False,
        "audio_arm": False,
    }


def test_route_toggle_takes_effect_next_read():
    nid = "777001"
    preview_state.init_preview_state(nid, tae=False, audio=False)

    resp = _run(
        http_routes.minimax_set_preview(
            _FakeRequest({"node_id": nid, "live_tae_preview": "true", "live_audio_preview": "on"})
        )
    )
    assert resp.status == 200
    pv = preview_state.get_preview(nid)
    assert pv["tae"] is True and pv["audio"] is True and pv["audio_arm"] is True


def test_route_omitted_flag_is_unchanged():
    nid = "777002"
    preview_state.init_preview_state(nid, tae=False, audio=True)
    # 只传 tae，audio 缺省（None）必须保持 True
    resp = _run(
        http_routes.minimax_set_preview(
            _FakeRequest({"node_id": nid, "live_tae_preview": True})
        )
    )
    assert resp.status == 200
    pv = preview_state.get_preview(nid)
    assert pv["tae"] is True and pv["audio"] is True
    # audio 一直开着，本次没有 关→开，不应置 arm
    assert pv["audio_arm"] is False


def test_route_rejects_bad_node_id_and_bad_json():
    bad_id = _run(
        http_routes.minimax_set_preview(
            _FakeRequest({"node_id": "abc", "live_tae_preview": True})
        )
    )
    assert bad_id.status == 400

    empty_id = _run(
        http_routes.minimax_set_preview(
            _FakeRequest({"node_id": "", "live_tae_preview": True})
        )
    )
    assert empty_id.status == 400

    bad_json = _run(http_routes.minimax_set_preview(_FakeRequest(raise_json=True)))
    assert bad_json.status == 400
