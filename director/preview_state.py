"""采样运行中的实时预览开关状态（进程内、按 node_id 索引、线程安全）。

实时预览（TAE 画面）/ 音频预览开关在 UI 上可于采样进行中随时切换。开关不再只走
「widget → Queue Prompt → plan.raw」这条静态快照路径，而是：

- 任务启动时 :func:`init_preview_state` 用 ``plan.raw`` 的提交值重置该节点状态，
  保证每次运行从提交时的开关起步、不继承上一次运行的残留；
- 采样进行中前端 toggle 通过 HTTP 路由 ``POST /minimax/director/set_preview``
  调 :func:`set_preview` 即时覆盖；
- 采样回调每步调 :func:`get_preview` 读最新值，决定是否解码/推送画面与音频。

ComfyUI 的 HTTP handler 跑在 aiohttp event loop 线程，采样跑在 worker 线程，
二者同进程，故用模块级 dict + :class:`threading.Lock` 即可，无需队列或文件。
node_id 数量等于画布上 Director 节点数，状态表不会无限增长；``init`` 每次运行
开头重置，也无需主动清理。
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
# node_id(str) -> {"tae": bool, "audio": bool, "audio_arm": bool}
_state: dict[str, dict[str, bool]] = {}


def init_preview_state(node_id: str | None, *, tae: bool, audio: bool) -> None:
    """任务启动时用提交值（plan.raw）重置该节点的运行时预览状态。

    会清掉 ``audio_arm`` 等一次性标志，确保新运行不受上次运行 / 运行前
    非采样期 HTTP 上报的残留影响。
    """
    if not node_id:
        return
    with _lock:
        _state[str(node_id)] = {"tae": bool(tae), "audio": bool(audio), "audio_arm": False}


def set_preview(
    node_id: str | None,
    *,
    tae: bool | None = None,
    audio: bool | None = None,
) -> None:
    """采样进行中由 HTTP 路由调用，更新某节点的预览开关。

    ``tae`` / ``audio`` 为 ``None`` 表示该项不变。当 ``audio`` 由关→开时置
    ``audio_arm=True``：音频解码有自适应步距节流（见 audio_preview.should_emit_audio_preview），
    arm 让「运行中刚打开」的下一采样步立即推一帧，不必等下一节流步。
    """
    if not node_id:
        return
    with _lock:
        cur = _state.setdefault(
            str(node_id), {"tae": False, "audio": False, "audio_arm": False}
        )
        if tae is not None:
            cur["tae"] = bool(tae)
        if audio is not None:
            new_audio = bool(audio)
            if new_audio and not cur["audio"]:
                cur["audio_arm"] = True
            cur["audio"] = new_audio


def get_preview(node_id: str | None) -> dict[str, bool]:
    """采样回调每步读取最新开关；返回 ``{"tae", "audio", "audio_arm"}``。

    ``audio_arm`` 为一次性标志：读取即消费（读后清零），保证只强制推送一帧。
    节点无状态（理论上 init 未跑到）时返回全 False。
    """
    if not node_id:
        return {"tae": False, "audio": False, "audio_arm": False}
    with _lock:
        cur = _state.get(str(node_id))
        if cur is None:
            return {"tae": False, "audio": False, "audio_arm": False}
        arm = bool(cur.get("audio_arm"))
        cur["audio_arm"] = False
        return {"tae": bool(cur.get("tae")), "audio": bool(cur.get("audio")), "audio_arm": arm}
