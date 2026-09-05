"""Lightweight in-sampling audio preview for MiniMax H3 (fl2va joint diffusion).

The per-step sampler callback receives ``x0`` (the predicted clean AV latent),
which is a NestedTensor whose last stream is the audio latent — the same stream
``VAEDecodeAudio`` decodes for the final clip. This module decodes that stream
with the official audio VAE and encodes a PCM16 WAV (base64) so the UI can offer
a manual "listen to the current step" preview during the late denoising steps.

Design notes:
- There is no tiny audio decoder analogous to the video TAE; we must run the full
  fp32 audio VAE, so callers decode sparingly (late steps only, throttled).
- Early-step audio is noise (and the official decode applies a std*5 gain), so we
  only decode in the last ~30% of steps.
- Everything here is best-effort: any failure returns None and never breaks sampling.
"""

from __future__ import annotations

import base64
import io
import logging
import wave
from typing import Any

import torch

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.audio_preview")

# Only decode in the late denoising phase (early audio is amplified white noise)
# and at most every N steps plus the final step.
LATE_STEP_FRAC = 0.7
PREVIEW_EVERY = 3


def should_emit_audio_preview(step: int, total_steps: int) -> bool:
    """Late-phase throttle: true on a sparse subset of steps near the end."""
    try:
        total = max(1, int(total_steps))
        s = int(step)
        start = int(total * LATE_STEP_FRAC)
        if s < start:
            return False
        last = total - 1
        return s % max(1, PREVIEW_EVERY) == 0 or s >= last
    except Exception:
        return False


def _import_vae_decode_audio():
    try:
        from comfy_extras.nodes_audio import vae_decode_audio

        return vae_decode_audio
    except ImportError:  # older ComfyUI layouts
        try:
            from comfy_extras.nodes_lt import vae_decode_audio  # type: ignore

            return vae_decode_audio
        except Exception:
            return None


def decode_preview_audio_dict(x0: Any, audio_vae: Any) -> dict | None:
    """Decode the current-step AV latent ``x0`` to a ComfyUI AUDIO dict.

    Mirrors the final decode path: ``vae_decode_audio`` unbinds the last nested
    stream (audio), runs ``vae.decode`` and normalizes loudness.
    """
    if audio_vae is None or x0 is None:
        return None
    fn = _import_vae_decode_audio()
    if fn is None:
        return None
    try:
        audio = fn(audio_vae, {"samples": x0})
        wave = audio.get("waveform") if isinstance(audio, dict) else None
        if not isinstance(wave, torch.Tensor) or wave.numel() <= 0:
            return None
        return {"waveform": wave, "sample_rate": int(audio.get("sample_rate") or 32000)}
    except Exception as exc:
        log.debug("Audio preview decode skipped: %s", exc)
        return None


def _waveform_to_wav_b64(audio: dict) -> tuple[str, int] | None:
    """Encode a ComfyUI AUDIO dict ([1,C,T] tensor) as PCM16 WAV base64."""
    try:
        wf = audio.get("waveform")
        sr = int(audio.get("sample_rate") or 32000)
        if not isinstance(wf, torch.Tensor) or wf.numel() <= 0:
            return None
        t = wf.detach().cpu().float()
        if t.ndim == 3:
            t = t[0]  # [C, T]
        if t.ndim != 2:
            return None
        # Keep at most stereo (H3 audio is mono/stereo).
        if t.shape[0] > 2:
            t = t[:2]
        ch = int(t.shape[0])
        n = int(t.shape[-1])
        if n <= 0 or ch <= 0:
            return None
        pcm = t.clamp(-1.0, 1.0).mul(32767.0).round().to(torch.int16)
        # [C, T] -> [T, C] interleaved -> little-endian PCM16 bytes
        arr = pcm.numpy().transpose(1, 0).reshape(-1).astype("<i2", copy=False)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(ch)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            wav.writeframes(arr.tobytes())
        return base64.b64encode(buf.getvalue()).decode("ascii"), sr
    except Exception as exc:
        log.debug("Audio preview WAV encode skipped: %s", exc)
        return None


def x0_to_audio_preview_b64(x0: Any, audio_vae: Any) -> tuple[str, int] | None:
    """Full path: current-step ``x0`` -> (WAV base64, sample_rate), or None."""
    audio = decode_preview_audio_dict(x0, audio_vae)
    if audio is None:
        return None
    return _waveform_to_wav_b64(audio)
