"""Encoder backends: command-line shape and the NVENC -> libx264 fallback."""

import logging

import pytest

from auto_redact.encode import (
    Libx264Encoder,
    NvencEncoder,
    make_encoder,
    run_with_fallback,
)
from auto_redact.errors import RenderError
from auto_redact.protocols import Encoder


def test_make_encoder_by_name_and_unknown_name():
    assert isinstance(make_encoder("libx264"), Libx264Encoder)
    assert isinstance(make_encoder("h264_nvenc"), NvencEncoder)
    with pytest.raises(RenderError, match="encoder must be one of"):
        make_encoder("nope")


def test_codec_args():
    assert Libx264Encoder().codec_args(18, "slow") == [
        "-c:v", "libx264", "-crf", "18", "-preset", "slow",
    ]
    args = NvencEncoder().codec_args(23, "medium")
    assert args[:2] == ["-c:v", "h264_nvenc"]
    assert args[args.index("-cq") + 1] == "23"
    assert args[args.index("-preset") + 1] == "p4"  # medium maps to p4
    assert NvencEncoder().codec_args(23, "p7")[-1] == "p7"  # already an nvenc preset


def test_libx264_has_no_fallback():
    assert Libx264Encoder().fallback("medium", RenderError("boom")) is None


def test_nvenc_falls_back_to_libx264_with_a_usable_preset(caplog):
    with caplog.at_level(logging.WARNING):
        fallback = NvencEncoder().fallback("p5", RenderError("no vram\nmore"))
    encoder, preset = fallback
    assert isinstance(encoder, Libx264Encoder) and preset == "medium"
    assert "falling back" in caplog.text
    assert NvencEncoder().fallback("slow", RenderError("x"))[1] == "slow"


def test_run_with_fallback_retries_once_and_reports_what_worked():
    calls = []

    def run(encoder, preset):
        calls.append((encoder.name, preset))
        if encoder.name == "h264_nvenc":
            raise RenderError("nvenc broke")

    encoder, preset = run_with_fallback(NvencEncoder(), "p4", run)
    assert encoder.name == "libx264" and preset == "medium"
    assert calls == [("h264_nvenc", "p4"), ("libx264", "medium")]


def test_run_with_fallback_reraises_without_a_fallback():
    def run(encoder, preset):
        raise RenderError("broke")

    with pytest.raises(RenderError, match="broke"):
        run_with_fallback(Libx264Encoder(), "medium", run)


def test_a_fake_encoder_satisfies_the_protocol():
    class Fake:
        name = "fake"

        def prepare(self):
            pass

        def codec_args(self, crf, preset):
            return ["-c:v", "fake"]

        def fallback(self, preset, error):
            return None

    encoder: Encoder = Fake()
    assert run_with_fallback(encoder, "x", lambda e, p: None) == (encoder, "x")
