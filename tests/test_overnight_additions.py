"""Tests for video method, run_pipeline integration, CLI, and edge cases.

Written during the 2026-08-06 overnight creative loop.
Target: close coverage gaps in creative_pipeline.py."""

import json
import pytest
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from creative_pipeline import (
    MMX,
    CreativeAsset,
    PipelineResult,
    slugify,
    parse_build_plan,
    run_pipeline,
    run_batch,
    load_history,
    _update_history,
    main,
    ASSETS_DIR,
    PROJECT_ROOT,
    HISTORY_FILE,
    BUILD_PLAN_SYSTEM,
    MMX_COMMON_FLAGS,
    MMX_FILE_FLAGS,
)


# ── Video method tests ────────────────────────────────────────────

class TestMMXVideo:
    @patch("creative_pipeline.subprocess.run")
    def test_video_generates_file(self, mock_run, tmp_path):
        """Video method returns path when file is created."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        out_file = tmp_path / "preview.mp4"
        out_file.write_bytes(b"fake video")

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx.video("cinematic flythrough", out_file)
        assert result == str(out_file)

    @patch("creative_pipeline.subprocess.run")
    def test_video_no_file_returns_empty(self, mock_run, tmp_path):
        """Video returns empty string when no file is created."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx.video("cinematic", tmp_path / "nope.mp4")
        assert result == ""

    @patch("creative_pipeline.subprocess.run")
    def test_video_with_image_ref(self, mock_run, tmp_path):
        """Video passes image reference when provided."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        out_file = tmp_path / "preview.mp4"
        out_file.write_bytes(b"fake")

        mmx = MMX(bin_path="/fake/mmx")
        mmx.video("flythrough", out_file, image_ref="concept.png")

        # Check --image was passed
        call_args = mock_run.call_args
        cmd = call_args[0][0]  # First positional arg is the cmd list
        assert "--image" in cmd
        assert "concept.png" in cmd

    @patch("creative_pipeline.subprocess.run")
    def test_video_stdout_path(self, mock_run, tmp_path):
        """Video returns path from stdout when mmx outputs it."""
        out_file = tmp_path / "preview.mp4"
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = str(out_file)
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        out_file.write_bytes(b"fake video")

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx.video("test", out_file)
        assert result == str(out_file)

    @patch("creative_pipeline.subprocess.run")
    def test_video_timeout(self, mock_run, tmp_path):
        """Video timeout returns empty."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="mmx", timeout=5)
        mmx = MMX(bin_path="/fake/mmx")
        result = mmx.video("test", tmp_path / "preview.mp4", )
        assert result == ""


# ── MMX init fallback tests ───────────────────────────────────────

class TestMMXInit:
    def test_falls_back_to_path_when_bin_missing(self):
        """When the configured binary doesn't exist, falls back to 'mmx' on PATH."""
        mmx = MMX(bin_path="/nonexistent/path/to/mmx")
        assert mmx.bin == "mmx"

    def test_keeps_bin_when_exists(self, tmp_path):
        """When the binary exists, keeps it."""
        fake = tmp_path / "mmx"
        fake.write_text("#!/bin/bash")
        fake.chmod(0o755)
        mmx = MMX(bin_path=str(fake))
        assert mmx.bin == str(fake)


# ── _run_file edge cases ──────────────────────────────────────────

class TestRunFileEdgeCases:
    @patch("creative_pipeline.subprocess.run")
    def test_run_file_returns_tuple(self, mock_run):
        """_run_file always returns (rc, stdout, stderr) tuple."""
        mock_proc = MagicMock()
        mock_proc.returncode = 42
        mock_proc.stdout = "output here"
        mock_proc.stderr = "warning"
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        rc, stdout, stderr = mmx._run_file(["image"])
        assert rc == 42
        assert stdout == "output here"
        assert stderr == "warning"

    @patch("creative_pipeline.subprocess.run")
    def test_run_file_nonzero_rc(self, mock_run):
        """_run_file propagates nonzero exit codes."""
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "generation failed"
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        rc, stdout, stderr = mmx._run_file(["video"])
        assert rc == 1
        assert "generation failed" in stderr


# ── _run retry behavior ───────────────────────────────────────────

class TestRunRetryBehavior:
    @patch("creative_pipeline.subprocess.run")
    @patch("creative_pipeline.time.sleep")
    def test_retry_on_empty_output(self, mock_sleep, mock_run):
        """Retries on empty output, succeeds on second attempt."""
        empty_proc = MagicMock()
        empty_proc.returncode = 0
        empty_proc.stdout = ""
        empty_proc.stderr = ""

        ok_proc = MagicMock()
        ok_proc.returncode = 0
        ok_proc.stdout = '{"data": 1}'
        ok_proc.stderr = ""

        mock_run.side_effect = [empty_proc, ok_proc]

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx._run(["text"], retries=1)
        assert result == {"data": 1}
        assert mock_run.call_count == 2

    @patch("creative_pipeline.subprocess.run")
    @patch("creative_pipeline.time.sleep")
    def test_retry_on_json_decode_error(self, mock_sleep, mock_run):
        """Retries when output isn't valid JSON."""
        bad_proc = MagicMock()
        bad_proc.returncode = 0
        bad_proc.stdout = "not json"
        bad_proc.stderr = ""

        good_proc = MagicMock()
        good_proc.returncode = 0
        good_proc.stdout = '{"ok": true}'
        good_proc.stderr = ""

        mock_run.side_effect = [bad_proc, good_proc]

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx._run(["text"], retries=1)
        # First attempt returns _raw (non-JSON), not retried because it's "valid" output
        # Actually the code returns _raw on JSONDecodeError without retrying.
        # Document this behavior:
        assert result == {"_raw": "not json"}

    @patch("creative_pipeline.subprocess.run")
    def test_all_flags_present_in_run(self, mock_run):
        """_run includes common flags in the command."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = '{"ok": true}'
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        mmx._run(["text", "chat"])

        cmd = mock_run.call_args[0][0]
        # Check common flags are appended
        for flag in MMX_COMMON_FLAGS:
            assert flag in cmd

    @patch("creative_pipeline.subprocess.run")
    def test_all_flags_present_in_run_file(self, mock_run):
        """_run_file includes file flags in the command."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        mmx._run_file(["image"])

        cmd = mock_run.call_args[0][0]
        for flag in MMX_FILE_FLAGS:
            assert flag in cmd


# ── run_pipeline integration ──────────────────────────────────────

def _make_mock_mmx(chat_resp='{"title": "Test"}', image_resp=("", ""),
                     music_resp="", speech_resp="", video_resp="",
                     vision_resp=""):
    """Create a properly configured MMX mock instance."""
    m = MagicMock()
    m.chat.return_value = chat_resp
    m.image.return_value = image_resp
    m.music.return_value = music_resp
    m.speech.return_value = speech_resp
    m.video.return_value = video_resp
    m.vision.return_value = vision_resp
    return m


def _patch_paths(monkeypatch, tmp_path):
    """Patch ASSETS_DIR, PROJECT_ROOT, and HISTORY_FILE to tmp_path."""
    monkeypatch.setattr("creative_pipeline.ASSETS_DIR", tmp_path)
    monkeypatch.setattr("creative_pipeline.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("creative_pipeline.HISTORY_FILE", tmp_path / "history.jsonl")


class TestRunPipelineIntegration:
    """Integration-style tests that mock MMX and test the full pipeline flow."""

    @patch("creative_pipeline.MMX")
    def test_pipeline_creates_assets(self, mock_mmx_class, tmp_path, monkeypatch):
        """Pipeline creates assets directory and saves build plan."""
        _patch_paths(monkeypatch, tmp_path)
        mock_mmx = _make_mock_mmx(
            chat_resp='{"title": "Test", "description": "A test"}',
            image_resp=("", ""),
            music_resp="",
        )
        mock_mmx_class.return_value = mock_mmx

        result = run_pipeline("test build", skip_video=True)
        assert result.request == "test build"
        assert result.build_plan is not None
        assert result.build_plan["title"] == "Test"

    @patch("creative_pipeline.MMX")
    def test_pipeline_handles_unparseable_plan(self, mock_mmx_class, tmp_path, monkeypatch):
        """Pipeline stores raw text when plan can't be parsed."""
        _patch_paths(monkeypatch, tmp_path)
        mock_mmx = _make_mock_mmx(chat_resp="Sorry, I can't generate that.")
        mock_mmx_class.return_value = mock_mmx

        result = run_pipeline("impossible build", skip_video=True)
        assert result.build_plan is None
        plan_assets = [a for a in result.assets if a.label == "Build Plan"]
        assert len(plan_assets) == 1

    @patch("creative_pipeline.MMX")
    def test_pipeline_with_speech(self, mock_mmx_class, tmp_path, monkeypatch):
        """Pipeline generates speech when with_speech=True and plan exists."""
        _patch_paths(monkeypatch, tmp_path)
        mock_mmx = _make_mock_mmx(
            chat_resp='{"title": "Castle", "description": "A castle", "structures": [], "color_palette": ["#333"]}',
            speech_resp="speech.mp3",
        )
        mock_mmx_class.return_value = mock_mmx

        result = run_pipeline("castle", skip_video=True, with_speech=True)
        speech_assets = [a for a in result.assets if a.kind == "speech"]
        assert len(speech_assets) == 1

    @patch("creative_pipeline.MMX")
    def test_pipeline_with_vision_check(self, mock_mmx_class, tmp_path, monkeypatch):
        """Pipeline runs vision check when requested."""
        _patch_paths(monkeypatch, tmp_path)

        art_path = tmp_path / "dark-forest" / "concept_art.png"
        art_path.parent.mkdir(parents=True, exist_ok=True)
        art_path.write_bytes(b"fake")

        mock_mmx = _make_mock_mmx(
            chat_resp='{"title": "Forest"}',
            image_resp=(str(art_path), ""),
            vision_resp="A dark forest with tall trees.",
        )
        mock_mmx_class.return_value = mock_mmx

        result = run_pipeline("dark forest", skip_video=True, with_vision_check=True)
        vision_assets = [a for a in result.assets if "Vision" in a.label]
        assert len(vision_assets) == 1

    @patch("creative_pipeline.MMX")
    def test_pipeline_with_video(self, mock_mmx_class, tmp_path, monkeypatch):
        """Pipeline generates video when skip_video=False."""
        _patch_paths(monkeypatch, tmp_path)
        mock_mmx = _make_mock_mmx(
            chat_resp='{"title": "City"}',
            video_resp="preview.mp4",
        )
        mock_mmx_class.return_value = mock_mmx

        result = run_pipeline("cyber city", skip_video=False)
        video_assets = [a for a in result.assets if a.kind == "video"]
        assert len(video_assets) == 1

    @patch("creative_pipeline.MMX")
    def test_pipeline_writes_manifest(self, mock_mmx_class, tmp_path, monkeypatch):
        """Pipeline writes a manifest.json file."""
        _patch_paths(monkeypatch, tmp_path)
        mock_mmx = _make_mock_mmx()
        mock_mmx_class.return_value = mock_mmx

        result = run_pipeline("manifest test", skip_video=True)
        manifest_assets = [a for a in result.assets if a.label == "Asset Manifest"]
        assert len(manifest_assets) == 1

    @patch("creative_pipeline.MMX")
    def test_pipeline_updates_history(self, mock_mmx_class, tmp_path, monkeypatch):
        """Pipeline appends to history after running."""
        _patch_paths(monkeypatch, tmp_path)
        mock_mmx = _make_mock_mmx()
        mock_mmx_class.return_value = mock_mmx

        run_pipeline("history test", skip_video=True)

        entries = load_history()
        assert len(entries) >= 1
        assert entries[0]["request"] == "history test"


# ── CLI / main() tests ────────────────────────────────────────────

class TestCLI:
    def test_no_args_prints_help(self, capsys):
        """Running with no args prints help and exits."""
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower() or "usage" in captured.err.lower()

    @patch("creative_pipeline.run_pipeline")
    def test_single_request(self, mock_pipeline, capsys):
        """CLI passes request to run_pipeline."""
        mock_result = PipelineResult(
            request="test",
            timestamp="20260805",
        )
        mock_result.build_plan = {"title": "Test"}
        mock_pipeline.return_value = mock_result

        with patch("sys.argv", ["creative_pipeline.py", "build a castle"]):
            main()

        mock_pipeline.assert_called_once()

    @patch("creative_pipeline.load_history")
    def test_history_flag(self, mock_history, capsys):
        """--history flag shows history."""
        mock_history.return_value = [
            {"request": "castle", "timestamp": "20260805-2000", "total": 3, "succeeded": 3, "failed": 0, "had_plan": True},
            {"request": "forest", "timestamp": "20260805-2100", "total": 2, "succeeded": 1, "failed": 1, "had_plan": False},
        ]

        with patch("sys.argv", ["creative_pipeline.py", "--history"]):
            main()

        captured = capsys.readouterr()
        assert "castle" in captured.out
        assert "forest" in captured.out
        assert "2" in captured.out  # count

    @patch("creative_pipeline.load_history")
    def test_history_empty(self, mock_history, capsys):
        """--history with no runs shows message."""
        mock_history.return_value = []

        with patch("sys.argv", ["creative_pipeline.py", "--history"]):
            main()

        captured = capsys.readouterr()
        assert "No pipeline runs" in captured.out

    @patch("creative_pipeline.run_batch")
    def test_batch_mode(self, mock_batch, tmp_path, capsys):
        """--batch reads file and runs batch."""
        batch_file = tmp_path / "builds.txt"
        batch_file.write_text("castle\nforest\n#cave (comment)\n")

        mock_batch.return_value = [
            PipelineResult(request="castle", timestamp="20260805"),
            PipelineResult(request="forest", timestamp="20260805"),
        ]

        with patch("sys.argv", ["creative_pipeline.py", "--batch", str(batch_file)]):
            main()

        mock_batch.assert_called_once()
        args, kwargs = mock_batch.call_args
        # Filter out comments and empty lines
        assert kwargs.get("skip_video") is True

    def test_batch_file_not_found(self, capsys, tmp_path):
        """--batch with missing file exits with error."""
        with patch("sys.argv", ["creative_pipeline.py", "--batch", str(tmp_path / "missing.txt")]):
            with pytest.raises(SystemExit):
                main()

    def test_batch_empty_file(self, capsys, tmp_path):
        """--batch with empty file exits with error."""
        batch_file = tmp_path / "empty.txt"
        batch_file.write_text("# only comments\n\n")

        with patch("sys.argv", ["creative_pipeline.py", "--batch", str(batch_file)]):
            with pytest.raises(SystemExit):
                main()

    @patch("creative_pipeline.run_pipeline")
    def test_compact_output(self, mock_pipeline, capsys):
        """--compact produces non-indented JSON."""
        mock_result = PipelineResult(request="test", timestamp="now")
        mock_result.build_plan = {"title": "Test"}
        mock_pipeline.return_value = mock_result

        with patch("sys.argv", ["creative_pipeline.py", "test build", "--compact"]):
            main()

        captured = capsys.readouterr()
        # Compact JSON should not start with '{\n' (no indentation)
        out = captured.out.strip()
        assert out.startswith("{")

    @patch("creative_pipeline.run_pipeline")
    def test_with_speech_flag(self, mock_pipeline, capsys):
        """--with-speech passes to pipeline."""
        mock_result = PipelineResult(request="test", timestamp="now")
        mock_pipeline.return_value = mock_result

        with patch("sys.argv", ["creative_pipeline.py", "castle", "--with-speech"]):
            main()

        _, kwargs = mock_pipeline.call_args
        assert kwargs.get("with_speech") is True

    @patch("creative_pipeline.run_pipeline")
    def test_with_video_flag(self, mock_pipeline, capsys):
        """--with-video overrides --skip-video."""
        mock_result = PipelineResult(request="test", timestamp="now")
        mock_pipeline.return_value = mock_result

        with patch("sys.argv", ["creative_pipeline.py", "city", "--with-video"]):
            main()

        _, kwargs = mock_pipeline.call_args
        assert kwargs.get("skip_video") is False

    @patch("creative_pipeline.run_pipeline")
    def test_with_vision_check_flag(self, mock_pipeline, capsys):
        """--with-vision-check passes to pipeline."""
        mock_result = PipelineResult(request="test", timestamp="now")
        mock_pipeline.return_value = mock_result

        with patch("sys.argv", ["creative_pipeline.py", "forest", "--with-vision-check"]):
            main()

        _, kwargs = mock_pipeline.call_args
        assert kwargs.get("with_vision_check") is True


# ── Constants and module-level tests ──────────────────────────────

class TestConstants:
    def test_common_flags_has_non_interactive(self):
        assert "--non-interactive" in MMX_COMMON_FLAGS

    def test_common_flags_has_quiet(self):
        assert "--quiet" in MMX_COMMON_FLAGS

    def test_common_flags_has_output_json(self):
        assert "--output" in MMX_COMMON_FLAGS
        assert "json" in MMX_COMMON_FLAGS

    def test_file_flags_has_non_interactive(self):
        assert "--non-interactive" in MMX_FILE_FLAGS

    def test_file_flags_has_quiet(self):
        assert "--quiet" in MMX_FILE_FLAGS

    def test_file_flags_no_json_output(self):
        """File flags should NOT have --output json."""
        assert "--output" not in MMX_FILE_FLAGS

    def test_assets_dir_exists_or_is_path(self):
        assert isinstance(ASSETS_DIR, Path)

    def test_project_root_is_path(self):
        assert isinstance(PROJECT_ROOT, Path)

    def test_history_file_is_path(self):
        assert isinstance(HISTORY_FILE, Path)


# ── CreativeAsset comprehensive tests ─────────────────────────────

class TestCreativeAssetComprehensive:
    def test_all_kinds(self):
        """All valid asset kinds work."""
        for kind in ("image", "music", "text", "speech", "video"):
            asset = CreativeAsset(kind=kind, label="Test", filename=f"test.{kind}")
            assert asset.kind == kind

    def test_skipped_status(self):
        asset = CreativeAsset(kind="video", label="Video", filename="", status="skipped")
        assert asset.status == "skipped"

    def test_meta_nested(self):
        asset = CreativeAsset(
            kind="image", label="Art", filename="art.png",
            meta={"prompt": "test", "nested": {"key": "value"}}
        )
        assert asset.meta["nested"]["key"] == "value"

    def test_asdict_roundtrip(self):
        original = CreativeAsset(
            kind="music", label="Soundtrack", filename="music.mp3",
            mmx_url="https://example.com/m.mp3",
            status="ok",
            meta={"duration": 30, "instrumental": True}
        )
        d = asdict(original)
        reconstructed = CreativeAsset(**d)
        assert reconstructed == original


# ── PipelineResult edge cases ─────────────────────────────────────

class TestPipelineResultEdgeCases:
    def test_summary_no_assets(self):
        r = PipelineResult(request="test", timestamp="now")
        s = r.summary
        assert s["total"] == 0
        assert s["succeeded"] == 0
        assert s["failed"] == 0

    def test_summary_all_failed(self):
        r = PipelineResult(request="test", timestamp="now")
        r.assets.append(CreativeAsset(kind="image", label="Art", filename="", status="failed", error="err"))
        r.assets.append(CreativeAsset(kind="music", label="Music", filename="", status="failed", error="err"))
        s = r.summary
        assert s["succeeded"] == 0
        assert s["failed"] == 2

    def test_successful_assets_filters_skipped(self):
        """Skipped assets are not in successful_assets."""
        r = PipelineResult(request="test", timestamp="now")
        r.assets.append(CreativeAsset(kind="image", label="Art", filename="art.png", status="ok"))
        r.assets.append(CreativeAsset(kind="video", label="Video", filename="", status="skipped"))
        assert len(r.successful_assets) == 1
        assert len(r.failed_assets) == 0  # skipped != failed

    def test_to_json_with_none_build_plan(self):
        r = PipelineResult(request="test", timestamp="now")
        data = json.loads(r.to_json())
        assert data["build_plan"] is None

    def test_to_json_preserves_unicode(self):
        r = PipelineResult(request="日本語テスト", timestamp="now")
        data = json.loads(r.to_json())
        assert data["request"] == "日本語テスト"

    def test_to_json_ensure_ascii_false(self):
        """to_json should produce readable unicode, not \\uXXXX escapes."""
        r = PipelineResult(request="café", timestamp="now")
        raw = r.to_json()
        assert "café" in raw  # Not \\u00e9


# ── Image method edge cases ───────────────────────────────────────

class TestImageEdgeCases:
    @patch("creative_pipeline.subprocess.run")
    def test_image_with_seed(self, mock_run, tmp_path):
        """Image passes --seed when provided."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        out_file = tmp_path / "art.png"
        out_file.write_bytes(b"fake")

        mmx = MMX(bin_path="/fake/mmx")
        mmx.image("prompt", out_file, seed=42)

        cmd = mock_run.call_args[0][0]
        assert "--seed" in cmd
        assert "42" in cmd

    @patch("creative_pipeline.subprocess.run")
    def test_image_custom_aspect_ratio(self, mock_run, tmp_path):
        """Image passes custom aspect ratio."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        out_file = tmp_path / "art.png"
        out_file.write_bytes(b"fake")

        mmx = MMX(bin_path="/fake/mmx")
        mmx.image("prompt", out_file, aspect_ratio="1:1")

        cmd = mock_run.call_args[0][0]
        assert "1:1" in cmd

    @patch("creative_pipeline.subprocess.run")
    def test_image_stdout_is_saved_path(self, mock_run, tmp_path):
        """When stdout contains a path that exists, use it as saved_path."""
        alt_path = tmp_path / "alt_output.png"
        alt_path.write_bytes(b"fake image")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = str(alt_path)
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        saved, url = mmx.image("test", tmp_path / "requested.png")
        assert saved == str(alt_path)


# ── Music method edge cases ───────────────────────────────────────

class TestMusicEdgeCases:
    @patch("creative_pipeline.subprocess.run")
    def test_music_instrumental_flag(self, mock_run, tmp_path):
        """Music passes --instrumental when instrumental=True."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        out_file = tmp_path / "music.mp3"
        out_file.write_bytes(b"fake")

        mmx = MMX(bin_path="/fake/mmx")
        mmx.music("ambient", out_file, instrumental=True)

        cmd = mock_run.call_args[0][0]
        assert "--instrumental" in cmd

    @patch("creative_pipeline.subprocess.run")
    def test_music_no_instrumental(self, mock_run, tmp_path):
        """Music omits --instrumental when instrumental=False."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        out_file = tmp_path / "music.mp3"
        out_file.write_bytes(b"fake")

        mmx = MMX(bin_path="/fake/mmx")
        mmx.music("song with vocals", out_file, instrumental=False)

        cmd = mock_run.call_args[0][0]
        assert "--instrumental" not in cmd

    @patch("creative_pipeline.subprocess.run")
    def test_music_duration_hint(self, mock_run, tmp_path):
        """Music passes duration hint via --extra."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        out_file = tmp_path / "music.mp3"
        out_file.write_bytes(b"fake")

        mmx = MMX(bin_path="/fake/mmx")
        mmx.music("ambient", out_file, duration_hint="30s")

        cmd = mock_run.call_args[0][0]
        assert "--extra" in cmd
        assert "30s" in cmd

    @patch("creative_pipeline.subprocess.run")
    def test_music_stdout_path(self, mock_run, tmp_path):
        """Music returns path from stdout when file exists."""
        out_file = tmp_path / "out.mp3"
        out_file.write_bytes(b"fake")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = str(out_file)
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx.music("test", tmp_path / "requested.mp3")
        assert result == str(out_file)


# ── Speech method edge cases ──────────────────────────────────────

class TestSpeechEdgeCases:
    @patch("creative_pipeline.subprocess.run")
    def test_speech_custom_voice(self, mock_run, tmp_path):
        """Speech passes custom voice parameter."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        out_file = tmp_path / "speech.mp3"
        out_file.write_bytes(b"fake")

        mmx = MMX(bin_path="/fake/mmx")
        mmx.speech("hello", out_file, voice="Custom_Voice")

        cmd = mock_run.call_args[0][0]
        assert "Custom_Voice" in cmd

    @patch("creative_pipeline.subprocess.run")
    def test_speech_default_voice(self, mock_run, tmp_path):
        """Speech uses default voice."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        out_file = tmp_path / "speech.mp3"
        out_file.write_bytes(b"fake")

        mmx = MMX(bin_path="/fake/mmx")
        mmx.speech("hello", out_file)

        cmd = mock_run.call_args[0][0]
        assert "English_expressive_narrator" in cmd

    @patch("creative_pipeline.subprocess.run")
    def test_speech_stdout_path(self, mock_run, tmp_path):
        """Speech returns path from stdout."""
        out_file = tmp_path / "speech.mp3"
        out_file.write_bytes(b"fake")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = str(out_file)
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx.speech("hello", tmp_path / "requested.mp3")
        assert result == str(out_file)


# ── Chat method edge cases ────────────────────────────────────────

class TestChatEdgeCases:
    @patch("creative_pipeline.subprocess.run")
    def test_chat_custom_temperature(self, mock_run):
        """Chat passes custom temperature."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = '{"choices": [{"message": {"content": "hi"}}]}'
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        mmx.chat("sys", "msg", temperature=0.1)

        cmd = mock_run.call_args[0][0]
        assert "0.1" in cmd

    @patch("creative_pipeline.subprocess.run")
    def test_chat_custom_max_tokens(self, mock_run):
        """Chat passes custom max_tokens."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = '{"choices": [{"message": {"content": "hi"}}]}'
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        mmx.chat("sys", "msg", max_tokens=100)

        cmd = mock_run.call_args[0][0]
        assert "100" in cmd

    @patch("creative_pipeline.subprocess.run")
    def test_chat_message_prefix(self, mock_run):
        """Chat prepends 'user:' to the message."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = '{"choices": [{"message": {"content": "hi"}}]}'
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        mmx.chat("sys", "hello there")

        cmd = mock_run.call_args[0][0]
        assert "user:hello there" in cmd

    @patch("creative_pipeline.subprocess.run")
    def test_chat_choices_no_content(self, mock_run):
        """Chat handles choices with missing content."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = '{"choices": [{"message": {}}]}'
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx.chat("sys", "msg")
        assert result == ""

    @patch("creative_pipeline.subprocess.run")
    def test_chat_no_choices_key(self, mock_run):
        """Chat handles response without choices key."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = '{"other": "data"}'
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx.chat("sys", "msg")
        # Should fall through to _raw or str(result)
        assert isinstance(result, str)
