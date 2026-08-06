"""Tests for new features: retry logic, asset history, batch mode, structured errors."""

import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from creative_pipeline import (
    MMX,
    CreativeAsset,
    PipelineResult,
    run_batch,
    load_history,
    _update_history,
    HISTORY_FILE,
    ASSETS_DIR,
)


# ── Retry logic tests ─────────────────────────────────────────────

class TestRetryLogic:
    @patch("creative_pipeline.subprocess.run")
    @patch("creative_pipeline.time.sleep")
    def test_retry_succeeds_on_second_attempt(self, mock_sleep, mock_run):
        """First call fails, second succeeds — should return valid result."""
        fail_proc = MagicMock()
        fail_proc.returncode = 1
        fail_proc.stdout = ""
        fail_proc.stderr = "transient error"

        ok_proc = MagicMock()
        ok_proc.returncode = 0
        ok_proc.stdout = '{"result": "ok"}'
        ok_proc.stderr = ""

        mock_run.side_effect = [fail_proc, ok_proc]

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx._run(["text", "chat"], retries=2)
        assert result == {"result": "ok"}
        assert mock_run.call_count == 2

    @patch("creative_pipeline.subprocess.run")
    @patch("creative_pipeline.time.sleep")
    def test_retry_exhausted_returns_error(self, mock_sleep, mock_run):
        """All retries fail — should return last error."""
        fail_proc = MagicMock()
        fail_proc.returncode = 1
        fail_proc.stdout = ""
        fail_proc.stderr = "persistent error"

        mock_run.side_effect = [fail_proc, fail_proc, fail_proc]

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx._run(["text"], retries=2)
        assert "_error" in result
        assert mock_run.call_count == 3  # initial + 2 retries

    @patch("creative_pipeline.subprocess.run")
    @patch("creative_pipeline.time.sleep")
    def test_retry_timeout_then_success(self, mock_sleep, mock_run):
        """Timeout on first try, success on second."""
        import subprocess as sp
        ok_proc = MagicMock()
        ok_proc.returncode = 0
        ok_proc.stdout = '{"ok": true}'
        ok_proc.stderr = ""

        mock_run.side_effect = [sp.TimeoutExpired(cmd="mmx", timeout=5), ok_proc]

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx._run(["text"], timeout=5, retries=1)
        assert result == {"ok": True}

    @patch("creative_pipeline.subprocess.run")
    def test_no_retries_by_default_works(self, mock_run):
        """retries=0 means single attempt."""
        ok_proc = MagicMock()
        ok_proc.returncode = 0
        ok_proc.stdout = '{"ok": true}'
        ok_proc.stderr = ""
        mock_run.return_value = ok_proc

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx._run(["text"], retries=0)
        assert result == {"ok": True}
        assert mock_run.call_count == 1

    @patch("creative_pipeline.subprocess.run")
    def test_file_not_found_no_retry(self, mock_run):
        """FileNotFoundError should not retry — binary doesn't exist."""
        import subprocess as sp
        mock_run.side_effect = FileNotFoundError("nope")

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx._run(["text"], retries=3)
        assert "_error" in result
        assert "not found" in result["_error"]
        assert mock_run.call_count == 1  # No retries for missing binary


# ── PipelineResult new properties ─────────────────────────────────

class TestPipelineResultProperties:
    def test_successful_assets_empty(self):
        r = PipelineResult(request="test", timestamp="now")
        assert r.successful_assets == []

    def test_successful_assets_with_ok(self):
        r = PipelineResult(request="test", timestamp="now")
        r.assets.append(CreativeAsset(
            kind="image", label="Art", filename="art.png", status="ok"
        ))
        r.assets.append(CreativeAsset(
            kind="music", label="Music", filename="", status="failed", error="timeout"
        ))
        ok = r.successful_assets
        assert len(ok) == 1
        assert ok[0].kind == "image"

    def test_failed_assets(self):
        r = PipelineResult(request="test", timestamp="now")
        r.assets.append(CreativeAsset(
            kind="music", label="Music", filename="", status="failed", error="timeout"
        ))
        failed = r.failed_assets
        assert len(failed) == 1
        assert failed[0].error == "timeout"

    def test_summary_basic(self):
        r = PipelineResult(request="castle", timestamp="20260805")
        r.build_plan = {"title": "Castle"}
        r.assets.append(CreativeAsset(
            kind="image", label="Art", filename="art.png", status="ok"
        ))
        r.assets.append(CreativeAsset(
            kind="text", label="Plan", filename="plan.json", status="ok"
        ))
        r.assets.append(CreativeAsset(
            kind="music", label="Music", filename="", status="failed", error="timeout"
        ))
        s = r.summary
        assert s["request"] == "castle"
        assert s["total"] == 3
        assert s["succeeded"] == 2
        assert s["failed"] == 1
        assert s["had_plan"] is True

    def test_summary_no_plan(self):
        r = PipelineResult(request="test", timestamp="now")
        s = r.summary
        assert s["had_plan"] is False

    def test_asset_status_default(self):
        """New assets default to 'ok' status."""
        asset = CreativeAsset(kind="image", label="Art", filename="art.png")
        assert asset.status == "ok"
        assert asset.error == ""


# ── History tests ─────────────────────────────────────────────────

class TestAssetHistory:
    def test_history_empty_when_no_file(self, tmp_path, monkeypatch):
        """History returns empty list when file doesn't exist."""
        monkeypatch.setattr("creative_pipeline.HISTORY_FILE", tmp_path / "nope.jsonl")
        assert load_history() == []

    def test_history_append_and_read(self, tmp_path, monkeypatch):
        """Can write and read history entries."""
        hist_file = tmp_path / "history.jsonl"
        monkeypatch.setattr("creative_pipeline.HISTORY_FILE", hist_file)
        monkeypatch.setattr("creative_pipeline.ASSETS_DIR", tmp_path)

        r = PipelineResult(request="castle", timestamp="20260805-2000")
        r.build_plan = {"title": "Castle"}
        r.assets.append(CreativeAsset(
            kind="image", label="Art", filename="art.png", status="ok"
        ))

        _update_history(r)

        entries = load_history()
        assert len(entries) == 1
        assert entries[0]["request"] == "castle"
        assert entries[0]["succeeded"] == 1

    def test_history_multiple_entries(self, tmp_path, monkeypatch):
        """Multiple entries are read in reverse order (newest first)."""
        hist_file = tmp_path / "history.jsonl"
        monkeypatch.setattr("creative_pipeline.HISTORY_FILE", hist_file)
        monkeypatch.setattr("creative_pipeline.ASSETS_DIR", tmp_path)

        for i in range(5):
            r = PipelineResult(request=f"build-{i}", timestamp=f"20260805-200{i}")
            _update_history(r)

        entries = load_history()
        assert len(entries) == 5
        assert entries[0]["request"] == "build-4"  # newest first

    def test_history_limit(self, tmp_path, monkeypatch):
        """History respects limit parameter."""
        hist_file = tmp_path / "history.jsonl"
        monkeypatch.setattr("creative_pipeline.HISTORY_FILE", hist_file)
        monkeypatch.setattr("creative_pipeline.ASSETS_DIR", tmp_path)

        for i in range(10):
            r = PipelineResult(request=f"build-{i}", timestamp=f"20260805-200{i}")
            _update_history(r)

        entries = load_history(limit=3)
        assert len(entries) == 3

    def test_history_skips_malformed_lines(self, tmp_path, monkeypatch):
        """Malformed JSON lines are skipped."""
        hist_file = tmp_path / "history.jsonl"
        monkeypatch.setattr("creative_pipeline.HISTORY_FILE", hist_file)
        monkeypatch.setattr("creative_pipeline.ASSETS_DIR", tmp_path)

        hist_file.write_text(
            '{"request": "good", "timestamp": "20260805"}\n'
            'BROKEN JSON\n'
            '{"request": "also-good", "timestamp": "20260806"}\n'
        )

        entries = load_history()
        assert len(entries) == 2


# ── Batch mode tests ──────────────────────────────────────────────

class TestBatchMode:
    @patch("creative_pipeline.run_pipeline")
    def test_batch_runs_all(self, mock_pipeline):
        """Batch mode calls run_pipeline for each request."""
        mock_pipeline.return_value = PipelineResult(
            request="test", timestamp="now"
        )
        results = run_batch(["castle", "forest", "cave"])
        assert len(results) == 3
        assert mock_pipeline.call_count == 3

    @patch("creative_pipeline.run_pipeline")
    def test_batch_continues_on_error(self, mock_pipeline):
        """Batch continues even if one request raises."""
        mock_pipeline.side_effect = [
            PipelineResult(request="castle", timestamp="now"),
            RuntimeError("boom"),
            PipelineResult(request="cave", timestamp="now"),
        ]
        results = run_batch(["castle", "forest", "cave"])
        assert len(results) == 3

    @patch("creative_pipeline.run_pipeline")
    def test_batch_empty_list(self, mock_pipeline):
        """Empty batch returns empty list."""
        results = run_batch([])
        assert results == []
        assert mock_pipeline.call_count == 0

    @patch("creative_pipeline.run_pipeline")
    def test_batch_passes_kwargs(self, mock_pipeline):
        """Batch forwards extra kwargs to run_pipeline."""
        mock_pipeline.return_value = PipelineResult(
            request="test", timestamp="now"
        )
        run_batch(["castle"], skip_video=True, with_speech=True)
        _, kwargs = mock_pipeline.call_args
        assert kwargs.get("skip_video") is True
        assert kwargs.get("with_speech") is True


# ── CreativeAsset with status/error ───────────────────────────────

class TestCreativeAssetStatus:
    def test_asset_with_error(self):
        asset = CreativeAsset(
            kind="image", label="Art", filename="",
            status="failed", error="mmx timed out"
        )
        assert asset.status == "failed"
        assert asset.error == "mmx timed out"

    def test_asset_serialization_with_status(self):
        asset = CreativeAsset(
            kind="music", label="Soundtrack", filename="ambient.mp3",
            status="ok", meta={"duration": 30}
        )
        from dataclasses import asdict
        d = asdict(asset)
        assert d["status"] == "ok"
        assert d["error"] == ""
        assert d["meta"]["duration"] == 30
