"""Tests for batch mode, history, and edge cases in lucineer-creative.

Covers run_batch, load_history edge cases, and pipeline error paths.
"""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from creative_pipeline import (
    PipelineResult,
    run_batch,
    load_history,
    HISTORY_FILE,
    slugify,
    parse_build_plan,
    MMX,
)


# ─── run_batch tests ──────────────────────────────────────────────

class TestRunBatch:
    @patch("creative_pipeline.run_pipeline")
    def test_empty_batch(self, mock_run):
        results = run_batch([])
        assert results == []
        mock_run.assert_not_called()

    @patch("creative_pipeline.run_pipeline")
    def test_single_item_batch(self, mock_run):
        mock_result = PipelineResult(request="test", timestamp="20260806")
        mock_run.return_value = mock_result
        results = run_batch(["test"])
        assert len(results) == 1
        assert results[0].request == "test"

    @patch("creative_pipeline.run_pipeline")
    def test_multi_item_batch(self, mock_run):
        mock_run.side_effect = [
            PipelineResult(request=f"req-{i}", timestamp="20260806")
            for i in range(3)
        ]
        results = run_batch(["a", "b", "c"])
        assert len(results) == 3
        assert results[0].request == "req-0"
        assert results[1].request == "req-1"
        assert results[2].request == "req-2"

    @patch("creative_pipeline.run_pipeline")
    def test_batch_with_exception(self, mock_run):
        """Batch should continue even if one item fails."""
        good = PipelineResult(request="good", timestamp="20260806")
        mock_run.side_effect = [good, RuntimeError("boom"), good]
        results = run_batch(["a", "b", "c"])
        assert len(results) == 3
        assert results[0].request == "good"
        assert results[1].request == "b"
        assert results[2].request == "good"

    @patch("creative_pipeline.run_pipeline")
    def test_batch_passes_kwargs(self, mock_run):
        mock_run.return_value = PipelineResult(request="x", timestamp="20260806")
        run_batch(["x"], skip_art=True, skip_music=True)
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("skip_art") is True
        assert call_kwargs.get("skip_music") is True


# ─── load_history tests ───────────────────────────────────────────

class TestLoadHistory:
    def test_no_history_file(self, tmp_path, monkeypatch):
        """When history file doesn't exist, return empty list."""
        monkeypatch.setattr("creative_pipeline.HISTORY_FILE", tmp_path / "noexist.jsonl")
        result = load_history()
        assert result == []

    def test_empty_history_file(self, tmp_path, monkeypatch):
        hf = tmp_path / "history.jsonl"
        hf.write_text("")
        monkeypatch.setattr("creative_pipeline.HISTORY_FILE", hf)
        result = load_history()
        assert result == []

    def test_single_entry(self, tmp_path, monkeypatch):
        hf = tmp_path / "history.jsonl"
        entry = {"request": "test", "timestamp": "20260806"}
        hf.write_text(json.dumps(entry) + "\n")
        monkeypatch.setattr("creative_pipeline.HISTORY_FILE", hf)
        result = load_history()
        assert len(result) == 1
        assert result[0]["request"] == "test"

    def test_multiple_entries(self, tmp_path, monkeypatch):
        hf = tmp_path / "history.jsonl"
        for i in range(5):
            with open(hf, "a") as f:
                f.write(json.dumps({"request": f"req-{i}", "timestamp": "20260806"}) + "\n")
        monkeypatch.setattr("creative_pipeline.HISTORY_FILE", hf)
        result = load_history()
        assert len(result) == 5

    def test_limit(self, tmp_path, monkeypatch):
        hf = tmp_path / "history.jsonl"
        for i in range(10):
            with open(hf, "a") as f:
                f.write(json.dumps({"request": f"req-{i}"}) + "\n")
        monkeypatch.setattr("creative_pipeline.HISTORY_FILE", hf)
        result = load_history(limit=3)
        assert len(result) == 3

    def test_invalid_json_skipped(self, tmp_path, monkeypatch):
        """Invalid JSON lines should be silently skipped."""
        hf = tmp_path / "history.jsonl"
        hf.write_text('{"valid": true}\n{invalid json}\n{"also_valid": true}\n')
        monkeypatch.setattr("creative_pipeline.HISTORY_FILE", hf)
        result = load_history()
        assert len(result) == 2

    def test_whitespace_only_lines(self, tmp_path, monkeypatch):
        hf = tmp_path / "history.jsonl"
        hf.write_text('{"a": 1}\n   \n{"b": 2}\n\n')
        monkeypatch.setattr("creative_pipeline.HISTORY_FILE", hf)
        result = load_history()
        assert len(result) == 2


# ─── PipelineResult tests ─────────────────────────────────────────

class TestPipelineResult:
    def test_successful_assets_empty(self):
        result = PipelineResult(request="test", timestamp="20260806")
        assert result.successful_assets == []  # property

    def test_failed_assets_empty(self):
        result = PipelineResult(request="test", timestamp="20260806")
        assert result.failed_assets == []  # property, not method


# ─── slugify edge cases ───────────────────────────────────────────

class TestSlugifyEdgeCases:
    def test_empty_string(self):
        result = slugify("")
        assert result == "build"  # default fallback

    def test_only_special_chars(self):
        result = slugify("!!!@@@###")
        assert result == "build"  # default fallback

    def test_numbers(self):
        assert slugify("build 42") == "build-42"

    def test_mixed_case(self):
        result = slugify("Hello World")
        assert result == "hello-world"

    def test_truncation(self):
        result = slugify("a" * 100)
        assert len(result) <= 50


# ─── parse_build_plan edge cases ──────────────────────────────────

class TestParseBuildPlan:
    def test_empty_string(self):
        assert parse_build_plan("") is None

    def test_plain_text(self):
        assert parse_build_plan("just some text") is None

    def test_malformed_json(self):
        assert parse_build_plan("{not json}") is None

    def test_json_without_required_fields(self):
        result = parse_build_plan('{"unrelated": "field"}')
        if result is not None:
            assert "reply" not in result or "commands" not in result

    def test_valid_minimal(self):
        plan = '{"reply": "Building it", "commands": []}'
        result = parse_build_plan(plan)
        assert result is not None
        assert "reply" in result
        assert "commands" in result


# ─── MMX vision tests ─────────────────────────────────────────────

class TestMMXVision:
    def test_vision_success(self):
        mmx = MMX(bin_path="/fake/mmx")
        with patch.object(mmx, "_run") as mock_run:
            mock_run.return_value = {"_raw": "A lighthouse at night"}
            result = mmx.vision("/path/to/image.jpg")
            assert "lighthouse" in result

    def test_vision_error(self):
        mmx = MMX(bin_path="/fake/mmx")
        with patch.object(mmx, "_run") as mock_run:
            mock_run.return_value = {"_error": "File not found"}
            result = mmx.vision("/nonexistent.jpg")
            assert "[ERROR]" in result

    def test_vision_custom_prompt(self):
        mmx = MMX(bin_path="/fake/mmx")
        with patch.object(mmx, "_run") as mock_run:
            mock_run.return_value = {"_raw": "response"}
            mmx.vision("/img.jpg", prompt="What color is the sky?")
            assert mock_run.called


# ─── MMX speech tests ─────────────────────────────────────────────

class TestMMXSpeech:
    def test_speech_success(self):
        mmx = MMX(bin_path="/fake/mmx")
        with patch.object(mmx, "_run_file") as mock_run:
            mock_run.return_value = (0, "", "")
            result = mmx.speech("Hello world", Path("/tmp/out.mp3"))
            assert isinstance(result, str)

    def test_speech_custom_voice(self):
        mmx = MMX(bin_path="/fake/mmx")
        with patch.object(mmx, "_run_file") as mock_run:
            mock_run.return_value = (0, "", "")
            mmx.speech("test", Path("/tmp/out.mp3"), voice="CustomVoice")
            assert mock_run.called

    def test_speech_error_returns_string(self):
        mmx = MMX(bin_path="/fake/mmx")
        with patch.object(mmx, "_run_file") as mock_run:
            mock_run.return_value = (1, "", "error")
            result = mmx.speech("test", Path("/tmp/out.mp3"))
            assert isinstance(result, str)
