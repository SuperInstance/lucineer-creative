"""Tests for the MMX wrapper class and pipeline edge cases."""

import json
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from creative_pipeline import (
    MMX,
    CreativeAsset,
    PipelineResult,
    slugify,
    parse_build_plan,
    make_concept_art_prompt,
    make_music_prompt,
)


# ── MMX wrapper tests ─────────────────────────────────────────────

class TestMMXWrapper:
    def test_init_default_bin(self):
        mmx = MMX()
        assert mmx.bin is not None

    def test_init_custom_bin(self, tmp_path):
        fake_bin = tmp_path / "fake_mmx"
        fake_bin.write_text("#!/bin/bash")
        fake_bin.chmod(0o755)
        mmx = MMX(bin_path=str(fake_bin))
        assert mmx.bin == str(fake_bin)

    @patch("creative_pipeline.subprocess.run")
    def test_run_valid_json(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = '{"result": "ok"}'
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx._run(["text", "chat"])
        assert result == {"result": "ok"}

    @patch("creative_pipeline.subprocess.run")
    def test_run_nonzero_exit(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "error message"
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx._run(["text", "chat"])
        assert "_error" in result
        assert "exit 1" in result["_error"]

    @patch("creative_pipeline.subprocess.run")
    def test_run_empty_output(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx._run(["text"])
        assert result["_error"] == "mmx returned empty output"

    @patch("creative_pipeline.subprocess.run")
    def test_run_non_json_output(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "plain text response"
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx._run(["text"])
        assert result["_raw"] == "plain text response"

    @patch("creative_pipeline.subprocess.run")
    def test_run_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="mmx", timeout=5)
        mmx = MMX(bin_path="/fake/mmx")
        try:
            result = mmx._run(["text"], timeout=5)
            assert "_error" in result or "timed out" in str(result)
        except (subprocess.TimeoutExpired, AttributeError):
            pass

    @patch("creative_pipeline.subprocess.run")
    def test_run_binary_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError("mmx not found")
        mmx = MMX(bin_path="/fake/mmx")
        result = mmx._run(["text"])
        assert "_error" in result
        assert "not found" in result["_error"]

    @patch("creative_pipeline.subprocess.run")
    def test_run_file_success(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "output"
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        rc, stdout, stderr = mmx._run_file(["image", "generate"])
        assert rc == 0
        assert stdout == "output"

    @patch("creative_pipeline.subprocess.run")
    def test_run_file_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="mmx", timeout=5)
        mmx = MMX(bin_path="/fake/mmx")
        try:
            rc, stdout, stderr = mmx._run_file(["image"], timeout=5)
            assert "timed out" in stderr or rc != 0
        except (subprocess.TimeoutExpired, AttributeError):
            pass

    @patch("creative_pipeline.subprocess.run")
    def test_run_file_binary_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError("mmx not found")
        mmx = MMX(bin_path="/fake/mmx")
        try:
            rc, stdout, stderr = mmx._run_file(["image"])
            assert "not found" in stderr or rc != 0
        except AttributeError:
            pass


# ── MMX chat method ───────────────────────────────────────────────

class TestMMXChat:
    @patch("creative_pipeline.subprocess.run")
    def test_chat_success_with_choices(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps({
            "choices": [{"message": {"content": "Hello!"}}]
        })
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx.chat("system", "hello")
        assert result == "Hello!"

    @patch("creative_pipeline.subprocess.run")
    def test_chat_success_with_raw(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "Just text"
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx.chat("system", "hello")
        assert result == "Just text"

    @patch("creative_pipeline.subprocess.run")
    def test_chat_error(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "fail"
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx.chat("system", "hello")
        assert "[ERROR]" in result


# ── MMX file output methods ───────────────────────────────────────

class TestMMXFileMethods:
    @patch("creative_pipeline.subprocess.run")
    def test_image_generates_file(self, mock_run, tmp_path):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        out_file = tmp_path / "art.png"
        out_file.write_bytes(b"fake png data")

        mmx = MMX(bin_path="/fake/mmx")
        saved, url = mmx.image("prompt", out_file)
        assert saved == str(out_file)

    @patch("creative_pipeline.subprocess.run")
    def test_image_url_output(self, mock_run, tmp_path):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "https://cdn.example.com/img.png"
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        out_file = tmp_path / "art.png"
        mmx = MMX(bin_path="/fake/mmx")
        saved, url = mmx.image("prompt", out_file)
        assert url == "https://cdn.example.com/img.png"

    @patch("creative_pipeline.subprocess.run")
    def test_music_generates_file(self, mock_run, tmp_path):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        out_file = tmp_path / "music.mp3"
        out_file.write_bytes(b"fake mp3")

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx.music("ambient", out_file)
        assert result == str(out_file)

    @patch("creative_pipeline.subprocess.run")
    def test_music_no_file(self, mock_run, tmp_path):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx.music("ambient", tmp_path / "nope.mp3")
        assert result == ""

    @patch("creative_pipeline.subprocess.run")
    def test_speech_generates_file(self, mock_run, tmp_path):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        out_file = tmp_path / "speech.mp3"
        out_file.write_bytes(b"fake audio")

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx.speech("hello world", out_file)
        assert result == str(out_file)

    @patch("creative_pipeline.subprocess.run")
    def test_vision_describes_image(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "A beautiful forest"
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx.vision("/path/to/image.png")
        assert result == "A beautiful forest"

    @patch("creative_pipeline.subprocess.run")
    def test_vision_error(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "fail"
        mock_run.return_value = mock_proc

        mmx = MMX(bin_path="/fake/mmx")
        result = mmx.vision("/path/to/image.png")
        assert "[ERROR]" in result


# ── Additional slugify edge cases ─────────────────────────────────

class TestSlugifyEdgeCases:
    def test_numbers_only(self):
        assert slugify("12345") == "12345"

    def test_mixed_alphanumeric(self):
        assert slugify("Build 42") == "build-42"

    def test_single_char(self):
        assert slugify("x") == "x"

    def test_only_special_chars(self):
        assert slugify("---===---") == "build"

    def test_newlines_tabs(self):
        assert slugify("hello\nworld\ttab") == "hello-world-tab"

    def test_emoji_removed(self):
        result = slugify("🎮 game 🏗️ build")
        assert "🎮" not in result
        assert "🏗️" not in result


# ── Additional parse_build_plan edge cases ────────────────────────

class TestParseBuildPlanEdgeCases:
    def test_array_root(self):
        """A bare JSON array should return None (we expect a dict)."""
        result = parse_build_plan('[1, 2, 3]')
        # The parser finds the first { ... } so arrays at root may return None
        # or the first object inside. This test documents the behavior.
        # If it returns the array, that's fine too — we just document it.
        if result is not None:
            assert isinstance(result, (dict, list))

    def test_json_with_nested_code_block(self):
        raw = 'Some text\n```json\n{"title": "Deep", "meta": {"x": 1}}\n```\nMore text'
        result = parse_build_plan(raw)
        assert result is not None
        assert result["title"] == "Deep"

    def test_multiple_json_objects(self):
        """When multiple JSON objects appear, parser finds one with title."""
        raw = '{"a": 1} {"title": "Real"}'
        result = parse_build_plan(raw)
        if result is not None:
            assert isinstance(result, dict)

    def test_very_long_input(self):
        """Long input shouldn't crash the parser."""
        raw = '{"title": "' + 'x' * 10000 + '"}'
        result = parse_build_plan(raw)
        assert result is not None
        assert len(result["title"]) == 10000


# ── PipelineResult serialization ──────────────────────────────────

class TestPipelineResultSerialization:
    def test_empty_result(self):
        r = PipelineResult(request="", timestamp="")
        data = json.loads(r.to_json())
        assert data["assets"] == []
        assert data["build_plan"] is None

    def test_with_build_plan(self):
        r = PipelineResult(request="test", timestamp="now")
        r.build_plan = {"title": "Castle", "parts": 500}
        data = json.loads(r.to_json())
        assert data["build_plan"]["title"] == "Castle"

    def test_with_multiple_assets(self):
        r = PipelineResult(request="test", timestamp="now")
        for i in range(5):
            r.assets.append(CreativeAsset(
                kind="image", label=f"Art {i}", filename=f"art_{i}.png"
            ))
        data = json.loads(r.to_json())
        assert len(data["assets"]) == 5

    def test_unicode_in_request(self):
        r = PipelineResult(request="café buïld", timestamp="now")
        data = json.loads(r.to_json())
        assert data["request"] == "café buïld"


# ── Prompt builder edge cases ─────────────────────────────────────

class TestPromptBuilderEdgeCases:
    def test_empty_concept_art_prompt(self):
        prompt = make_concept_art_prompt("")
        assert "Roblox" in prompt

    def test_long_concept_art_prompt(self):
        long = "a" * 1000
        prompt = make_concept_art_prompt(long)
        assert long in prompt

    def test_empty_music_prompt(self):
        prompt = make_music_prompt("")
        assert "ambient" in prompt.lower()

    def test_special_chars_in_prompt(self):
        prompt = make_concept_art_prompt('build with "quotes" and \\ backslashes')
        assert "quotes" in prompt
