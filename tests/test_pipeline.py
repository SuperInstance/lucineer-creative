"""Tests for the Lucineer Creative Pipeline."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Ensure the pipeline module is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from creative_pipeline import (
    slugify,
    parse_build_plan,
    make_concept_art_prompt,
    make_music_prompt,
    CreativeAsset,
    PipelineResult,
    BUILD_PLAN_SYSTEM,
    SAFE_RE,
)


class TestSlugify:
    def test_basic(self):
        assert slugify("Spooky Forest") == "spooky-forest"

    def test_special_chars(self):
        assert slugify("Castle on a Hill!") == "castle-on-a-hill"

    def test_multiple_spaces(self):
        assert slugify("cyberpunk   city") == "cyberpunk-city"

    def test_empty_string(self):
        assert slugify("") == "build"

    def test_unicode(self):
        # Non-ASCII characters are replaced with hyphens
        result = slugify("Café Müller")
        assert "café" not in result  # é is not a-z0-9

    def test_truncation(self):
        long = "a" * 100
        result = slugify(long)
        assert len(result) <= 50

    def test_leading_trailing_hyphens(self):
        assert slugify("--- test ---") == "test"


class TestParseBuildPlan:
    def test_valid_json(self):
        raw = '{"title": "Forest", "structures": []}'
        result = parse_build_plan(raw)
        assert result == {"title": "Forest", "structures": []}

    def test_markdown_fenced_json(self):
        raw = '```json\n{"title": "Castle"}\n```'
        result = parse_build_plan(raw)
        assert result == {"title": "Castle"}

    def test_markdown_fenced_no_language(self):
        raw = '```\n{"title": "Cave"}\n```'
        result = parse_build_plan(raw)
        assert result == {"title": "Cave"}

    def test_json_with_surrounding_text(self):
        raw = 'Here is the plan:\n{"title": "Town"}\nThat is the plan.'
        result = parse_build_plan(raw)
        assert result == {"title": "Town"}

    def test_invalid_json(self):
        assert parse_build_plan("not json at all") is None

    def test_empty_string(self):
        assert parse_build_plan("") is None

    def test_nested_json(self):
        raw = '{"title": "X", "lighting": {"ambient": "#FFF"}}'
        result = parse_build_plan(raw)
        assert result["lighting"]["ambient"] == "#FFF"


class TestPromptBuilders:
    def test_concept_art_prompt_contains_request(self):
        prompt = make_concept_art_prompt("haunted house")
        assert "haunted house" in prompt

    def test_concept_art_prompt_mentions_roblox(self):
        prompt = make_concept_art_prompt("test")
        assert "Roblox" in prompt

    def test_music_prompt_contains_request(self):
        prompt = make_music_prompt("peaceful meadow")
        assert "peaceful meadow" in prompt

    def test_music_prompt_mentions_ambient(self):
        prompt = make_music_prompt("test")
        assert "ambient" in prompt.lower()


class TestCreativeAsset:
    def test_defaults(self):
        asset = CreativeAsset(kind="image", label="Art", filename="art.png")
        assert asset.kind == "image"
        assert asset.mmx_url == ""
        assert asset.meta == {}

    def test_with_all_fields(self):
        asset = CreativeAsset(
            kind="music",
            label="Soundtrack",
            filename="music.mp3",
            mmx_url="https://example.com/x.mp3",
            meta={"bpm": 120},
        )
        assert asset.meta["bpm"] == 120


class TestPipelineResult:
    def test_to_json(self):
        result = PipelineResult(request="test build", timestamp="2026-08-05")
        data = json.loads(result.to_json())
        assert data["request"] == "test build"
        assert data["assets"] == []
        assert data["build_plan"] is None

    def test_with_assets(self):
        result = PipelineResult(request="test", timestamp="now")
        result.assets.append(CreativeAsset(kind="text", label="Plan", filename="plan.json"))
        data = json.loads(result.to_json())
        assert len(data["assets"]) == 1
        assert data["assets"][0]["kind"] == "text"


class TestBuildPlanSystem:
    def test_mentions_json(self):
        assert "JSON" in BUILD_PLAN_SYSTEM

    def test_mentions_roblox(self):
        assert "Roblox" in BUILD_PLAN_SYSTEM

    def test_has_required_fields(self):
        for field_name in ("title", "structures", "lighting", "build_steps"):
            assert field_name in BUILD_PLAN_SYSTEM
