#!/usr/bin/env python3
"""
Lucineer Creative Pipeline — MMX (MiniMax) Integration
=======================================================

Turns a natural-language build request into a complete creative asset pack:
  1. Concept art (image generation)
  2. Ambient music / soundscape (music generation)
  3. Build plan (text generation — structured, Roblox-ready)
  4. Narration (speech synthesis — optional voice-over for the plan)

Usage:
    python3 creative_pipeline.py "build a spooky forest"
    python3 creative_pipeline.py "medieval castle on a hill" --skip-video
    python3 creative_pipeline.py "cyberpunk city street" --with-video
    python3 creative_pipeline.py "fantasy potion shop" --with-speech

Outputs are saved under ./assets/ and a JSON summary is printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MMX_BIN = os.environ.get("MMX_BIN", str(Path.home() / ".npm-global" / "bin" / "mmx"))

# Shared MMX flags for non-interactive agent use
MMX_COMMON_FLAGS = [
    "--non-interactive",
    "--quiet",
    "--output", "json",
]

# Consistent agent flags for non-JSON (file-output) calls
MMX_FILE_FLAGS = [
    "--non-interactive",
    "--quiet",
]


# ── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class CreativeAsset:
    kind: str          # "image" | "music" | "text" | "speech" | "video"
    label: str         # human-friendly name
    filename: str      # relative path under assets/
    mmx_url: str = ""  # MiniMax CDN url (if available)
    status: str = "ok"  # "ok" | "failed" | "skipped"
    error: str = ""    # error message if status == "failed"
    meta: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    request: str
    timestamp: str
    assets: list[CreativeAsset] = field(default_factory=list)
    build_plan: Optional[dict] = None

    def to_json(self) -> str:
        return json.dumps({
            "request": self.request,
            "timestamp": self.timestamp,
            "assets": [asdict(a) for a in self.assets],
            "build_plan": self.build_plan,
        }, indent=2, ensure_ascii=False)

    @property
    def successful_assets(self) -> list[CreativeAsset]:
        """Assets that completed successfully."""
        return [a for a in self.assets if a.status == "ok" and a.filename]

    @property
    def failed_assets(self) -> list[CreativeAsset]:
        """Assets that failed."""
        return [a for a in self.assets if a.status == "failed"]

    @property
    def summary(self) -> dict:
        """Quick summary for logging/history."""
        return {
            "request": self.request,
            "timestamp": self.timestamp,
            "total": len(self.assets),
            "succeeded": len(self.successful_assets),
            "failed": len(self.failed_assets),
            "had_plan": self.build_plan is not None,
        }


# ── MMX Wrapper ──────────────────────────────────────────────────────────────

class MMX:
    """Thin subprocess wrapper around the mmx CLI."""

    def __init__(self, bin_path: str = MMX_BIN):
        self.bin = bin_path
        if not Path(self.bin).exists():
            # Fall back to PATH lookup
            self.bin = "mmx"

    def _run(self, args: list[str], timeout: int = 300, retries: int = 2) -> dict:
        """Run mmx with JSON output and parse the result. Retries on transient failures."""
        cmd = [self.bin] + args + MMX_COMMON_FLAGS
        last_error = {}
        for attempt in range(retries + 1):
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                last_error = {"_error": f"mmx timed out after {timeout}s"}
                if attempt < retries:
                    wait = 2 ** attempt  # 1s, 2s, 4s...
                    time.sleep(wait)
                    continue
                return last_error
            except FileNotFoundError:
                return {"_error": f"mmx binary not found: {self.bin}"}

            if proc.returncode != 0:
                last_error = {
                    "_error": f"mmx exit {proc.returncode}",
                    "stderr": proc.stderr.strip(),
                }
                if attempt < retries:
                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue
                return last_error

            stdout = proc.stdout.strip()
            if not stdout:
                last_error = {"_error": "mmx returned empty output"}
                if attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                return last_error
            try:
                return json.loads(stdout)
            except json.JSONDecodeError:
                # Some commands output plain text even with --output json
                return {"_raw": stdout}
        return last_error

    def _run_file(self, args: list[str], timeout: int = 300) -> tuple[int, str, str]:
        """Run mmx with file output (no JSON parsing). Returns (rc, stdout, stderr)."""
        cmd = [self.bin] + args + MMX_FILE_FLAGS
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
        except subprocess.TimeoutExpired:
            return 5, "", f"mmx timed out after {timeout}s"
        except FileNotFoundError:
            return 1, "", f"mmx binary not found: {self.bin}"

    # ── Text ─────────────────────────────────────────────────────────────────

    def chat(self, system: str, message: str, temperature: float = 0.8,
             max_tokens: int = 4096) -> str:
        """Generate text via MiniMax chat."""
        result = self._run([
            "text", "chat",
            "--system", system,
            "--message", f"user:{message}",
            "--temperature", str(temperature),
            "--max-tokens", str(max_tokens),
        ])
        if "_error" in result:
            return f"[ERROR] {result['_error']}: {result.get('stderr', '')}"
        # JSON response — extract text content
        if "choices" in result:
            return result["choices"][0].get("message", {}).get("content", "")
        return result.get("_raw", str(result))

    # ── Image ────────────────────────────────────────────────────────────────

    def image(self, prompt: str, out_path: Path, aspect_ratio: str = "16:9",
              seed: Optional[int] = None) -> tuple[str, str]:
        """Generate an image. Returns (saved_path, mmx_url_or_empty)."""
        args = [
            "image", "generate",
            "--prompt", prompt,
            "--aspect-ratio", aspect_ratio,
            "--out", str(out_path),
        ]
        if seed is not None:
            args += ["--seed", str(seed)]

        rc, stdout, stderr = self._run_file(args, timeout=120)
        saved_path = str(out_path) if out_path.exists() else ""
        url = stdout if stdout.startswith("http") else ""
        # If stdout isn't a URL it might be the saved path
        if not saved_path and stdout and Path(stdout).exists():
            saved_path = stdout
        return saved_path, url

    # ── Music ────────────────────────────────────────────────────────────────

    def music(self, prompt: str, out_path: Path, instrumental: bool = True,
              duration_hint: str = "") -> str:
        """Generate music/soundscape. Returns saved file path."""
        args = [
            "music", "generate",
            "--prompt", prompt,
            "--out", str(out_path),
            "--format", "mp3",
        ]
        if instrumental:
            args += ["--instrumental"]
        if duration_hint:
            args += ["--extra", duration_hint]

        rc, stdout, stderr = self._run_file(args, timeout=300)
        if out_path.exists():
            return str(out_path)
        # mmx might output the path
        if stdout and Path(stdout.strip()).exists():
            return stdout.strip()
        return ""

    # ── Speech ───────────────────────────────────────────────────────────────

    def speech(self, text: str, out_path: Path, voice: str = "English_expressive_narrator") -> str:
        """Synthesize speech. Returns saved file path."""
        args = [
            "speech", "synthesize",
            "--text", text,
            "--voice", voice,
            "--out", str(out_path),
        ]
        rc, stdout, stderr = self._run_file(args, timeout=120)
        if out_path.exists():
            return str(out_path)
        if stdout and Path(stdout.strip()).exists():
            return stdout.strip()
        return ""

    # ── Video ────────────────────────────────────────────────────────────────

    def video(self, prompt: str, out_path: Path, image_ref: Optional[str] = None) -> str:
        """Generate a short video clip. Returns saved file path."""
        args = [
            "video", "generate",
            "--prompt", prompt,
            "--download", str(out_path),
        ]
        if image_ref:
            args += ["--image", image_ref]

        rc, stdout, stderr = self._run_file(args, timeout=600)
        if out_path.exists():
            return str(out_path)
        if stdout and Path(stdout.strip()).exists():
            return stdout.strip()
        return ""

    # ── Vision ───────────────────────────────────────────────────────────────

    def vision(self, image_path: str, prompt: str = "Describe this image in detail.") -> str:
        """Describe an image using vision model."""
        result = self._run([
            "vision", "describe",
            "--image", image_path,
            "--prompt", prompt,
        ])
        if "_error" in result:
            return f"[ERROR] {result['_error']}"
        return result.get("_raw", str(result))


# ── Pipeline Stages ──────────────────────────────────────────────────────────

SAFE_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Make a filesystem-safe slug from text."""
    slug = SAFE_RE.sub("-", text.lower()).strip("-")
    return slug[:50] or "build"


def make_concept_art_prompt(build_request: str) -> str:
    return (
        f"Concept art for a Roblox game environment: {build_request}. "
        f"Detailed, atmospheric, game-ready lighting, vibrant colors, "
        f"isometric perspective showing the full scene layout, "
        f"stylized 3D render quality suitable as reference for building in Roblox Studio."
    )


def make_music_prompt(build_request: str) -> str:
    return (
        f"Atmospheric ambient background music for a game environment: {build_request}. "
        f"Looping-friendly, immersive, cinematic game soundtrack, subtle and evocative, "
        f"suitable for continuous background play in a Roblox experience."
    )


BUILD_PLAN_SYSTEM = (
    "You are Lucineer, an expert Roblox builder and game designer. "
    "Given a build request, produce a detailed, structured build plan in JSON format "
    "with the following fields:\n"
    "  - title: short creative name for the build\n"
    "  - description: 2-3 sentence overview\n"
    "  - biome: the Roblox biome/terrain setting\n"
    "  - structures: list of {name, description, materials} for each major structure\n"
    "  - props: list of decorative elements (trees, rocks, lights, particles)\n"
    "  - lighting: {ambient_color, skybox, fog, brightness} for Roblox Lighting service\n"
    "  - color_palette: list of hex color codes that define the visual theme\n"
    "  - sound_design: list of {element, description} for audio cues\n"
    "  - build_steps: ordered list of steps to construct this in Roblox Studio\n"
    "  - estimated_parts: rough number of Roblox parts needed\n"
    "Respond with ONLY valid JSON, no markdown fences."
)


def parse_build_plan(raw: str) -> Optional[dict]:
    """Try to extract JSON from the LLM response."""
    # Strip markdown fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find first { ... last }
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                return json.loads(cleaned[first:last + 1])
            except json.JSONDecodeError:
                pass
    return None


def run_pipeline(
    build_request: str,
    skip_video: bool = True,
    with_speech: bool = False,
    with_vision_check: bool = False,
) -> PipelineResult:
    """Run the full creative pipeline for a build request."""
    mmx = MMX()
    slug = slugify(build_request)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result = PipelineResult(request=build_request, timestamp=timestamp)

    print(f"\n🎨 Lucineer Creative Pipeline", file=sys.stderr)
    print(f"   Request: {build_request}", file=sys.stderr)
    print(f"   Output:  {ASSETS_DIR}/{slug}/", file=sys.stderr)
    print(f"   Time:    {timestamp}\n", file=sys.stderr)

    # Create per-build asset directory
    build_dir = ASSETS_DIR / slug
    build_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Build Plan (text) ───────────────────────────────────────────
    print("📝 Stage 1/4: Generating build plan...", file=sys.stderr)
    raw_plan = mmx.chat(
        system=BUILD_PLAN_SYSTEM,
        message=build_request,
        temperature=0.8,
        max_tokens=4096,
    )
    plan = parse_build_plan(raw_plan)
    result.build_plan = plan

    # Save raw and parsed plan
    plan_file = build_dir / "build_plan.json"
    if plan:
        plan_file.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    else:
        plan_file.write_text(raw_plan)
    result.assets.append(CreativeAsset(
        kind="text",
        label="Build Plan",
        filename=str(plan_file.relative_to(PROJECT_ROOT)),
        meta={"parsed": plan is not None},
    ))
    print(f"   ✓ Saved build plan ({'structured' if plan else 'raw text'})", file=sys.stderr)

    # ── Stage 2: Concept Art (image) ─────────────────────────────────────────
    print("🖼️ Stage 2/4: Generating concept art...", file=sys.stderr)
    art_prompt = make_concept_art_prompt(build_request)
    art_path = build_dir / "concept_art.png"
    saved, url = mmx.image(art_prompt, art_path, aspect_ratio="16:9")
    result.assets.append(CreativeAsset(
        kind="image",
        label="Concept Art",
        filename=str(art_path.relative_to(PROJECT_ROOT)) if saved else "",
        mmx_url=url,
        meta={"prompt": art_prompt, "aspect_ratio": "16:9"},
    ))
    print(f"   ✓ Concept art {'saved' if saved else 'generated'}"
          + (f" → {saved}" if saved else ""), file=sys.stderr)

    # Optional: vision check on generated art
    if with_vision_check and saved:
        print("   🔍 Vision check on concept art...", file=sys.stderr)
        vision_desc = mmx.vision(saved, "Describe this concept art. What are the key visual elements, colors, and atmosphere?")
        vision_file = build_dir / "concept_art_analysis.txt"
        vision_file.write_text(vision_desc)
        result.assets.append(CreativeAsset(
            kind="text",
            label="Concept Art Analysis (Vision)",
            filename=str(vision_file.relative_to(PROJECT_ROOT)),
            meta={"model": "vision"},
        ))
        print(f"   ✓ Vision analysis saved", file=sys.stderr)

    # ── Stage 3: Ambient Music (audio) ───────────────────────────────────────
    print("🎵 Stage 3/4: Generating ambient music...", file=sys.stderr)
    music_prompt = make_music_prompt(build_request)
    music_path = build_dir / "ambient.mp3"
    saved_music = mmx.music(music_prompt, music_path, instrumental=True)
    result.assets.append(CreativeAsset(
        kind="music",
        label="Ambient Soundtrack",
        filename=str(music_path.relative_to(PROJECT_ROOT)) if saved_music else "",
        meta={"prompt": music_prompt, "instrumental": True},
    ))
    print(f"   ✓ Music {'saved' if saved_music else 'generated'}"
          + (f" → {saved_music}" if saved_music else ""), file=sys.stderr)

    # ── Stage 4: Optional Speech Narration ───────────────────────────────────
    if with_speech and plan:
        print("🎙️ Stage 4: Generating build plan narration...", file=sys.stderr)
        narration_text = (
            f"Build plan for {plan.get('title', build_request)}. "
            f"{plan.get('description', '')} "
            f"The build features {len(plan.get('structures', []))} main structures "
            f"and uses a color palette of {', '.join(plan.get('color_palette', [])[:5])}. "
            f"Estimated parts needed: {plan.get('estimated_parts', 'unknown')}."
        )
        speech_path = build_dir / "plan_narration.mp3"
        saved_speech = mmx.speech(narration_text, speech_path)
        result.assets.append(CreativeAsset(
            kind="speech",
            label="Build Plan Narration",
            filename=str(speech_path.relative_to(PROJECT_ROOT)) if saved_speech else "",
            meta={"voice": "English_expressive_narrator"},
        ))
        print(f"   ✓ Narration {'saved' if saved_speech else 'generated'}", file=sys.stderr)

    # ── Optional Video Preview ───────────────────────────────────────────────
    if not skip_video:
        print("🎬 Generating video preview...", file=sys.stderr)
        video_prompt = (
            f"Cinematic flythrough of: {build_request}. "
            f"Slow camera pan, atmospheric lighting, game environment showcase."
        )
        video_path = build_dir / "preview.mp4"
        image_ref = str(art_path) if art_path.exists() else None
        saved_video = mmx.video(video_prompt, video_path, image_ref=image_ref)
        result.assets.append(CreativeAsset(
            kind="video",
            label="Build Preview Video",
            filename=str(video_path.relative_to(PROJECT_ROOT)) if saved_video else "",
            meta={"prompt": video_prompt, "image_ref": image_ref},
        ))
        print(f"   ✓ Video {'saved' if saved_video else 'generated'}"
              + (f" → {saved_video}" if saved_video else ""), file=sys.stderr)

    # ── Write manifest ───────────────────────────────────────────────────────
    manifest_path = build_dir / "manifest.json"
    manifest_path.write_text(result.to_json())
    result.assets.append(CreativeAsset(
        kind="text",
        label="Asset Manifest",
        filename=str(manifest_path.relative_to(PROJECT_ROOT)),
    ))

    print(f"\n✅ Pipeline complete! Assets saved to {build_dir}/", file=sys.stderr)
    print(f"   Manifest: {manifest_path}", file=sys.stderr)
    print(f"   Assets: {len(result.successful_assets)} ok, {len(result.failed_assets)} failed", file=sys.stderr)
    print(file=sys.stderr)

    # ── Update asset history ─────────────────────────────────────────────
    _update_history(result)

    return result


# ── Asset History ────────────────────────────────────────────────────────────

HISTORY_FILE = ASSETS_DIR / "history.jsonl"


def _update_history(result: PipelineResult) -> None:
    """Append a pipeline run summary to the history file."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    entry = json.dumps(result.summary, ensure_ascii=False)
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


def load_history(limit: int = 50) -> list[dict]:
    """Load recent pipeline runs from history."""
    if not HISTORY_FILE.exists():
        return []
    lines = HISTORY_FILE.read_text(encoding="utf-8").strip().split("\n")
    entries = []
    for line in reversed(lines):
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if len(entries) >= limit:
            break
    return entries


# ── Batch Mode ───────────────────────────────────────────────────────────────

def run_batch(requests: list[str], **kwargs) -> list[PipelineResult]:
    """Run the pipeline for multiple build requests. Returns all results."""
    results = []
    total = len(requests)
    for i, req in enumerate(requests, 1):
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  Batch {i}/{total}: {req}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        try:
            result = run_pipeline(req, **kwargs)
            results.append(result)
        except Exception as e:
            print(f"  ❌ Batch item {i} failed: {e}", file=sys.stderr)
            results.append(PipelineResult(
                request=req,
                timestamp=time.strftime("%Y%m%d-%H%M%S"),
            ))
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Lucineer Creative Pipeline — MMX-powered asset generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 creative_pipeline.py "build a spooky forest"
  python3 creative_pipeline.py "medieval castle on a hill" --skip-video
  python3 creative_pipeline.py "cyberpunk city" --with-video --with-speech
  python3 creative_pipeline.py "fantasy potion shop" --with-vision-check
  python3 creative_pipeline.py --batch builds.txt
  python3 creative_pipeline.py --history

Output:
  Prints a JSON summary to stdout.
  Assets are saved under assets/<slug>/ directory.
        """,
    )
    parser.add_argument(
        "request",
        nargs="?",
        default=None,
        help='The build request, e.g. "build a spooky forest"',
    )
    parser.add_argument(
        "--batch", metavar="FILE",
        help="Run pipeline for each line in the given file",
    )
    parser.add_argument(
        "--history", action="store_true",
        help="Show recent pipeline run history",
    )
    parser.add_argument(
        "--skip-video", action="store_true", default=True,
        help="Skip video generation (default: video is skipped)",
    )
    parser.add_argument(
        "--with-video", action="store_true", default=False,
        help="Enable video preview generation (slower, uses more quota)",
    )
    parser.add_argument(
        "--with-speech", action="store_true", default=False,
        help="Generate speech narration of the build plan",
    )
    parser.add_argument(
        "--with-vision-check", action="store_true", default=False,
        help="Run vision analysis on generated concept art",
    )
    parser.add_argument(
        "--pretty", action="store_true", default=True,
        help="Pretty-print JSON output (default)",
    )
    parser.add_argument(
        "--compact", action="store_true", default=False,
        help="Compact JSON output (no indentation)",
    )

    args = parser.parse_args()

    # ── History mode ─────────────────────────────────────────────
    if args.history:
        history = load_history()
        if not history:
            print("No pipeline runs in history.")
        else:
            print(f"Recent pipeline runs ({len(history)}):\n")
            for entry in history:
                status = f"{entry['succeeded']}/{entry['total']} ok"
                plan = "✓" if entry.get("had_plan") else "✗"
                print(f"  [{entry['timestamp']}] {entry['request'][:50]}  {status}  plan:{plan}")
        return

    # ── Batch mode ───────────────────────────────────────────────
    if args.batch:
        batch_file = Path(args.batch)
        if not batch_file.exists():
            print(f"Error: batch file not found: {batch_file}", file=sys.stderr)
            sys.exit(1)
        requests = [line.strip() for line in batch_file.read_text().splitlines() if line.strip() and not line.startswith("#")]
        if not requests:
            print("Error: batch file is empty", file=sys.stderr)
            sys.exit(1)
        skip_video = args.skip_video and not args.with_video
        results = run_batch(
            requests,
            skip_video=skip_video,
            with_speech=args.with_speech,
            with_vision_check=args.with_vision_check,
        )
        indent = None if args.compact else 2
        summaries = [r.summary for r in results]
        print(json.dumps(summaries, indent=indent, ensure_ascii=False))
        return

    # ── Single request mode ──────────────────────────────────────
    if not args.request:
        parser.print_help()
        sys.exit(1)

    skip_video = args.skip_video and not args.with_video

    result = run_pipeline(
        build_request=args.request,
        skip_video=skip_video,
        with_speech=args.with_speech,
        with_vision_check=args.with_vision_check,
    )

    indent = None if args.compact else 2
    print(json.dumps(json.loads(result.to_json()), indent=indent, ensure_ascii=False))


if __name__ == "__main__":
    main()
