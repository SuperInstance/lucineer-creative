# lucineer-creative

**MMX-powered creative asset pipeline that generates concept art, ambient music, build plans, and narration from a single natural-language request.**

When Lucineer builds, the creative pipeline produces a complete atmospheric asset pack — image, soundtrack, structured plan, and optional voice-over — saved to disk for integration into the Roblox experience.

---

## Architecture

```
Build Request ("build a spooky forest")
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 1: BUILD PLAN (text generation)                    │
│ Model: MiniMax-M3 via MMX chat                           │
│ Output: Structured JSON (title, structures, palette,     │
│         lighting, sound_design, build_steps, est_parts)  │
│ Save: assets/<slug>/build_plan.json                      │
└──────────────────────────┬───────────────────────────────┘
                           │
    ┌──────────────────────┼──────────────────────┐
    ▼                      ▼                      ▼
┌────────────┐    ┌───────────────┐    ┌──────────────────┐
│ Stage 2:   │    │ Stage 3:      │    │ Stage 4 (opt):   │
│ CONCEPT ART│    │ AMBIENT MUSIC │    │ SPEECH NARRATION │
│            │    │               │    │                  │
│ MMX image  │    │ MMX music     │    │ MMX speech       │
│ 16:9 ratio │    │ instrumental  │    │ Expressive       │
│ Stylized   │    │ Looping-friendly│  │ narrator voice   │
│ game render│    │ Cinematic OST │    │                  │
│            │    │               │    │ Triggered by     │
│ Save:      │    │ Save:         │    │ --with-speech    │
│ concept_   │    │ ambient.mp3   │    │                  │
│ art.png    │    │               │    │ Save:            │
└────────────┘    └───────────────┘    │ plan_narration   │
                                       │ .mp3             │
                                       └──────────────────┘
                           │
                           ▼ (optional)
                   ┌───────────────┐
                   │ Stage 5:      │
                   │ VIDEO PREVIEW │
                   │               │
                   │ MMX video     │
                   │ Cinematic     │
                   │ flythrough    │
                   │               │
                   │ Image ref     │
                   │ from Stage 2  │
                   └───────────────┘
```

---

## Asset Manifest

Each pipeline run produces a `manifest.json`:

```json
{
  "request": "build a spooky forest",
  "timestamp": "20260802-205800",
  "assets": [
    {
      "kind": "text",
      "label": "Build Plan",
      "filename": "assets/spooky-forest/build_plan.json",
      "meta": { "parsed": true }
    },
    {
      "kind": "image",
      "label": "Concept Art",
      "filename": "assets/spooky-forest/concept_art.png",
      "mmx_url": "https://...",
      "meta": { "prompt": "Concept art for a Roblox game environment...", "aspect_ratio": "16:9" }
    },
    {
      "kind": "music",
      "label": "Ambient Soundtrack",
      "filename": "assets/spooky-forest/ambient.mp3",
      "meta": { "prompt": "Atmospheric ambient background music...", "instrumental": true }
    }
  ],
  "build_plan": {
    "title": "Whispering Pines",
    "description": "A dense fog-shrouded forest with ancient trees...",
    "biome": "Forest",
    "structures": [...],
    "color_palette": ["#1a2e1a", "#0d1a0d", "#2d4a2d", ...],
    "estimated_parts": 45
  }
}
```

---

## Build Plan Schema

The text-generation stage produces structured JSON:

```typescript
interface BuildPlan {
  title: string;                    // Creative name
  description: string;              // 2-3 sentence overview
  biome: string;                    // Roblox biome/terrain setting
  structures: {                     // Major structural elements
    name: string;
    description: string;
    materials: string[];
  }[];
  props: string[];                  // Decorative elements
  lighting: {
    ambient_color: string;
    skybox: string;
    fog: string;
    brightness: number;
  };
  color_palette: string[];          // Hex color codes
  sound_design: {
    element: string;
    description: string;
  }[];
  build_steps: string[];            // Ordered construction steps
  estimated_parts: number;          // Rough part count
}
```

---

## CLI

```bash
# Standard pipeline (text + image + music)
python3 creative_pipeline.py "build a spooky forest"

# Include video preview (slower, uses more quota)
python3 creative_pipeline.py "cyberpunk city street" --with-video

# Include speech narration of the build plan
python3 creative_pipeline.py "fantasy potion shop" --with-speech

# Run vision analysis on generated concept art
python3 creative_pipeline.py "medieval castle" --with-vision-check

# Compact JSON output
python3 creative_pipeline.py "tower" --compact
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--skip-video` | `true` | Skip video generation |
| `--with-video` | `false` | Enable video preview |
| `--with-speech` | `false` | Generate speech narration |
| `--with-vision-check` | `false` | Vision analysis on concept art |
| `--pretty` | `true` | Pretty-print JSON |
| `--compact` | `false` | Compact JSON output |

---

## MMX Integration

The pipeline wraps the `mmx` CLI (`~/.npm-global/bin/mmx`) via subprocess calls:

| Stage | MMX Command | Output | Timeout |
|-------|-------------|--------|---------|
| Build Plan | `mmx text chat` | JSON via stdout | 300s |
| Concept Art | `mmx image generate` | PNG file | 120s |
| Ambient Music | `mmx music generate` | MP3 file | 300s |
| Speech | `mmx speech synthesize` | MP3 file | 120s |
| Video | `mmx video generate` | MP4 file | 600s |
| Vision | `mmx vision describe` | Text via stdout | 300s |

All calls use `--non-interactive --quiet` flags for agent use. JSON-returning calls add `--output json`.

### Prompt Engineering

**Concept art prompt template:**
```
Concept art for a Roblox game environment: {request}.
Detailed, atmospheric, game-ready lighting, vibrant colors,
isometric perspective showing the full scene layout,
stylized 3D render quality suitable as reference for building in Roblox Studio.
```

**Music prompt template:**
```
Atmospheric ambient background music for a game environment: {request}.
Looping-friendly, immersive, cinematic game soundtrack, subtle and evocative,
suitable for continuous background play in a Roblox experience.
```

---

## File Layout

```
creative_pipeline.py    # Full pipeline implementation (~400 lines)
README.md               # This file
assets/                 # Generated assets, organized by slug
  └── <slug>/
      ├── build_plan.json       # Structured build plan
      ├── concept_art.png       # Generated concept art
      ├── ambient.mp3           # Ambient soundtrack
      ├── plan_narration.mp3    # Speech narration (optional)
      ├── preview.mp4           # Video preview (optional)
      ├── concept_art_analysis.txt  # Vision analysis (optional)
      └── manifest.json         # Complete asset manifest
```

---

## Dependencies

- **mmx CLI** (`~/.npm-global/bin/mmx`) — MiniMax AI platform access
- **MMX Starter plan** — quota-limited; plan asset generation carefully, batch efficiently
- No Python dependencies beyond stdlib (`json`, `subprocess`, `pathlib`, `argparse`)

---

## Related Repositories

| Repository | Role |
|-----------|------|
| [lucineer-brain](../lucineer-brain) | Build command generation (consumes build plans) |
| [lucineer-worker](../lucineer-worker) | Processor daemon orchestration |
| [lucineer-system](../lucineer-system) | Design docs for creative integration |
| [casting-call](../casting-call) | MMX_M3 profile in the model atlas |

---

## License

MIT
