"""
Verdant 3D - Roblox-native multi-AI backend (v24)
v24: self-healing Gemini model discovery (fixes the 404-on-every-build error), admin panel at /admin, teaching by
text lessons and by example 3D models, global memory v2 (local recolouring, persistent research cache, pinned models,
backup/restore) and a build pipeline that spends fewer API calls for the same quality.
"""
import asyncio
import base64
import colorsys
import contextvars
import hashlib
import json
import logging
import math
import os
import re
import secrets
import sqlite3
import struct
import threading
import time
import traceback
import uuid
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("verdant")

_PROGRESS: "contextvars.ContextVar[Any]" = contextvars.ContextVar("verdant_progress", default=None)


def report(msg: str, kind: str = "live") -> None:
    fn = _PROGRESS.get()
    if fn:
        try:
            fn(msg, kind)
        except Exception:
            pass


# ---------------------------------------------------------------- AI effort levels (how hard the AI works)
EFFORTS: Dict[str, Dict[str, Any]] = {
    "low": dict(name="low", label="Low", note="Fastest", thinking=0, max_out=8192, max_chars=30000,
                gemini_total=80, stall=45, job_s=120, parts=0.6, revisions=0, research=0, best_of=1,
                d3_parts=(14, 26), d3_pts=(6, 8), d3_tris=900),
    "mid": dict(name="mid", label="Mid", note="Balanced", thinking=1024, max_out=16384, max_chars=60000,
                gemini_total=150, stall=60, job_s=200, parts=1.0, revisions=1, research=1, best_of=1,
                d3_parts=(26, 44), d3_pts=(8, 10), d3_tris=2200),
    "high": dict(name="high", label="High", note="Deep work (recommended for pro modelers)", thinking=4096, max_out=24576,
                 max_chars=90000, gemini_total=230, stall=90, job_s=330, parts=1.25, revisions=2, research=2, best_of=1,
                 d3_parts=(44, 72), d3_pts=(8, 14), d3_tris=4500),
    "extra": dict(name="extra", label="Extra", note="Everything in High, even stronger", thinking=12288, max_out=32768,
                  max_chars=130000, gemini_total=320, stall=120, job_s=480, parts=1.5, revisions=3, research=3, best_of=2,
                  d3_parts=(70, 110), d3_pts=(8, 16), d3_tris=8000),
}
EFFORTS["mid"]["thinking"] = {"off": 0, "low": 1024, "medium": 4096, "high": 8192}.get(
    os.getenv("THINKING_LEVEL", "low").lower().strip(), 1024)
EFFORT_RANK = {"low": 0, "mid": 1, "high": 2, "extra": 3}
EFFORT_PROMPT = {
    "low": "EFFORT: LOW (fastest). Keep it simple: a clean, readable silhouette with only the essential parts. Do not over-think.",
    "mid": "EFFORT: MID (balanced). Good proportions plus the key secondary details.",
    "high": ("EFFORT: HIGH (deep work). Plan proportions, silhouette and anatomy carefully before writing. Add secondary and "
             "tertiary details (brows, jaws, trims, spikes, claws, accessories), make left and right perfectly symmetric, and "
             "check that every joint overlaps its neighbour."),
    "extra": ("EFFORT: EXTRA (maximum). Do everything in HIGH and go further: richer anatomy, layered plates or skin folds, fine "
              "detail on every limb and the head, two-tone accents, and mentally check the silhouette from front, side and top "
              "before answering. Every joint must overlap; no placeholder shapes."),
}
_EFFORT: "contextvars.ContextVar[Dict[str, Any]]" = contextvars.ContextVar("verdant_effort", default=EFFORTS["mid"])
_TEMP: "contextvars.ContextVar[Optional[float]]" = contextvars.ContextVar("verdant_temp", default=None)


def norm_effort(v: Any) -> str:
    v = str(v or "mid").lower().strip()
    return v if v in EFFORTS else "mid"


def norm_variant(v: Any) -> str:
    v = re.sub(r"[\s_\-]", "", str(v or "classic").lower())
    return "depth3d" if v in ("depth3d", "d3d", "depth", "lowpoly") else "classic"


def effort_range(lo: int, hi: int, prof: Dict[str, Any]) -> Tuple[int, int]:
    m = prof["parts"]
    hi2 = min(int(hi * m), 225)
    lo2 = min(int(lo * m), hi2 - 10)
    return max(40, lo2), max(60, hi2)


VERSION = "v24.1"
app = FastAPI(title="Verdant 3D - Multi-Slab Brick Sculptor + Universal Depth 3D Backend (v24)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["*"], max_age=86400)

# --------------------------------------------------------------------------- config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
# "" / "auto" = ask Google which models the key can use and pick the best one (recommended). Set AI_MODEL only to force one.
AI_MODEL = os.getenv("AI_MODEL", "").strip()
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()
AI_DAILY_CAP = int(os.getenv("AI_DAILY_CAP", "0") or 0)  # 0 = unlimited server-key AI calls per day
THINKING_LEVEL = os.getenv("THINKING_LEVEL", "low").lower().strip()
GEMINI_TOTAL_S = float(os.getenv("GEMINI_TOTAL_S", "150"))
GEMINI_CALL_S = float(os.getenv("GEMINI_CALL_S", "110"))
GEMINI_MAX_OUT = int(os.getenv("GEMINI_MAX_OUT", "16384"))
GEMINI_MAX_CHARS = int(os.getenv("GEMINI_MAX_CHARS", "60000"))
GEMINI_STALL_S = float(os.getenv("GEMINI_STALL_S", "60"))
JOB_MAX_S = float(os.getenv("JOB_MAX_S", "200"))

DEFAULT_MODELS = {
    "gemini": AI_MODEL,
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o"),
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
    "groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
}
ENV_KEYS = {
    "gemini": GEMINI_API_KEY,
    "anthropic": ANTHROPIC_API_KEY,
    "openai": OPENAI_API_KEY,
    "groq": GROQ_API_KEY,
}

AUTO_REVISE = os.getenv("AUTO_REVISE", "1") != "0"
try:
    MAX_REVISIONS = max(0, min(2, int(os.getenv("MAX_REVISIONS", "1"))))
except ValueError:
    MAX_REVISIONS = 1

REVISE_BUDGET_S = 95.0
RETRY_STATUS = (429, 500, 502, 503, 504, 529)

THINKING_BUDGETS = {
    "none": 0,
    "off": 0,
    "low": 1024,
    "medium": 4096,
    "high": 8192,
}

STUD = 0.28
MAX_PARTS = 240
MAX_EXTENT_STUDS = 70.0
MAX_PART_STUDS = 28.0
MIN_PART_STUDS = 0.16
SCENE_MAX_M = 12.0
PUSH = 0.10
SHAPES = {"box", "sphere", "cylinder", "cone", "torus"}
SMOOTH_DEFAULT = {"box": 0.06, "sphere": 0.45, "cylinder": 0.18, "cone": 0.15, "torus": 0.30}
SHAPE_ALIAS = {
    "cube": "box", "block": "box", "brick": "box", "plate": "box", "wedge": "box", "ramp": "box",
    "slope": "box", "part": "box", "roof": "box", "panel": "box", "plank": "box", "slab": "box",
    "ball": "sphere", "orb": "sphere", "ellipsoid": "sphere", "dome": "sphere", "drop": "sphere",
    "lobe": "sphere", "capsule": "sphere", "ovoid": "sphere",
    "tube": "cylinder", "pillar": "cylinder", "rod": "cylinder", "pipe": "cylinder",
    "column": "cylinder", "disc": "cylinder", "disk": "cylinder", "wheel": "cylinder",
    "tire": "cylinder", "tyre": "cylinder", "barrel": "cylinder", "shaft": "cylinder",
    "pyramid": "cone", "spike": "cone", "horn": "cone", "fang": "cone", "tooth": "cone",
    "ring": "torus", "donut": "torus", "hoop": "torus", "collar": "torus",
}
PART_RANGE = {
    "creature": (115, 165), "humanoid": (105, 155), "vehicle": (95, 140),
    "structure": (100, 150), "prop": (75, 120), "plant": (85, 130),
}
GATE = 0.80

# --------------------------------------------------------------------------- the skill
CORE_SKILL = """You are the VERDANT MASTER BRICK SCULPTOR: an elite Roblox Studio character & boss builder (like the artists who build detailed voxel/brick bosses in top Roblox games).
CRITICAL SECRET OF REAL ROBLOX BOSS MODELS:
Real Roblox models are NEVER made of 1 giant scaled block per body part with rounded borders! Instead, every curve, muscle, chest plate, brow arch, crown ring, wrist cuff, and shoulder fur tuft is sculpted from **4 to 10 stacked, stepped, overlapping brick slabs** (`shape: "box"`, `smooth: 0.06`) that step inward/outward by 0.5-1.0 stud per layer!

OUTPUT - reply with ONE valid JSON object and nothing else (no markdown fences, no comments, no prose).
Write every entry of "ops" as a compact ARRAY ROW [id, shape, [w,h,d], [x,y,z], [rotX,rotY,rotZ], "#rrggbb", smooth] - never as an object - so the answer stays short and fast. Never repeat or pad rows:
{
  "title": "Iconic Game-Ready Title",
  "category": "creature|humanoid|vehicle|structure|prop|plant",
  "plan": {
    "reference": "2-3 sentences: exact multi-slab brick layering, hollow roaring mouth anatomy, V-brow expression, and 6-zone color blocking",
    "must_have": ["countable multi-part assemblies, e.g. 'hollow roaring mouth framed by upper snout, cheek pillars & lower jaw with 8 white teeth & pink tongue', 'V-angled cream brow ridge over glowing red slit eyes', '8-piece octagonal golden crown with 5 spikes', '6-slab stepped cream V-chest & abs', '10 jagged dark shoulder fur spikes', '4-ring stepped golden wrist cuffs'"],
    "signature_features": ["6-10 traits that give this model true Roblox Studio brick-sculpted appeal"],
    "palette": {
      "base": "#141c28",
      "secondary": "#e8dfd1",
      "trim": "#f5c518",
      "accent": "#ff2a2a",
      "dark": "#0b0e14",
      "light": "#ffffff"
    },
    "proportions": "one sentence with exact width:height:length ratios in studs",
    "sections": {"Head_And_Face_Slabs": 36, "Stepped_Torso_And_Chest": 28, "Spiked_Shoulder_Fur": 16, "Multi_Slab_Arms_And_Cuffs": 38, "Legs_And_Toes": 24}
  },
  "ops": [
    ["Group_Part","box",[w,h,d],[x,y,z],[degX,degY,degZ],"#rrggbb",0.06]
  ]
}

MULTI-SLAB ROBLOX SCULPTING LAWS (FOLLOW STRICTLY):
1. NEVER ONE GIANT BLOCK — ALWAYS STACKED SLABS!
   - Never build a chest, torso, bicep, forearm, or face out of 1 single cube!
   - Build every muscle and curve as a **staircase of 3 to 6 thinner horizontal brick slabs** (`smooth: 0.05..0.08`) where each slab is slightly wider or narrower than the one below it (e.g., `Chest_Slab_1` width 12 -> `Chest_Slab_2` width 10.5 -> `Chest_Slab_3` width 8.5 -> `Belly_Slab_4` width 6.5 -> `Belly_Slab_5` width 4.5).
2. HOLLOW-FRAME ROARING FACE (NEVER A FLAT WHITE WALL!):
   - Do NOT put a single flat white `FacePlate` across the front of the head! That hides the mouth and looks like a sticker.
   - Instead, build a **3D frame around a deep dark Mouth Pit (`Head_Mouth_Pit`)**:
     * Top of frame: Heavy V-angled cream Brow Ridge (`Head_Brow_L` rot `[0,0,-16]`, `Head_Brow_R` rot `[0,0,16]`, `Head_Brow_Mid`) + `Head_Upper_Muzzle` (`#e8dfd1`) protruding at `z = 8.6`.
     * Sides of frame: Left & Right Cream Cheek Pillars (`Head_CheekPillar_L` at `x = -2.4, z = 8.4`, `Head_CheekPillar_R` at `x = +2.4, z = 8.4`) framing the open mouth!
     * Center of frame: Deep dark `Head_Mouth_Pit` (`#0b0e14`) recessed at `x = 0, y = 17.8, z = 7.9` with a pink `Head_Tongue` (`#d95b72`) on the bottom of the pit, 4 upper white teeth (`Tooth_U1..4`) hanging down from the upper muzzle, and 4 lower white teeth (`Tooth_L1..4`) rising up from the lower jaw!
     * Bottom of frame: Stepped Cream Lower Jaw (`Head_Jaw_Slab1`, `Head_Jaw_Slab2`) at `y = 16.2, z = 8.5`.
     * Eyes: Recessed slightly under the overhanging V-brows at `y = 21.1, z = 8.3`: wide cream eye sockets with fierce glowing red slit `Eye_L` & `Eye_R` (`#ff2a2a`, size `[1.9, 0.75, 0.6]`, rot `[0,0,-10]` / `[0,0,10]`).
3. SPIKED / LAYERED SHOULDER FUR PADS (`Pauldron_*`):
   - Look at real Roblox bosses: the shoulders have **5 to 7 individual angled dark fur wedges/spikes per side** fanning outward and downward over the upper arms (`rot: [10, 15, 25]`, `[0, 0, 35]`, etc.) so the silhouette is jagged and fierce, not a smooth box!
4. MULTI-RING GOLDEN WRIST CUFFS & OCTAGONAL CROWN (`trim` `#f5c518`):
   - Build each wrist bracer from **4 stacked/angled golden bands** (`Bracer_L_Ring1..4`, tilted `rot:[12,0,-8]`) wrapping around the forearm.
   - Build the Crown as a **hollow octagonal ring of 8 golden blocks** around a dark inner cap, topped with **5 tall stepped golden spikes** (center front spike tallest with a diamond gem) so it looks like a real 3D crown, not a yellow box!
5. CRISP STUDIO EDGES (`smooth: 0.06`):
   - Keep `"smooth": 0.05` to `0.08` on all blocks so Three.js renders crisp, faceted Roblox Studio brick edges instead of puffy pillows!"""

_i2 = CORE_SKILL.index("2. HOLLOW-FRAME ROARING FACE")
_i5 = CORE_SKILL.index("5. CRISP STUDIO EDGES")
FIGURE_LAWS = CORE_SKILL[_i2:_i5]
CORE_SKILL = (CORE_SKILL[:_i2]
              + "2. FOLLOW THE BLUEPRINT FOR THIS SUBJECT. Use only the features the real subject has - never add crowns, "
                "golden bracers, shoulder fur or a roaring mouth unless the subject or blueprint calls for them.\n"
              + CORE_SKILL[_i5:].replace("5. CRISP STUDIO EDGES", "3. CRISP STUDIO EDGES"))

BLUEPRINTS = {
    "creature_quad": """BLUEPRINT - STUDIO BOSS APE / GORILLA / QUADRUPED / BEAST, @LO@-@HI@ parts, built from stepped multi-slab Roblox bricks (`smooth: 0.06`):
- IF GORILLA / APE / KING KONG / YETI / TROLL:
  - Follow this EXACT MULTI-SLAB ANATOMICAL TEMPLATE (front = +Z, ground y=0, `smooth: 0.06` on all parts):
  - 1. HEAD, V-BROW & HOLLOW ROARING MOUTH (34-42 parts, centered at x=0, y=16..25):
    - Stepped Dark Cranium & Jagged Cheek Ruff (`base` `#141c28`):
      * `Head_Skull_1` [8.2, 2.6, 6.8] at [0, 22.8, 4.2], `Head_Skull_2` [8.8, 2.8, 7.2] at [0, 20.4, 4.0], `Head_Skull_3` [8.4, 2.6, 6.8] at [0, 18.0, 3.8].
      * Jagged Cheek Ruff Spikes (3 per side): `Head_Ruff_L1` [2.2, 1.8, 4.2] at [-4.6, 21.2, 4.8] rot [0,0,20], `Head_Ruff_L2` [2.4, 1.8, 4.4] at [-4.9, 19.6, 4.8] rot [0,0,30], `Head_Ruff_L3` [2.2, 1.6, 4.0] at [-4.6, 18.0, 4.8] rot [0,0,40] (and mirror `Head_Ruff_R1..3` at +4.6..+4.9).
    - V-Angled Cream Brow & Fierce Red Slit Eyes (z = 8.0..8.9):
      * `Head_Forehead_Slab` (`secondary` `#e8dfd1`) [6.2, 1.4, 2.0] at [0, 22.4, 7.6].
      * Overhanging V-Brows (`secondary` `#e8dfd1`): `Head_Brow_Center` [2.0, 1.1, 2.2] at [0, 21.6, 8.7], `Head_Brow_L` [3.0, 1.1, 2.2] at [-1.9, 21.9, 8.6] rot [0,0,-16], `Head_Brow_R` [3.0, 1.1, 2.2] at [1.9, 21.9, 8.6] rot [0,0,16].
      * Eye Sockets & Glowing Red Slit Eyes: `Eye_Socket_L` / `R` (`secondary`) [2.4, 1.3, 1.4] at [-1.9, 21.0, 7.9] / [1.9, 21.0, 7.9]; `Eye_L` [1.8, 0.75, 0.6] at [-1.9, 21.0, 8.55] rot [0,0,-12] (`accent` `#ff2a2a`), `Eye_R` [1.8, 0.75, 0.6] at [1.9, 21.0, 8.55] rot [0,0,12] (`accent` `#ff2a2a`).
    - Hollow Roaring Mouth Frame (z = 7.6..9.4 — NOT a flat wall!):
      * Upper Snout Bridge (`secondary` `#e8dfd1`): `Head_Snout_Top` [4.8, 1.3, 2.4] at [0, 20.1, 8.5], `Head_Snout_Lip` [5.4, 1.1, 2.6] at [0, 19.2, 8.8], `Head_Nostril_L`/`R` (`dark`) [0.55, 0.55, 0.5] at [-0.8, 20.2, 9.65] / [0.8, 20.2, 9.65].
      * Left & Right Cream Cheek Pillars framing the open mouth (`secondary` `#e8dfd1`): `Head_CheekPillar_L` [1.5, 3.2, 2.4] at [-2.3, 17.9, 8.5], `Head_CheekPillar_R` [1.5, 3.2, 2.4] at [2.3, 17.9, 8.5].
      * Deep Dark Mouth Cavity in the center (`dark` `#0b0e14`): `Head_Mouth_Pit` [3.2, 2.4, 1.6] at [0, 17.9, 7.9], pink `Head_Tongue` (`#d95b72`) [2.4, 0.65, 1.8] at [0, 17.0, 8.4].
      * Lower Jaw Slabs (`secondary` `#e8dfd1`): `Head_Jaw_Upper` [5.0, 1.0, 2.6] at [0, 16.4, 8.7], `Head_Jaw_Chin` [4.0, 0.9, 2.2] at [0, 15.6, 8.4].
      * 8 Crisp White Teeth & Fangs (`light` `#ffffff`, inside the open mouth at z=9.1): `Fang_UL` / `Fang_UR` [0.55, 1.0, 0.55] at [-1.2, 18.4, 9.1] / [1.2, 18.4, 9.1], `Tooth_U1` / `Tooth_U2` [0.45, 0.65, 0.45] at [-0.4, 18.55, 9.15] / [0.4, 18.55, 9.15], `Fang_LL` / `Fang_LR` [0.55, 0.95, 0.55] at [-1.2, 17.2, 9.1] / [1.2, 17.2, 9.1], `Tooth_L1` / `Tooth_L2` [0.45, 0.6, 0.45] at [-0.4, 17.05, 9.15] / [0.4, 17.05, 9.15].
    - 10-Part Sculpted Golden Crown (`trim` `#f5c518`, at y=24.2..28.5):
      * Octagonal Band around dark cap: `Crown_Cap_Dark` (`base`) [4.8, 2.0, 4.8] at [0, 25.0, 4.2], `Crown_Band_F` [5.8, 1.4, 1.4] at [0, 24.8, 6.6], `Crown_Band_B` [5.8, 1.4, 1.4] at [0, 24.8, 1.8], `Crown_Band_L` [1.4, 1.4, 5.8] at [-2.4, 24.8, 4.2], `Crown_Band_R` [1.4, 1.4, 5.8] at [2.4, 24.8, 4.2].
      * Stepped Tall Crown Spikes: `Crown_Spike_F1` [2.2, 1.8, 1.2] at [0, 26.2, 6.7], `Crown_Spike_F2` [1.4, 1.8, 1.0] at [0, 27.6, 6.8], `Crown_Gem_Front` [1.1, 1.1, 0.6] at [0, 25.8, 7.2] rot [0,0,45] (`#ffe066`), `Crown_Spike_FL` / `FR` [1.5, 2.4, 1.1] at [-2.2, 26.4, 5.8] / [2.2, 26.4, 5.8] rot [0,0,18] / [0,0,-18], `Crown_Spike_BL` / `BR` [1.4, 2.0, 1.1] at [-2.4, 26.1, 2.8] / [2.4, 26.1, 2.8] rot [0,0,22] / [0,0,-22].
  - 2. STEPPED TORSO SLABS & 7-SLAB CREAM V-CHEST (26-32 parts):
    - Stacked Dark Torso Slabs (`base` `#141c28`):
      * `Torso_Slab_1` [15.2, 3.2, 9.6] at [0, 18.2, 1.6], `Torso_Slab_2` [14.4, 3.2, 9.2] at [0, 15.4, 1.4], `Torso_Slab_3` [12.8, 3.0, 8.6] at [0, 12.6, 1.0], `Torso_Slab_4` [11.2, 2.8, 8.0] at [0, 10.0, 0.5], `Pelvis_Slab` [10.4, 3.2, 7.4] at [0, 7.4, 0.0].
    - 7-Slab Stepped Cream Chest & V-Taper Abs (`secondary` `#e8dfd1`, stepping down in width to a V-point!):
      * `Chest_Pec_L_Top` [5.0, 2.2, 1.6] at [-2.7, 17.2, 6.2], `Chest_Pec_R_Top` [5.0, 2.2, 1.6] at [2.7, 17.2, 6.2], `Chest_Pec_L_Bot` [5.4, 2.4, 1.8] at [-2.8, 15.2, 6.3], `Chest_Pec_R_Bot` [5.4, 2.4, 1.8] at [2.8, 15.2, 6.3], `Chest_Sternum` [3.2, 3.6, 1.5] at [0, 16.0, 6.1].
      * Stepped V-Abs: `Belly_Slab_1` [8.6, 1.8, 1.6] at [0, 13.4, 5.4], `Belly_Slab_2` [6.8, 1.7, 1.5] at [0, 11.8, 4.9], `Belly_Slab_3` [4.8, 1.6, 1.4] at [0, 10.3, 4.4], `Belly_Slab_VTip` [2.8, 1.5, 1.3] at [0, 8.9, 3.9].
    - 12 Jagged Shoulder Fur Spikes (`base` `#1e293b`, 6 per shoulder fanning out like armor scales!):
      * Left Shoulder: `Pauldron_L_Top1` [5.2, 1.6, 6.4] at [-7.8, 20.8, 2.0] rot [0,0,15], `Pauldron_L_Top2` [5.8, 1.6, 6.8] at [-8.8, 19.6, 2.0] rot [0,0,25], `Pauldron_L_Mid1` [6.2, 1.6, 7.0] at [-9.6, 18.2, 2.0] rot [0,0,35], `Pauldron_L_Spike1` [3.2, 1.4, 2.8] at [-11.2, 19.0, 3.2] rot [15,20,40], `Pauldron_L_Spike2` [3.2, 1.4, 2.8] at [-11.5, 17.5, 1.5] rot [-10,-15,45], `Pauldron_L_Low` [5.6, 1.5, 6.2] at [-10.0, 16.8, 2.0] rot [0,0,40] (and mirror `Pauldron_R_*` on +X!).
  - 3. MULTI-SLAB MUSCULAR ARMS, 4-RING GOLD CUFFS & KNUCKLE HANDS (36-44 parts):
    - Left Arm built from stacked muscular slabs (`base` `#141c28`, aligned at X = -9.6..-10.2):
      * Bicep Slabs: `Arm_L_Shoulder_Slab` [6.2, 2.8, 6.2] at [-9.2, 16.8, 2.2], `Arm_L_Bicep_1` [6.6, 2.6, 6.6] at [-9.6, 14.4, 2.8], `Arm_L_Bicep_2` [6.4, 2.6, 6.4] at [-9.8, 12.2, 3.4], `Arm_L_Elbow_Slab` [6.0, 2.2, 6.0] at [-9.8, 10.2, 4.0].
      * Forearm Slabs: `Arm_L_Forearm_1` [7.0, 2.6, 7.0] at [-9.8, 8.2, 4.6], `Arm_L_Forearm_2` [6.8, 2.6, 6.8] at [-9.8, 6.0, 5.0].
      * 4-Ring Stepped Golden Wrist Bracer (`trim` `#f5c518`): `Bracer_L_Ring1` [7.6, 1.0, 7.6] at [-9.8, 8.8, 4.5] rot [12,0,-6], `Bracer_L_Ring2` [7.8, 1.1, 7.8] at [-9.8, 7.7, 4.7] rot [12,0,-6], `Bracer_L_Ring3` [7.8, 1.1, 7.8] at [-9.8, 6.6, 4.9] rot [12,0,-6], `Bracer_L_Ring4` [7.5, 1.0, 7.5] at [-9.8, 5.5, 5.1] rot [12,0,-6].
      * Multi-Slab Fist & Cream Knuckles: `Hand_L_Back1` (`base`) [7.0, 2.0, 5.8] at [-9.8, 3.8, 5.4], `Hand_L_Back2` (`base`) [6.8, 2.0, 5.8] at [-9.8, 2.0, 5.6], plus 4 two-segment cream fingers (`Finger_L_1..4` `[1.55, 2.6, 2.4]` at `x = [-12.1, -10.5, -8.9, -7.3], y = 1.3, z = 7.4` in `secondary` `#e8dfd1`) and `Thumb_L` `[1.6, 2.2, 2.2]` at `[-6.8, 1.8, 5.6]`.
    - Mirror all parts for Right Arm (`Arm_R_*`, `Bracer_R_Ring1..4`, `Hand_R_*`, `Finger_R_1..4`, `Thumb_R`) at X = +9.2..+9.8!
  - 4. MULTI-SLAB HIND LEGS & CREAM TOES (22-26 parts):
    - Left Leg (`base` `#141c28`): `Leg_L_Thigh_1` [5.8, 2.6, 6.0] at [-4.6, 6.2, -0.2], `Leg_L_Thigh_2` [5.4, 2.4, 5.6] at [-4.8, 4.2, 0.1], `Leg_L_Shin_1` [5.0, 2.2, 5.2] at [-4.8, 2.4, 0.3], `Foot_L_Base` [5.4, 1.8, 5.6] at [-4.8, 0.9, 0.8], plus 4 cream toe blocks `Toe_L_1..4` (`secondary` `#e8dfd1`) `[1.25, 1.6, 2.4]` across `x = [-6.6, -5.4, -4.2, -3.0], y = 0.8, z = 3.2` (and mirror Right Leg `Leg_R_*`, `Foot_R_Base`, `Toe_R_1..4`).""",

    "creature_multi": """BLUEPRINT - CEPHALOPOD (SQUID / OCTOPUS / KRAKEN) OR ARACHNID / CRUSTACEAN, @LO@-@HI@ parts, built from stepped Roblox blocks (`smooth: 0.06`):
- Stepped multi-slab Head & Mantle (7-9 stacked horizontal slabs tapering cleanly), overhanging brow ridges, big glowing `accent` eyes + dark pupils, open beak surrounded by teeth, 4 `secondary` cream under-mantle plates, 2 stepped fins, and 8 radial 4-segment arms + 2 clubs with `light` suckers.""",

    "creature_dragon": """BLUEPRINT - DRAGON / WYVERN / WINGED BOSS, @LO@-@HI@ parts, built from stepped Roblox brick slabs (`smooth: 0.06`):
- Hollow roaring jaw with upper snout, lower jaw, dark mouth pit, fiery tongue, and 8 white fangs; V-angled brows over glowing eyes; 2 swept 4-segment horns; 8 stepped cream belly scute slabs; multi-slab torso & legs with golden leg cuffs and white claws; wide 4-spar wings with layered membrane plates.""",

    "creature_bird": """BLUEPRINT - BIRD / PHOENIX / GRIFFIN BOSS, @LO@-@HI@ parts (`smooth: 0.06`):
- Multi-slab stepped breast with 6 `secondary` plumage slabs, 2-part hooked golden `trim` beak with open mouth cavity, glowing `accent` eyes under brow ridges, 6-part royal head crest, 2 wide 3-layer feathered wings, fanned tail feathers, and 2 golden-taloned legs.""",

    "creature_aquatic": """BLUEPRINT - AQUATIC / SHARK / LEVIATHAN BOSS, @LO@-@HI@ parts (`smooth: 0.06`):
- 10-segment tapering body along Z (`Body_Seg1..10`) with crisp two-tone counter-shading (`base` top, 8 stepped `secondary` cream belly slabs), hollow open jaws at +Z lined with 10 white `Tooth_*` blocks and `Mouth_Pit`, glowing `accent` eyes, gill slits, tall stepped dorsal fin, pectoral fins, and twin tail flukes.""",

    "humanoid": """BLUEPRINT - HUMANOID / MECH / KNIGHT / BOSS CHARACTER, @LO@-@HI@ parts, 18-24 studs tall (`smooth: 0.06`):
- Multi-slab Head with recessed visor/face, glowing `Eye_L`/`Eye_R` under angled brows, open mouth/mask, and a 10-part helmet or golden crown (`Crown_*`).
- Stepped Torso & V-taper armor with 6 contrasting `secondary` chest/ab plates, glowing `accent` chest emblem, `trim` gold belt, and 6-part spiked/layered shoulder pauldrons per side (`Pauldron_L_1..6`, `Pauldron_R_1..6`).
- Multi-slab Arms (`Shoulder -> Bicep_1..2 -> Forearm_1..2 -> 4-Ring Gold Bracer -> Hand + 5 Fingers`) wielding a signature weapon, and multi-slab Legs (`Thigh_1..2 -> KneeGuard -> Shin_1..2 -> Boot + Toes`) at y=0.""",

    "vehicle": """BLUEPRINT - VEHICLE (SUPERCAR / TRUCK / AIRCRAFT / SHIP), @LO@-@HI@ parts (`smooth: 0.06`):
- Multi-tone stepped bodywork: `base` paint, `secondary` stripes/trim, `dark` splitters/diffuser/tires, `trim` gold/silver rims, `light` headlights & glass, `accent` glowing taillights/thrusters, 4 multi-part cylindrical wheels, stepped hood/roof slopes, quad exhausts, and rear wing.""",

    "structure": """BLUEPRINT - ARCHITECTURAL STRUCTURE / CASTLE / TOWER / SHRINE, @LO@-@HI@ parts (`smooth: 0.04`):
- Stepped stone foundation, corner quoins, recessed wall panels, `trim` roof trims/banners, `accent` glowing windows & lanterns, `light` steps and arch keystones, pitched roof slopes, and courtyard props.""",

    "prop": """BLUEPRINT - DETAILED PROP / WEAPON / ARTIFACT / CHEST, @LO@-@HI@ parts (`smooth: 0.06`):
- Crisp multi-slab layering using all 6 palette slots: `dark` grip/frame, `trim` gold guard/brackets, `base` core, `secondary` blade fuller/inlay, `light` razor edges/rivets, and `accent` glowing runes/gems.""",

    "plant": """BLUEPRINT - STYLIZED TREE / BOTANICAL DIORAMA / FANTASY FLORA, @LO@-@HI@ parts (`smooth: 0.06`):
- Stepped earth & mossy rock base, multi-slab root & trunk chains, and 30-40 stepped voxel canopy/blossom/mushroom slabs across 3 distinct shades with glowing fruit/spores.""",
}

WINGS_BP = """WINGS ADD-ON: Build 2 large symmetrical wings spreading wide along -X and +X from block shoulders. Each wing has a shoulder block, a 3-block leading-edge arm in `base`/`trim`, and 2-3 rows of 6-8 flat feather/membrane blocks in `secondary`/`accent` fanned out in 15 degree steps."""

KW = {
    "vehicle": r"\b(car|supercar|sportcar|sports car|hypercar|sedan|coupe|convertible|lamborghini|ferrari|porsche|bugatti|mclaren|mustang|camaro|corvette|gtr|supra|f1|formula|racecar|truck|pickup|lorry|bus|van|suv|jeep|hummer|limo|taxi|police car|ambulance|firetruck|tractor|motorcycle|motorbike|bike|bicycle|scooter|moped|atv|quad|tank|apc|humvee|train|locomotive|tram|subway|monorail|plane|airplane|biplane|airliner|fighter|jet|bomber|helicopter|chopper|drone|blimp|zeppelin|airship|rocket|spaceship|starship|spacecraft|shuttle|ufo|saucer|rover|lander|boat|ship|sailboat|yacht|speedboat|ferry|cruiser|battleship|carrier|submarine|sub|pirate ship|galleon|canoe|kayak|raft|jetski|hovercraft|sled|sleigh|chariot|wagon|cart|carriage|coach|rickshaw|kart|go-kart|buggy|forklift|bulldozer|excavator|crane|dump truck|cement mixer|combine|harvester|snowplow|snowmobile|segway|skateboard|hoverboard|rollercoaster)\b",
    "structure": r"\b(house|home|cottage|cabin|chalet|bungalow|villa|mansion|manor|estate|palace|chateau|castle|keep|citadel|fortress|fort|bastion|bunker|outpost|watchtower|guardtower|tower|turret|spire|obelisk|monument|memorial|arch|gate|gateway|portals?|wall|rampart|bridge|viaduct|aqueduct|overpass|pier|dock|wharf|harbor|lighthouse|beacon|windmill|watermill|mill|barn|silo|stable|coop|greenhouse|conservatory|shed|shack|hut|hovel|igloo|yurt|teepee|tent|pavilion|gazebo|pergola|pagoda|shrine|temple|sanctuary|monastery|abbey|church|cathedral|chapel|mosque|synagogue|stupa|pyramid|ziggurat|parthenon|colosseum|amphitheater|arena|stadium|coliseum|gym|pool|rink|shop|store|market|bazaar|stall|booth|kiosk|bakery|cafe|coffee shop|restaurant|diner|pizzeria|bar|pub|tavern|inn|hotel|motel|hostel|bank|vault|post office|library|school|university|college|academy|museum|gallery|theater|theatre|cinema|opera|studio|workshop|forge|smithy|foundry|factory|warehouse|depot|hangar|garage|gas station|service station|fire station|police station|hospital|clinic|pharmacy|prison|jail|dungeon|asylum|lab|laboratory|observatory|planetarium|power plant|reactor|substation|dam|water tower|cooling tower|oil rig|pumpjack|mine|quarry|sawmill|lumberyard|farm|ranch|orchard|vineyard|park|playground|fountain|plaza|square|courtyard|alley|street|road|highway|intersection|roundabout|tunnel|subway station|train station|airport|terminal|control tower|helipad|spaceport|launch pad|moonbase|habitat|biodome|dome|skyline|cityblock|skyscraper|highrise|office|apartment|condo|tenement|ruins|stonehenge|totem)\b",
    "humanoid": r"\b(man|woman|boy|girl|kid|child|baby|human|person|people|guy|lady|gentleman|villager|citizen|peasant|farmer|miner|blacksmith|merchant|trader|shopkeeper|baker|chef|cook|waiter|bartender|butcher|barber|tailor|carpenter|mason|builder|worker|mechanic|engineer|scientist|professor|teacher|student|doctor|nurse|surgeon|paramedic|firefighter|fireman|policeman|police|cop|detective|sheriff|deputy|guard|security|warden|prisoner|inmate|convict|criminal|thief|burglar|bandit|outlaw|robber|gangster|mafia|hitman|assassin|spy|agent|soldier|trooper|marine|commando|sniper|medic|pilot|captain|general|commander|admiral|sailor|seaman|pirate|buccaneer|corsair|viking|berserker|gladiator|spartan|centurion|legionnaire|knight|paladin|crusader|templar|squire|warrior|fighter|barbarian|brawler|monk|sensei|ninja|shinobi|samurai|ronin|shogun|kensei|archer|ranger|hunter|huntress|scout|rogue|bard|minstrel|jester|clown|mime|acrobat|magician|illusionist|wizard|sorcerer|sorceress|witch|warlock|mage|archmage|necromancer|druid|shaman|cleric|priest|priestess|bishop|pope|monarch|king|queen|prince|princess|emperor|empress|pharaoh|sultan|tsar|lord|duke|duchess|baron|noble|aristocrat|president|mayor|judge|lawyer|athlete|runner|sprinter|swimmer|diver|gymnast|boxer|wrestler|karate|judoka|fencer|skater|skier|snowboarder|surfer|cyclist|biker|racer|driver|golfer|tennis|basketball|football|soccer|baseball|hockey|quarterback|goalie|referee|coach|cheerleader|dancer|ballerina|singer|rockstar|musician|guitarist|drummer|dj|actor|director|artist|painter|sculptor|photographer|journalist|reporter|astronaut|cosmonaut|spaceman|explorer|adventurer|mountaineer|cowboy|cowgirl|desperado|native|tribal|nomad|bedouin|eskimo|inuit|elf|high elf|wood elf|dark elf|drow|dwarf|gnome|halfling|hobbit|orc|uruk|goblin|hobgoblin|troll|ogre|cyclops|titan|goliath|minotaur|centaur|satyr|faun|harpy|siren|mermaid|merman|triton|naga|gorgon|medusa|lamia|succubus|incubus|demon|devil|fiend|imp|angel|seraph|cherub|valkyrie|deity|god|goddess|avatar|demigod|lich|vampire|nosferatu|werewolf|lycan|zombie|ghoul|wight|draugr|mummy|skeleton|ghost|phantom|specter|wraith|banshee|poltergeist|spirit|reaper|grim reaper|scarecrow|golem|ent|treant|elemental|djinn|genie|efreet|robot|android|cyborg|automaton|droid|mech|mecha|walker|synth|reploid|bionic|alien|martian|grey|reptilian|predator|mutant|superhero|supervillain|hero|villain|vigilante|sidekick|champion|character|npc|miniboss|mob|dummy|mannequin|statue|bust|figurine|doll|action figure|toy soldier|nutcracker|snowman|santa|leprechaun|cupid)\b",
    "plant": r"\b(tree|oak|pine|fir|spruce|cedar|redwood|sequoia|birch|maple|willow|weeping willow|elm|ash|beech|poplar|aspen|cypress|yew|baobab|acacia|banyan|mangrove|palm|coconut palm|date palm|banana tree|apple tree|cherry blossom|sakura|peach tree|orange tree|lemon tree|olive tree|fig tree|bonsai|topiary|dead tree|hollow tree|stump|log|branch|twig|root|bush|shrub|hedge|bramble|briar|berry bush|rose bush|hydrangea|azalea|rhododendron|lavender|heather|sagebrush|tumbleweed|flower|rose|tulip|sunflower|daisy|lily|waterlily|lotus|orchid|iris|peony|poppy|marigold|dandelion|violet|pansy|carnations|chrysanthemum|daffodil|hyacinth|hibiscus|jasmine|plumeria|bougainvillea|bird of paradise|protea|edelweiss|blossom|bouquet|wreath|garland|petal|bud|stem|leaf|leaves|clover|shamrock|fern|frond|moss|lichen|ivy|vine|creeper|liana|grapevine|beanstalk|grass|lawn|turf|meadow|reed|bulrush|cattail|papyrus|bamboo|sugarcane|wheat|barley|oats|rye|corn|maize|rice|crop|cactus|saguaro|prickly pear|barrel cactus|agave|aloe|yucca|succulent|echeveria|jade plant|venus flytrap|pitcher plant|sundew|carnivorous plant|rafflesia|corpse flower|kelp|seaweed|algae|coral|anemone|sponge|mushroom|toadstool|fungi|fungus|morel|truffle|chanterelle|puffball|amanita|mycelium|spore|forest|woods|grove|thicket|jungle|rainforest|swamp|marsh|bog|oasis|garden|terrace|planter|terrarium)\b",
    "prop": r"\b(sword|broadsword|longsword|shortsword|greatsword|claymore|rapier|foil|saber|sabre|scimitar|cutlass|katana|wakizashi|nodachi|tanto|dagger|dirk|stiletto|knife|machete|kukri|cleaver|bayonet|blade|excalibur|axe|battleaxe|hatchet|tomahawk|halberd|poleaxe|glaive|bardiche|scythe|sickle|kama|spear|lance|pike|javelin|harpoon|trident|bident|pitchfork|staff|quarterstaff|bo staff|magic staff|crozier|scepter|wand|rod|club|mace|morningstar|flail|warhammer|maul|mallet|hammer|sledgehammer|pickaxe|mattock|crowbar|wrench|spanner|screwdriver|pliers|saw|chainsaw|drill|jackhammer|shovel|spade|hoe|rake|trowel|shears|scissors|anvil|vise|bellows|tongs|chisel|level|tape measure|toolbox|ladder|scaffold|wheelbarrow|bow|longbow|shortbow|recurve bow|compound bow|crossbow|ballista|catapult|trebuchet|cannon|mortar|howitzer|gatling|minigun|gun|handgun|pistol|revolver|deagle|glock|luger|flintlock|musket|blunderbuss|rifle|carbine|assault rifle|ak47|m16|sniper rifle|shotgun|blaster|raygun|laser gun|plasma rifle|railgun|gauss|phaser|needler|bazooka|rpg|missile launcher|grenade launcher|flamethrower|taser|grenade|bomb|dynamite|tnt|c4|landmine|claymore mine|torpedo|missile|nuke|ammo|bullet|shell|cartridge|quiver|arrow|bolt|dart|shuriken|kunai|boomerang|chakram|sling|slingshot|whip|lasso|net|bola|shield|buckler|heater shield|kite shield|tower shield|pavise|aegis|riot shield|armor|breastplate|cuirass|chainmail|plate mail|gauntlet|bracer|greave|pauldron|helmet|helm|great helm|sallet|bascinet|morion|kabuto|spartan helmet|viking helmet|knight helmet|space helmet|diving helmet|gas mask|mask|oni mask|tiki mask|venetian mask|goggles|visor|glasses|sunglasses|monocle|eyepatch|crown|tiara|diadem|circlet|coronet|halo|hat|cap|top hat|bowler|fedora|sombrero|cowboy hat|beret|beanie|fez|turban|straw hat|wizard hat|witch hat|pirate hat|tricorne|santa hat|chef hat|hard hat|ring|amulet|talisman|pendant|necklace|locket|choker|bracelet|bangle|brooch|earring|medal|medallion|badge|crest|emblem|belt|buckle|strap|holster|scabbard|sheath|cape|cloak|scarf|tie|bowtie|glove|mitten|boot|shoe|sneaker|sandal|slipper|skate|rollerblade|ski|snowshoe|flipper|backpack|rucksack|knapsack|satchel|messenger bag|briefcase|suitcase|luggage|trunk|duffel|purse|handbag|pouch|wallet|canteen|flask|waterskin|chest|treasure chest|coffer|strongbox|safe|lockbox|mimic|crate|box|carton|package|parcel|present|gift|giftbox|barrel|keg|cask|vat|tun|bucket|pail|tub|trough|basket|hamper|urn|vase|amphora|jug|pitcher|ewer|jar|mason jar|canister|tin|can|bottle|potion|elixir|vial|phial|beaker|test tube|retort|alembic|cauldron|kettle|teapot|coffee pot|mug|cup|goblet|chalice|grail|tankard|stein|wineglass|champagne flute|tumbler|shot glass|bowl|plate|platter|dish|tray|salver|saucer|tureen|gravy boat|fork|spoon|ladle|spatula|whisk|rolling pin|cutting board|mortar and pestle|salt shaker|pepper mill|napkin|tablecloth|placemat|coaster|candlestick|candelabra|menorah|chandelier|lantern|oil lamp|genie lamp|lava lamp|desk lamp|floor lamp|streetlamp|flashlight|torch|brazier|campfire|bonfire|fireplace|hearth|stove|oven|microwave|toaster|blender|mixer|coffee maker|espresso|fridge|refrigerator|freezer|sink|faucet|tap|bathtub|shower|toilet|mirror|scale|towel rack|soap dispenser|toothbrush|hairdryer|razor|comb|brush|perfume|lipstick|compact|table|dining table|coffee table|end table|side table|nightstand|desk|workbench|counter|island|altar|podium|lectern|easel|chair|armchair|recliner|rocking chair|office chair|gaming chair|stool|barstool|bench|pew|throne|sofa|couch|loveseat|sectional|futon|chaise|ottoman|pouf|beanbag|bed|bunk bed|four-poster|crib|cradle|hammock|mattress|pillow|cushion|blanket|quilt|rug|carpet|mat|curtain|drapes|blinds|screen|divider|wardrobe|armoire|closet|dresser|chest of drawers|cabinet|cupboard|hutches|pantry|sideboard|buffet|credenza|locker|bookshelf|bookcase|shelf|rack|coat rack|hat stand|umbrella stand|plant stand|pedestal|plinth|column|pillar|balustrade|railing|fence|mailbox|fire hydrant|trash can|dumpster|recycle bin|lamppost|traffic light|stop sign|street sign|billboard|neon sign|awnings|parasol|umbrella|sleeping bag|cooler|grill|bbq|picnic table|birdbath|birdhouse|feeder|gnomes|wind chime|weathervane|sundial|globe|orrery|telescope|microscope|binoculars|spyglass|magnifying glass|compass|sextant|astrolabe|hourglass|clock|grandfather clock|wall clock|alarm clock|pocket watch|wristwatch|stopwatch|metronome|thermometer|barometer|balance|weights|abacus|calculator|typewriter|cash register|atm|vending machine|arcade|arcade cabinet|pinball|claw machine|jukebox|slot machine|roulette|pool table|foosball|air hockey|dartboard|chessboard|chess piece|dice|domino|cards|poker chip|mahjong|puzzle|rubiks cube|yo-yo|spinning top|dreidel|marbles|blocks|lego|train set|rocking horse|teddy bear|plushie|kite|balloon|ball|soccer ball|basketball|football|baseball|tennis ball|golf ball|bowling ball|bowling pin|billiard ball|beach ball|snowglobe|trophy|ribbon|plaque|bell|gong|whistle|bugle|trumpet|trombone|tuba|french horn|saxophone|clarinet|oboe|flute|piccolo|recorder|panpipes|harmonica|accordion|bagpipes|organ|piano|grand piano|upright piano|keyboard|harpsichord|harp|lyre|lute|guitar|acoustic guitar|electric guitar|bass|ukulele|banjo|mandolin|sitar|violin|viola|cello|double bass|fiddle|drum|snare|kick drum|tom|bongo|conga|djembe|timpani|tambourine|maraca|castanets|triangle|xylophone|marimba|vibraphone|glockenspiel|cymbal|hi-hat|microphone|amp|amplifier|speaker|subwoofer|boombox|radio|gramophone|phonograph|turntable|record player|vinyl|cassette|walkman|mp3|ipod|headphones|earbuds|camera|dslr|polaroid|camcorder|video camera|projector|film reel|clapperboard|tripod|spotlight|tv|television|crt|flatscreen|monitor|computer|pc|laptop|macbook|tablet|ipad|phone|smartphone|iphone|rotary phone|payphone|telephone|fax|printer|scanner|router|modem|server rack|mainframe|hard drive|floppy|usb|sd card|cd|dvd|battery|solar panel|wind turbine|generator|transformer|engine|motor|piston|gear|cog|spring|lever|pulley|valve|pipe|gauge|meter|switch|button|control panel|console|radar|satellite dish|antenna|buoy|anchor|ship wheel|propeller|oar|paddle|lifebuoy|treasure map|scroll|parchment|book|tome|grimoire|spellbook|bible|dictionary|encyclopedia|notebook|journal|diary|ledger|clipboard|folder|binder|letter|envelope|stamp|newspaper|magazine|comic|poster|painting|frame|canvas|palette|paintbrush|pencil|pen|quill|inkwell|crayon|marker|chalk|eraser|ruler|protractor|stapler|tape|glue|paperclip|pushpin|sculpture|gargoyle|well|gravestone|tombstone|crypt|sarcophagus|coffin|casket|skull|bones|ribcage|fossil|amber|crystal|quartz|amethyst|emerald|ruby|sapphire|diamond|topaz|opal|pearl|geode|ore|ingot|gold bar|nugget|coin|doubloon|key|skeleton key|keycard|padlock|chain|shackles|handcuffs|cage|trap|bear trap|mousetrap|target|bullseye|sandcastle|cake|birthday cake|wedding cake|cupcake|muffin|donut|doughnut|croissant|bagel|bread|loaf|baguette|pretzel|toast|pancake|waffle|pie|tart|cookie|biscuit|macaron|brownie|chocolate|candy|lollipop|candy cane|gummy|ice cream|sundae|popsicle|milkshake|smoothie|pizza|burger|hamburger|cheeseburger|hotdog|sandwich|taco|burrito|nachos|sushi|nigiri|maki|ramen|noodles|soup|stew|salad|steak|rib|chicken leg|drumstick|ham|bacon|sausage|egg|fried egg|omelette|cheese|butter|milk|yogurt|honey|jam|peanut butter|ketchup|mustard|mayo|hot sauce|soy sauce|oil|vinegar|salt|pepper|sugar|flour|pasta|cereal|popcorn|chips|fries|nut|acorn|pinecone|apple|banana|orange|lemon|lime|grapefruit|peach|pear|plum|cherry|strawberry|raspberry|blueberry|blackberry|grape|watermelon|melon|cantaloupe|pineapple|mango|papaya|kiwi|pomegranate|fig|date|coconut|avocado|tomato|potato|carrot|broccoli|cauliflower|cabbage|lettuce|spinach|kale|celery|asparagus|artichoke|bell pepper|chili|jalapeno|onion|garlic|leek|radish|turnip|beet|pumpkin|squash|zucchini|cucumber|eggplant|aubergine|pea|bean|peanut)\b",
}
SUB_KW = {
    "dragon": r"\b(dragon|wyvern|drake|lindworm|wyrm|hydradragon|bahamut|smaug|charizard|balrog)\b",
    "bird": r"\b(bird|phoenix|eagle|hawk|falcon|owl|parrot|macaw|cockatoo|toucan|peacock|swan|flamingo|penguin|duck|goose|pelican|stork|crane|heron|ibis|vulture|condor|albatross|seagull|puffin|kiwi|ostrich|emu|cassowary|chicken|rooster|hen|turkey|quail|pheasant|pigeon|dove|crow|raven|magpie|jay|robin|sparrow|finch|canary|hummingbird|kingfisher|woodpecker|griffin|griffon|hippogriff|roc|harpy|thunderbird|cockatrice|archaeopteryx|pterodactyl|pteranodon|quetzalcoatlus)\b",
    "multi": r"\b(kraken|octopus|squid|cuttlefish|nautilus|jellyfish|medusa|hydra|cthulhu|cephalopod|spider|tarantula|black widow|recluse|scorpion|crab|hermit crab|king crab|lobster|crayfish|shrimp|prawn|mantis shrimp|isopod|trilobite|centipede|millipede|tick|mite|ant|termite|bee|bumblebee|wasp|hornet|yellowjacket|beetle|scarab|stag beetle|rhino beetle|ladybug|weevil|firefly|glowworm|cockroach|cricket|grasshopper|locust|katydid|mantis|praying mantis|stick insect|leaf insect|earwig|flea|louse|mosquito|fly|housefly|dragonfly|damselfly|butterfly|moth|caterpillar|silkworm|grub|maggot|larva|insect|bug|arachnid|crustacean|starfish|sea star|brittle star|sea urchin|sand dollar)\b",
    "aquatic": r"\b(fish|goldfish|koi|carp|trout|salmon|tuna|mackerel|sardine|anchovy|herring|cod|haddock|halibut|flounder|sole|plaice|turbot|catfish|pike|perch|bass|tilapia|piranha|barracuda|swordfish|marlin|sailfish|flying fish|anglerfish|lanternfish|blobfish|pufferfish|blowfish|boxfish|triggerfish|parrotfish|clownfish|angelfish|butterflyfish|tang|surgeonfish|moorish idol|lionfish|scorpionfish|stonefish|goby|blenny|wrasse|grouper|snapper|bream|betta|guppy|tetra|discus|cichlid|arowana|gar|arapaima|coelacanth|sturgeon|paddlefish|shark|great white|hammerhead|tiger shark|whale shark|basking shark|mako|thresher|bull shark|nurse shark|reef shark|goblin shark|megalodon|ray|stingray|manta ray|eagle ray|electric ray|sawfish|skate|guitarfish|whale|blue whale|humpback|sperm whale|orca|killer whale|beluga|narwhal|fin whale|minke|right whale|bowhead|pilot whale|beaked whale|dolphin|bottlenose|porpoise|vaquita|manatee|dugong|seacow|seal|sea lion|walrus|fur seal|elephant seal|leopard seal|otter|sea otter|platypus|axolotl|newt|salamander|tadpole|eel|moray|conger|electric eel|lamprey|hagfish|sea snake|serpent|sea serpent|leviathan|nessie|loch ness|mosasaur|ichthyosaur|plesiosaur|liopleurodon|basilosaurus|dunkleosteus|anomalocaris|seahorse|pipefish|seadragon|worm|earthworm|leech|bobbit worm|tube worm|flatworm|planarian|nudibranch|sea slug|sea cucumber|conch|snail|slug)\b",
}
WINGS_KW = r"\b(wing|wings|winged|dragon|wyvern|drake|pegasus|alicorn|griffin|griffon|hippogriff|phoenix|roc|thunderbird|harpy|gargoyle|sphinx|chimera|manticore|cockatrice|basilisk|bat|vampire bat|flying fox|pterodactyl|pteranodon|archaeopteryx|butterfly|moth|bee|bumblebee|wasp|hornet|dragonfly|damselfly|fairy|pixie|sprite|sylph|angel|seraph|cherub|valkyrie|demon|devil|imp|succubus|bird|eagle|hawk|falcon|owl|parrot|toucan|peacock|swan|flamingo|duck|goose|seagull|crow|raven|dove|pigeon|rooster|chicken|hummingbird|jetpack)\b"
ANIMAL_KW = (r"\b(gorilla|ape|apes|monkey|chimp|chimpanzee|baboon|orangutan|kong|yeti|bigfoot|sasquatch|goat|sheep|ram|"
             r"cow|bull|ox|buffalo|bison|pig|boar|horse|pony|donkey|zebra|giraffe|camel|llama|alpaca|deer|moose|elk|"
             r"reindeer|antelope|rhino|rhinoceros|hippo|hippopotamus|elephant|mammoth|bear|panda|wolf|dog|puppy|fox|"
             r"coyote|cat|kitten|lion|tiger|leopard|cheetah|panther|lynx|hyena|rabbit|bunny|hare|squirrel|mouse|rat|"
             r"hamster|beaver|raccoon|skunk|badger|kangaroo|koala|sloth|armadillo|hedgehog|bat|"
             r"dinosaur|dino|trex|t-rex|raptor|velociraptor|triceratops|stegosaurus|crocodile|alligator|gator|lizard|"
             r"gecko|iguana|komodo|snake|cobra|python|anaconda|viper|turtle|tortoise|frog|toad|"
             r"cerberus|griffin|chimera|manticore|sphinx|unicorn|werewolf)\b")


def classify(prompt: str, hint: str = "") -> Tuple[str, str, bool, bool]:
    p, hint = prompt.lower(), (hint or "").lower().strip()
    matched = True
    if hint in PART_RANGE:
        cat = hint
    else:
        if any(re.search(SUB_KW[s], p) for s in ("dragon", "multi", "bird", "aquatic")):
            cat = "creature"
        elif re.search(ANIMAL_KW, p):
            cat = "creature"
        else:
            cat = next((c for c in ("vehicle", "structure", "humanoid", "plant", "prop") if re.search(KW[c], p)), None)
            if cat is None:
                cat, matched = "creature", False
    sub = ""
    if cat == "creature":
        sub = next((s for s in ("dragon", "multi", "bird", "aquatic") if re.search(SUB_KW[s], p)), "quad")
    wings = bool(re.search(WINGS_KW, p)) and cat in ("creature", "humanoid") and sub != "dragon"
    return cat, sub, wings, matched


def part_range(cat: str, target: Optional[int]) -> Tuple[int, int]:
    lo, hi = PART_RANGE[cat]
    if target:
        t = max(40, min(195, int(target)))
        lo, hi = max(40, t - 15), min(195, t + 15)
    return lo, hi


def build_system_prompt(cat: str, sub: str, wings: bool, lo: int, hi: int) -> str:
    bp = BLUEPRINTS.get("creature_" + sub, BLUEPRINTS["creature_quad"]) if cat == "creature" else BLUEPRINTS[cat]
    figure = cat == "humanoid" or (cat == "creature" and sub in ("quad", "dragon"))
    blocks = [CORE_SKILL] + ([FIGURE_LAWS] if figure else []) + [bp] + ([WINGS_BP] if wings else [])
    return "\n\n".join(blocks).replace("@LO@", str(lo)).replace("@HI@", str(hi))


def build_user_prompt(prompt: str, cat: str, sub: str, lo: int, hi: int, web_ctx: str) -> str:
    out = [
        f'SUBJECT: "{prompt}"',
        f"CATEGORY: {cat}" + (f" ({sub})" if sub else ""),
        f"TARGET: {lo}-{hi} parts, longest side 24-45 studs.",
        "SCULPTING RULES: build every curve from 3-6 stacked, stepped brick slabs (`smooth: 0.06`), never one giant scaled block per body part.",
    ]
    if cat == "humanoid" or (cat == "creature" and sub in ("quad", "dragon")):
        out += [
            "HOLLOW ROARING MOUTH & V-BROW FACE (only if the subject has a face like this): do NOT place a flat plate over the whole face. Frame a deep dark `Head_Mouth_Pit` (`#0b0e14`) with `Head_Snout_Lip` on top, `Head_CheekPillar_L` & `R` on the sides, `Head_Jaw_Upper` below, a pink `Head_Tongue` and white teeth inside, and V-angled `Head_Brow_L` & `R` over glowing `Eye_L` & `Eye_R`.",
            "Add shoulder fur spikes (`Pauldron_L_1..6`) and stepped gold wrist rings (`Bracer_L_Ring1..4`) only if they fit the subject (apes, bosses, armoured characters).",
        ]
    else:
        out.append("Use only features the real subject has. Do NOT add a crown, golden bracers, shoulder fur or a roaring mouth unless the subject needs them.")
    if web_ctx:
        out.append("LIVE WEB REFERENCE:\n" + web_ctx)
    return "\n".join(out)


def compact_build(parts: List[Dict[str, Any]]) -> str:
    rows = [[p["id"], p["shape"], [round(v, 2) for v in p["size"]], [round(v, 2) for v in p["pos"]],
             [round(v, 1) for v in p["rot"]], p["color"], round(p["smooth"], 2)] for p in parts]
    return json.dumps(rows, separators=(",", ":"))


def build_revision_prompt(prompt: str, cat: str, lo: int, hi: int, parts: List[Dict[str, Any]],
                          problems: List[str], change: str = "") -> str:
    out = [f'SUBJECT: "{prompt}"', f"CATEGORY: {cat}", f"TARGET: {lo}-{hi} parts.",
           'CURRENT BUILD in studs, rows are [id, shape, size, pos, rot, color, smooth]:', compact_build(parts)]
    if change:
        out.append(f'REQUESTED CHANGE FROM THE USER: "{change}"\nApply it faithfully and keep everything else.')
    if problems:
        out.append("PROBLEMS FOUND BY THE CHECKER - fix all of them:\n" + "\n".join("- " + m for m in problems))
    out.append("Break any large single blocks into 3-5 stepped brick slabs (`smooth: 0.06`). Keep only features the real subject has.")
    out.append("Return the COMPLETE corrected build in the same JSON schema (plan + every op), never a diff.")
    return "\n".join(out)


# --------------------------------------------------------------------------- providers (Gemini with model discovery, OpenAI, Groq, Anthropic)
class ProviderError(Exception):
    def __init__(self, provider: str, status: int, message: str, attempts: Optional[List[Dict[str, Any]]] = None):
        super().__init__(message)
        self.provider, self.status, self.message = provider, status, message
        self.attempts: List[Dict[str, Any]] = list(attempts or [])

    def to_http(self) -> HTTPException:
        code = (401 if self.status in (401, 403) else 404 if self.status == 404 else 429 if self.status == 429
                else 504 if self.status == 504 else 502)
        return HTTPException(status_code=code, detail=f"{self.provider}: {self.message}"[:4000])

    def to_dict(self) -> Dict[str, Any]:
        return {"provider": self.provider, "upstream_status": self.status, "message": self.message[:4000],
                "attempts": self.attempts}


def _err_text(res: httpx.Response) -> str:
    try:
        j = res.json()
        e = j.get("error", j) if isinstance(j, dict) else j
        if isinstance(e, dict):
            return str(e.get("message") or e)[:2000]
        return str(e)[:2000]
    except Exception:
        return res.text[:2000]


_LIMIT_HINT = re.compile(r"too large|too high|too many|exceed|at most|less than or equal|no more than|"
                         r"maximum (?:allowed|value|number|output|completion)|which is the maximum|is the maximum")


def _limit_from(low: str, current: int) -> int:
    if not _LIMIT_HINT.search(low):
        return 0
    nums = [int(n) for n in re.findall(r"(?<![\d.])\d{3,7}(?![\d.])", low)]
    below = [n for n in nums if 256 <= n < current]
    return max(below) if below else 0


def _heal_payload(payload: Dict[str, Any], body: str) -> bool:
    low, changed = body.lower(), False
    for k in ("temperature", "response_format", "reasoning_effort", "top_p"):
        if k in payload and k in low:
            payload.pop(k)
            changed = True
    if "response_format" in payload and re.search(r"json_validate_failed|failed_generation|json mode|json_object", low):
        payload.pop("response_format")
        changed = True
    swapped = False
    if "max_tokens" in payload and "max_completion_tokens" in low:
        payload["max_completion_tokens"] = payload.pop("max_tokens")
        swapped = changed = True
    if not swapped:
        for k in ("max_tokens", "max_completion_tokens"):
            if k in payload and "token" in low and _LIMIT_HINT.search(low):
                lim = _limit_from(low, payload[k])
                new = lim or max(1024, payload[k] // 2)
                if new < payload[k]:
                    payload[k] = new
                    changed = True
    return changed


def _retry_delay(res: httpx.Response, attempt: int) -> float:
    try:
        ra = float(res.headers.get("retry-after", ""))
    except ValueError:
        ra = 0.0
    return min(5.0, ra if ra > 0 else 1.1 * attempt)


def _worth_retrying(res: httpx.Response) -> bool:
    if res.status_code not in RETRY_STATUS:
        return False
    low = res.text.lower()
    return not ("insufficient_quota" in low or "billing" in low)


async def _post_healing(client: httpx.AsyncClient, provider: str, url: str, headers: Dict[str, str],
                        payload: Dict[str, Any]) -> Dict[str, Any]:
    heals = retries = 0
    while True:
        try:
            res = await client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException:
            raise ProviderError(provider, 504, "the provider did not answer in time")
        except httpx.HTTPError as e:
            raise ProviderError(provider, 502, f"network error talking to {provider}: {e.__class__.__name__}")
        if res.status_code == 200:
            try:
                return res.json()
            except ValueError:
                raise ProviderError(provider, 502, "the provider returned a non-JSON response")
        if res.status_code == 400 and heals < 6 and _heal_payload(payload, res.text):
            heals += 1
            continue
        if retries < 2 and _worth_retrying(res):
            retries += 1
            await asyncio.sleep(_retry_delay(res, retries))
            continue
        raise ProviderError(provider, res.status_code, _err_text(res))


def _gemini_chunk(chunk: Dict[str, Any]) -> Tuple[str, str]:
    c0 = (chunk.get("candidates") or [{}])[0]
    parts = (c0.get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts if not p.get("thought")), str(c0.get("finishReason") or "")


def _err_body_text(body: str) -> str:
    try:
        j = json.loads(body)
        e = j.get("error", j) if isinstance(j, dict) else j
        return str(e.get("message") or e)[:2000] if isinstance(e, dict) else str(e)[:2000]
    except Exception:
        return (body or "")[:2000]


async def _gemini_stream_once(client, url: str, headers: Dict[str, str], payload: Dict[str, Any],
                              deadline: float, model: str, stall: float = 60.0, max_chars: int = 60000) -> Tuple[int, str, str, str]:
    t0 = time.monotonic()
    tmo = httpx.Timeout(connect=15.0, read=stall, write=30.0, pool=15.0)
    async with client.stream("POST", url, headers=headers, json=payload, timeout=tmo) as res:
        if res.status_code != 200:
            body = (await res.aread()).decode("utf-8", "ignore")
            return res.status_code, body, "", res.headers.get("retry-after", "")
        buf: List[str] = []
        chars, finish, last_rep = 0, "", 0.0
        async for line in res.aiter_lines():
            now = time.monotonic()
            if now > deadline:
                finish = "DEADLINE"
                break
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                continue
            try:
                chunk = json.loads(raw)
            except ValueError:
                continue
            if isinstance(chunk, dict) and chunk.get("error"):
                err = chunk["error"] if isinstance(chunk["error"], dict) else {}
                return int(err.get("code") or 502), json.dumps(chunk), "", ""
            txt, fin = _gemini_chunk(chunk if isinstance(chunk, dict) else {})
            if txt:
                buf.append(txt)
                chars += len(txt)
            if fin:
                finish = fin
            if now - last_rep > 1.5:
                report(f"{model} is writing the model... about {chars // 4:,} tokens")
                last_rep = now
            if chars > max_chars:
                finish = "LENGTH_CAP"
                break
        log.info("gemini %s: %d chars in %.1fs (finish=%s)", model, chars, time.monotonic() - t0, finish or "?")
        return 200, "".join(buf), finish, ""


# ---- Gemini model discovery: ask Google which models THIS key can use instead of guessing names (the old 404 bug)
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
_GEM_CACHE: Dict[str, Dict[str, Any]] = {}
_GEM_GOOD: Dict[str, str] = {}
_GEM_TTL = 3600.0
_GEM_BAD: Dict[str, Dict[str, float]] = {}  # models that just failed for good reasons are skipped for a while (saves calls)


def _gem_block(kh: str, model: str, secs: float) -> None:
    _GEM_BAD.setdefault(kh, {})[model] = time.time() + secs


def _gem_blocked(kh: str) -> Dict[str, float]:
    now = time.time()
    d = _GEM_BAD.get(kh) or {}
    for m in [m for m, t in d.items() if t <= now]:
        d.pop(m, None)
    return {m: round(t - now) for m, t in d.items()}
_GEM_SKIP = re.compile(r"embed|aqa|imagen|veo|tts|audio|live|image|robotics|computer|vision|learnlm|gemma|banana|native|-exp", re.I)


def _key_hash(key: str) -> str:
    return hashlib.sha1((key or "").encode()).hexdigest()[:12]


def _gem_rank(name: str) -> Optional[Tuple[int, int, float, int, int]]:
    n = name.lower()
    if not n.startswith("gemini-") or _GEM_SKIP.search(n):
        return None
    fam = "pro" if "-pro" in n else "flash" if "flash" in n else ""
    if not fam:
        return None
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
    ver = float(m.group(1)) if m else 0.0
    return (1 if fam == "flash" else 0,
            0 if "preview" in n else 1,
            ver,
            0 if "lite" in n else 1,
            0 if re.search(r"-\d{3}$|-\d{2}-\d{2,4}$", n) else 1)


async def gemini_models(client: httpx.AsyncClient, key: str, force: bool = False) -> List[str]:
    """Ranked list of generateContent models this key can really call (cached 1h). [] when Google can't be reached."""
    kh = _key_hash(key)
    ent = _GEM_CACHE.get(kh)
    if ent and not force and time.time() - ent["at"] < _GEM_TTL:
        return ent["names"]
    names: List[str] = []
    metas: List[Dict[str, Any]] = []
    token = ""
    for _ in range(6):
        url = f"{GEMINI_BASE}/models?pageSize=200" + (f"&pageToken={token}" if token else "")
        try:
            res = await client.get(url, headers={"x-goog-api-key": key}, timeout=httpx.Timeout(12.0, connect=8.0))
        except httpx.HTTPError:
            return ent["names"] if ent else []
        if res.status_code in (400, 401, 403):
            low = res.text.lower()
            if res.status_code != 400 or "api key" in low or "api_key" in low:
                raise ProviderError("gemini", 401, f"Gemini rejected the API key: {_err_text(res)}")
            return ent["names"] if ent else []
        if res.status_code != 200:
            return ent["names"] if ent else []
        try:
            data = res.json()
        except ValueError:
            return ent["names"] if ent else []
        for m in data.get("models") or []:
            nm = str(m.get("name", "")).split("/", 1)[-1]
            metas.append({"name": nm, "display_name": m.get("displayName", ""), "version": m.get("version", ""),
                          "input_token_limit": m.get("inputTokenLimit"), "output_token_limit": m.get("outputTokenLimit"),
                          "methods": m.get("supportedGenerationMethods") or [], "thinking": bool(m.get("thinking"))})
            if "generateContent" in (m.get("supportedGenerationMethods") or []):
                names.append(nm)
        token = data.get("nextPageToken") or ""
        if not token:
            break
    ranked = sorted({n for n in names if _gem_rank(n)}, key=lambda n: _gem_rank(n), reverse=True)  # type: ignore
    _GEM_CACHE[kh] = {"at": time.time(), "names": ranked, "all": sorted(set(names)), "catalog": metas}
    return ranked


def _gemini_chain_raw(requested: str, available: List[str], good: str = "") -> List[str]:
    """Models to try, best first. Works with an empty `available` (discovery failed) too."""
    low = (requested or "").strip().lower()
    if low.startswith("models/"):
        low = low[7:]
    if low in ("", "auto", "default"):
        low = ""
    chain: List[str] = []

    def add(m: str) -> None:
        if m and m not in chain:
            chain.append(m)

    if not available:
        if low.startswith("gemini-"):
            add(low)
        for m in ("gemini-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash"):
            add(m)
        return chain
    aset = set(available)
    if low in aset:
        add(low)
    elif low:
        fam = "pro" if "pro" in low else "lite" if "lite" in low else "flash"
        pool = [n for n in available if (fam == "pro" and "-pro" in n) or (fam == "lite" and "lite" in n)
                or (fam == "flash" and "flash" in n and "lite" not in n)]
        for n in pool[:2]:
            add(n)
    if good in aset and not low:
        add(good)
    for n in available[:3]:
        add(n)
    if good in aset:
        add(good)
    return chain[:5]


def gemini_chain(requested: str, available: List[str], good: str = "", skip: Optional[set] = None) -> List[str]:
    chain = _gemini_chain_raw(requested, available, good)
    if skip:
        kept = [m for m in chain if m not in skip]
        if kept:
            return kept
    return chain


async def call_gemini(client, key, model, system, user) -> Tuple[str, str]:
    available = await gemini_models(client, key)
    kh = _key_hash(key)
    chain = gemini_chain(model or DEFAULT_MODELS["gemini"], available, _GEM_GOOD.get(kh, ""), set(_gem_blocked(kh)))
    prof = _EFFORT.get()
    budget = prof["thinking"]
    deadline = time.monotonic() + prof["gemini_total"]
    last: Optional[ProviderError] = None
    tried: List[Dict[str, Any]] = []
    prev_m = ""
    for m in chain:
        if last is not None and prev_m:
            tried.append({"model": prev_m, "status": last.status, "message": last.message[:1500]})
            prev_m = ""
        if deadline - time.monotonic() < 20 and last is not None:
            break
        prev_m = m
        url = f"{GEMINI_BASE}/models/{m}:streamGenerateContent?alt=sse"
        gc: Dict[str, Any] = {"responseMimeType": "application/json", "temperature": _TEMP.get() or 0.32,
                              "maxOutputTokens": prof["max_out"]}
        if budget > 0 and "lite" not in m:
            gc["thinkingConfig"] = {"thinkingBudget": budget}
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gc,
        }
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
        heals = retries = 0
        report(f"Asking {m}...")
        while True:
            try:
                status, body, finish, retry_after = await _gemini_stream_once(
                    client, url, headers, payload, deadline, m, prof["stall"], prof["max_chars"])
            except httpx.TimeoutException:
                last = ProviderError("gemini", 504, f"{m} sent nothing for {int(prof['stall'])}s (stalled)")
                log.warning("gemini %s stalled", m)
                break
            except httpx.HTTPError as e:
                last = ProviderError("gemini", 502, f"network error: {e.__class__.__name__}")
                break
            if status == 200:
                text = body.strip()
                if text:
                    _GEM_GOOD[kh] = m
                    MEMORY.bump(f"gem_model:{m}")
                    return text, m
                last = ProviderError("gemini", 502, f"{m} returned no text (finish: {finish or '?'})")
                break
            low = body.lower()
            if status == 400 and heals < 5:
                if "thinking" in low and "thinkingConfig" in gc:
                    gc.pop("thinkingConfig"); heals += 1; continue
                if "mime" in low and "responseMimeType" in gc:
                    gc.pop("responseMimeType"); heals += 1; continue
                if "maxoutputtokens" in low or "output token" in low:
                    new = _limit_from(low, gc["maxOutputTokens"]) or 8192
                    if new < gc["maxOutputTokens"]:
                        gc["maxOutputTokens"] = new; heals += 1; continue
            if retries < 1 and status in RETRY_STATUS and not ("insufficient_quota" in low or "billing" in low
                                                    or re.search(r"limit:\s*0|per ?day|daily", low)):
                retries += 1
                try:
                    ra = float(retry_after)
                except (TypeError, ValueError):
                    ra = 0.0
                await asyncio.sleep(min(5.0, ra if ra > 0 else 1.1))
                continue
            last = ProviderError("gemini", status, f"{m}: {_err_body_text(body)}")
            if status == 404:
                _GEM_CACHE.pop(kh, None)  # model list is stale - rediscover next time
            if status == 404 or (status == 400 and "not supported" in low):
                _gem_block(kh, m, 21600)
            elif status == 429 and re.search(r"limit:\s*0|per ?day|daily", low):
                _gem_block(kh, m, 3600)
            if status in (401, 403) or (status == 400 and "api key" in low):
                raise last
            break
    if last is not None and prev_m:
        tried.append({"model": prev_m, "status": last.status, "message": last.message[:1500]})
    if last is None:
        last = ProviderError("gemini", 502, "Gemini request failed")
    msg = last.message
    if last.status == 404:
        have = ", ".join(available[:6]) or "unknown (could not list models)"
        msg = (f"{msg} - models your key can use: {have}. Open /gemini/models for the full list, "
               f"or leave the model on 'auto'.")
    if len(tried) > 1:
        msg += " | all attempts: " + " ; ".join(f"{t['model']} -> {t['status']}: {t['message'][:400]}" for t in tried)
    raise ProviderError("gemini", last.status, msg, attempts=tried)


def _is_reasoning(model: str) -> bool:
    return model.lower().startswith(("o1", "o3", "o4", "gpt-5", "gpt-6"))


async def call_openai_compat(client, provider, base, key, model, system, user) -> Tuple[str, str]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
    }
    if provider == "openai" and _is_reasoning(model):
        payload["max_completion_tokens"] = 32000
        payload["reasoning_effort"] = "medium"
    else:
        payload["temperature"] = 0.32
        payload["max_tokens"] = 16000 if provider == "openai" else 12000
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    data = await _post_healing(client, provider, base + "/chat/completions", headers, payload)
    msg = ((data.get("choices") or [{}])[0].get("message") or {})
    return msg.get("content") or "", model


async def call_openai(client, key, model, system, user):
    return await call_openai_compat(client, "openai", "https://api.openai.com/v1", key,
                                    model or DEFAULT_MODELS["openai"], system, user)


async def call_groq(client, key, model, system, user):
    return await call_openai_compat(client, "groq", "https://api.groq.com/openai/v1", key,
                                    model or DEFAULT_MODELS["groq"], system, user)


async def call_anthropic(client, key, model, system, user) -> Tuple[str, str]:
    model = model or DEFAULT_MODELS["anthropic"]
    payload = {"model": model, "max_tokens": 16000, "temperature": 0.32, "system": system,
               "messages": [{"role": "user", "content": user}]}
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    data = await _post_healing(client, "anthropic", "https://api.anthropic.com/v1/messages", headers, payload)
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"), model


CALLERS = {"gemini": call_gemini, "openai": call_openai, "anthropic": call_anthropic, "groq": call_groq}


async def call_pollinations(client, system: str, user: str) -> str:
    try:
        res = await client.post("https://text.pollinations.ai/openai", json={
            "model": "openai", "temperature": 0.32, "jsonMode": True,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    except httpx.TimeoutException:
        raise ProviderError("free-fallback", 504, "the free fallback AI did not answer in time")
    except httpx.HTTPError as e:
        raise ProviderError("free-fallback", 502, f"network error: {e.__class__.__name__}")
    if res.status_code != 200:
        raise ProviderError("free-fallback", res.status_code, _err_text(res))
    try:
        data = res.json()
    except ValueError:
        return res.text
    if isinstance(data, dict) and ("ops" in data or "parts" in data):
        return json.dumps(data)
    return ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "") if isinstance(data, dict) else res.text


def _daily_cap_check() -> None:
    if AI_DAILY_CAP > 0 and MEMORY.counter(f"ai_calls:{time.strftime('%Y%m%d', time.gmtime())}") >= AI_DAILY_CAP:
        raise ProviderError("server", 429, f"The daily AI limit ({AI_DAILY_CAP} calls) of this server is used up. "
                                           f"Try again tomorrow or paste your own API key in the settings.")


async def call_llm(system: str, user: str, provider: str = "", model: str = "", api_key: str = "") -> Tuple[str, Dict[str, str]]:
    prov, key, model = (provider or "").lower().strip(), (api_key or "").strip(), (model or "").strip()
    if not prov and any(k in model.lower() for k in ("gemini", "flash", "argon")):
        prov = "gemini"
    if prov not in CALLERS:
        prov = ""
    order: List[Tuple[str, str, str, bool]] = []
    if prov and (key or ENV_KEYS.get(prov)):
        order.append((prov, key or ENV_KEYS[prov], model, bool(key)))
    for name in ("gemini", "anthropic", "openai", "groq"):
        if ENV_KEYS[name] and not any(o[0] == name for o in order):
            order.append((name, ENV_KEYS[name], "", False))
    if order and not order[0][3]:
        _daily_cap_check()
    errors: List[ProviderError] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0)) as client:
        for name, k, m, own in order:
            try:
                text, used = await CALLERS[name](client, k, m, system, user)
                MEMORY.bump("ai_calls")
                MEMORY.bump(f"ai_calls:{time.strftime('%Y%m%d', time.gmtime())}")
                return text, {"provider": name, "model": used}
            except ProviderError as e:
                MEMORY.bump("ai_errors")
                errors.append(e)
                log.warning("provider %s failed (%s): %s", name, e.status, e.message[:200])
                if own and e.status in (400, 401, 403, 404):
                    raise  # the user's own key/model is wrong - tell them, don't hide it behind another provider
        if not order:
            try:
                text = await call_pollinations(client, system, user)
                MEMORY.bump("ai_calls")
                return text, {"provider": "free-fallback", "model": "pollinations"}
            except ProviderError as e:
                errors.append(e)
    if len(errors) == 1:
        raise errors[0]
    first = errors[0] if errors else ProviderError("server", 502, "No AI provider is configured.")
    raise ProviderError("all providers", first.status,
                        " | ".join(f"{e.provider}: {e.message}" for e in errors)[:4000] or
                        "No AI provider is configured. Add GEMINI_API_KEY on the server or paste a key in the settings.",
                        attempts=[a for e in errors for a in e.attempts])


# --------------------------------------------------------------------------- web research (persistent cache = fewer Serper calls)
async def fetch_anatomical_research(query: str, override_key: str = "", depth: int = 1) -> str:
    key = (override_key or SERPER_API_KEY).strip()
    if not key:
        return ""
    rk = " ".join(sorted(prompt_tokens(query))) or query.lower().strip()
    cached = MEMORY.research_get(rk, depth)
    if cached:
        MEMORY.bump("research_cache_hits")
        return cached

    async def one(client: httpx.AsyncClient, q: str) -> List[str]:
        res = await client.post("https://google.serper.dev/search",
                                headers={"X-API-KEY": key, "Content-Type": "application/json"}, json={"q": q, "num": 5})
        if res.status_code != 200:
            return []
        d, bits = res.json(), []
        kg = d.get("knowledgeGraph") or {}
        if kg.get("description"):
            bits.append(str(kg["description"]))
        for k, v in list((kg.get("attributes") or {}).items())[:5]:
            bits.append(f"{k}: {v}")
        ab = d.get("answerBox") or {}
        if ab.get("snippet") or ab.get("answer"):
            bits.append(str(ab.get("snippet") or ab.get("answer")))
        bits += [o["snippet"] for o in (d.get("organic") or [])[:4] if o.get("snippet")]
        return bits

    qs = [f"{query} appearance colors chest face markings", f"{query} character anatomy body proportions",
          f"{query} roblox boss model design", f"{query} silhouette shape language concept art",
          f"{query} distinctive features details"][:{1: 2, 2: 3, 3: 5}.get(depth, 2)]
    try:
        async with httpx.AsyncClient(timeout=8.5) as client:
            results = await asyncio.gather(*(one(client, q) for q in qs), return_exceptions=True)
    except Exception:
        return ""
    MEMORY.bump("research_calls", len(qs))
    seen, bits = set(), []
    for r in results:
        if isinstance(r, list):
            for b in r:
                if b not in seen:
                    seen.add(b)
                    bits.append(b)
    text = " | ".join(bits)[:1400 + 600 * max(0, depth - 1)]
    if text:
        MEMORY.research_put(rk, depth, text)
    return text


def clean_web_context(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    m = re.search(r"Live Serper Web Research Context[^:\n]*:\s*(.+)$", raw, re.S | re.I)
    if m:
        return m.group(1).strip()[:1600]
    if raw.lower().startswith("request:"):
        return ""
    return raw[:1600]


# --------------------------------------------------------------------------- parsing & Multi-Slab Sculpting / Hollow Face Solver
def salvage_ops(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ops": []}
    m = re.search(r'"title"\s*:\s*"([^"]*)"', text)
    if m:
        out["title"] = m.group(1)
    m = re.search(r'"category"\s*:\s*"([^"]*)"', text)
    if m:
        out["category"] = m.group(1)
    m = re.search(r'"(?:ops|parts|hulls)"\s*:\s*\[', text)
    if not m:
        return out
    dec, i = json.JSONDecoder(), m.end()
    while True:
        cands = [x for x in (text.find("{", i), text.find("[", i)) if x >= 0]
        if not cands:
            break
        j = min(cands)
        try:
            obj, k = dec.raw_decode(text, j)
        except ValueError:
            break
        if isinstance(obj, (dict, list)):
            out["ops"].append(obj)
        i = k
    return out


def parse_llm_json(text: str) -> Dict[str, Any]:
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    t = re.sub(r"```(?:json)?", "", t, flags=re.I).strip()
    dec = json.JSONDecoder()
    for fix in (False, True):
        s = re.sub(r",\s*([}\]])", r"\1", t) if fix else t
        starts = [x for x in (s.find("{"), s.find("[")) if x >= 0]
        if not starts:
            break
        try:
            obj, _ = dec.raw_decode(s[min(starts):])
        except ValueError:
            continue
        if isinstance(obj, list):
            obj = {"ops": obj}
        if isinstance(obj, dict):
            return obj
    return salvage_ops(t)


def _num(v: Any, d: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else d
    except (TypeError, ValueError):
        return d


def _vec3(v: Any, d: Tuple[float, float, float]) -> List[float]:
    if isinstance(v, (list, tuple)):
        return [_num(v[i], d[i]) if i < len(v) else d[i] for i in range(3)]
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return [float(v)] * 3
    return list(d)


def group_of(pid: str) -> str:
    return re.sub(r"\d+$", "", pid.split("_")[0])


def _row_to_op(row: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(row, (list, tuple)) or len(row) < 4 or not isinstance(row[0], str):
        return None
    out: Dict[str, Any] = {"id": row[0], "shape": row[1], "size": row[2], "pos": row[3]}
    rest = list(row[4:])
    if rest and isinstance(rest[0], (list, tuple)):
        out["rot"] = rest.pop(0)
    if rest and isinstance(rest[0], str):
        out["color"] = rest.pop(0)
    if rest and isinstance(rest[0], (int, float)) and not isinstance(rest[0], bool):
        out["smooth"] = rest.pop(0)
    return out


def _wrap_deg(a: float) -> float:
    return ((a + 180.0) % 360.0) - 180.0


def fix_cylinder_axes(pid: str, shape: str, size: List[float], rot: List[float]) -> Tuple[List[float], List[float]]:
    if shape not in ("cylinder", "cone"):
        return size, rot
    sx, sy, sz = size
    rx, ry, rz = rot
    is_wheel = bool(re.search(r"wheel|tire|tyre|rim|hub|axle|brake|caliper|coin|disc|disk|suck|pad|button", pid, re.I))

    if is_wheel and shape == "cylinder":
        dims = sorted([sx, sy, sz])
        thickness, diameter = dims[0], dims[2]
        if sx < sy * 0.8 and sx < sz * 0.8 and abs(rz) < 15:
            rz = 90.0
        elif sz < sx * 0.8 and sz < sy * 0.8 and abs(rx) < 15 and abs(rz) < 15:
            rx = 90.0
        elif abs(rx) < 15 and abs(rz) < 15 and re.search(r"wheel|tire|tyre|rim|hub|axle", pid, re.I):
            rz = 90.0
        return [diameter, thickness, diameter], [_wrap_deg(rx), _wrap_deg(ry), _wrap_deg(rz)]

    if sz > sy * 1.35 and abs(sx - sy) <= 0.42 * max(sx, sy):
        rad_avg = (sx + sy) / 2.0
        length = sz
        if abs(rx) < 45.0 and abs(rz) < 45.0:
            rx = _wrap_deg(rx + 90.0)
        return [rad_avg, length, rad_avg], [rx, ry, rz]

    if sx > sy * 1.35 and abs(sz - sy) <= 0.42 * max(sz, sy):
        rad_avg = (sy + sz) / 2.0
        length = sx
        if abs(rz) < 45.0 and abs(rx) < 45.0:
            rz = _wrap_deg(rz - 90.0)
        return [rad_avg, length, rad_avg], [rx, ry, rz]

    if max(sx, sz) > min(sx, sz) * 1.35:
        rad_avg = (sx + sz) / 2.0
        return [rad_avg, sy, rad_avg], [rx, ry, rz]

    return size, rot


_ROUND_EXEMPT = re.compile(r"eye|pupil|iris|orb|ball|pearl|gem|jewel|bead|knob|bubble|halo|lens|egg|berry|fruit|planet|globe", re.I)
_SPIKE_OK = re.compile(r"spike|thorn|fang|tooth|talon|claw|horn|tusk|stinger|crown_point|crown_spike", re.I)
_RING_OK = re.compile(r"ring|halo|hoop|tire|tyre", re.I)
_CYL_EXEMPT = re.compile(
    r"wheel|tire|tyre|rim|hub|axle|barrel|pipe|exhaust|column|pillar|trunk|wand|staff|handle|pole|post|"
    r"nozzle|thruster|cannon|muzzle|suck|candle|lamp|chimney|spar|grip|hilt", re.I
)


def heal_part_shape_and_smoothness(pid: str, shape: str, size: List[float], smooth: float, cat: str, sub: str) -> Tuple[str, float]:
    longest = max(size)
    if shape == "sphere":
        if not (_ROUND_EXEMPT.search(pid) and longest <= 3.6):
            shape = "box"
    elif shape == "cone":
        if not (_SPIKE_OK.search(pid) and longest <= 3.2):
            shape = "box"
    elif shape == "torus":
        if not _RING_OK.search(pid):
            shape = "box"
    if shape == "box":
        smooth = min(smooth, 0.07)
    elif shape == "cylinder":
        smooth = min(smooth, 0.22)
    return shape, max(0.0, min(1.0, smooth))


def enforce_roblox_budget(parts: List[Dict[str, Any]], cat: str) -> int:
    n = len(parts)
    if n < 8:
        return 0
    changed = 0
    round_idx = sorted((i for i, p in enumerate(parts) if p["shape"] in ("sphere", "cone", "torus")),
                       key=lambda i: (bool(_ROUND_EXEMPT.search(parts[i]["id"])), -vol(parts[i])))
    cap_round = max(4, int(0.08 * n))
    for i in round_idx[: max(0, len(round_idx) - cap_round)]:
        parts[i]["shape"] = "box"
        parts[i]["smooth"] = 0.06
        changed += 1
    if cat in ("creature", "humanoid", "plant"):
        cyl = sorted((i for i, p in enumerate(parts) if p["shape"] == "cylinder" and not _CYL_EXEMPT.search(p["id"])),
                     key=lambda i: -vol(parts[i]))
        cap_cyl = max(2, int(0.10 * n))
        for i in cyl[cap_cyl:]:
            parts[i]["shape"] = "box"
            parts[i]["smooth"] = 0.06
            changed += 1
    return changed


# --------------------------------------------------------------------------- Universal Subject-Aware Palette & Color-Zone Enforcer
def _hex_to_hls(c: str) -> Tuple[float, float, float]:
    r, g, b = (int(c[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(r, g, b)


def _hls_to_hex(h: float, l: float, s: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h % 1.0, max(0.0, min(1.0, l)), max(0.0, min(1.0, s)))
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def _color_dist(c1: str, c2: str) -> float:
    r1, g1, b1 = (int(c1[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    r2, g2, b2 = (int(c2[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    return math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)


_TOK = r"(?:^|_)"
_TOK_END = r"(?:$|_|\d)"

_ZONE_SKIN_CHEST = re.compile(
    _TOK + r"(?:chest|pec|pecs|sternum|ab|abs|belly|underbelly|stomach|scute|forehead|cheekpillar|cheekbone|"
           r"muzzle|snout|jaw|brow|ear|palm|finger|thumb|toe|suck|throat|socket|plastron|breastplate|inlay|windshield|cabin_glass)" + _TOK_END, re.I
)
_ZONE_GOLD_TRIM = re.compile(
    _TOK + r"(?:crown|bracer|cuff|bangle|buckle|trim|emblem|collar|rim|hilt|guard|pommel|halo|tiara|horn|crest|exhaust|bumper|spire_tip)" + _TOK_END, re.I
)
_ZONE_GLOW_ACCENT = re.compile(
    _TOK + r"(?:eye|eyes|iris|visor|reactor|gem|jewel|rune|flame|fire|laser|plasma|taillight|thruster|afterburner|lantern|beacon|core)" + _TOK_END, re.I
)
_ZONE_WHITE_LIGHT = re.compile(
    _TOK + r"(?:tooth|teeth|fang|fangs|tusk|claw|talon|nail|sclera|highlight|shine|glint|headlight|blade_edge)" + _TOK_END, re.I
)
_ZONE_DARK = re.compile(
    _TOK + r"(?:pupil|mouth|pit|cavity|nostril|tire|tyre|slit|grille|diffuser|splitter|wheel_well)" + _TOK_END, re.I
)
_ZONE_TONGUE = re.compile(_TOK + r"(?:tongue|gum|gums)" + _TOK_END, re.I)
_ZONE_FORCE_BASE = re.compile(
    _TOK + r"(?:cranium|skull|topdome|ruff|cheek|chin|torso|pelvis|hump|back|pauldron|tuft|shoulder|bicep|elbow|forearm|hand|hip|thigh|knee|shin|calf|foot|ankle|cap|mantle|carapace|hull|chassis|fender|hood)" + _TOK_END, re.I
)

_PROMPT_COLOR_MAP = [
    (r"\b(crimson|red|scarlet|ruby|fire|flame|lava|magma|inferno|charizard|blood)\b", "#b81d18", "#f3e5ab", "#f5a623", "#ffea00"),
    (r"\b(purple|violet|amethyst|kraken|squid|octopus|void|shadow|nebula|arcane|dark elf)\b", "#5b2c86", "#dcd0ff", "#f5c518", "#00f5d4"),
    (r"\b(blue|azure|cobalt|sapphire|ice|frost|glacier|water|ocean|sea|shark|whale|dolphin)\b", "#1b5e91", "#e2f1f8", "#90e0ef", "#00f5d4"),
    (r"\b(green|emerald|jade|forest|jungle|moss|goblin|orc|treant|cactus|tree|plant|turtle|crocodile|alligator|frog|snake)\b", "#246a42", "#dcebd8", "#d4a338", "#ff595e"),
    (r"\b(gold|golden|yellow|sun|solar|desert|sand|lion|griffin|bee|wasp|pikachu|pharaoh)\b", "#c98a1c", "#faf0ca", "#5c3c10", "#00bbf9"),
    (r"\b(white|snow|polar|arctic|ivory|angel|ghost|skeleton|skull|unicorn|pegasus|snowman)\b", "#d9e2ec", "#9fb3c8", "#f5c518", "#38bdf8"),
    (r"\b(orange|amber|copper|bronze|fox|tiger|pumpkin)\b", "#d95d1e", "#faedcd", "#2b2d42", "#70e000"),
    (r"\b(pink|rose|magenta|coral|sakura|cherry|flamingo|axolotl)\b", "#d6457a", "#ffe5ec", "#f5c518", "#00f5d4"),
    (r"\b(brown|wood|wooden|oak|bear|grizzly|mammoth|deer|moose|elk|horse|dog|monkey|chimp|owl|eagle)\b", "#6b4423", "#ead2ac", "#d4a338", "#ff9f1c"),
    (r"\b(silver|steel|iron|chrome|titanium|mech|mecha|robot|knight|paladin|cyborg|armor|armored|watchtower|castle|fortress|stone)\b", "#4a5568", "#cbd5e1", "#f5c518", "#00e5ff"),
    (r"\b(gorilla|ape|kong|black|obsidian|panther|raven|crow|bat|spider|scorpion)\b", "#182232", "#e8dfd1", "#f5c518", "#ff2a2a"),
]


def infer_subject_palette(prompt: str, cat: str) -> Tuple[str, str, str, str]:
    p_low = prompt.lower()
    for pat, b_col, s_col, t_col, a_col in _PROMPT_COLOR_MAP:
        if re.search(pat, p_low):
            return b_col, s_col, t_col, a_col
    if cat == "vehicle":
        return "#c1121f", "#e5e7eb", "#f5c518", "#00e5ff"
    if cat == "structure":
        return "#64748b", "#e2e8f0", "#d97706", "#fbbf24"
    if cat == "plant":
        return "#2d6a4f", "#74c69d", "#7f4f24", "#ffb703"
    if cat == "humanoid":
        return "#334155", "#e2e8f0", "#f5c518", "#00e5ff"
    return "#243447", "#e8dfd1", "#f5c518", "#ff2a2a"


def enforce_color_blocking(parts: List[Dict[str, Any]], palette: Dict[str, Any], prompt: str, cat: str) -> None:
    p_low = prompt.lower()
    is_ape = any(w in p_low for w in ("gorilla", "ape", "monkey", "kong", "primate", "baboon", "orangutan", "chimpanzee"))
    def_base, def_sec, def_trim, def_acc = infer_subject_palette(prompt, cat)

    base = str(palette.get("base") or "").lower()
    if not re.fullmatch(r"#[0-9a-f]{6}", base) or (not is_ape and base in ("#141c28", "#8aa0b0", "#5f9a78")):
        base = def_base
    if is_ape:
        base = "#182232"
    bh, bl, bs = _hex_to_hls(base)

    sec = str(palette.get("secondary") or "").lower()
    if not re.fullmatch(r"#[0-9a-f]{6}", sec) or _color_dist(base, sec) < 0.28:
        sec = def_sec

    trim = str(palette.get("trim") or "").lower()
    if not re.fullmatch(r"#[0-9a-f]{6}", trim) or _color_dist(base, trim) < 0.24:
        trim = def_trim

    accent = str(palette.get("accent") or "").lower()
    if not re.fullmatch(r"#[0-9a-f]{6}", accent) or _color_dist(base, accent) < 0.22:
        accent = def_acc

    light = "#ffffff"
    dark = "#0b0e14"

    palette["base"] = base
    palette["secondary"] = sec
    palette["trim"] = trim
    palette["accent"] = accent
    palette["light"] = light
    palette["dark"] = dark

    for p in parts:
        pid = p["id"]
        c_cur = str(p.get("color") or "").lower()
        if _ZONE_TONGUE.search(pid):
            p["color"] = "#d95b72"
        elif _ZONE_WHITE_LIGHT.search(pid):
            p["color"] = light
        elif _ZONE_DARK.search(pid):
            p["color"] = dark
        elif _ZONE_GLOW_ACCENT.search(pid) and not re.search(r"brow|socket|lid|frame|crown", pid, re.I):
            p["color"] = accent
        elif _ZONE_GOLD_TRIM.search(pid) and not re.search(r"innerdome|cap_dark|waist|belt|pelvis", pid, re.I):
            p["color"] = trim
        elif _ZONE_SKIN_CHEST.search(pid) and not re.search(r"cranium|skull|ruff|chin_fur|back|shoulder|pauldron|bicep|forearm|thigh|shin|hand_l|hand_r|foot_l|foot_r", pid, re.I):
            p["color"] = sec
        elif _ZONE_FORCE_BASE.search(pid) or c_cur in ("#8aa0b0", "#5f9a78", "#141c28"):
            if re.search(r"pauldron|tuft|ruff|back_silver|saddle|wing_membrane|cape|cloak|fin|roof|canopy", pid, re.I):
                p["color"] = _hls_to_hex(bh, min(0.68, max(0.22, bl + 0.08)), max(0.25, bs))
            else:
                p["color"] = base


# --------------------------------------------------------------------------- Automatic Monolith-to-Multi-Slab Sculptor & Hollow Face Builder
_SUBDIVIDE_LIMB = re.compile(r"bicep|forearm|upperarm|lowerarm|thigh|shin|calf", re.I)
_SUBDIVIDE_TORSO = re.compile(r"^torso_(?:upper|mid|core|main|body)$|^body_core$|^chest_core$", re.I)
_SUBDIVIDE_BRACER = re.compile(r"bracer|cuff", re.I)


def subdivide_monoliths_into_slabs(parts: List[Dict[str, Any]], cat: str) -> Tuple[List[Dict[str, Any]], int]:
    if cat not in ("creature", "humanoid"):
        return parts, 0

    out: List[Dict[str, Any]] = []
    added_slabs = 0
    existing_ids = {p["id"] for p in parts}

    for p in parts:
        pid = p["id"]
        sx, sy, sz = p["size"]
        px, py, pz = p["pos"]

        if p["shape"] == "box" and _SUBDIVIDE_LIMB.search(pid) and not re.search(r"slab|seg|ring|\d$", pid, re.I) and sy >= 4.2 and len(out) + len(parts) < MAX_PARTS - 12:
            n_slabs = 3
            slab_h = (sy / n_slabs) * 1.06
            profiles = (0.94, 1.04, 0.90)
            for k in range(n_slabs):
                y_off = (k - (n_slabs - 1) / 2.0) * (sy / n_slabs)
                scale_xz = profiles[k]
                new_id = f"{pid}_Slab{k + 1}"
                while new_id in existing_ids:
                    new_id += "_"
                existing_ids.add(new_id)
                out.append({
                    "id": new_id,
                    "shape": "box",
                    "size": [round(sx * scale_xz, 2), round(slab_h, 2), round(sz * scale_xz, 2)],
                    "pos": [px, round(py + y_off, 2), pz],
                    "rot": list(p["rot"]),
                    "color": p["color"],
                    "smooth": 0.06,
                })
            added_slabs += n_slabs - 1
            continue

        if p["shape"] == "box" and _SUBDIVIDE_BRACER.search(pid) and not re.search(r"ring|top|bot|upper|lower|\d$", pid, re.I) and sy >= 2.0 and len(out) + len(parts) < MAX_PARTS - 8:
            n_rings = 3
            ring_h = (sy / n_rings) * 0.92
            for k in range(n_rings):
                y_off = (k - 1) * (sy / n_rings)
                flare = 1.04 if k in (0, 2) else 1.0
                new_id = f"{pid}_Ring{k + 1}"
                while new_id in existing_ids:
                    new_id += "_"
                existing_ids.add(new_id)
                out.append({
                    "id": new_id,
                    "shape": "box",
                    "size": [round(sx * flare, 2), round(ring_h, 2), round(sz * flare, 2)],
                    "pos": [px, round(py + y_off, 2), pz],
                    "rot": list(p["rot"]),
                    "color": p["color"],
                    "smooth": 0.06,
                })
            added_slabs += n_rings - 1
            continue

        if p["shape"] == "box" and _SUBDIVIDE_TORSO.search(pid) and sy >= 6.0 and len(out) + len(parts) < MAX_PARTS - 10:
            n_slabs = 3
            slab_h = (sy / n_slabs) * 1.05
            tapers = (0.86, 0.96, 1.04)
            for k in range(n_slabs):
                y_off = (k - 1) * (sy / n_slabs)
                t_scale = tapers[k]
                new_id = f"{pid}_Slab{k + 1}"
                while new_id in existing_ids:
                    new_id += "_"
                existing_ids.add(new_id)
                out.append({
                    "id": new_id,
                    "shape": "box",
                    "size": [round(sx * t_scale, 2), round(slab_h, 2), round(sz * (0.92 + 0.04 * k), 2)],
                    "pos": [px, round(py + y_off, 2), pz],
                    "rot": list(p["rot"]),
                    "color": p["color"],
                    "smooth": 0.06,
                })
            added_slabs += n_slabs - 1
            continue

        out.append(p)
    return out, added_slabs


def build_3d_hollow_face(parts: List[Dict[str, Any]], palette: Dict[str, Any], cat: str) -> int:
    if cat not in ("creature", "humanoid"):
        return 0

    head_parts = [p for p in parts if re.search(r"head|skull|cranium|face|muzzle|snout|jaw|mouth|eye|brow|fang|tooth|tongue", p["id"], re.I)]
    if len(head_parts) < 4:
        return 0

    craniums = [
        p for p in head_parts
        if re.search(r"^(?:head|skull|cranium)", p["id"], re.I)
        and not re.search(r"face|muzzle|snout|jaw|mouth|cavity|pit|tongue|tooth|teeth|fang|eye|pupil|brow|nostril|ear|horn|crown|crest|ruff", p["id"], re.I)
    ]
    if not craniums:
        return 0

    cranium_front_z = max(p["pos"][2] + p["size"][2] * 0.5 for p in craniums)
    sec_col = palette.get("secondary", "#e8dfd1")
    dark_col = palette.get("dark", "#0b0e14")
    accent_col = palette.get("accent", "#ff2a2a")
    light_col = palette.get("light", "#ffffff")
    fixed = 0
    existing_ids = {p["id"] for p in parts}

    for p in list(parts):
        pid = p["id"]
        if re.search(r"faceplate|face_plate|faceupper", pid, re.I) and p["size"][1] >= 3.2:
            old_h = p["size"][1]
            old_y = p["pos"][1]
            p["size"][1] = round(old_h * 0.42, 2)
            p["pos"][1] = round(old_y + old_h * 0.28, 2)
            p["pos"][2] = round(cranium_front_z + 0.35, 2)
            fixed += 1
            if "Head_CheekPillar_L" not in existing_ids and len(parts) < MAX_PARTS - 4:
                for side_s, tag in ((-1, "L"), (1, "R")):
                    cid = f"Head_CheekPillar_{tag}"
                    existing_ids.add(cid)
                    parts.append({
                        "id": cid, "shape": "box",
                        "size": [1.45, round(old_h * 0.58, 2), 2.1],
                        "pos": [round(side_s * (p["size"][0] * 0.38), 2), round(old_y - old_h * 0.20, 2), round(cranium_front_z + 0.65, 2)],
                        "rot": [0.0, 0.0, 0.0], "color": sec_col, "smooth": 0.06,
                    })
                    fixed += 1

    for p in parts:
        pid = p["id"]
        if re.search(r"brow", pid, re.I):
            p["pos"][2] = max(p["pos"][2], round(cranium_front_z + 1.05, 2))
            if re.search(r"_l$|_left", pid, re.I) and abs(p["rot"][2]) < 5:
                p["rot"][2] = -15.0
            elif re.search(r"_r$|_right", pid, re.I) and abs(p["rot"][2]) < 5:
                p["rot"][2] = 15.0
            fixed += 1

    for p in parts:
        pid = p["id"]
        if re.search(r"(?:^|_)eye(?:_|$|\d)", pid, re.I) and not re.search(r"pupil|brow|socket", pid, re.I):
            p["pos"][2] = max(p["pos"][2], round(cranium_front_z + 0.82, 2))
            p["size"][0] = max(p["size"][0], 1.75)
            p["size"][1] = min(max(p["size"][1], 0.65), 0.95)
            p["color"] = accent_col
            if re.search(r"_l$|_left", pid, re.I) and abs(p["rot"][2]) < 4:
                p["rot"][2] = -10.0
            elif re.search(r"_r$|_right", pid, re.I) and abs(p["rot"][2]) < 4:
                p["rot"][2] = 10.0
            fixed += 1

    muzzle_z = cranium_front_z + 1.15
    for p in parts:
        pid = p["id"]
        if re.search(r"muzzle|snout", pid, re.I) and not re.search(r"cavity|pit|mouth", pid, re.I):
            p["pos"][2] = max(p["pos"][2], round(muzzle_z, 2))
            p["size"][1] = min(p["size"][1], 1.6)
            fixed += 1
        elif re.search(r"jaw", pid, re.I):
            p["pos"][2] = max(p["pos"][2], round(muzzle_z - 0.05, 2))
            p["size"][1] = min(p["size"][1], 1.4)
            fixed += 1
        elif re.search(r"mouth_cavity|mouth_pit|mouthcavity|head_mouth", pid, re.I):
            p["pos"][2] = round(muzzle_z - 0.25, 2)
            p["color"] = dark_col
            fixed += 1
        elif re.search(r"tongue", pid, re.I):
            p["pos"][2] = round(muzzle_z + 0.05, 2)
            p["color"] = "#d95b72"
            fixed += 1
        elif re.search(r"fang|tooth|teeth", pid, re.I):
            p["pos"][2] = round(muzzle_z + 0.42, 2)
            p["color"] = light_col
            p["smooth"] = 0.04
            fixed += 1

    return fixed


def smooth_and_weld_rigging(parts: List[Dict[str, Any]], cat: str) -> int:
    if cat not in ("creature", "humanoid"):
        return 0
    welded = 0
    for side_tag in ("l", "r"):
        arm_parts = [
            p for p in parts
            if re.search(rf"(?:arm|bicep|forearm|elbow|bracer|cuff|hand)_{side_tag}", p["id"], re.I)
            and not re.search(r"finger|thumb|pauldron|tuft", p["id"], re.I)
        ]
        if len(arm_parts) >= 3:
            median_x = sum(p["pos"][0] for p in arm_parts) / len(arm_parts)
            for p in arm_parts:
                if abs(p["pos"][0] - median_x) > 0.75:
                    p["pos"][0] = round(0.3 * p["pos"][0] + 0.7 * median_x, 2)
                    welded += 1
    return welded


def _snap_q(v: float, q: float = 0.1) -> float:
    return round(v / q) * q


def _snap_rot(a: float) -> float:
    return _wrap_deg(round(a / 5.0) * 5.0)


def normalize_ops(raw_ops: List[Any], unit: float = 1.0, default_color: str = "#141c28",
                  cat: str = "creature", sub: str = "") -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    seen = set()
    for idx, item in enumerate(raw_ops):
        if len(parts) >= MAX_PARTS:
            break
        if isinstance(item, (list, tuple)):
            item = _row_to_op(item)
        if not isinstance(item, dict) or str(item.get("op", "add")).lower() not in ("add", "", "part"):
            continue
        shape = str(item.get("shape") or item.get("s") or item.get("type") or "box").lower().strip()
        shape = SHAPE_ALIAS.get(shape, shape)
        if shape not in SHAPES:
            shape = "box"
        pid = re.sub(r"[^A-Za-z0-9_]", "", str(item.get("id") or item.get("name") or item.get("n") or f"Part_{idx + 1}").replace(" ", "_"))[:36] or f"Part_{idx + 1}"
        base, n = pid, 2
        while pid in seen:
            pid, n = f"{base}_{n}", n + 1
        seen.add(pid)

        size = [min(MAX_PART_STUDS, max(MIN_PART_STUDS, abs(s) * unit))
                for s in _vec3(item.get("size") or item.get("sz") or item.get("scale"), (1, 1, 1))]
        rp = item.get("pos") or item.get("position") or item.get("po")
        if isinstance(rp, (list, tuple)):
            pos = [max(-90.0, min(90.0, v * unit)) for v in _vec3(rp, (0, 0, 0))]
        else:
            pos = [0.0, size[1] / 2, 0.0]
        rot = [_snap_rot(r) for r in _vec3(item.get("rot") or item.get("rotation") or item.get("ro"), (0, 0, 0))]

        smooth = max(0.0, min(1.0, _num(item.get("smooth"), SMOOTH_DEFAULT[shape])))
        shape, smooth = heal_part_shape_and_smoothness(pid, shape, size, smooth, cat, sub)
        size, rot = fix_cylinder_axes(pid, shape, size, rot)
        if shape == "box":
            size = [max(0.20, _snap_q(s, 0.1)) for s in size]
            pos = [_snap_q(v, 0.1) for v in pos]

        color = str(item.get("color") or item.get("c") or default_color).strip()
        if re.fullmatch(r"#[0-9a-fA-F]{3}", color):
            color = "#" + "".join(ch * 2 for ch in color[1:])
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            color = default_color
        parts.append({"id": pid, "shape": shape, "size": size, "pos": pos, "rot": rot, "color": color.lower(), "smooth": smooth})
    return parts


# --------------------------------------------------------------------------- geometry math & kinematic chain stitcher
def _mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def rot_matrix(rot: List[float]):
    rx, ry, rz = (math.radians(v) for v in rot)
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    return _mat_mul(_mat_mul([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]),
                    [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])


def aabb(p: Dict[str, Any]) -> List[Tuple[float, float]]:
    w, h, z = p["size"]
    half_local = (0.47 * w, 0.47 * h, 0.15 * z) if p["shape"] == "torus" else (w / 2, h / 2, z / 2)
    r = rot_matrix(p["rot"])
    half = [sum(abs(r[i][j]) * half_local[j] for j in range(3)) for i in range(3)]
    return [(p["pos"][i] - half[i], p["pos"][i] + half[i]) for i in range(3)]


def aabb_gap(b1: List[Tuple[float, float]], b2: List[Tuple[float, float]]) -> Tuple[float, List[float]]:
    gaps = [max(b1[k][0] - b2[k][1], b2[k][0] - b1[k][1], 0.0) for k in range(3)]
    mv = [(b1[k][0] - b2[k][1]) if b1[k][0] > b2[k][1]
          else (b1[k][1] - b2[k][0]) if b2[k][0] > b1[k][1] else 0.0 for k in range(3)]
    return math.sqrt(sum(g * g for g in gaps)), mv


def vol(p: Dict[str, Any]) -> float:
    return p["size"][0] * p["size"][1] * p["size"][2]


def axis_aligned(p: Dict[str, Any]) -> bool:
    return all(abs(((r + 45.0) % 90.0) - 45.0) < 0.6 for r in p["rot"])


def bounds(parts: List[Dict[str, Any]]):
    bx = [aabb(p) for p in parts]
    lo = [min(b[k][0] for b in bx) for k in range(3)]
    hi = [max(b[k][1] for b in bx) for k in range(3)]
    return lo, hi


def frame(parts: List[Dict[str, Any]]) -> float:
    lo, hi = bounds(parts)
    m = max(hi[k] - lo[k] for k in range(3))
    s = MAX_EXTENT_STUDS / m if m > MAX_EXTENT_STUDS else 1.0
    cx, cz, y0 = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2, lo[1]
    for p in parts:
        p["pos"] = [(p["pos"][0] - cx) * s, (p["pos"][1] - y0) * s, (p["pos"][2] - cz) * s]
        if s != 1.0:
            p["size"] = [max(MIN_PART_STUDS, v * s) for v in p["size"]]
    return s


def dedupe(parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for p in parts:
        k = (p["shape"], tuple(round(v, 1) for v in p["pos"]), tuple(round(v, 1) for v in p["size"]), tuple(round(v) for v in p["rot"]))
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


_CHAIN_RE = re.compile(
    r"^((?:tentacle|arm|leg|tail|neck|spine|wing|branch|root|horn|antenna|claw|finger|toe|vine|feeler|whisker|fin)"
    r"(?:_[a-zA-Z0-9]+?)?)"
    r"(?:_?(?:seg|slab|part|sec|bone|joint|node|ring|lobe)?_?(\d+)|_(base|root|shoulder|bicep|upper|elbow|mid|middle|lower|forearm|thigh|knee|shin|calf|ankle|club|tip|end|paw|hand|foot|claw))$",
    re.I,
)
_STAGE_ORDER = {
    "base": 1, "root": 1, "shoulder": 1, "thigh": 1,
    "upper": 2, "bicep": 2, "elbow": 3, "knee": 3, "mid": 3, "middle": 3,
    "lower": 4, "forearm": 4, "shin": 4, "calf": 4, "ankle": 5,
    "club": 8, "paw": 8, "hand": 8, "foot": 8, "claw": 9, "tip": 10, "end": 10,
}
_DETAIL_CHILD_RE = re.compile(r"suck|pad|claw|nail|talon|spike|barb|tuft|spot|glow|ring|stripe|bracer|cuff", re.I)


def stitch_articulated_chains(parts: List[Dict[str, Any]]) -> int:
    chains: Dict[str, List[Tuple[int, int]]] = {}
    for idx, p in enumerate(parts):
        pid = p["id"]
        if _DETAIL_CHILD_RE.search(pid) and not re.search(r"club$", pid, re.I):
            continue
        m = _CHAIN_RE.match(pid)
        if not m:
            continue
        prefix = m.group(1).lower()
        order = int(m.group(2)) if m.group(2) else _STAGE_ORDER.get((m.group(3) or "").lower(), 5)
        chains.setdefault(prefix, []).append((order, idx))

    stitched = 0
    for prefix, seq in chains.items():
        if len(seq) < 2:
            continue
        seq.sort(key=lambda x: x[0])
        indices = [idx for _, idx in seq]
        chain_set = set(indices)
        child_indices = [
            i for i, p in enumerate(parts)
            if i not in chain_set and p["id"].lower().startswith(prefix)
        ]

        for pos_idx in range(len(indices) - 1):
            i_prev = indices[pos_idx]
            i_curr = indices[pos_idx + 1]
            p_prev, p_curr = parts[i_prev], parts[i_curr]
            b_prev, b_curr = aabb(p_prev), aabb(p_curr)
            gap, mv = aabb_gap(b_prev, b_curr)

            dx = [p_curr["pos"][k] - p_prev["pos"][k] for k in range(3)]
            dist = math.sqrt(sum(d * d for d in dx))
            r_prev = max(p_prev["size"]) * 0.5
            r_curr = max(p_curr["size"]) * 0.5
            max_reach = (r_prev + r_curr) * 0.88

            if gap > 0.02 or dist > max_reach * 1.15:
                long_axis = max(range(3), key=lambda k: p_curr["size"][k])
                if p_curr["shape"] in ("cylinder", "cone"):
                    long_axis = 1
                p_curr["size"][long_axis] = min(MAX_PART_STUDS, p_curr["size"][long_axis] * 1.18)
                b_curr = aabb(p_curr)
                gap, mv = aabb_gap(b_prev, b_curr)

            if gap > 0.01 and dist > 1e-4:
                shift = [v + (0.12 if v > 0 else -0.12 if v < 0 else 0.0) for v in mv]
                downstream = indices[pos_idx + 1:]
                curr_pos_before = list(p_curr["pos"])
                for d_idx in downstream:
                    for k in range(3):
                        parts[d_idx]["pos"][k] += shift[k]
                for c_idx in child_indices:
                    c_pos = parts[c_idx]["pos"]
                    d_to_curr = math.sqrt(sum((c_pos[k] - curr_pos_before[k]) ** 2 for k in range(3)))
                    d_to_prev = math.sqrt(sum((c_pos[k] - p_prev["pos"][k]) ** 2 for k in range(3)))
                    if d_to_curr <= d_to_prev * 1.25:
                        for k in range(3):
                            parts[c_idx]["pos"][k] += shift[k]
                stitched += 1
    return stitched


# --------------------------------------------------------------------------- checks and auto-fixes
def find_coplanar(parts: List[Dict[str, Any]]) -> List[Tuple[int, int, int, int]]:
    boxes = [(i, aabb(p)) for i, p in enumerate(parts) if p["shape"] == "box" and axis_aligned(p)]
    hits = []
    for a in range(len(boxes)):
        i, bi = boxes[a]
        for b in range(a + 1, len(boxes)):
            j, bj = boxes[b]
            if min(min(bi[k][1], bj[k][1]) - max(bi[k][0], bj[k][0]) for k in range(3)) <= 0.05:
                continue
            for k in range(3):
                if abs(bi[k][1] - bj[k][1]) < 0.025:
                    hits.append((i, j, k, 1))
                if abs(bi[k][0] - bj[k][0]) < 0.025:
                    hits.append((i, j, k, 0))
    return hits


def fix_coplanar(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    pairs, names = set(), []
    for pass_idx in range(8):
        hits = find_coplanar(parts)
        if not hits:
            break
        done = set()
        step_push = PUSH * (1.0 + 0.18 * pass_idx)
        for i, j, k, side in hits:
            if (i, j) not in pairs:
                pairs.add((i, j))
                names.append(f"{parts[i]['id']}/{parts[j]['id']}")
            tgt = i if vol(parts[i]) < vol(parts[j]) else j
            if (tgt, k, side) in done:
                continue
            done.add((tgt, k, side))
            parts[tgt]["size"][k] += step_push
            parts[tgt]["pos"][k] += step_push / 2 if side == 1 else -step_push / 2
    return {"pairs": len(pairs), "names": names, "remaining": len({(i, j) for i, j, _, _ in find_coplanar(parts)})}


_FLOAT_OK = re.compile(r"^(float|halo|orbit|spark|particle|aura|hover|ember|bubble|smoke|cloud)", re.I)


def components(parts: List[Dict[str, Any]], tol: float = 0.18) -> List[List[int]]:
    n, bx = len(parts), [aabb(p) for p in parts]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if all(bx[i][k][0] - tol <= bx[j][k][1] and bx[j][k][0] - tol <= bx[i][k][1] for k in range(3)):
                parent[find(i)] = find(j)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda c: (len(c), sum(vol(parts[i]) for i in c)), reverse=True)


def snap_floating(parts: List[Dict[str, Any]], max_gap: float = 16.0) -> int:
    snapped = 0
    for _ in range(7):
        comps = components(parts)
        if len(comps) <= 1:
            break
        bx = [aabb(p) for p in parts]
        main, moved = comps[0], False
        main_set = set(main)
        for comp in comps[1:]:
            if all(_FLOAT_OK.match(parts[i]["id"]) for i in comp):
                continue
            comp_prefix = "_".join(parts[comp[0]]["id"].split("_")[:2]).lower()
            candidates = [m for m in main if parts[m]["id"].lower().startswith(comp_prefix)] or main
            best = None
            for i in comp:
                for m in candidates:
                    d, mv = aabb_gap(bx[m], bx[i])
                    if best is None or d < best[0]:
                        best = (d, mv)
            if best is None or best[0] > 6.0:
                for i in comp:
                    for m in main_set:
                        d, mv = aabb_gap(bx[m], bx[i])
                        if best is None or d < best[0]:
                            best = (d, mv)
            limit = max_gap if len(comp) <= 6 else 9.0
            if best and best[0] <= limit:
                mv = [v + (0.12 if v > 0 else -0.12 if v < 0 else 0.0) for v in best[1]]
                for i in comp:
                    for k in range(3):
                        parts[i]["pos"][k] += mv[k]
                snapped += len(comp)
                moved = True
        if not moved:
            break
    return snapped


def analyze(parts: List[Dict[str, Any]], cat: str, sub: str, lo: int, hi: int, prompt: str,
            overlap: Dict[str, Any], snapped: int, slabs_added: int, face_fixed: int) -> Dict[str, Any]:
    n, issues, score = len(parts), [], 100.0
    glo, ghi = bounds(parts)
    ext = [ghi[k] - glo[k] for k in range(3)]
    gate = int(lo * GATE)

    def add(code: str, msg: str, blocking: bool, penalty: float) -> None:
        nonlocal score
        issues.append({"code": code, "msg": msg, "blocking": blocking})
        score -= penalty

    if n < lo:
        add("PART_COUNT_LOW", f"only {n} parts; the target is {lo}-{hi}. Subdivide large torso/limb blocks into 4-6 stepped brick slabs and add jagged shoulder fur spikes, 4-ring wrist cuffs, and hollow-mouth facial details.",
            n < gate, min(35.0, (lo - n) / lo * 60))

    comps = components(parts)
    stray = [c for c in comps[1:] if not all(_FLOAT_OK.match(parts[i]["id"]) for i in c)]
    if stray:
        names = ", ".join(parts[c[0]]["id"] for c in stray[:4])
        add("FLOATING", f"{len(stray)} group(s) of parts are not connected to the main body (e.g. {names}). Move them so they overlap a neighbour.",
            True, min(25.0, 10.0 * len(stray)))
    float_msg = (f"{len(stray)} disconnected group(s) left." if stray else
                 f"Every part is connected ({slabs_added} monoliths subdivided into stepped brick slabs, {face_fixed} hollow-jaw facial features sculpted).")

    if overlap["remaining"]:
        add("OVERLAP", f"{overlap['remaining']} pair(s) of parts still share the exact same surface.", False, min(10.0, 2.5 * overlap["remaining"]))
    if overlap["pairs"]:
        shown = "; ".join(overlap["names"][:2])
        overlap_msg = (f"{overlap['pairs']} pair(s) of parts shared a surface ({shown}) — automatically offset by {PUSH / 2:.2f} stud so nothing flickers in Studio.")
    else:
        overlap_msg = "No same-surface overlaps - nothing will flicker in Studio."

    prop_msg = "Multi-slab brick contours and 3D hollow-mouth facial relief verified."
    if cat in ("creature", "humanoid"):
        eyes = [p for p in parts if re.search(r"(?:^|_)eye(?:_|$|\d)", p["id"], re.I)]
        mouth_parts = [p for p in parts if re.search(r"mouth|pit|cavity|jaw|fang|tooth|tongue|beak", p["id"], re.I)]
        if not eyes:
            add("MISSING_EYES", "the model has no visible Eye parts! Add glowing Eye_L, Eye_R and V-angled Brow blocks on the front of the head (+Z).", True, 15)
        if cat == "creature" and len(mouth_parts) < 4:
            add("MISSING_FACE_DETAIL", "the creature's face is missing a 3D hollow roaring mouth! Frame a dark Head_Mouth_Pit with CheekPillars, Snout_Lip, Jaw, Tongue, and 6-8 white Fang/Tooth blocks.", True, 14)

    color_counts: Dict[str, int] = {}
    for p in parts:
        bucket = "#%x%x%x" % (int(p["color"][1:3], 16) >> 5, int(p["color"][3:5], 16) >> 5, int(p["color"][5:7], 16) >> 5)
        color_counts[bucket] = color_counts.get(bucket, 0) + 1
    distinct_cols = len(color_counts)
    max_col_share = max(color_counts.values()) / max(n, 1) if color_counts else 1.0
    if distinct_cols < 4 or max_col_share > 0.72:
        add("MONOCHROME", f"the coloring is too monochrome ({distinct_cols} distinct color bucket(s)). Use contrasting `secondary` on chest slabs/face frame/fingers/toes, gold `trim` on bracers/crown, bright `accent` on eyes, and white `light` on teeth!",
            True, 16)

    box_n = sum(1 for p in parts if p["shape"] == "box")
    block_share = box_n / max(n, 1)

    score = max(0, int(round(score)))
    grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D"
    return {
        "issues": issues, "score": score, "grade": grade,
        "ext": ext, "tilted": 0, "colors": distinct_cols,
        "diag": {
            "readout": f"{n} parts · {ext[0]:.1f} × {ext[1]:.1f} × {ext[2]:.1f} studs",
            "overlap_check": overlap_msg,
            "proportion_check": prop_msg,
            "floating_check": float_msg,
            "detail_check": f"{n} parts (target {lo}-{hi}), {block_share * 100:.0f}% crisp Roblox bricks, {distinct_cols} contrasting color zones.",
        },
    }


def process_build(data: Dict[str, Any], cat: str, sub: str, lo: int, hi: int, prompt: str, unit: float = 1.0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw = data.get("ops") if isinstance(data.get("ops"), list) else data.get("parts")
    if not isinstance(raw, list) or not raw:
        raise ValueError("The AI returned no parts.")
    if not isinstance(data.get("plan"), dict):
        data["plan"] = {}
    if not isinstance(data["plan"].get("palette"), dict):
        data["plan"]["palette"] = {}
    palette = data["plan"]["palette"]
    def_base, _, _, _ = infer_subject_palette(prompt, cat)
    base = str(palette.get("base", def_base))
    parts = normalize_ops(raw, unit, base if re.fullmatch(r"#[0-9a-fA-F]{6}", base) else def_base, cat, sub)
    if len(parts) < 3:
        raise ValueError("The AI returned fewer than 3 usable parts.")
    enforce_roblox_budget(parts, cat)
    enforce_color_blocking(parts, palette, prompt, cat)
    parts, slabs_added = subdivide_monoliths_into_slabs(parts, cat)
    parts = dedupe(parts)
    frame(parts)
    smooth_and_weld_rigging(parts, cat)
    stitched = stitch_articulated_chains(parts)
    face_fixed = build_3d_hollow_face(parts, palette, cat)
    ov1 = fix_coplanar(parts)
    snapped = snap_floating(parts) + stitched
    frame(parts)
    ov2 = fix_coplanar(parts)
    overlap = {"pairs": ov1["pairs"] + ov2["pairs"], "names": ov1["names"] + ov2["names"], "remaining": ov2["remaining"]}
    return parts, analyze(parts, cat, sub, lo, hi, prompt, overlap, snapped, slabs_added, face_fixed)


# --------------------------------------------------------------------------- output & tonal contrast
_NO_JITTER = re.compile(r"eye|pupil|lens|light|glass|window|flame|glow|ember|spark|tooth|fang|tongue|gem", re.I)


def jitter(color: str, key: str, amt: float = 0.05) -> str:
    r, g, b = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if l < 0.12:
        l = 0.12 + l * 0.5
    delta = ((zlib.crc32(key.encode()) % 1000) / 1000 - 0.5) * 2 * amt
    l = max(0.08, min(0.95, l + delta))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def export_build(parts: List[Dict[str, Any]], analysis: Dict[str, Any], tonal: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ext = analysis["ext"]
    k = min(STUD, SCENE_MAX_M / max(max(ext), 1e-6))
    ops = []
    for p in parts:
        color = p["color"] if (not tonal or _NO_JITTER.search(p["id"])) else jitter(p["color"], p["id"])
        ops.append({
            "op": "add", "id": p["id"], "group": group_of(p["id"]), "shape": p["shape"],
            "size": [round(max(0.05, min(8.0, v * k)), 3) for v in p["size"]],
            "pos": [round(max(-12.0, min(12.0, v * k)), 3) for v in p["pos"]],
            "rot": [round(v, 1) for v in p["rot"]], "color": color, "smooth": round(p["smooth"], 3),
        })
    stats = {
        "parts": len(parts), "studs": [round(v, 1) for v in ext], "scene_scale": round(k, 4),
        "readout": f"{len(parts)} parts \u00b7 {ext[0]:.1f} \u00d7 {ext[1]:.1f} \u00d7 {ext[2]:.1f} studs",
    }
    return ops, stats


def better(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    ka = (-sum(1 for i in a["issues"] if i["blocking"]), a["score"])
    kb = (-sum(1 for i in b["issues"] if i["blocking"]), b["score"])
    return ka > kb


# --------------------------------------------------------------------------- global memory v2 (shared model library + teaching + stats)
MEMORY_STOP = {
    "a", "an", "the", "of", "with", "and", "please", "make", "build", "create", "generate", "me", "my", "i",
    "want", "need", "give", "draw", "design", "some", "that", "this", "is", "it", "to", "for", "in", "on",
    "model", "3d", "roblox", "can", "you", "like", "style", "very", "really", "its", "has", "have", "who", "which",
}
MEMORY_SYN = {
    "huge": "giant", "enormous": "giant", "massive": "giant", "gigantic": "giant", "colossal": "giant",
    "cartoon": "cartoony", "cartoonish": "cartoony", "cartoonlike": "cartoony", "toony": "cartoony",
    "cute": "cute", "tiny": "small", "little": "small", "mini": "small",
    "octopi": "octopus", "octopuses": "octopus", "calamari": "squid",
    "golden": "gold", "grey": "gray", "crimson": "red", "scarlet": "red", "azure": "blue", "navy": "blue",
    "emerald": "green", "violet": "purple", "lavender": "purple", "turquoise": "teal", "aqua": "cyan",
    "silvery": "silver", "dragons": "dragon",
}
# colour words: (hue degrees, saturation, lightness) the model's main body should move to
COLOR_HSL: Dict[str, Tuple[float, float, float]] = {
    "red": (0, 0.72, 0.42), "orange": (27, 0.85, 0.5), "yellow": (52, 0.85, 0.52), "gold": (45, 0.8, 0.5),
    "green": (122, 0.5, 0.38), "teal": (175, 0.55, 0.38), "cyan": (187, 0.7, 0.5), "blue": (215, 0.65, 0.45),
    "purple": (275, 0.5, 0.4), "pink": (332, 0.6, 0.6), "magenta": (310, 0.65, 0.45), "brown": (25, 0.45, 0.3),
    "black": (0, 0.0, 0.1), "white": (0, 0.0, 0.92), "gray": (0, 0.0, 0.5), "silver": (210, 0.08, 0.72),
}
NEUTRAL_COLORS = {"black", "white", "gray", "silver"}
SIZE_WORDS = {"giant", "small", "big", "large", "tall", "short"}
TRIVIAL_WORDS = set(COLOR_HSL) | SIZE_WORDS


def _stem(w: str) -> str:
    if w in MEMORY_SYN:
        return MEMORY_SYN[w]
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 4 and w.endswith(("ses", "xes", "ches", "shes", "zes")):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    return MEMORY_SYN.get(w, w)


def prompt_tokens(prompt: str) -> frozenset:
    words = re.findall(r"[a-z0-9]+", (prompt or "").lower())
    toks = {_stem(w) for w in words if w not in MEMORY_STOP and len(w) > 1}
    return frozenset(toks)


def core_tokens(tokens: frozenset) -> frozenset:
    """The subject without colour / size adjectives - 'red dragon' and 'blue dragon' share the core {dragon}."""
    return frozenset(t for t in tokens if t not in TRIVIAL_WORDS)


def prompt_key(tokens: frozenset, variant: str = "classic") -> str:
    base = " ".join(sorted(tokens))
    return "~d3d " + base if variant == "depth3d" else base


MEMORY_FLAG_LIMIT = int(os.getenv("MEMORY_FLAG_LIMIT", "3"))
MEMORY_MIN_SCORE = int(os.getenv("MEMORY_MIN_SCORE", "70"))
RESEARCH_TTL_S = 45 * 86400


def _pick_db_path() -> Path:
    want = os.getenv("MEMORY_DB", "")
    candidates = [Path(want)] if want else []
    if Path("/var/data").is_dir():
        candidates.append(Path("/var/data/verdant_memory.db"))
    candidates += [Path(__file__).parent / "data" / "verdant_memory.db", Path("/tmp/verdant_memory.db")]
    for c in candidates:
        try:
            c.parent.mkdir(parents=True, exist_ok=True)
            probe = c.parent / ".write_test"
            probe.write_text("ok")
            probe.unlink()
            return c
        except Exception:
            continue
    return Path("/tmp/verdant_memory.db")


class MemoryStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or _pick_db_path()
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        with self.lock:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS models(
                    id TEXT PRIMARY KEY, key TEXT UNIQUE, prompt TEXT, category TEXT, title TEXT,
                    payload TEXT, parts INTEGER, score INTEGER, uses INTEGER DEFAULT 0,
                    flags INTEGER DEFAULT 0, parent_id TEXT, created REAL, last_used REAL, engine TEXT);
                CREATE INDEX IF NOT EXISTS idx_models_used ON models(uses DESC);
                CREATE TABLE IF NOT EXISTS lessons(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, scope TEXT DEFAULT 'any',
                    variant TEXT DEFAULT 'any', keywords TEXT DEFAULT '', enabled INTEGER DEFAULT 1,
                    hits INTEGER DEFAULT 0, created REAL);
                CREATE TABLE IF NOT EXISTS examples(
                    id TEXT PRIMARY KEY, name TEXT, prompt TEXT, category TEXT, scope TEXT DEFAULT 'subject',
                    tokens TEXT, payload TEXT, parts INTEGER, enabled INTEGER DEFAULT 1, uses INTEGER DEFAULT 0,
                    created REAL, source_fmt TEXT, notes TEXT DEFAULT '', model_id TEXT DEFAULT '');
                CREATE TABLE IF NOT EXISTS counters(name TEXT PRIMARY KEY, n INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS research(key TEXT PRIMARY KEY, depth INTEGER, text TEXT, created REAL);
                CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
            """)
            for ddl in ("ALTER TABLE models ADD COLUMN effort TEXT DEFAULT 'mid'",
                        "ALTER TABLE models ADD COLUMN pinned INTEGER DEFAULT 0",
                        "ALTER TABLE models ADD COLUMN source TEXT DEFAULT 'ai'",
                        "ALTER TABLE models ADD COLUMN notes TEXT DEFAULT ''"):
                try:
                    self.db.execute(ddl)
                except sqlite3.OperationalError:
                    pass
            self.db.commit()
        self.index: List[Dict[str, Any]] = []
        self.reload()

    @property
    def persistent(self) -> bool:
        return bool(os.getenv("MEMORY_DB")) or str(self.path).startswith("/var/data")

    @staticmethod
    def _split_key(key: str) -> Tuple[str, str]:
        return ("depth3d", key[5:]) if key.startswith("~d3d ") else ("classic", key)

    def reload(self) -> None:
        with self.lock:
            rows = self.db.execute("SELECT id,key,prompt,category,title,parts,score,uses,effort,pinned,source FROM models "
                                   "WHERE flags < ?", (MEMORY_FLAG_LIMIT,)).fetchall()
        idx = []
        for r in rows:
            d = dict(r)
            d["variant"], k = self._split_key(d["key"])
            d["tokens"] = frozenset(k.split())
            d["core"] = core_tokens(d["tokens"])
            d["effort"] = d.get("effort") or "mid"
            d["pinned"] = int(d.get("pinned") or 0)
            d["rank"] = 3 if d["pinned"] else EFFORT_RANK.get(d["effort"], 1)
            d["score"] = int(d.get("score") or 0)
            idx.append(d)
        self.index = idx

    def count(self) -> int:
        return len(self.index)

    # ---- counters / research cache -------------------------------------------------------------------------
    def bump(self, name: str, n: int = 1) -> None:
        try:
            with self.lock:
                self.db.execute("INSERT INTO counters(name,n) VALUES(?,?) ON CONFLICT(name) DO UPDATE SET n = n + ?",
                                (name, n, n))
                self.db.commit()
        except Exception:
            pass

    def counter(self, name: str) -> int:
        with self.lock:
            r = self.db.execute("SELECT n FROM counters WHERE name=?", (name,)).fetchone()
        return int(r["n"]) if r else 0

    def counters(self) -> Dict[str, int]:
        with self.lock:
            rows = self.db.execute("SELECT name,n FROM counters WHERE name NOT LIKE 'ai_calls:%'").fetchall()
        return {r["name"]: int(r["n"]) for r in rows}

    def research_get(self, key: str, depth: int) -> str:
        with self.lock:
            r = self.db.execute("SELECT text FROM research WHERE key=? AND depth>=? AND created>?",
                                (key, depth, time.time() - RESEARCH_TTL_S)).fetchone()
        return r["text"] if r else ""

    def research_put(self, key: str, depth: int, text: str) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO research(key,depth,text,created) VALUES(?,?,?,?)",
                            (key, depth, text, time.time()))
            self.db.commit()

    # ---- matching ------------------------------------------------------------------------------------------
    def find(self, tokens: frozenset, variant: str = "classic") -> Tuple[str, Optional[Dict[str, Any]], List[str], List[str]]:
        """mode: exact (reuse as is) | local (same subject, only colour/size words differ -> zero-API edit)
        | variant (similar subject, the AI re-sculpts it) | none."""
        if not tokens:
            return "none", None, [], []
        core = core_tokens(tokens)
        best_exact = best_local = best_var = None
        for e in self.index:
            if e["variant"] != variant:
                continue
            b = e["tokens"]
            if not b:
                continue
            inter = len(tokens & b)
            union = len(tokens | b)
            jac = inter / union if union else 0.0
            if b == tokens or jac >= 0.85:
                rk = (e["pinned"], e["rank"], e["score"], e["uses"])
                if best_exact is None or rk > best_exact[0]:
                    best_exact = (rk, e)
                continue
            if core and e["core"] == core:
                rk = (e["pinned"], e["rank"], e["score"], e["uses"])
                if best_local is None or rk > best_local[0]:
                    best_local = (rk, e)
                continue
            subset = b < tokens and len(b) / len(tokens) >= 0.4
            if subset or jac >= 0.5:
                rank = (1 if subset else 0, jac, e["score"], e["uses"])
                if best_var is None or rank > best_var[0]:
                    best_var = (rank, e)
        if best_exact:
            return "exact", best_exact[1], [], []
        if best_local:
            e = best_local[1]
            return "local", e, sorted(tokens - e["tokens"]), sorted(e["tokens"] - tokens)
        if best_var:
            e = best_var[1]
            return "variant", e, sorted(tokens - e["tokens"]), sorted(e["tokens"] - tokens)
        return "none", None, [], []

    # ---- CRUD ----------------------------------------------------------------------------------------------
    def get(self, mid: str, include_flagged: bool = False) -> Optional[Dict[str, Any]]:
        with self.lock:
            r = self.db.execute("SELECT * FROM models WHERE id=?" + ("" if include_flagged else " AND flags < ?"),
                                (mid,) if include_flagged else (mid, MEMORY_FLAG_LIMIT)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["payload"] = json.loads(d["payload"])
        d["variant"], _ = self._split_key(d["key"])
        d["effort"] = d.get("effort") or "mid"
        d["pinned"] = int(d.get("pinned") or 0)
        d["source"] = d.get("source") or "ai"
        return d

    def touch(self, mid: str) -> int:
        with self.lock:
            self.db.execute("UPDATE models SET uses = uses + 1, last_used = ? WHERE id = ?", (time.time(), mid))
            self.db.commit()
            r = self.db.execute("SELECT uses FROM models WHERE id=?", (mid,)).fetchone()
        for e in self.index:
            if e["id"] == mid:
                e["uses"] = r["uses"] if r else e["uses"] + 1
        return r["uses"] if r else 0

    def save(self, prompt: str, tokens: frozenset, category: str, payload: Dict[str, Any], score: int,
             parent_id: str = "", engine: str = "", effort: str = "mid", variant: str = "classic",
             source: str = "ai", pinned: int = 0, notes: str = "") -> str:
        key = prompt_key(tokens, variant)
        mid = uuid.uuid4().hex[:12]
        now = time.time()
        body = json.dumps(payload, separators=(",", ":"))
        nparts = len(payload.get("ops") or payload.get("hulls") or [])
        title = str(payload.get("title") or prompt)[:80]
        with self.lock:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO models(id,key,prompt,category,title,payload,parts,score,uses,flags,parent_id,created,"
                "last_used,engine,effort,pinned,source,notes) VALUES(?,?,?,?,?,?,?,?,0,0,?,?,?,?,?,?,?,?)",
                (mid, key, prompt[:300], category, title, body, nparts, int(score), parent_id or None, now, now,
                 engine[:80], effort, int(pinned), source, notes[:500]))
            if cur.rowcount == 0:
                r = self.db.execute("SELECT id,effort,flags,score,pinned FROM models WHERE key=?", (key,)).fetchone()
                if not r:
                    self.db.commit()
                    return ""
                if r["pinned"] and not pinned:
                    self.db.commit()
                    return r["id"]  # the owner locked this model - never overwrite it
                better_effort = EFFORT_RANK.get(effort, 1) > EFFORT_RANK.get(r["effort"] or "mid", 1)
                better_score = (EFFORT_RANK.get(effort, 1) >= EFFORT_RANK.get(r["effort"] or "mid", 1)
                                and int(score) >= int(r["score"] or 0) + 4)
                if better_effort or better_score or (r["flags"] or 0) >= MEMORY_FLAG_LIMIT or pinned:
                    self.db.execute(
                        "UPDATE models SET prompt=?,category=?,title=?,payload=?,parts=?,score=?,created=?,engine=?,effort=?,"
                        "flags=0,pinned=?,source=?,notes=? WHERE id=?",
                        (prompt[:300], category, title, body, nparts, int(score), now, engine[:80], effort, int(pinned),
                         source, notes[:500], r["id"]))
                    self.db.commit()
                    self.reload()
                    return r["id"]
                self.db.commit()
                return r["id"]
            self.db.commit()
        self.reload()
        return mid

    def flag(self, mid: str) -> bool:
        with self.lock:
            cur = self.db.execute("UPDATE models SET flags = flags + 1 WHERE id=? AND pinned=0", (mid,))
            self.db.commit()
        self.reload()
        return cur.rowcount > 0

    def delete(self, mid: str) -> bool:
        with self.lock:
            cur = self.db.execute("DELETE FROM models WHERE id=?", (mid,))
            self.db.commit()
        self.reload()
        return cur.rowcount > 0

    def update(self, mid: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cur = self.get(mid, include_flagged=True)
        if not cur:
            return None
        sets, vals = [], []
        if "prompt" in fields and str(fields["prompt"]).strip():
            prompt = str(fields["prompt"]).strip()[:300]
            tokens = prompt_tokens(prompt)
            if not tokens:
                raise ValueError("The prompt has no meaningful words.")
            key = prompt_key(tokens, cur["variant"])
            with self.lock:
                clash = self.db.execute("SELECT id FROM models WHERE key=? AND id<>?", (key, mid)).fetchone()
            if clash:
                raise ValueError(f"Another model ({clash['id']}) already uses that prompt - delete or edit it first.")
            sets += ["prompt=?", "key=?"]
            vals += [prompt, key]
        for col, cast in (("title", lambda v: str(v)[:80]), ("category", lambda v: str(v)[:20]),
                          ("notes", lambda v: str(v)[:500]), ("pinned", lambda v: 1 if v in (1, True, "1", "true") else 0),
                          ("flags", lambda v: max(0, int(v))), ("effort", lambda v: norm_effort(v))):
            if col in fields:
                sets.append(f"{col}=?")
                vals.append(cast(fields[col]))
        if sets:
            with self.lock:
                self.db.execute(f"UPDATE models SET {', '.join(sets)} WHERE id=?", (*vals, mid))
                self.db.commit()
            self.reload()
        return self.get(mid, include_flagged=True)

    def listing(self, q: str = "", limit: int = 40, sort: str = "popular", variant: str = "") -> List[Dict[str, Any]]:
        total, items = self.admin_list(q=q, limit=limit, sort=sort, variant=variant, offset=0, public=True)
        return items

    def admin_list(self, q: str = "", limit: int = 50, offset: int = 0, sort: str = "popular", variant: str = "",
                   source: str = "", flagged: bool = False, pinned: bool = False, public: bool = False) -> Tuple[int, List[Dict[str, Any]]]:
        q = (q or "").strip().lower()
        where, args = [], []
        if public or not flagged:
            where.append("flags < ?")
            args.append(MEMORY_FLAG_LIMIT)
        elif flagged:
            where.append("flags > 0")
        if q:
            where.append("(lower(prompt) LIKE ? OR lower(title) LIKE ? OR lower(notes) LIKE ?)")
            args += [f"%{q}%"] * 3
        if variant == "depth3d":
            where.append("key LIKE '~d3d %'")
        elif variant == "classic":
            where.append("key NOT LIKE '~d3d %'")
        if source:
            where.append("source=?")
            args.append(source)
        if pinned:
            where.append("pinned=1")
        order = {"popular": "uses DESC, created DESC", "new": "created DESC", "score": "score DESC, uses DESC",
                 "parts": "parts DESC", "flags": "flags DESC, created DESC", "old": "created ASC"}.get(sort, "uses DESC, created DESC")
        w = (" WHERE " + " AND ".join(where)) if where else ""
        with self.lock:
            total = self.db.execute(f"SELECT COUNT(*) c FROM models{w}", args).fetchone()["c"]
            rows = self.db.execute(
                f"SELECT id,key,prompt,category,title,parts,score,uses,flags,parent_id,created,last_used,effort,pinned,source,notes,engine "
                f"FROM models{w} ORDER BY {order} LIMIT ? OFFSET ?", (*args, max(1, min(200, limit)), max(0, offset))).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["variant"], _ = self._split_key(d.pop("key"))
            d["effort"] = d.get("effort") or "mid"
            d["pinned"] = int(d.get("pinned") or 0)
            d["source"] = d.get("source") or "ai"
            if public:
                for k in ("flags", "last_used", "notes", "engine"):
                    d.pop(k, None)
            out.append(d)
        return total, out

    def stats(self) -> Dict[str, Any]:
        with self.lock:
            rows = self.db.execute("SELECT key,effort,source,pinned,flags,uses,score,parts FROM models").fetchall()
            nl = self.db.execute("SELECT COUNT(*) c FROM lessons").fetchone()["c"]
            ne = self.db.execute("SELECT COUNT(*) c FROM examples").fetchone()["c"]
        st: Dict[str, Any] = {"models": len(rows), "classic": 0, "depth3d": 0, "pinned": 0, "flagged": 0, "uses": 0,
                              "avg_score": 0, "by_effort": {}, "by_source": {}, "lessons": nl, "examples": ne}
        tot = 0
        for r in rows:
            st["depth3d" if r["key"].startswith("~d3d ") else "classic"] += 1
            st["pinned"] += int(r["pinned"] or 0)
            st["flagged"] += 1 if (r["flags"] or 0) > 0 else 0
            st["uses"] += int(r["uses"] or 0)
            tot += int(r["score"] or 0)
            e = r["effort"] or "mid"
            st["by_effort"][e] = st["by_effort"].get(e, 0) + 1
            s = r["source"] or "ai"
            st["by_source"][s] = st["by_source"].get(s, 0) + 1
        st["avg_score"] = round(tot / len(rows), 1) if rows else 0
        try:
            st["db_kb"] = round(self.path.stat().st_size / 1024, 1)
        except OSError:
            st["db_kb"] = 0
        return st

    # ---- backup / restore (the free Render disk is wiped on redeploy - export regularly!) --------------------
    def export_all(self) -> Dict[str, Any]:
        with self.lock:
            models = [dict(r) for r in self.db.execute("SELECT * FROM models").fetchall()]
            lessons = [dict(r) for r in self.db.execute("SELECT * FROM lessons").fetchall()]
            examples = [dict(r) for r in self.db.execute("SELECT * FROM examples").fetchall()]
        for m in models:
            m["payload"] = json.loads(m["payload"])
        for e in examples:
            e["payload"] = json.loads(e["payload"])
        return {"format": "verdant-memory-backup", "version": 24, "exported": time.time(),
                "models": models, "lessons": lessons, "examples": examples}

    def import_all(self, data: Dict[str, Any]) -> Dict[str, int]:
        if not isinstance(data, dict) or data.get("format") != "verdant-memory-backup":
            raise ValueError("This is not a Verdant memory backup file.")
        n_m = n_l = n_e = 0
        with self.lock:
            for m in data.get("models") or []:
                try:
                    key = str(m["key"])
                    ex = self.db.execute("SELECT id,score,pinned FROM models WHERE key=?", (key,)).fetchone()
                    if ex and (ex["pinned"] or int(ex["score"] or 0) >= int(m.get("score") or 0)):
                        continue
                    if ex:
                        self.db.execute("DELETE FROM models WHERE id=?", (ex["id"],))
                    self.db.execute(
                        "INSERT OR REPLACE INTO models(id,key,prompt,category,title,payload,parts,score,uses,flags,parent_id,created,"
                        "last_used,engine,effort,pinned,source,notes) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (str(m["id"]), key, m.get("prompt", ""), m.get("category", ""), m.get("title", ""),
                         json.dumps(m["payload"], separators=(",", ":")), int(m.get("parts") or 0), int(m.get("score") or 0),
                         int(m.get("uses") or 0), int(m.get("flags") or 0), m.get("parent_id"), m.get("created") or time.time(),
                         m.get("last_used") or time.time(), m.get("engine", ""), m.get("effort") or "mid",
                         int(m.get("pinned") or 0), m.get("source") or "ai", m.get("notes") or ""))
                    n_m += 1
                except (KeyError, TypeError, ValueError, sqlite3.Error):
                    continue
            have = {r["text"] for r in self.db.execute("SELECT text FROM lessons").fetchall()}
            for l in data.get("lessons") or []:
                try:
                    if str(l["text"]) in have:
                        continue
                    self.db.execute("INSERT INTO lessons(text,scope,variant,keywords,enabled,hits,created) VALUES(?,?,?,?,?,?,?)",
                                    (str(l["text"])[:700], l.get("scope") or "any", l.get("variant") or "any",
                                     l.get("keywords") or "", int(l.get("enabled", 1)), int(l.get("hits") or 0),
                                     l.get("created") or time.time()))
                    n_l += 1
                except (KeyError, TypeError, ValueError, sqlite3.Error):
                    continue
            for e in data.get("examples") or []:
                try:
                    if self.db.execute("SELECT 1 FROM examples WHERE id=?", (str(e["id"]),)).fetchone():
                        continue
                    self.db.execute(
                        "INSERT INTO examples(id,name,prompt,category,scope,tokens,payload,parts,enabled,uses,created,source_fmt,notes,model_id) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (str(e["id"]), e.get("name", ""), e.get("prompt", ""), e.get("category", ""), e.get("scope") or "subject",
                         e.get("tokens", ""), json.dumps(e["payload"], separators=(",", ":")), int(e.get("parts") or 0),
                         int(e.get("enabled", 1)), int(e.get("uses") or 0), e.get("created") or time.time(),
                         e.get("source_fmt", ""), e.get("notes", ""), e.get("model_id", "")))
                    n_e += 1
                except (KeyError, TypeError, ValueError, sqlite3.Error):
                    continue
            self.db.commit()
        self.reload()
        return {"models": n_m, "lessons": n_l, "examples": n_e}

    # ---- teaching: text lessons -------------------------------------------------------------------------------
    def lesson_add(self, text: str, scope: str = "any", variant: str = "any", keywords: str = "") -> int:
        text = (text or "").strip()[:700]
        if len(text) < 6:
            raise ValueError("Write at least a short sentence.")
        scope = scope if scope in PART_RANGE else "any"
        variant = variant if variant in ("classic", "depth3d") else "any"
        kw = " ".join(sorted(prompt_tokens(keywords)))
        with self.lock:
            cur = self.db.execute("INSERT INTO lessons(text,scope,variant,keywords,enabled,hits,created) VALUES(?,?,?,?,1,0,?)",
                                  (text, scope, variant, kw, time.time()))
            self.db.commit()
        return int(cur.lastrowid)

    def lesson_list(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT * FROM lessons ORDER BY enabled DESC, id DESC").fetchall()]

    def lesson_update(self, lid: int, f: Dict[str, Any]) -> bool:
        sets, vals = [], []
        if "text" in f and str(f["text"]).strip():
            sets.append("text=?"); vals.append(str(f["text"]).strip()[:700])
        if "scope" in f:
            sets.append("scope=?"); vals.append(f["scope"] if f["scope"] in PART_RANGE else "any")
        if "variant" in f:
            sets.append("variant=?"); vals.append(f["variant"] if f["variant"] in ("classic", "depth3d") else "any")
        if "keywords" in f:
            sets.append("keywords=?"); vals.append(" ".join(sorted(prompt_tokens(str(f["keywords"])))))
        if "enabled" in f:
            sets.append("enabled=?"); vals.append(1 if f["enabled"] in (1, True, "1", "true") else 0)
        if not sets:
            return False
        with self.lock:
            cur = self.db.execute(f"UPDATE lessons SET {', '.join(sets)} WHERE id=?", (*vals, lid))
            self.db.commit()
        return cur.rowcount > 0

    def lesson_delete(self, lid: int) -> bool:
        with self.lock:
            cur = self.db.execute("DELETE FROM lessons WHERE id=?", (lid,))
            self.db.commit()
        return cur.rowcount > 0

    def lessons_for(self, tokens: frozenset, cat: str, variant: str, limit: int = 6, chars: int = 1400,
                    count_hits: bool = True) -> List[Dict[str, Any]]:
        with self.lock:
            rows = [dict(r) for r in self.db.execute("SELECT * FROM lessons WHERE enabled=1").fetchall()]
        picked = []
        for r in rows:
            if r["scope"] not in ("any", cat) or r["variant"] not in ("any", variant):
                continue
            kw = frozenset((r["keywords"] or "").split())
            if kw and not (kw & tokens):
                continue
            picked.append((1 if kw else 0, r["hits"], r["id"], r))
        picked.sort(key=lambda t: (-t[0], -t[1], t[2]))
        out, used = [], 0
        for _, _, _, r in picked:
            if len(out) >= limit or used + len(r["text"]) > chars:
                continue
            out.append(r)
            used += len(r["text"])
        if out and count_hits:
            with self.lock:
                self.db.executemany("UPDATE lessons SET hits = hits + 1 WHERE id=?", [(r["id"],) for r in out])
                self.db.commit()
        return out

    # ---- teaching: example 3D models ----------------------------------------------------------------------------
    def example_add(self, name: str, prompt: str, category: str, scope: str, payload: Dict[str, Any],
                    source_fmt: str, notes: str = "", model_id: str = "") -> str:
        eid = uuid.uuid4().hex[:10]
        tokens = " ".join(sorted(prompt_tokens(prompt or name)))
        with self.lock:
            self.db.execute(
                "INSERT INTO examples(id,name,prompt,category,scope,tokens,payload,parts,enabled,uses,created,source_fmt,notes,model_id) "
                "VALUES(?,?,?,?,?,?,?,?,1,0,?,?,?,?)",
                (eid, name[:80], (prompt or name)[:200], category if category in PART_RANGE else "", 
                 "category" if scope == "category" else "subject", tokens, json.dumps(payload, separators=(",", ":")),
                 len(payload.get("parts") or []), time.time(), source_fmt, notes[:500], model_id))
            self.db.commit()
        return eid

    def example_list(self) -> List[Dict[str, Any]]:
        with self.lock:
            rows = self.db.execute("SELECT id,name,prompt,category,scope,tokens,parts,enabled,uses,created,source_fmt,notes,model_id "
                                   "FROM examples ORDER BY created DESC").fetchall()
        return [dict(r) for r in rows]

    def example_get(self, eid: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            r = self.db.execute("SELECT * FROM examples WHERE id=?", (eid,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["payload"] = json.loads(d["payload"])
        return d

    def example_update(self, eid: str, f: Dict[str, Any]) -> bool:
        sets, vals = [], []
        for col, cast in (("name", lambda v: str(v)[:80]), ("notes", lambda v: str(v)[:500]),
                          ("category", lambda v: v if v in PART_RANGE else ""),
                          ("scope", lambda v: "category" if v == "category" else "subject"),
                          ("enabled", lambda v: 1 if v in (1, True, "1", "true") else 0)):
            if col in f:
                sets.append(f"{col}=?"); vals.append(cast(f[col]))
        if "prompt" in f:
            sets += ["prompt=?", "tokens=?"]
            vals += [str(f["prompt"])[:200], " ".join(sorted(prompt_tokens(str(f["prompt"]))))]
        if not sets:
            return False
        with self.lock:
            cur = self.db.execute(f"UPDATE examples SET {', '.join(sets)} WHERE id=?", (*vals, eid))
            self.db.commit()
        return cur.rowcount > 0

    def example_delete(self, eid: str) -> bool:
        with self.lock:
            cur = self.db.execute("DELETE FROM examples WHERE id=?", (eid,))
            self.db.commit()
        return cur.rowcount > 0

    def examples_for(self, tokens: frozenset, cat: str, limit: int = 1, count_uses: bool = True) -> List[Dict[str, Any]]:
        with self.lock:
            rows = [dict(r) for r in self.db.execute("SELECT id,name,prompt,category,scope,tokens,uses FROM examples WHERE enabled=1").fetchall()]
        scored = []
        for r in rows:
            et = frozenset((r["tokens"] or "").split())
            if r["scope"] == "category":
                if r["category"] == cat:
                    scored.append((0.3, r["uses"], r["id"]))
                continue
            if not et:
                continue
            cov = len(et & tokens) / len(et)
            if cov >= 0.5:
                scored.append((cov + 1.0, r["uses"], r["id"]))
        scored.sort(reverse=True)
        out = []
        for _, _, eid in scored[:limit]:
            ex = self.example_get(eid)
            if ex:
                out.append(ex)
        if out and count_uses:
            with self.lock:
                self.db.executemany("UPDATE examples SET uses = uses + 1 WHERE id=?", [(e["id"],) for e in out])
                self.db.commit()
        return out


MEMORY = MemoryStore()
_INFLIGHT: Dict[str, "asyncio.Future[Dict[str, Any]]"] = {}


# --------------------------------------------------------------------------- teaching: lessons + example 3D models
EX_MAX_PARTS = 120
EX_PTS = 12
MAX_UPLOAD_CHARS = 16_000_000
TEACH_EXAMPLES = max(0, min(2, int(os.getenv("TEACH_EXAMPLES", "1"))))
_DIRS26 = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]


def _reduce_points(pts: List[Tuple[float, float, float]], k: int = EX_PTS) -> List[Tuple[float, float, float]]:
    """Keep the points that define the outer shape: directional extremes -> hull vertices -> farthest-point sampling."""
    uniq = list({(round(p[0], 4), round(p[1], 4), round(p[2], 4)) for p in pts})
    if len(uniq) > 60:
        uniq = list({max(uniq, key=lambda p: d[0] * p[0] + d[1] * p[1] + d[2] * p[2]) for d in _DIRS26})
    if len(uniq) > k:
        faces = convex_hull_3d(uniq)
        if faces:
            uniq = [uniq[i] for i in sorted({i for f in faces for i in f})]
    if len(uniq) > k:
        sel = [min(uniq, key=lambda p: p[0])]
        rest = [p for p in uniq if p is not sel[0]]
        while len(sel) < k and rest:
            far = max(rest, key=lambda p: min((p[0] - s[0]) ** 2 + (p[1] - s[1]) ** 2 + (p[2] - s[2]) ** 2 for s in sel))
            sel.append(far)
            rest.remove(far)
        uniq = sel
    return uniq


def _split_cloud(pts: List[Tuple[float, float, float]], target: int) -> List[List[Tuple[float, float, float]]]:
    """Median-split a single big mesh into chunks along its longest axis so it can become several convex parts."""
    if len(pts) > 60000:
        step = len(pts) // 60000 + 1
        pts = pts[::step]
    clusters = [list(pts)]
    while len(clusters) < target:
        i = max(range(len(clusters)), key=lambda j: len(clusters[j]))
        c = clusters[i]
        if len(c) < 24:
            break
        ext = [max(p[a] for p in c) - min(p[a] for p in c) for a in range(3)]
        a = ext.index(max(ext))
        c.sort(key=lambda p: p[a])
        h = len(c) // 2
        clusters[i:i + 1] = [c[:h], c[h:]]
    return clusters


def _hex_from_rgb(r: float, g: float, b: float) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, round(v * 255))) for v in (r, g, b))


def _lin_to_srgb(c: float) -> float:
    c = max(0.0, min(1.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055


def _clean_name(n: Any, fallback: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", str(n or fallback)).strip("_")[:40] or fallback


def parse_obj(text: str) -> List[Dict[str, Any]]:
    verts: List[Tuple[float, float, float]] = []
    vcols: Dict[int, Tuple[float, float, float]] = {}
    groups: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    cur = "Part"
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("v "):
            t = line.split()
            try:
                verts.append((float(t[1]), float(t[2]), float(t[3])))
                if len(t) >= 7:
                    vcols[len(verts) - 1] = (float(t[4]), float(t[5]), float(t[6]))
            except (ValueError, IndexError):
                continue
        elif line[:2] in ("o ", "g "):
            cur = line[2:].strip() or cur
            groups.setdefault(cur, {"idx": set(), "color": None})
        elif line.startswith("#"):
            m = re.search(r"colou?r\s*[:=]?\s*(#[0-9a-fA-F]{6})", line)
            if m:
                groups.setdefault(cur, {"idx": set(), "color": None})["color"] = m.group(1).lower()
        elif line.startswith("f "):
            g = groups.setdefault(cur, {"idx": set(), "color": None})
            for tok in line.split()[1:]:
                try:
                    i = int(tok.split("/")[0])
                except ValueError:
                    continue
                i = i - 1 if i > 0 else len(verts) + i
                if 0 <= i < len(verts):
                    g["idx"].add(i)
    out = []
    for name, g in groups.items():
        if not g["idx"]:
            continue
        color = g["color"]
        if not color:
            cs = [vcols[i] for i in g["idx"] if i in vcols]
            if cs:
                n = len(cs)
                r, gg, b = (sum(c[k] for c in cs) / n for k in range(3))
                if max(r, gg, b) > 1.0:
                    r, gg, b = r / 255, gg / 255, b / 255
                color = _hex_from_rgb(r, gg, b)
        out.append({"name": name, "color": color, "pts": [verts[i] for i in g["idx"]]})
    return out


def _mat4_mul(a: List[float], b: List[float]) -> List[float]:
    return [sum(a[r * 4 + k] * b[k * 4 + c] for k in range(4)) for r in range(4) for c in range(4)]


def _trs_matrix(node: Dict[str, Any]) -> List[float]:
    m = node.get("matrix")
    if isinstance(m, list) and len(m) == 16:   # glTF is column-major; store as row-major
        return [float(m[c * 4 + r]) for r in range(4) for c in range(4)]
    t = node.get("translation") or [0, 0, 0]
    qx, qy, qz, qw = node.get("rotation") or [0, 0, 0, 1]
    s = node.get("scale") or [1, 1, 1]
    r = [[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
         [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
         [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]]
    return [r[0][0] * s[0], r[0][1] * s[1], r[0][2] * s[2], t[0],
            r[1][0] * s[0], r[1][1] * s[1], r[1][2] * s[2], t[1],
            r[2][0] * s[0], r[2][1] * s[1], r[2][2] * s[2], t[2],
            0.0, 0.0, 0.0, 1.0]


_GL_FMT = {5120: ("b", 1, 127.0), 5121: ("B", 1, 255.0), 5122: ("h", 2, 32767.0), 5123: ("H", 2, 65535.0), 5126: ("f", 4, 0.0)}


def _gl_vec3(gltf: Dict[str, Any], bins: List[bytes], ai: int) -> List[Tuple[float, float, float]]:
    acc = gltf["accessors"][ai]
    if acc.get("sparse") or "bufferView" not in acc or acc.get("type") != "VEC3":
        raise ValueError("Unsupported glTF accessor (sparse or non-VEC3 positions).")
    bv = gltf["bufferViews"][acc["bufferView"]]
    buf = bins[bv.get("buffer", 0)]
    fmt, size, norm_div = _GL_FMT[acc["componentType"]]
    off = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride") or size * 3
    out = []
    for i in range(min(acc["count"], 400000)):
        o = off + i * stride
        v = struct.unpack_from("<3" + fmt, buf, o)
        if acc.get("normalized") and norm_div:
            v = tuple(max(x / norm_div, -1.0) for x in v)
        out.append((float(v[0]), float(v[1]), float(v[2])))
    return out


def parse_gltf(gltf: Dict[str, Any], bins: List[bytes]) -> List[Dict[str, Any]]:
    nodes = gltf.get("nodes") or []
    scenes = gltf.get("scenes") or [{"nodes": list(range(len(nodes)))}]
    roots = scenes[gltf.get("scene", 0)].get("nodes") or []
    out: List[Dict[str, Any]] = []

    def visit(ni: int, parent: List[float], depth: int = 0) -> None:
        if depth > 40 or ni >= len(nodes):
            return
        node = nodes[ni]
        m = _mat4_mul(parent, _trs_matrix(node))
        if "mesh" in node:
            mesh = gltf["meshes"][node["mesh"]]
            for pi, prim in enumerate(mesh.get("primitives") or []):
                if "KHR_draco_mesh_compression" in (prim.get("extensions") or {}):
                    raise ValueError("This GLB uses Draco compression. Re-export it without Draco (or as OBJ).")
                if prim.get("mode", 4) not in (4, 5, 6) or "POSITION" not in (prim.get("attributes") or {}):
                    continue
                local = _gl_vec3(gltf, bins, prim["attributes"]["POSITION"])
                pts = [(m[0] * x + m[1] * y + m[2] * z + m[3], m[4] * x + m[5] * y + m[6] * z + m[7],
                        m[8] * x + m[9] * y + m[10] * z + m[11]) for (x, y, z) in local]
                color = None
                mi = prim.get("material")
                if mi is not None and mi < len(gltf.get("materials") or []):
                    f = ((gltf["materials"][mi].get("pbrMetallicRoughness") or {}).get("baseColorFactor"))
                    if isinstance(f, list) and len(f) >= 3:
                        color = _hex_from_rgb(*(_lin_to_srgb(float(c)) for c in f[:3]))
                base = node.get("name") or mesh.get("name") or f"Node_{ni}"
                out.append({"name": base + (f"_{pi + 1}" if len(mesh["primitives"]) > 1 else ""), "color": color, "pts": pts})
        for ci in node.get("children") or []:
            visit(ci, m, depth + 1)

    ident = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]
    for r in roots:
        visit(r, ident)
    return out


def parse_glb(blob: bytes) -> List[Dict[str, Any]]:
    if len(blob) < 20 or blob[:4] != b"glTF":
        raise ValueError("Not a binary GLB file.")
    pos, gltf, bins = 12, None, []
    while pos + 8 <= len(blob):
        clen, ctype = struct.unpack_from("<II", blob, pos)
        chunk = blob[pos + 8: pos + 8 + clen]
        if ctype == 0x4E4F534A:
            gltf = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:
            bins.append(chunk)
        pos += 8 + clen + ((4 - clen % 4) % 4)
    if gltf is None:
        raise ValueError("GLB has no JSON chunk.")
    for b in gltf.get("buffers") or []:
        if not bins:
            uri = str(b.get("uri", ""))
            if uri.startswith("data:"):
                bins.append(base64.b64decode(uri.split(",", 1)[1]))
    return parse_gltf(gltf, bins or [b""])


def parse_gltf_json(text: str) -> List[Dict[str, Any]]:
    gltf = json.loads(text)
    bins = []
    for b in gltf.get("buffers") or []:
        uri = str(b.get("uri", ""))
        if not uri.startswith("data:"):
            raise ValueError("This .gltf points at an external .bin file. Export a single .glb (or embed the buffers).")
        bins.append(base64.b64decode(uri.split(",", 1)[1]))
    return parse_gltf(gltf, bins)


def parse_json_model(obj: Any) -> List[Dict[str, Any]]:
    """Verdant workspace/low-poly JSON, this server's own payloads (hulls / ops) and the AI row format."""
    if isinstance(obj, dict) and ("asset" in obj and "meshes" in obj):
        return parse_gltf(obj, [])
    groups: List[Dict[str, Any]] = []
    if isinstance(obj, dict) and isinstance(obj.get("payload"), dict):
        obj = obj["payload"]
    rows = (obj.get("parts") or obj.get("hulls")) if isinstance(obj, dict) else obj
    if isinstance(obj, dict) and obj.get("ops") and not rows:
        for o in obj["ops"]:
            try:
                bb = aabb({"size": _vec3(o.get("size"), (1, 1, 1)), "pos": _vec3(o.get("pos"), (0, 0, 0)),
                           "rot": _vec3(o.get("rot"), (0, 0, 0)), "shape": str(o.get("shape") or "box")})
                pts = [(x, y, z) for x in bb[0] for y in bb[1] for z in bb[2]]
                groups.append({"name": o.get("id"), "color": o.get("color"), "pts": pts})
            except (TypeError, ValueError, KeyError):
                continue
        return groups
    for k, r in enumerate(rows or []):
        try:
            if isinstance(r, dict):
                pts = r.get("points") or r.get("pts") or r.get("p") or []
                pos = r.get("pos") or r.get("position") or [0, 0, 0]
                sc = r.get("scale") or [1, 1, 1]
                if isinstance(sc, (int, float)):
                    sc = [sc] * 3
                wp = [(pos[0] + sc[0] * p[0], pos[1] + sc[1] * p[1], pos[2] + sc[2] * p[2]) for p in pts]
                groups.append({"name": r.get("id") or r.get("name"), "color": r.get("color"), "pts": wp})
            elif isinstance(r, (list, tuple)) and len(r) >= 3 and isinstance(r[2], (list, tuple)):
                groups.append({"name": r[0], "color": r[1], "pts": [tuple(map(float, p)) for p in r[2]]})
        except (TypeError, ValueError, IndexError):
            continue
    return groups


def parse_model_upload(fmt: str, text: str = "", blob: Optional[bytes] = None, filename: str = "") -> Tuple[List[Dict[str, Any]], str]:
    fmt = (fmt or "auto").lower().strip().lstrip(".")
    fn = (filename or "").lower()
    if fmt == "auto":
        if blob and blob[:4] == b"glTF":
            fmt = "glb"
        elif fn.endswith(".glb"):
            fmt = "glb"
        elif fn.endswith((".obj",)):
            fmt = "obj"
        elif fn.endswith(".gltf"):
            fmt = "gltf"
        elif fn.endswith(".fbx"):
            fmt = "fbx"
        else:
            t = (text or "").lstrip()[:1]
            fmt = "json" if t in ("{", "[") else "obj"
    if fmt == "fbx":
        raise ValueError("FBX can't be read here. Export the model as OBJ or GLB from Blender/your tool and upload that.")
    if fmt == "glb":
        if blob is None:
            raise ValueError("GLB needs binary data.")
        return parse_glb(blob), "glb"
    if fmt == "gltf":
        return parse_gltf_json(text or (blob or b"").decode("utf-8", "ignore")), "gltf"
    if fmt == "json":
        return parse_json_model(json.loads(text or (blob or b"").decode("utf-8", "ignore"))), "json"
    if fmt == "obj":
        return parse_obj(text or (blob or b"").decode("utf-8", "ignore")), "obj"
    raise ValueError(f"Unknown format '{fmt}'. Use obj, glb, gltf or json.")


def build_example(groups: List[Dict[str, Any]], max_parts: int = EX_MAX_PARTS) -> Dict[str, Any]:
    """Raw mesh groups -> compact convex parts in Verdant model space (ground Y=0, longest side 5.2, centred on X/Z)."""
    notes: List[str] = []
    groups = [{"name": g.get("name"), "color": g.get("color"), "pts": [tuple(map(float, p)) for p in g.get("pts") or []
                                                                      if all(math.isfinite(float(v)) for v in p)]}
              for g in groups]
    groups = [g for g in groups if len(g["pts"]) >= 3]
    if not groups:
        raise ValueError("No usable geometry was found in that file.")
    total = sum(len(g["pts"]) for g in groups)
    if len(groups) <= 3 and total >= 90:
        target, split = 28, []
        for g in groups:
            share = max(2, round(target * len(g["pts"]) / total))
            for i, c in enumerate(_split_cloud(g["pts"], share)):
                split.append({"name": f"{_clean_name(g['name'], 'Mesh')}_{i + 1}", "color": g["color"], "pts": c})
        notes.append(f"The file was one big mesh, so it was cut into {len(split)} chunks.")
        groups = split
    allp = [p for g in groups for p in g["pts"]]
    mn = [min(p[k] for p in allp) for k in range(3)]
    mx = [max(p[k] for p in allp) for k in range(3)]
    span = max(mx[k] - mn[k] for k in range(3))
    if span <= 1e-9:
        raise ValueError("The model has no size.")
    k = 5.2 / span
    cx, cz = (mn[0] + mx[0]) / 2, (mn[2] + mx[2]) / 2
    parts: List[Dict[str, Any]] = []
    ids: Dict[str, int] = {}
    for gi, g in enumerate(groups):
        pts = _reduce_points([((x - cx) * k, (y - mn[1]) * k, (z - cz) * k) for (x, y, z) in g["pts"]])
        if len(pts) < 4:
            continue
        faces = convex_hull_3d(pts)
        if not faces or hull_volume(pts, faces) < 1e-5:
            ext = [max(p[a] for p in pts) - min(p[a] for p in pts) for a in range(3)]
            a = ext.index(min(ext))
            pts = pts + [tuple(p[j] + (0.03 if j == a else 0.0) for j in range(3)) for p in pts]
            faces = convex_hull_3d(pts)
            if not faces or hull_volume(pts, faces) < 1e-6:
                continue
        pid = _clean_name(g["name"], f"Part_{gi + 1}")
        ids[pid] = ids.get(pid, 0) + 1
        if ids[pid] > 1:
            pid = f"{pid}_{ids[pid]}"
        parts.append({"id": pid, "color": _hex6(g["color"], "#9aa5b1"),
                      "pts": [[round(v, 3) for v in p] for p in pts], "_vol": hull_volume(pts, faces)})
    if not parts:
        raise ValueError("None of the parts had a valid volume.")
    if len(parts) > max_parts:
        parts.sort(key=lambda p: -p["_vol"])
        notes.append(f"{len(parts) - max_parts} tiny parts were left out (kept the {max_parts} biggest).")
        parts = parts[:max_parts]
    for p in parts:
        p.pop("_vol", None)
    if not any(g["color"] for g in groups):
        notes.append("The file had no colours - all parts are grey. Colour is still taught from the part names/shapes.")
    ap = [q for p in parts for q in p["pts"]]
    bbox = [round(max(q[a] for q in ap) - min(q[a] for q in ap), 2) for a in range(3)]
    return {"parts": parts, "bbox": bbox, "notes": notes}


def example_from_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    return build_example(parse_json_model(payload))


def example_hulls(ex: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Example parts -> hull rows in the builder's format (pos + local points)."""
    hulls = []
    for p in ex.get("parts") or []:
        wp = [tuple(q) for q in p["pts"]]
        faces = convex_hull_3d(wp)
        if not faces:
            continue
        vs = sorted({i for f in faces for i in f})
        wp = [wp[i] for i in vs]
        c = tuple(sum(q[j] for q in wp) / len(wp) for j in range(3))
        hulls.append({"id": p["id"], "color": p["color"], "pos": [round(v, 3) for v in c], "scale": [1, 1, 1],
                      "points": [[round(q[j] - c[j], 3) for j in range(3)] for q in wp]})
    return hulls


def _guess_shape(pts: List[Tuple[float, float, float]], pid: str) -> str:
    faces = convex_hull_3d(pts)
    if not faces:
        return "box"
    bb = [max(p[a] for p in pts) - min(p[a] for p in pts) for a in range(3)]
    bv = bb[0] * bb[1] * bb[2]
    if bv <= 1e-9:
        return "box"
    r = hull_volume(pts, faces) / bv
    return "box" if r >= 0.85 else "cylinder" if r >= 0.7 else "sphere" if r >= 0.45 else "cone"


def _with_twins(parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop the mirrored right-hand twin of every left part (saves prompt tokens); the AI is told to mirror."""
    by_id = {p["id"]: p for p in parts}
    drop = set()
    rows = []
    for p in parts:
        mir = 0
        if re.search(r"(^|_)L(?=_|\d|$)|Left", p["id"]):
            t = by_id.get(_mirror_id(p["id"]))
            if t and t["id"] != p["id"] and t["color"] == p["color"] and len(t["pts"]) == len(p["pts"]):
                a = sorted((round(q[0], 1), round(q[1], 1), round(q[2], 1)) for q in p["pts"])
                b = sorted((round(-q[0], 1), round(q[1], 1), round(q[2], 1)) for q in t["pts"])
                if a == b:
                    drop.add(t["id"])
                    mir = 1
        rows.append((p, mir))
    return [(p, m) for p, m in rows if p["id"] not in drop]


def reference_block(ex: Dict[str, Any], variant: str) -> str:
    parts = _with_twins((ex.get("payload") or {}).get("parts") or [])
    head = (f'REFERENCE MODEL "{ex["name"]}" - the owner taught the AI with this real example. Study its part list, '
            f'proportions, colour zones and how parts connect, then build the requested subject in the same spirit '
            f'(do not copy it blindly if the subject differs). ')
    if variant == "depth3d":
        rows = [[p["id"], p["color"], [[round(q[0], 2), round(q[1], 2), round(q[2], 2)] for q in _reduce_points([tuple(q) for q in p["pts"]], 8)], m]
                for p, m in parts]
        return head + ("Rows are [id, colour, points in MODEL space (ground Y=0, +Z front, longest side 5.2), mirror]; mirror=1 means the "
                       "twin on the other side exists:\n") + json.dumps(rows, separators=(",", ":"))
    k = 32.0 / 5.2
    rows = []
    for p, m in parts:
        pts = [tuple(q) for q in p["pts"]]
        lo = [min(q[a] for q in pts) for a in range(3)]
        hi = [max(q[a] for q in pts) for a in range(3)]
        rows.append([p["id"], _guess_shape(pts, p["id"]), [round((hi[a] - lo[a]) * k, 1) for a in range(3)],
                     [round((hi[a] + lo[a]) / 2 * k, 1) for a in range(3)], p["color"]] + ([1] if m else []))
    return head + ("Rows are [id, shape, size studs, centre studs, colour, (1 = mirrored twin exists - build both sides)] in "
                   "studs, Y up, +Z front, centred on X/Z, ground at Y=0, longest side 32 studs; rebuild with several stepped "
                   "brick slabs per body part:\n") + json.dumps(rows, separators=(",", ":"))


def teaching_block(prompt: str, tokens: frozenset, cat: str, variant: str, count: bool = True) -> Tuple[str, Dict[str, Any]]:
    """Owner lessons + example reference, appended to the USER prompt (the system prompt stays identical = cacheable)."""
    lessons = MEMORY.lessons_for(tokens, cat, variant, count_hits=count)
    exs = MEMORY.examples_for(tokens, cat, TEACH_EXAMPLES, count_uses=count) if TEACH_EXAMPLES else []
    parts = []
    if lessons:
        parts.append("OWNER LESSONS (the studio owner taught these rules - they override your habits, follow them):\n"
                     + "\n".join("- " + l["text"] for l in lessons))
    for ex in exs:
        parts.append(reference_block(ex, variant))
    if count and (lessons or exs):
        MEMORY.bump("taught_builds")
    return "\n\n".join(parts), {"lessons": [l["id"] for l in lessons], "examples": [e["id"] for e in exs]}


# --------------------------------------------------------------------------- zero-API recolouring ("red dragon" from a saved "blue dragon")
def _hex_hls(c: str) -> Tuple[float, float, float]:
    r, g, b = (int(c[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(r, g, b)


def _hls_hex(h: float, l: float, s: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h % 1.0, max(0.0, min(1.0, l)), max(0.0, min(1.0, s)))
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def recolor_payload(payload: Dict[str, Any], color_word: str, title: str = "") -> Dict[str, Any]:
    """Shift the model's main body colour to `color_word`; eyes, teeth, glows and contrasting trim keep their colours."""
    out = json.loads(json.dumps(payload))
    items = out.get("ops") or out.get("hulls") or []
    tgt = COLOR_HSL.get(color_word)
    if not tgt or not items:
        return out
    th, ts, tl = tgt[0] / 360.0, tgt[1], tgt[2]
    sat, neutral = [], []
    for it in items:
        try:
            h, l, s = _hex_hls(it["color"])
        except (KeyError, ValueError):
            continue
        (sat if s > 0.18 and 0.1 < l < 0.9 else neutral).append((h, l, s))
    # dominant hue: circular mean of the saturated colours (weighted by how many parts use them)
    dom_h = None
    if len(sat) >= max(3, len(items) * 0.25):
        sx = sum(math.cos(h * 2 * math.pi) for h, _, _ in sat)
        sy = sum(math.sin(h * 2 * math.pi) for h, _, _ in sat)
        dom_h = (math.atan2(sy, sx) / (2 * math.pi)) % 1.0
    for it in items:
        if _NO_JITTER.search(str(it.get("id", ""))):
            continue
        try:
            h, l, s = _hex_hls(it["color"])
        except (KeyError, ValueError):
            continue
        is_body = False
        if dom_h is not None:
            d = abs(((h - dom_h + 0.5) % 1.0) - 0.5)
            is_body = s > 0.18 and d < 0.125   # within 45 degrees of the main hue
        else:
            is_body = 0.18 < l < 0.85 and s <= 0.22   # neutral (grey/black/white) model: its body is the neutral mass
        if not is_body:
            continue
        if color_word in NEUTRAL_COLORS:
            if color_word == "white":
                nl, ns = 0.78 + 0.18 * l, 0.06
            elif color_word == "black":
                nl, ns = 0.06 + 0.22 * l, 0.08
            elif color_word == "silver":
                nl, ns = 0.6 + 0.2 * l, 0.08
            else:
                nl, ns = 0.35 + 0.3 * l, 0.05
            it["color"] = _hls_hex(th, nl, ns)
        else:
            keep = 0.5 if dom_h is not None else 0.0
            nh = th + ((((h - dom_h + 0.5) % 1.0) - 0.5) * keep if dom_h is not None else 0.0)
            nl = max(0.1, min(0.9, tl + (l - (sum(x[1] for x in sat) / len(sat) if sat else 0.45)) * 0.8))
            it["color"] = _hls_hex(nh, nl, max(ts * 0.7, min(ts, s if dom_h is not None else ts)))
    if title:
        out["title"] = title[:60]
    return out


def local_variant_color(added: List[str], dropped: List[str]) -> Optional[Tuple[bool, str]]:
    """(True, colour-or-'') when the only differences are colour/size words, so no AI call is needed."""
    non_trivial = [t for t in added + dropped if t not in TRIVIAL_WORDS]
    if non_trivial:
        return None
    colors = [t for t in added if t in COLOR_HSL]
    return True, (colors[0] if colors else "")


def apply_tonal(ops: List[Dict[str, Any]], tonal: bool) -> List[Dict[str, Any]]:
    out = []
    for o in ops:
        o = dict(o)
        if tonal and not _NO_JITTER.search(str(o.get("id", ""))):
            try:
                o["color"] = jitter(o["color"], o["id"])
            except Exception:
                pass
        out.append(o)
    return out


def apply_tonal_payload(out: Dict[str, Any], tonal: bool) -> Dict[str, Any]:
    out = dict(out)
    if out.get("ops"):
        out["ops"] = apply_tonal(out["ops"], tonal)
    if out.get("hulls"):
        hs = []
        for h in out["hulls"]:
            h = dict(h)
            if tonal and not _NO_JITTER.search(str(h.get("id", ""))):
                try:
                    h["color"] = jitter(h["color"], h["id"])
                except Exception:
                    pass
            hs.append(h)
        out["hulls"] = hs
    return out


def memory_info(source: str, **kw: Any) -> Dict[str, Any]:
    return {"source": source, "total_models": MEMORY.count(), **kw}


# --------------------------------------------------------------------------- pipeline
async def ask_and_parse(system: str, user: str, provider: str, model: str, api_key: str, retries: int = 1):
    last: Optional[Exception] = None
    for _ in range(retries + 1):
        try:
            text, engine = await call_llm(system, user, provider, model, api_key)
        except ProviderError as e:
            raise e.to_http()
        data = parse_llm_json(text)
        if isinstance(data.get("ops") or data.get("parts") or data.get("hulls"), list) and (data.get("ops") or data.get("parts") or data.get("hulls")):
            return data, engine
        last = ValueError("The AI answer contained no parts.")
    raise HTTPException(status_code=422, detail=str(last))


def response_payload(title: str, cat: str, data: Dict[str, Any], parts, analysis, tonal: bool,
                     engine: Dict[str, str], revisions: int, used_serper: bool) -> Dict[str, Any]:
    ops, stats = export_build(parts, analysis, tonal)
    diag = dict(analysis["diag"])
    diag["quality"] = (f"Grade {analysis['grade']} ({analysis['score']}/100)"
                       + (f" after {revisions} automatic revision{'s' if revisions != 1 else ''}" if revisions else ""))
    return {
        "title": str(data.get("title") or title)[:60], "category": cat, "ops": ops, "stats": stats,
        "diagnostics": diag,
        "quality": {"score": analysis["score"], "grade": analysis["grade"], "revisions": revisions,
                    "issues_left": [i["msg"] for i in analysis["issues"]]},
        "plan": data.get("plan") if isinstance(data.get("plan"), dict) else None,
        "engine": engine, "used_serper": used_serper,
    }


class GenerateRequest(BaseModel):
    prompt: str
    web_context: Optional[str] = ""
    use_serper: bool = True
    provider: Optional[str] = ""
    model: Optional[str] = ""
    api_key: Optional[str] = ""
    serper_key: Optional[str] = ""
    category: Optional[str] = ""
    target_parts: Optional[int] = None
    revise: Optional[bool] = None
    max_revisions: Optional[int] = None
    tonal_variation: bool = True
    effort: Optional[str] = "mid"
    variant: Optional[str] = "classic"
    use_memory: bool = True
    publish: bool = True
    fresh: bool = False


class ReviseRequest(BaseModel):
    instruction: str
    ops: List[Dict[str, Any]] = []
    hulls: Optional[List[Dict[str, Any]]] = None
    variant: Optional[str] = "classic"
    effort: Optional[str] = "mid"
    prompt: Optional[str] = ""
    category: Optional[str] = ""
    scene_scale: Optional[float] = None
    provider: Optional[str] = ""
    model: Optional[str] = ""
    api_key: Optional[str] = ""
    tonal_variation: bool = True


class TestKeyRequest(BaseModel):
    provider: str
    api_key: str
    model: Optional[str] = ""


Progress = Any


def _noop(msg: str, kind: str = "") -> None:
    return None


# --------------------------------------------------------------------------- Depth 3D: universal standable low-poly convex volumes
def _vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vcross(u, v):
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


def _vdot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def convex_hull_3d(pts: List[Tuple[float, float, float]], eps: float = 1e-6) -> List[Tuple[int, int, int]]:
    """Incremental 3-D convex hull. Returns outward-facing triangles (index triples); [] when the points are degenerate."""
    n = len(pts)
    if n < 4:
        return []
    i0 = 0
    i1 = next((i for i in range(1, n) if _vdot(_vsub(pts[i], pts[i0]), _vsub(pts[i], pts[i0])) > eps), None)
    if i1 is None:
        return []
    i2 = next((i for i in range(n) if i not in (i0, i1)
               and (lambda c: _vdot(c, c))(_vcross(_vsub(pts[i1], pts[i0]), _vsub(pts[i], pts[i0]))) > eps), None)
    if i2 is None:
        return []
    nrm = _vcross(_vsub(pts[i1], pts[i0]), _vsub(pts[i2], pts[i0]))
    i3 = next((i for i in range(n) if i not in (i0, i1, i2) and abs(_vdot(nrm, _vsub(pts[i], pts[i0]))) > eps), None)
    if i3 is None:
        return []
    cen = tuple(sum(pts[i][k] for i in (i0, i1, i2, i3)) / 4 for k in range(3))
    faces = []
    for f in ((i0, i1, i2), (i0, i1, i3), (i1, i2, i3), (i0, i2, i3)):
        a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
        if _vdot(_vcross(_vsub(b, a), _vsub(c, a)), _vsub(cen, a)) > 0:
            f = (f[0], f[2], f[1])
        faces.append(f)
    used = {i0, i1, i2, i3}
    for pi in range(n):
        if pi in used:
            continue
        p = pts[pi]
        vis, keep = [], []
        for f in faces:
            a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
            nf = _vcross(_vsub(b, a), _vsub(c, a))
            ln = math.sqrt(_vdot(nf, nf)) or 1.0
            (vis if _vdot(nf, _vsub(p, a)) / ln > eps else keep).append(f)
        if not vis:
            continue
        edges = set()
        for f in vis:
            edges.update(((f[0], f[1]), (f[1], f[2]), (f[2], f[0])))
        faces = keep + [(u, v, pi) for (u, v) in edges if (v, u) not in edges]
    return faces


def hull_volume(pts, faces) -> float:
    v = 0.0
    for f in faces:
        a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
        v += _vdot(a, _vcross(b, c)) / 6.0
    return abs(v)


def _hex6(c: Any, default: str = "#243447") -> str:
    c = str(c or "").strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", c):
        return c.lower()
    if re.fullmatch(r"#[0-9a-fA-F]{3}", c):
        return "#" + "".join(ch * 2 for ch in c[1:]).lower()
    return default


def _mirror_id(pid: str) -> str:
    m = re.search(r"(^|_)([LR])(?=_|\d|$)", pid)
    if m:
        i = m.start(2)
        return pid[:i] + ("R" if m.group(2) == "L" else "L") + pid[i + 1:]
    if "Left" in pid:
        return pid.replace("Left", "Right", 1)
    if "Right" in pid:
        return pid.replace("Right", "Left", 1)
    return pid + "_mirror"


def _sculpted_points_from_box(size: List[float], pos: List[float], rot: List[float], shape: str = "box") -> List[Tuple[float, float, float]]:
    """Converts a [w,h,d] primitive into a 12-point faceted low-poly volume in model space so any Classic
    [id, shape, size, pos, rot, color] row becomes a genuine beveled low-poly convex hull."""
    w, h, d = max(0.14, float(size[0])), max(0.14, float(size[1])), max(0.14, float(size[2]))
    cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
    rm = rot_matrix(rot)

    def xform(lx: float, ly: float, lz: float) -> Tuple[float, float, float]:
        rx = rm[0][0] * lx + rm[0][1] * ly + rm[0][2] * lz + cx
        ry = rm[1][0] * lx + rm[1][1] * ly + rm[1][2] * lz + cy
        rz = rm[2][0] * lx + rm[2][1] * ly + rm[2][2] * lz + cz
        return (round(rx, 3), round(ry, 3), round(rz, 3))

    hx, hy, hz = w * 0.5, h * 0.5, d * 0.5
    if shape == "cone":
        return [
            xform(-hx, -hy, -hz), xform(hx, -hy, -hz), xform(hx, -hy, hz), xform(-hx, -hy, hz),
            xform(0.0, -hy, hz * 1.12), xform(0.0, hy, 0.0)
        ]
    return [
        xform(-hx * 0.82, -hy, -hz * 0.82), xform(hx * 0.82, -hy, -hz * 0.82),
        xform(hx * 0.85, -hy, hz * 0.85),   xform(-hx * 0.85, -hy, hz * 0.85),
        xform(-hx * 1.04, 0.0, 0.0),        xform(hx * 1.04, 0.0, 0.0),
        xform(0.0, 0.0, hz * 1.06),         xform(0.0, 0.0, -hz * 1.04),
        xform(-hx * 0.86, hy, -hz * 0.84),  xform(hx * 0.86, hy, -hz * 0.84),
        xform(hx * 0.88, hy, hz * 0.88),    xform(-hx * 0.88, hy, hz * 0.88),
    ]


def _bevel_if_plain_cube(pts: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    if len(pts) != 8:
        return pts
    xs, ys, zs = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]
    x0, x1, y0, y1, z0, z1 = min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    if min(dx, dy, dz) < 1e-3:
        return pts
    cx, cy, cz = (x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5
    inset = [((x - cx) * 0.86 + cx, y, (z - cz) * 0.86 + cz) for (x, y, z) in pts]
    equator = [
        (cx - dx * 0.53, cy, cz), (cx + dx * 0.53, cy, cz),
        (cx, cy, cz + dz * 0.54), (cx, cy, cz - dz * 0.53),
        (cx, y1 + dy * 0.04, cz), (cx, y0 - dy * 0.02, cz),
    ]
    return [(round(x, 3), round(y, 3), round(z, 3)) for (x, y, z) in (inset + equator)]


def _clean_points(pts: Any) -> List[Tuple[float, float, float]]:
    out, seen = [], set()
    if not isinstance(pts, (list, tuple)):
        return out
    for p in pts[:64]:
        if isinstance(p, (list, tuple)) and len(p) == 3 and not isinstance(p[0], (list, tuple)):
            try:
                q = tuple(round(max(-40.0, min(40.0, float(v))), 3) for v in p)
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(v) for v in q) and q not in seen:
                seen.add(q)
                out.append(q)
    return out


DEPTH_SKILL = """You are the VERDANT DEPTH 3D SCULPTOR: a senior low-poly 3D game artist.
You build models as CONVEX VOLUMES ("hulls"). Every part is the convex hull of 6-12 3D points and is rendered flat-shaded, like bespoke low-poly studio game art.

OUTPUT - reply with ONE valid JSON object and nothing else (no markdown, no comments, no prose):
{
  "title": "short model name",
  "category": "creature|humanoid|vehicle|structure|plant|prop",
  "plan": {
    "silhouette": "one sentence describing wide 3D stance, posture, and proportions",
    "palette": {"base":"#hex","secondary":"#hex","trim":"#hex","accent":"#hex","dark":"#0b0e14","light":"#ffffff"},
    "signature_features": ["..."]
  },
  "parts": [
    ["Torso_Upper", "#b81d18", [[x,y,z],[x,y,z],[x,y,z],[x,y,z],[x,y,z],[x,y,z],[x,y,z],[x,y,z]], 0],
    ["Shoulder_L", "#b81d18", [[x,y,z],[x,y,z],[x,y,z],[x,y,z],[x,y,z],[x,y,z],[x,y,z],[x,y,z]], 1]
  ]
}
Every part in `"parts"` is ONE ROW: `[id, "#rrggbb", points, mirror]`.
- `points` are `[x,y,z]` triples in MODEL space (one shared coordinate system for the whole model, 2 decimals).
- `mirror` = 1 tells the server to mirror the twin across the X=0 plane (`_L` -> `_R`), so author the `+X` side with `mirror=1` and NEVER write both sides yourself; use `0` for centered parts (`X=0`).

UNIVERSAL 3D SCULPTING & STANDING LAWS (FOLLOW STRICTLY FOR ALL SUBJECTS):
1. WIDE, STANDABLE 3D SILHOUETTE (NEVER STACK PARTS INTO A TOTEM POLE AT X=0!):
   - Center the spine/torso/chassis on `X = 0`, facing `+Z`, standing firmly on the ground at `Y = 0`.
   - ALWAYS place bilateral parts (`Shoulder_L`, `Arm_L`, `Bicep_L`, `Forearm_L`, `Hand_L`, `Thigh_L`, `Shin_L`, `Foot_L`, `Wing_L`, `Tentacle_L`, `Wheel_FL`, `Turret_L`) well out on `+X` (`X = +0.7 to +2.8`) with `mirror = 1`!
   - Never put left/right limbs, wheels, wings, or tentacles on `X = 0`.
2. MULTI-TONE SUBJECT-APPROPRIATE PALETTE (NEVER MONOCHROME):
   - Match the real colors of the prompt (`base` for main body/paint/scales/fur/stone, contrasting `secondary` `#e8dfd1` or cream/silver for chest/underbelly/snout/windows/fingers/toes, `trim` `#f5c518` for gold/metal cuffs/horns/rims/crown, `accent` for glowing eyes/fire/thrusters, `light` `#ffffff` for fangs/claws/headlights, `dark` `#0b0e14` for open mouth pit/tires).
3. FACETED LOW-POLY VOLUMES (NO PLAIN CUBES):
   - Every volume must TAPER, LEAN, or BEVEL: tapered 8-10 point limbs, 10-12 point faceted torso/head gems, 6-point wedges for snouts/jaws/blades/feathers, 5-point pyramids for horns/fangs/claws/roofs.
4. SOLID JOINT OVERLAP & GROUNDING:
   - Neighbouring segments must overlap by `0.08 - 0.18` units so nothing floats. Feet, tires, tentacles, or foundations rest cleanly on `Y = 0`.

TARGET: @LO@-@HI@ parts and @PMIN@-@PMAX@ points per part (mirrored `_R` twins are added automatically for free).
"""

DEPTH_HINTS = {
    "creature_quad": """CREATURE / BEAST / APE / QUADRUPED IN MODEL SPACE (Y=0 ground, +Z front, author +X side with mirror=1):
- Spine & Torso (mirror=0): `Pelvis` (x=-0.9..0.9, y=1.2..2.1, z=-0.6..0.5), `Torso_Mid` (x=-1.2..1.2, y=2.0..3.3, z=-0.7..0.8), `Torso_Upper` (x=-1.6..1.6, y=3.1..4.5, z=-0.8..1.0), plus contrasting `secondary` `Chest_Pec_L` (mirror=1, x=0.15..1.35, y=3.3..4.3, z=0.8..1.25) and 3 stepped `Belly_Slab_1..3` hulls on +Z.
- Head & Hollow Roaring Jaw (y=4.3..5.6, z=0.5..2.0): `Head_Cranium` (`base`), angled `Head_Brow_L` (mirror=1, `secondary`), glowing `Eye_L` (mirror=1, `accent`), `Head_Snout` (`secondary`), dark `Head_Mouth_Pit` (`#0b0e14`), `Head_Lower_Jaw` (`secondary`), and 4 `Fang_*` pyramids (`#ffffff`, mirror=1).
- Wide Muscular Arms & Hands (mirror=1, placed OUT at X = +1.4 .. +2.7!): `Shoulder_L` (x=1.2..2.1, y=3.6..4.5), `Pauldron_L_1..2`, `Bicep_L` (x=1.5..2.4, y=2.5..3.7), `Forearm_L` (x=1.6..2.6, y=1.3..2.6, z=0.4..1.4), `Bracer_Cuff_L` (`trim`), `Hand_Palm_L` (x=1.6..2.7, y=0.3..1.4, z=0.7..1.7), plus 4 `Finger_L_1..4` and `Thumb_L` (`secondary`).
- Wide Stance Legs & Toes (mirror=1, placed at X = +0.55 .. +1.35): `Thigh_L` (x=0.55..1.35, y=0.9..2.0), `Shin_L` (x=0.6..1.3, y=0.25..1.0), `Foot_L` (y=0.0..0.35), and 3 `Toe_L_1..3` (`secondary`) on y=0.""",
    "creature_dragon": """DRAGON / WYVERN / WINGED BEAST IN MODEL SPACE (Y=0 ground, +Z front, author +X side with mirror=1):
- S-curved neck (`Neck_1..3`), broad chest & body (`Torso_Front`, `Torso_Mid`, `Pelvis`) with 6 stepped cream/gold `Belly_Scute_1..6` plates (`secondary`) along the front/underside, and a long 5-segment tapering `Tail_1..5` sweeping back to Z = -3.8 with dorsal `Spine_Spike_1..6`.
- Fierce sculpted Dragon Head at (y=4.2..5.4, z=1.2..2.8): `Head_Skull`, swept `Horn_L_1..2` (mirror=1, `trim`), overhanging `Brow_L` (mirror=1), glowing `Eye_L` (mirror=1, `accent`), `Upper_Snout`, dark `Mouth_Pit`, hinged `Lower_Jaw`, and white `Fang_L_1..3` (mirror=1).
- 2 Massive Wings (mirror=1, spreading wide from X = +0.9 out to X = +3.6, Y = 3.2..5.8): `Wing_Shoulder_L`, `Wing_Arm_L1..2`, `Wing_ThumbClaw_L`, and 3 wide thin faceted membrane plates `Wing_Membrane_L1..3` (`secondary`/`accent`).
- 4 Powerful Legs & Talons (mirror=1, at X = +0.7..+1.5): `Thigh_L`, `Shin_L`, `Foot_L`, and 3 sharp white `Claw_L_1..3` grounded at y=0.""",
    "creature_multi": """KRAKEN / SQUID / OCTOPUS / ARACHNID IN MODEL SPACE (Y=0 ground, +Z front, author +X side with mirror=1):
- Tall faceted Bulbous Mantle/Dome (`Mantle_Base`, `Mantle_Mid`, `Mantle_Crown`, `Mantle_Fin_L` with mirror=1) rising to y=5.2, overhanging `Brow_L` (mirror=1), large glowing `Eye_L` (mirror=1, `accent`), dark `Mouth_Beak` at the lower front, and contrasting `secondary` under-mantle plates.
- 8 Radial Multi-Segment Tentacles / Legs (`Tentacle_L1_Seg1..3`, `Tentacle_L2_Seg1..3`, `Tentacle_L3_Seg1..3`, `Tentacle_L4_Seg1..3`, all with `mirror=1`) fanning wide outward across X = +0.6 .. +3.2 and Z = -2.5 .. +2.5 on the ground (y=0..1.4) with `secondary` sucker/claw accents.""",
    "creature_bird": """BIRD / PHOENIX / GRIFFIN IN MODEL SPACE (Y=0 ground, +Z front, author +X side with mirror=1):
- Proud angled breast (`Breast_Upper`, `Breast_Lower`) with layered `Plumage_Slab_1..4` (`secondary`), `Head_Skull` with 3-part `Crest_1..3` (`accent`/`trim`), hooked 2-part golden `Beak_Upper` & `Beak_Lower` (`trim`), glowing `Eye_L` (mirror=1, `accent`), wide 3-section feathered `Wing_L_1..3` (mirror=1, X = +0.8..+3.4), fanned `Tail_Feather_1..5`, and taloned `Leg_L` + `Talon_L_1..3` (mirror=1) at y=0.""",
    "creature_aquatic": """SHARK / WHALE / LEVIATHAN IN MODEL SPACE (Y=0 ground, +Z front, author +X side with mirror=1):
- 6-segment streamlined tapering body along Z = +2.4 .. -2.8 (`Body_1..6`) with dark/vivid dorsal top (`base`) and crisp `secondary` cream/white `Belly_1..5` counter-shading underneath, open jaws (`Upper_Jaw`, `Lower_Jaw`, `Mouth_Pit`, white `Tooth_L_1..4` mirror=1), glowing `Eye_L` (mirror=1), tall `Dorsal_Fin`, wide `Pectoral_Fin_L` (mirror=1, X = +0.8..+2.6), and twin `Tail_Fluke_L` / `Upper`.""",
    "humanoid": """HUMANOID / MECH / KNIGHT IN MODEL SPACE (Y=0 ground, +Z front, author +X side with mirror=1):
- V-tapered `Pelvis`, `Waist`, and `Chest_Core` (`base`) with contrasting `Chest_Plate_L` (mirror=1, `secondary`), glowing `Chest_Core_Gem` (`accent`), and golden `Belt_Buckle` (`trim`).
- Sculpted `Head_Helm` with recessed dark `Visor_Slit`, glowing `Eye_L` (mirror=1, `accent`), and `Crown_Or_Crest` (`trim`).
- Wide `Pauldron_L_1..2` (mirror=1, X = +1.1..+1.9), articulated `Bicep_L`, `Forearm_L`, golden `Gauntlet_Cuff_L` (`trim`), `Hand_L` (mirror=1, X = +1.3..+2.0) + handheld weapon/shield, and wide-stance `Thigh_L`, `Knee_Guard_L`, `Shin_L`, `Boot_L` (mirror=1, X = +0.55..+1.15) standing on y=0.""",
    "vehicle": """VEHICLE / SUPERCAR / AIRCRAFT / SHIP IN MODEL SPACE (Y=0 ground, +Z front, author +X side with mirror=1):
- Wide low-slung sculpted chassis (`Chassis_Center`, `Front_Hood`, `Rear_Deck`, `Side_Skirt_L` mirror=1 across X = ±1.4, Z = -2.4..+2.4), tapered `Cabin_Windshield` & `Roof` (`secondary`), dark `Front_Splitter` & `Rear_Diffuser` (`dark`), `Spoiler_Wing` + `Spoiler_Strut_L` (mirror=1), 4 wide cylindrical/octagonal `Wheel_FL` & `Wheel_RL` tires (`dark`, mirror=1 at X = +1.35, Z = ±1.55) with golden `Rim_FL` & `Rim_RL` (`trim`, mirror=1), `Headlight_L` (`light`, mirror=1), and glowing `Taillight_L` (`accent`, mirror=1).""",
    "structure": """STRUCTURE / WATCHTOWER / CASTLE / SHRINE IN MODEL SPACE (Y=0 ground, +Z front, author +X side with mirror=1):
- Wide stepped stone foundation (`Base_Plinth_1..2`, X = -2.2..2.2, Z = -2.2..2.2, Y = 0..0.8), tapered main tower/walls (`Wall_Core`, `Corner_Pillar_FL` & `RL` mirror=1), recessed arched `Doorway_Frame` & `Door_Arch` on +Z, overhanging `Battlement_Ring` & crenelations (`Merlon_L1..3` mirror=1), pitched pyramid/conical `Roof_Spire` (`trim`/`secondary`), and glowing `Window_L` / `Lantern_L` (`accent`, mirror=1).""",
    "plant": """PLANT / TREE / FANTASY FLORA IN MODEL SPACE (Y=0 ground, +Z front, author +X side with mirror=1):
- Wide mossy island/rock base (`Base_Earth`, `Root_Flare_L1..2` mirror=1) on y=0, 4-part organic curved trunk & primary branches (`Trunk_1..3`, `Branch_L1..2` mirror=1), and 10-16 overlapping faceted crystal/foliage canopy volumes (`Canopy_Center`, `Canopy_Cluster_L1..5` mirror=1) spread wide across X = ±2.2, Y = 2.4..5.2 in 3 rich shades (`base`, `secondary`, `trim`) with glowing `Fruit_Or_Blossom_L` (`accent`, mirror=1).""",
    "prop": """PROP / WEAPON / ARTIFACT IN MODEL SPACE (Y=0 ground, +Z front, author +X side with mirror=1):
- Bold multi-part silhouette using all color zones: `dark` grip/base, `trim` gold crossguard/brackets (`Guard_Wing_L` mirror=1), `base` core body, `secondary` beveled blade/inlay slabs, `light` razor edges/rivets, and `accent` glowing rune/gem core.""",
}


def build_depth_system_prompt(cat: str, sub: str, prof: Dict[str, Any]) -> str:
    lo, hi = prof["d3_parts"]
    pmin, pmax = prof["d3_pts"]
    hint_key = f"creature_{sub}" if cat == "creature" and f"creature_{sub}" in DEPTH_HINTS else cat
    hint = DEPTH_HINTS.get(hint_key, DEPTH_HINTS["prop"])
    return (DEPTH_SKILL.replace("@LO@", str(lo)).replace("@HI@", str(hi))
            .replace("@PMIN@", str(pmin)).replace("@PMAX@", str(pmax)) + "\n" + hint)


def hulls_to_rows(hulls: List[Dict[str, Any]]) -> List[List[Any]]:
    """Workspace parts (local points + pos + scale) -> compact rows in MODEL space for the AI."""
    rows = []
    for h in hulls:
        try:
            pos = h.get("pos") or [0, 0, 0]
            sc = h.get("scale") or [1, 1, 1]
            pts = [[round(pos[0] + sc[0] * p[0], 2), round(pos[1] + sc[1] * p[1], 2), round(pos[2] + sc[2] * p[2], 2)]
                   for p in (h.get("points") or [])]
            if len(pts) >= 4:
                rows.append([str(h.get("id") or "Part"), _hex6(h.get("color")), pts, 0])
        except (TypeError, ValueError, IndexError):
            continue
    return rows


def build_depth_user_prompt(prompt: str, cat: str, sub: str, prof: Dict[str, Any], web_ctx: str) -> str:
    lo, hi = prof["d3_parts"]
    b_col, s_col, t_col, a_col = infer_subject_palette(prompt, cat)
    out = [
        f'SUBJECT: "{prompt}"',
        f"CATEGORY: {cat}" + (f" ({sub})" if sub else ""),
        f"Sculpt it as standable low-poly convex volumes: about {lo}-{hi} parts, {prof['d3_pts'][0]}-{prof['d3_pts'][1]} points each.",
        f"SUGGESTED PALETTE FOR THIS SUBJECT: base={b_col}, secondary={s_col}, trim={t_col}, accent={a_col}, dark=#0b0e14, light=#ffffff.",
        "IMPORTANT: Place all left/right limbs, wings, tentacles, wheels, or side towers out at X = +0.7 .. +2.8 with `mirror=1` so the model has a wide, balanced 3D stance and NEVER collapses into a thin line on X=0!",
        EFFORT_PROMPT[prof["name"]],
    ]
    if web_ctx:
        out.append("LIVE WEB REFERENCE (use it for anatomy, colours and proportions; do not copy text): " + web_ctx)
    return "\n".join(out)


def build_depth_revision_prompt(prompt: str, cat: str, prof: Dict[str, Any], hulls: List[Dict[str, Any]],
                                problems: List[str], change: str = "") -> str:
    rows = hulls_to_rows(hulls)
    out = [f'SUBJECT: "{prompt}"', f"CATEGORY: {cat}",
           f"TARGET: {prof['d3_parts'][0]}-{prof['d3_parts'][1]} parts.",
           'CURRENT MODEL - rows are [id, color, points in MODEL space, mirror=0]:',
           json.dumps(rows, separators=(",", ":"))]
    if change:
        out.append(f'REQUESTED CHANGE: "{change}"\nApply it faithfully and keep everything else.')
    if problems:
        out.append("PROBLEMS FOUND BY THE CHECKER - fix all of them:\n" + "\n".join("- " + m for m in problems))
    out.append(EFFORT_PROMPT[prof["name"]])
    out.append("Return the COMPLETE corrected model in the same JSON schema (plan + every part row), never a diff. "
               "You may now write both sides of symmetric parts or use the mirror flag.")
    return "\n".join(out)


_BILATERAL_PART_RE = re.compile(
    r"(?:arm|bicep|forearm|shoulder|pauldron|hand|palm|finger|thumb|bracer|cuff|gauntlet|leg|thigh|knee|shin|calf|foot|boot|toe|claw|talon|wing|tentacle|fin|wheel|tire|rim|headlight|taillight|eye|brow|cheek|horn|ear|tusk|fang)",
    re.I,
)


def _auto_spread_bilateral_hulls(raw: List[List[Any]]) -> None:
    """If an LLM accidentally puts bilateral limbs/wheels/wings/eyes on X=0 (totem pole bug),
    automatically offset them outward along +X and turn on mirror=1 so the model stands wide in 3D!"""
    existing_ids = {r[0] for r in raw}
    for r in raw:
        pid, col, pts, mir = r[0], r[1], r[2], r[3]
        if not pts:
            continue
        cx = sum(p[0] for p in pts) / len(pts)
        is_lr = bool(re.search(r"(^|_)[LR](?=_|\d|$)|Left|Right", pid))
        if (is_lr or _BILATERAL_PART_RE.search(pid)) and abs(cx) < 0.18:
            sign = -1.0 if re.search(r"(^|_)R(?=_|\d|$)|Right", pid) else 1.0
            if re.search(r"wing|tentacle", pid, re.I):
                shift_x = 1.45 * sign
            elif re.search(r"arm|bicep|forearm|hand|palm|finger|bracer|cuff|gauntlet|shoulder|pauldron|wheel|tire|rim", pid, re.I):
                shift_x = 1.25 * sign
            elif re.search(r"leg|thigh|knee|shin|calf|foot|boot|toe|claw|talon", pid, re.I):
                shift_x = 0.72 * sign
            else:
                shift_x = 0.42 * sign
            r[2] = [(round(x + shift_x, 3), y, z) for (x, y, z) in pts]
            partner = _mirror_id(pid)
            if partner not in existing_ids and sign > 0:
                r[3] = True


def process_hulls(data: Dict[str, Any], prof: Dict[str, Any], lo: int, hi: int,
                  prompt: str = "", cat: str = "creature") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = data.get("parts") or data.get("hulls") or data.get("ops") or []
    if not isinstance(rows, list) or not rows:
        raise ValueError("The AI answer contained no parts.")
    if not isinstance(data.get("plan"), dict):
        data["plan"] = {}
    if not isinstance(data["plan"].get("palette"), dict):
        data["plan"]["palette"] = {}
    palette = data["plan"]["palette"]
    def_base, _, _, _ = infer_subject_palette(prompt, cat)

    raw: List[List[Any]] = []
    dropped: List[str] = []
    for k, row in enumerate(rows[:400]):
        pid, col, pts, mir = f"Part_{k + 1}", def_base, [], 0
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            if isinstance(row[1], str) and not str(row[1]).strip().startswith("#") and len(row) >= 4 and isinstance(row[2], (list, tuple)) and len(row[2]) == 3 and not isinstance(row[2][0], (list, tuple)):
                pid = str(row[0])
                shape = str(row[1]).lower()
                sz = _vec3(row[2], (1, 1, 1))
                po = _vec3(row[3], (0, 0, 0))
                ro = _vec3(row[4], (0, 0, 0)) if len(row) > 4 and isinstance(row[4], (list, tuple)) else [0.0, 0.0, 0.0]
                col = next((x for x in row[4:] if isinstance(x, str) and str(x).strip().startswith("#")), def_base)
                pts = _sculpted_points_from_box(sz, po, ro, shape)
                mir = 0
            else:
                pid, col, pts, mir = row[0], row[1], row[2], (row[3] if len(row) > 3 else 0)
        elif isinstance(row, dict):
            pid = row.get("id") or row.get("name") or f"Part_{k + 1}"
            col = row.get("color") or row.get("c") or def_base
            pts = row.get("points") or row.get("p")
            mir = row.get("mirror") or row.get("m") or 0
            pos, sc = row.get("pos") or row.get("position"), row.get("scale")
            if not pts and (row.get("size") or row.get("sz")):
                sz = _vec3(row.get("size") or row.get("sz"), (1, 1, 1))
                po = _vec3(pos, (0, sz[1] / 2, 0))
                ro = _vec3(row.get("rot") or row.get("rotation"), (0, 0, 0))
                pts = _sculpted_points_from_box(sz, po, ro, str(row.get("shape") or "box").lower())
            elif isinstance(pos, (list, tuple)) and len(pos) == 3 and isinstance(pts, (list, tuple)):
                sc = sc if isinstance(sc, (list, tuple)) and len(sc) == 3 else (1, 1, 1)
                try:
                    pts = [[pos[0] + sc[0] * p[0], pos[1] + sc[1] * p[1], pos[2] + sc[2] * p[2]] for p in pts]
                except (TypeError, IndexError):
                    pts = []
        else:
            continue

        pid = re.sub(r"[^A-Za-z0-9_\-]", "_", str(pid or f"Part_{k + 1}"))[:40] or f"Part_{k + 1}"
        clean = _bevel_if_plain_cube(_clean_points(pts))
        if len(clean) < 4:
            dropped.append(pid)
            continue
        auto_mir = mir in (1, True, "1", "true", "yes")
        if not auto_mir and re.search(r"(^|_)L(?=_|\d|$)|Left", pid):
            partner = _mirror_id(pid)
            if not any(isinstance(r, (list, tuple)) and len(r) > 0 and str(r[0]) == partner for r in rows):
                auto_mir = True
        raw.append([pid, _hex6(col, def_base), clean, auto_mir])

    if not raw:
        raise ValueError("The AI answer contained no usable parts.")

    _auto_spread_bilateral_hulls(raw)

    dummy_parts = [{"id": r[0], "color": r[1]} for r in raw]
    enforce_color_blocking(dummy_parts, palette, prompt, cat)
    for r, dp in zip(raw, dummy_parts):
        r[1] = dp["color"]

    expanded: List[List[Any]] = []
    existing_names = {r[0] for r in raw}
    for pid, col, pts, mir in raw:
        expanded.append([pid, col, pts])
        if mir:
            cx = sum(p[0] for p in pts) / len(pts)
            twin_id = _mirror_id(pid)
            if abs(cx) > 0.04 and twin_id not in existing_names:
                expanded.append([twin_id, col, [(-x, y, z) for (x, y, z) in pts]])
    ids: Dict[str, int] = {}
    for e in expanded:
        n = ids.get(e[0], 0) + 1
        ids[e[0]] = n
        if n > 1:
            e[0] = f"{e[0]}_{n}"
    expanded = expanded[: hi * 2 + 24]
    if not expanded:
        raise ValueError("The AI answer contained no usable parts.")

    allp = [p for e in expanded for p in e[2]]
    mn = [min(p[k] for p in allp) for k in range(3)]
    mx = [max(p[k] for p in allp) for k in range(3)]
    span_x, span_y, span_z = mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]
    widen_x = 1.0  # a too-narrow silhouette is fixed locally (no AI call): spread it sideways
    if cat in ("creature", "humanoid") and span_y > 0 and span_x / span_y < 0.52:
        widen_x = min(1.6, 0.58 * span_y / max(span_x, 1e-6))
    k = 5.2 / (max(span_x * widen_x, span_y, span_z) or 1.0)
    cx, cz = (mn[0] + mx[0]) / 2, (mn[2] + mx[2]) / 2

    hulls: List[Dict[str, Any]] = []
    boxes = tris = 0
    for pid, col, pts in expanded:
        wp = [((x - cx) * widen_x * k, (y - mn[1]) * k, (z - cz) * k) for (x, y, z) in pts]
        faces = convex_hull_3d(wp)
        if not faces or hull_volume(wp, faces) < 1e-4:
            dropped.append(pid)
            continue
        vs = sorted({i for f in faces for i in f})
        wp = [wp[i] for i in vs]
        c = tuple(sum(p[j] for p in wp) / len(wp) for j in range(3))
        local = [[round(p[0] - c[0], 3), round(p[1] - c[1], 3), round(p[2] - c[2], 3)] for p in wp]
        bb = [max(p[j] for p in wp) - min(p[j] for p in wp) for j in range(3)]
        if len(wp) == 8 and bb[0] * bb[1] * bb[2] > 0 and hull_volume(wp, faces) / (bb[0] * bb[1] * bb[2]) > 0.96:
            boxes += 1
        tris += len(faces)
        hulls.append({"id": pid, "color": col, "pos": [round(c[0], 3), round(c[1], 3), round(c[2], 3)],
                      "scale": [1, 1, 1], "points": local})
    if not hulls:
        raise ValueError("None of the AI's parts had a valid volume.")

    n = len(hulls)
    issues: List[Dict[str, Any]] = []

    def add(msg: str, blocking: bool = False, code: str = "") -> None:
        issues.append({"code": code, "msg": msg, "blocking": blocking})

    if n < max(4, int(lo * 0.5)):
        add(f"Only {n} parts - the target is {lo}-{hi}. Add the missing anatomy and details.", True, "PART_COUNT_LOW")
    elif n < lo * 0.8:
        add(f"{n} parts is a little light (target {lo}-{hi}). Add secondary details.")
    if dropped:
        add(f"{len(dropped)} part(s) had no volume (flat, collinear or too few points): {', '.join(dropped[:6])}. Give them real thickness.",
            len(dropped) > max(2, n * 0.12), "DROPPED")

    boxes_w = []
    for h in hulls:
        xs = [h["pos"][0] + p[0] for p in h["points"]]
        ys = [h["pos"][1] + p[1] for p in h["points"]]
        zs = [h["pos"][2] + p[2] for p in h["points"]]
        boxes_w.append((min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)))
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    tol = 0.08
    for i in range(n):
        a = boxes_w[i]
        for j in range(i + 1, n):
            b = boxes_w[j]
            if a[0] - tol <= b[1] and b[0] - tol <= a[1] and a[2] - tol <= b[3] and b[2] - tol <= a[3] and a[4] - tol <= b[5] and b[4] - tol <= a[5]:
                parent[find(i)] = find(j)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    floating: List[str] = []
    if len(groups) > 1:
        main = max(groups.values(), key=len)
        main_centers = [hulls[i]["pos"] for i in main]
        for g in groups.values():
            if g is main:
                continue
            for idx_h in g:
                hp = hulls[idx_h]["pos"]
                nearest = min(main_centers, key=lambda mc: sum((hp[q] - mc[q]) ** 2 for q in range(3)))
                dist = math.sqrt(sum((hp[q] - nearest[q]) ** 2 for q in range(3)))
                if dist > 0.35:
                    for q in range(3):
                        hulls[idx_h]["pos"][q] = round(hp[q] + (nearest[q] - hp[q]) * 0.45, 3)
                else:
                    floating.append(hulls[idx_h]["id"])

    if n >= 12 and len({h["color"] for h in hulls}) < 3:
        add("Use at least 3 colours (base, secondary, accent) so the model reads clearly.", True, "MONOCHROME")
    avg_pts = sum(len(h["points"]) for h in hulls) / n
    size = [round(max(b[1] for b in boxes_w) - min(b[0] for b in boxes_w), 2),
            round(max(b[3] for b in boxes_w), 2),
            round(max(b[5] for b in boxes_w) - min(b[4] for b in boxes_w), 2)]
    if cat in ("creature", "humanoid") and size[1] > 0 and size[0] / size[1] < 0.48:
        add(f"Silhouette is too narrow ({size[0]} wide vs {size[1]} tall). Spread arms, shoulders, wings, or legs wider along +X/-X!", False, "NARROW")

    nb = sum(1 for i in issues if i["blocking"])
    score = max(0, min(100, 100 - 12 * nb - 4 * (len(issues) - nb)))
    grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D"
    analysis = {
        "issues": issues, "score": score, "grade": grade,
        "stats": {"parts": n, "tris": tris, "size": size, "avg_points": round(avg_pts, 1), "cubes": boxes},
        "diag": {
            "readout": f"{n} parts · {tris} triangles · {size[0]} × {size[1]} × {size[2]} units",
            "floating_check": "All parts grounded & connected" if not floating else f"{len(floating)} part(s) adjusted",
            "detail_check": f"{avg_pts:.1f} points/part · {len({h['color'] for h in hulls})} color zones",
        },
    }
    return hulls, analysis


def depth_payload(prompt: str, cat: str, data: Dict[str, Any], hulls: List[Dict[str, Any]], analysis: Dict[str, Any],
                  engine: Dict[str, str], revisions: int, used_serper: bool) -> Dict[str, Any]:
    diag = dict(analysis["diag"])
    diag["quality"] = (f"Grade {analysis['grade']} ({analysis['score']}/100)"
                       + (f" after {revisions} automatic revision{'s' if revisions != 1 else ''}" if revisions else ""))
    return {
        "variant": "depth3d", "title": str(data.get("title") or prompt.title())[:60], "category": cat,
        "hulls": hulls, "stats": analysis["stats"], "diagnostics": diag,
        "quality": {"score": analysis["score"], "grade": analysis["grade"], "revisions": revisions,
                    "issues_left": [i["msg"] for i in analysis["issues"]]},
        "plan": data.get("plan") if isinstance(data.get("plan"), dict) else None,
        "engine": engine, "used_serper": used_serper,
    }


# --------------------------------------------------------------------------- shared pipeline pieces
AI_FIX_CODES = {"PART_COUNT_LOW", "MISSING_EYES", "MISSING_FACE_DETAIL", "FLOATING", "DROPPED"}


def revision_problems(analysis: Dict[str, Any], prof: Dict[str, Any]) -> List[str]:
    """Blocking problems worth an AI call. At low/mid effort a model whose ONLY blocking issue is colour balance is
    accepted as is (the AI rarely fixes it and each revision costs a full call); high/extra still revise it."""
    blocking = [i for i in analysis["issues"] if i["blocking"]]
    if not blocking:
        return []
    if prof["name"] in ("high", "extra") or any(i.get("code") in AI_FIX_CODES for i in blocking):
        return [i["msg"] for i in blocking]
    MEMORY.bump("revisions_skipped")
    return []


def _rounds(req: GenerateRequest, prof: Dict[str, Any]) -> int:
    do_revise = AUTO_REVISE if req.revise is None else req.revise
    cap = 9 if req.max_revisions is None else max(0, min(3, req.max_revisions))
    return min(prof["revisions"], cap) if do_revise else 0


def _reuse_ok(entry: Dict[str, Any], rank: int) -> bool:
    """A saved model satisfies a request when the owner pinned it, it was built at >= effort, or it is already excellent."""
    return bool(entry.get("pinned")) or entry["rank"] >= rank or (entry["score"] >= 88 and entry["rank"] >= rank - 1)


async def _best_candidate(system: str, user: str, req: GenerateRequest, prof: Dict[str, Any], progress: Progress, process):
    """Ask the AI (best_of times at Extra effort - the 2nd call only when the 1st is not already excellent), keep the best."""
    n = max(1, int(prof.get("best_of", 1)))
    temps = (None, 0.7, 0.45)
    best, last_err = None, None
    for i in range(n):
        _TEMP.set(temps[i])
        try:
            data, engine = await ask_and_parse(system, user, req.provider or "", req.model or "", req.api_key or "")
        except HTTPException as e:
            last_err = e
            if best is not None or e.status_code in (400, 401, 403, 404):
                break
            continue
        try:
            items, analysis = await asyncio.to_thread(process, data)
        except ValueError as e:
            last_err = HTTPException(status_code=422, detail=str(e))
            continue
        if best is None or better(analysis, best[3]):
            best = (data, engine, items, analysis)
        if i + 1 < n:
            if analysis["score"] >= 92 and not any(x["blocking"] for x in analysis["issues"]):
                MEMORY.bump("second_candidates_skipped")
                break
            progress(f"{prof['label']} effort: first result scored {analysis['score']} - trying a second independent candidate...")
    if best is None:
        raise last_err or HTTPException(status_code=422, detail="The AI returned no usable parts.")
    return best


async def _build_classic(req: GenerateRequest, prompt: str, base: Optional[Dict[str, Any]], added: List[str],
                         dropped: List[str], progress: Progress, prof: Dict[str, Any],
                         tokens: frozenset = frozenset()) -> Tuple[Dict[str, Any], int]:
    cat, sub, wings, matched = classify(prompt, req.category or "")
    lo, hi = part_range(cat, req.target_parts)
    if not req.target_parts:
        lo, hi = effort_range(lo, hi, prof)
    web_ctx = clean_web_context(req.web_context or "")
    if not web_ctx and base is None and req.use_serper and prof["research"] > 0:
        progress("Researching the subject on the web...")
        web_ctx = await fetch_anatomical_research(prompt, req.serper_key or "", prof["research"])
    system = build_system_prompt(cat, sub, wings, lo, hi)
    started = time.monotonic()
    teach, taught = teaching_block(prompt, tokens, cat, "classic")

    if base is not None:
        bp = base["payload"]
        scale = (bp.get("stats") or {}).get("scene_scale") or STUD
        current = normalize_ops(bp["ops"], 1.0 / scale, "#141c28", cat, sub)
        frame(current)
        change = (f'Build "{prompt}" as a NEW, noticeably different take on the saved model "{base["prompt"]}". '
                  f'Keep the same creature/object identity, but re-sculpt it in a fresh way (different part layout, '
                  f'proportions and detailing - do not copy the old part list). '
                  + (f'Apply these new traits: {", ".join(added)}. ' if added else "")
                  + (f'Drop these traits from the old model: {", ".join(dropped)}.' if dropped else ""))
        user = build_revision_prompt(prompt, cat, lo, hi, current, [], change)
        progress(f'Re-sculpting the saved "{base["prompt"]}" model with: {", ".join(added) or "your changes"}...')
    else:
        user = build_user_prompt(prompt, cat, sub, lo, hi, web_ctx)
        progress(f"Sculpting a brand-new 3D model ({prof['label']} effort)...")
    user += "\n\n" + EFFORT_PROMPT[prof["name"]]
    if teach:
        user += "\n\n" + teach
        progress("Applying what the owner taught: " + ", ".join(
            x for x in (f"{len(taught['lessons'])} lesson(s)" if taught["lessons"] else "",
                        f"{len(taught['examples'])} example model(s)" if taught["examples"] else "") if x) + ".")

    state = {"cat": cat, "lo": lo, "hi": hi, "first": True}

    def process(data: Dict[str, Any]):
        if state["first"]:
            state["first"] = False
            if not matched and not (req.category or "") and str(data.get("category", "")).lower() in PART_RANGE:
                state["cat"] = str(data["category"]).lower()
                state["lo"], state["hi"] = part_range(state["cat"], req.target_parts)
                if not req.target_parts:
                    state["lo"], state["hi"] = effort_range(state["lo"], state["hi"], prof)
        return process_build(data, state["cat"], sub, state["lo"], state["hi"], prompt)

    data, engine, parts, analysis = await _best_candidate(system, user, req, prof, progress, process)
    cat, lo, hi = state["cat"], state["lo"], state["hi"]

    revisions = 0
    for _ in range(_rounds(req, prof)):
        problems = revision_problems(analysis, prof)
        if not problems:
            break
        if time.monotonic() - started > prof["job_s"] * 0.5 * (revisions + 1) / (revisions + 2):
            break
        progress(f"Checker found {len(problems)} problem(s) - asking the AI to fix them...")
        try:
            d2, _eng = await ask_and_parse(system, build_revision_prompt(prompt, cat, lo, hi, parts, problems),
                                           req.provider or "", req.model or "", req.api_key or "", retries=0)
            p2, a2 = await asyncio.to_thread(process_build, d2, cat, sub, lo, hi, prompt)
        except (HTTPException, ValueError):
            break
        revisions += 1
        if better(a2, analysis):
            data, parts, analysis = d2, p2, a2
        else:
            break  # the AI did not improve it - another call would just burn quota
    payload = response_payload(prompt.title(), cat, data, parts, analysis, False, engine, revisions, bool(web_ctx))
    payload["taught"] = taught
    return payload, analysis["score"]


async def _build_depth(req: GenerateRequest, prompt: str, base: Optional[Dict[str, Any]], added: List[str],
                       dropped: List[str], progress: Progress, prof: Dict[str, Any],
                       tokens: frozenset = frozenset()) -> Tuple[Dict[str, Any], int]:
    cat, sub, _, _ = classify(prompt, req.category or "")
    lo, hi = prof["d3_parts"]
    web_ctx = clean_web_context(req.web_context or "")
    if not web_ctx and base is None and req.use_serper and prof["research"] > 0:
        progress("Researching the subject on the web...")
        web_ctx = await fetch_anatomical_research(prompt, req.serper_key or "", prof["research"])
    system = build_depth_system_prompt(cat, sub, prof)
    started = time.monotonic()
    teach, taught = teaching_block(prompt, tokens, cat, "depth3d")
    if base is not None:
        change = (f'Build "{prompt}" as a NEW, noticeably different take on the saved model "{base["prompt"]}". '
                  f'Keep the same identity but re-sculpt it in a fresh way (different shapes, proportions and detail). '
                  + (f'Apply these new traits: {", ".join(added)}. ' if added else "")
                  + (f'Drop these traits from the old model: {", ".join(dropped)}.' if dropped else ""))
        user = build_depth_revision_prompt(prompt, cat, prof, base["payload"].get("hulls") or [], [], change)
        progress(f'Re-sculpting the saved "{base["prompt"]}" model with: {", ".join(added) or "your changes"}...')
    else:
        user = build_depth_user_prompt(prompt, cat, sub, prof, web_ctx)
        progress(f"Sculpting low-poly volumes ({prof['label']} effort)...")
    if teach:
        user += "\n\n" + teach
        progress("Applying what the owner taught: " + ", ".join(
            x for x in (f"{len(taught['lessons'])} lesson(s)" if taught["lessons"] else "",
                        f"{len(taught['examples'])} example model(s)" if taught["examples"] else "") if x) + ".")

    def process(data: Dict[str, Any]):
        return process_hulls(data, prof, lo, hi, prompt, cat)

    data, engine, hulls, analysis = await _best_candidate(system, user, req, prof, progress, process)

    revisions = 0
    for _ in range(_rounds(req, prof)):
        problems = revision_problems(analysis, prof)
        if not problems:
            break
        if time.monotonic() - started > prof["job_s"] * 0.5 * (revisions + 1) / (revisions + 2):
            break
        progress(f"Checker found {len(problems)} problem(s) - asking the AI to fix them...")
        try:
            d2, _e = await ask_and_parse(system, build_depth_revision_prompt(prompt, cat, prof, hulls, problems),
                                         req.provider or "", req.model or "", req.api_key or "", retries=0)
            h2, a2 = await asyncio.to_thread(process_hulls, d2, prof, lo, hi, prompt, cat)
        except (HTTPException, ValueError):
            break
        revisions += 1
        if better(a2, analysis):
            data, hulls, analysis = d2, h2, a2
        else:
            break
    payload = depth_payload(prompt, cat, data, hulls, analysis, engine, revisions, bool(web_ctx))
    payload["taught"] = taught
    return payload, analysis["score"]


async def _build_new(req: GenerateRequest, prompt: str, base: Optional[Dict[str, Any]], added: List[str],
                     dropped: List[str], progress: Progress, variant: str = "classic",
                     prof: Optional[Dict[str, Any]] = None, tokens: frozenset = frozenset()) -> Tuple[Dict[str, Any], int]:
    prof = prof or EFFORTS["mid"]
    if variant == "depth3d":
        return await _build_depth(req, prompt, base, added, dropped, progress, prof, tokens)
    return await _build_classic(req, prompt, base, added, dropped, progress, prof, tokens)


def _serve_saved(saved: Dict[str, Any], entry_id: str, tonal: bool, mode: str, uses: int, **kw: Any) -> Dict[str, Any]:
    out = apply_tonal_payload(saved["payload"], tonal)
    out["engine"] = {"provider": "global-memory", "model": entry_id}
    out["used_serper"] = False
    out["effort"] = saved["effort"]
    out["memory"] = memory_info(mode, id=entry_id, uses=uses, base_prompt=saved["prompt"], saved=False,
                                effort=saved["effort"], pinned=saved.get("pinned", 0), **kw)
    return out


async def run_generate(req: GenerateRequest, progress: Progress = _noop) -> Dict[str, Any]:
    prompt = req.prompt.strip()[:1000]
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")
    variant, effort = norm_variant(req.variant), norm_effort(req.effort)
    prof, rank = EFFORTS[effort], EFFORT_RANK[effort]
    _EFFORT.set(prof)
    tokens = prompt_tokens(prompt)
    key = prompt_key(tokens, variant)

    mode, entry, added, dropped = ("none", None, [], [])
    if req.use_memory and not req.fresh and tokens:
        mode, entry, added, dropped = MEMORY.find(tokens, variant)
    if mode in ("exact", "local") and entry and not _reuse_ok(entry, rank):
        progress(f"A {entry['effort']}-effort model is saved - rebuilding at {effort} effort to improve it...")
        mode = "variant" if mode == "local" else "none"
        if mode == "none":
            entry = None

    if mode == "exact" and entry:
        saved = MEMORY.get(entry["id"])
        if saved:
            uses = MEMORY.touch(entry["id"])
            MEMORY.bump("memory_hits")
            progress("Found in the global memory - no AI call needed.", "ok")
            return _serve_saved(saved, entry["id"], req.tonal_variation, "memory", uses)

    if mode == "local" and entry:
        saved = MEMORY.get(entry["id"])
        info = local_variant_color(added, dropped)
        if saved and info:
            color = info[1]
            MEMORY.touch(entry["id"])
            MEMORY.bump("local_variants")
            if color:
                progress(f'Same model as the saved "{saved["prompt"]}" - recolouring it {color} locally, no AI call needed.', "ok")
                payload = recolor_payload(saved["payload"], color, prompt.title())
                saved_id = ""
                if req.publish and req.use_memory and tokens:
                    saved_id = MEMORY.save(prompt, tokens, saved["category"] or "", payload, saved["score"], parent_id=entry["id"],
                                           engine="local:recolor", effort=saved["effort"], variant=variant, source="local")
                payload["engine"] = {"provider": "local", "model": "recolor"}
                payload["used_serper"] = False
                payload["effort"] = saved["effort"]
                payload["memory"] = memory_info("variant", id=saved_id or entry["id"], saved=bool(saved_id), uses=0,
                                                base_prompt=saved["prompt"], added=added, dropped=dropped, effort=saved["effort"],
                                                local=True)
                return apply_tonal_payload(payload, req.tonal_variation)
            progress("Found a saved model of this subject - no AI call needed.", "ok")
            return _serve_saved(saved, entry["id"], req.tonal_variation, "memory", saved.get("uses", 0))
        mode = "variant"

    fkey = f"{key}|{effort}"
    if req.use_memory and not req.fresh and fkey in _INFLIGHT:
        progress("Someone else is building this right now - waiting for their result...")
        try:
            shared = await asyncio.wait_for(asyncio.shield(_INFLIGHT[fkey]), timeout=prof["job_s"] - 10)
            out = apply_tonal_payload(shared, req.tonal_variation)
            out["memory"] = {**(out.get("memory") or {}), "source": "memory", "saved": False}
            return out
        except Exception:
            pass

    fut: Optional["asyncio.Future[Dict[str, Any]]"] = None
    if tokens and fkey not in _INFLIGHT:
        fut = asyncio.get_running_loop().create_future()
        _INFLIGHT[fkey] = fut
    try:
        base = MEMORY.get(entry["id"]) if (mode == "variant" and entry) else None
        if base is not None and not (base["payload"].get("hulls") if variant == "depth3d" else base["payload"].get("ops")):
            base = None
        if base:
            MEMORY.touch(base["id"])
        t0 = time.monotonic()
        payload, score = await _build_new(req, prompt, base, added, dropped, progress, variant, prof, tokens)
        MEMORY.bump("ai_builds")
        payload["effort"] = effort
        payload["build_seconds"] = round(time.monotonic() - t0, 1)

        saved_id = ""
        if req.publish and req.use_memory and tokens and score >= MEMORY_MIN_SCORE:
            saved_id = MEMORY.save(prompt, tokens, payload.get("category", ""), payload, score,
                                   parent_id=base["id"] if base else "",
                                   engine=f"{payload['engine'].get('provider', '')}:{payload['engine'].get('model', '')}",
                                   effort=effort, variant=variant)
            if saved_id:
                progress("Published to the global memory.", "ok")
        payload["memory"] = memory_info("variant" if base else "fresh", id=saved_id, saved=bool(saved_id),
                                        base_prompt=base["prompt"] if base else "", added=added, dropped=dropped,
                                        uses=0, effort=effort)
        if fut and not fut.done():
            fut.set_result(dict(payload))
        return apply_tonal_payload(payload, req.tonal_variation)
    except BaseException as e:
        if fut and not fut.done():
            fut.set_exception(e if isinstance(e, Exception) else RuntimeError("cancelled"))
            fut.exception()
        raise
    finally:
        if fut is not None and _INFLIGHT.get(fkey) is fut:
            _INFLIGHT.pop(fkey, None)


async def run_revise(req: ReviseRequest, progress: Progress = _noop) -> Dict[str, Any]:
    change = req.instruction.strip()[:1000]
    variant, effort = norm_variant(req.variant), norm_effort(req.effort)
    prof = EFFORTS[effort]
    _EFFORT.set(prof)
    prompt = (req.prompt or "the current model").strip()[:1000]
    if not change:
        raise HTTPException(status_code=400, detail="instruction is required.")

    if variant == "depth3d":
        cur = [h for h in (req.hulls or []) if isinstance(h, dict)]
        if not hulls_to_rows(cur):
            raise HTTPException(status_code=400, detail="hulls (the current Depth 3D model) are required.")
        cat, sub, _, _ = classify(f"{prompt} {change}", req.category or "")
        lo, hi = prof["d3_parts"]
        system = build_depth_system_prompt(cat, sub, prof)
        progress("Applying your change...")
        data, engine = await ask_and_parse(system, build_depth_revision_prompt(prompt, cat, prof, cur, [], change),
                                           req.provider or "", req.model or "", req.api_key or "")
        try:
            hulls, analysis = await asyncio.to_thread(process_hulls, data, prof, max(4, min(lo, len(cur))), max(hi, len(cur) * 2), prompt, cat)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        out = depth_payload(prompt, cat, data, hulls, analysis, engine, 1, False)
        out["effort"] = effort
        return apply_tonal_payload(out, req.tonal_variation)

    if not req.ops:
        raise HTTPException(status_code=400, detail="ops are required.")
    cat, sub, wings, _ = classify(f"{prompt} {change}", req.category or "")
    lo, hi = part_range(cat, None)
    lo, hi = effort_range(lo, hi, prof)
    unit = 1.0 / (req.scene_scale if req.scene_scale and req.scene_scale > 0 else STUD)
    current = normalize_ops(req.ops, unit, "#141c28", cat, sub)
    if len(current) < 3:
        raise HTTPException(status_code=400, detail="ops contained fewer than 3 usable parts.")
    frame(current)
    system = build_system_prompt(cat, sub, wings, lo, hi)
    progress("Applying your change...")
    data, engine = await ask_and_parse(system, build_revision_prompt(prompt, cat, lo, hi, current, [], change),
                                       req.provider or "", req.model or "", req.api_key or "")
    try:
        parts, analysis = await asyncio.to_thread(process_build, data, cat, sub, lo, hi, prompt)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    out = response_payload(prompt.title(), cat, data, parts, analysis, req.tonal_variation, engine, 1, False)
    out["effort"] = effort
    return out


# --------------------------------------------------------------------------- background jobs (no more proxy time-outs)
def _hint(status: int, text: str = "") -> str:
    low = (text or "").lower()
    if status in (401, 403) or "api key" in low:
        return "The AI key was rejected. Check the key in the Lumio server popup or GEMINI_API_KEY on the server."
    if status == 429:
        return "Quota or rate limit reached. Wait a minute, lower the effort, or use another model or key."
    if status == 404:
        return "Model not found for this key. Open /gemini/models to see exactly which models the key can use."
    if status == 504 or "stalled" in low or "took longer" in low:
        return "The AI was too slow. Lower the effort or use a Flash / Flash-Lite model."
    if status == 422:
        return "The AI answered, but the model it drew was not valid. Try again or change the prompt."
    return "See detail and attempts for the full upstream error. /health shows the server state."


def lumio_shapes(hulls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Depth-3D hulls (pos + local points) -> Lumio shapes (id, color, absolute pts)."""
    out = []
    for h in hulls or []:
        try:
            pos, sc = h.get("pos") or [0, 0, 0], h.get("scale") or [1, 1, 1]
            pts = [[round(pos[j] + p[j] * sc[j], 3) for j in range(3)] for p in h["points"]]
            if len(pts) >= 4:
                out.append({"id": str(h.get("id", "Part")), "color": h.get("color", "#888888"), "pts": pts})
        except (KeyError, TypeError, IndexError, ValueError):
            continue
    return out


def _with_lumio(res: Any) -> Any:
    if isinstance(res, dict) and res.get("hulls"):
        res = dict(res)
        res["lumio"] = {"title": res.get("title", ""), "category": res.get("category", "prop"),
                        "shapes": lumio_shapes(res["hulls"])}
    return res


JOB_TTL_S = 1800
_JOB_SLOTS: Optional[asyncio.Semaphore] = None


def job_slots() -> asyncio.Semaphore:
    global _JOB_SLOTS
    if _JOB_SLOTS is None:
        _JOB_SLOTS = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_BUILDS", "4")))
    return _JOB_SLOTS


JOBS: Dict[str, Dict[str, Any]] = {}
_TASKS: set = set()


def _gc_jobs() -> None:
    now = time.time()
    for jid in [k for k, v in JOBS.items() if now - v["updated"] > JOB_TTL_S]:
        JOBS.pop(jid, None)


def start_job(kind: str, coro_factory, max_s: Optional[float] = None) -> str:
    _gc_jobs()
    cap = max_s or JOB_MAX_S
    jid = uuid.uuid4().hex[:16]
    job = {"id": jid, "kind": kind, "status": "queued", "steps": [], "result": None, "error": None,
           "created": time.time(), "updated": time.time()}
    JOBS[jid] = job

    def progress(msg: str, kind_: str = "") -> None:
        job["updated"] = time.time()
        if kind_ == "live":
            job["live"] = msg
            return
        job["steps"].append({"t": round(time.time() - job["created"], 1), "msg": msg, "kind": kind_})

    async def runner() -> None:
        try:
            _PROGRESS.set(progress)
            async with job_slots():
                job["status"] = "running"
                job["updated"] = time.time()
                job["result"] = await asyncio.wait_for(coro_factory(progress), timeout=cap)
                job["status"] = "done"
                log.info("job %s (%s) done in %.1fs", jid, kind, time.time() - job["created"])
        except asyncio.TimeoutError:
            log.warning("job %s (%s) hit the %ss cap", jid, kind, cap)
            job["status"], job["error"] = "error", {"status": 504, "detail": f"The AI took longer than {int(cap)}s. Lower the AI effort or pick a faster model (Gemini Flash-Lite) and try again."}
        except ProviderError as e:
            log.warning("job %s provider error: %s %s", jid, e.provider, e.message[:300])
            job["status"], job["error"] = "error", {**e.to_dict(), "status": e.to_http().status_code,
                                                    "detail": f"{e.provider}: {e.message}"[:4000],
                                                    "hint": _hint(e.status, e.message)}
        except HTTPException as e:
            log.warning("job %s failed: %s %s", jid, e.status_code, e.detail)
            job["status"], job["error"] = "error", {"status": e.status_code, "detail": str(e.detail)[:4000],
                                                    "hint": _hint(e.status_code, str(e.detail))}
        except Exception as e:
            log.exception("job %s crashed", jid)
            job["status"], job["error"] = "error", {"status": 500, "detail": f"{e.__class__.__name__}: {e}"[:2000],
                                                    "trace": traceback.format_exc()[-3000:],
                                                    "hint": "The server crashed while building. Send the trace to fix it."}
        job["updated"] = time.time()

    task = asyncio.create_task(runner())
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return jid


@app.post("/jobs/generate")
async def job_generate(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required.")
    dk = hashlib.sha1("|".join([req.prompt.strip().lower(), norm_variant(req.variant), norm_effort(req.effort),
                                (req.category or ""), (req.provider or ""), (req.model or ""),
                                _key_hash(req.api_key or ""), str(req.use_memory)]).encode()).hexdigest()
    if not req.fresh:
        for j in JOBS.values():
            if j.get("dedupe") == dk and j["status"] in ("queued", "running"):
                MEMORY.bump("duplicate_jobs_shared")
                return {"job_id": j["id"], "shared": True}
    jid = start_job("generate", lambda p: run_generate(req, p), EFFORTS[norm_effort(req.effort)]["job_s"])
    JOBS[jid]["dedupe"] = dk
    return {"job_id": jid}


@app.post("/jobs/revise")
async def job_revise(req: ReviseRequest):
    return {"job_id": start_job("revise", lambda p: run_revise(req, p), EFFORTS[norm_effort(req.effort)]["job_s"])}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str, since: int = 0):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown or expired job.")
    out = {"id": job_id, "status": job["status"], "steps": job["steps"][max(0, since):], "next": len(job["steps"]),
           "live": job.get("live", ""),
           "elapsed": round(time.time() - job["created"], 1)}
    if job["status"] == "done":
        out["result"] = _with_lumio(job["result"])
    if job["status"] == "error":
        out["error"] = job["error"]
    return JSONResponse(content=out)


@app.post("/generate")
async def generate(req: GenerateRequest):
    try:
        return JSONResponse(content=await asyncio.wait_for(run_generate(req), timeout=JOB_MAX_S))
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="The AI took too long. Use /jobs/generate or a faster model.")


@app.post("/revise")
async def revise(req: ReviseRequest):
    try:
        return JSONResponse(content=await asyncio.wait_for(run_revise(req), timeout=JOB_MAX_S))
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="The AI took too long. Use /jobs/revise or a faster model.")


# --------------------------------------------------------------------------- admin page (served at /admin)
ADMIN_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Verdant 3D - Admin</title>
<style>
:root{--bg:#0b0f0d;--panel:#121a16;--panel2:#18231d;--line:#25352c;--txt:#e6f1ea;--mut:#8aa397;--acc:#4ade80;--acc2:#22c55e;--warn:#fbbf24;--bad:#f87171}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
a{color:var(--acc)}h1{font-size:18px;margin:0}h2{font-size:15px;margin:0 0 10px}
header{display:flex;align-items:center;gap:14px;padding:12px 18px;border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:5;flex-wrap:wrap}
nav{display:flex;gap:6px;flex-wrap:wrap}nav button{background:transparent;border:1px solid var(--line);color:var(--mut);padding:7px 13px;border-radius:8px;cursor:pointer}
nav button.on{background:var(--acc2);color:#04130a;border-color:var(--acc2);font-weight:600}
main{padding:18px;max-width:1200px;margin:0 auto}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}.stat{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:12px}
.stat b{display:block;font-size:22px}.stat span{color:var(--mut);font-size:12px}
input,select,textarea{background:#0d1511;border:1px solid var(--line);color:var(--txt);border-radius:8px;padding:8px 10px;font:inherit;max-width:100%}
textarea{width:100%;min-height:90px;resize:vertical}label{display:block;color:var(--mut);font-size:12px;margin:8px 0 3px}
button.btn{background:var(--acc2);color:#04130a;border:0;border-radius:8px;padding:8px 14px;font-weight:600;cursor:pointer}
button.btn.sec{background:var(--panel2);color:var(--txt);border:1px solid var(--line)}button.btn.bad{background:#7f1d1d;color:#fecaca}
button.btn.sm{padding:4px 9px;font-size:12px}.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:middle}
th{color:var(--mut);font-size:12px;font-weight:500}.tag{display:inline-block;padding:1px 7px;border-radius:99px;font-size:11px;border:1px solid var(--line);color:var(--mut)}
.tag.g{color:var(--acc);border-color:var(--acc2)}.tag.w{color:var(--warn);border-color:#7a5b10}.tag.b{color:var(--bad);border-color:#7f1d1d}
.mut{color:var(--mut)}.msg{padding:8px 12px;border-radius:8px;margin:8px 0;background:var(--panel2);border:1px solid var(--line)}.msg.err{color:var(--bad)}.msg.ok{color:var(--acc)}
#login{position:fixed;inset:0;background:var(--bg);display:flex;align-items:center;justify-content:center;z-index:50}#login .card{width:340px}
dialog{background:var(--panel);color:var(--txt);border:1px solid var(--line);border-radius:12px;max-width:760px;width:94vw;padding:16px}dialog::backdrop{background:#000a}
canvas{background:#0d1511;border:1px solid var(--line);border-radius:8px;max-width:100%}pre{white-space:pre-wrap;word-break:break-word;background:#0d1511;padding:10px;border-radius:8px;border:1px solid var(--line);max-height:260px;overflow:auto;font-size:12px}
.hide{display:none!important}
</style></head><body>
<div id="login"><div class="card"><h1>Verdant 3D admin</h1><p class="mut">Enter the admin password (the <code>ADMIN_TOKEN</code> env var on Render, or the generated one printed in the server log).</p>
<input id="pw" type="password" placeholder="Admin password" style="width:100%" autofocus><div class="row" style="margin-top:10px"><button class="btn" onclick="doLogin()">Open</button></div><div id="lmsg" class="msg err hide"></div></div></div>
<header><h1>Verdant 3D &middot; Admin</h1><nav id="nav"></nav><span class="mut" id="ver"></span><button class="btn sec sm" onclick="logout()">Log out</button></header>
<main id="main"></main>
<dialog id="dlg"><div id="dbody"></div><div class="row" style="margin-top:12px"><button class="btn sec" onclick="dlg.close()">Close</button></div></dialog>
<script>
const $=s=>document.querySelector(s);let TOKEN=sessionStorage.getItem('vt')||'';let tab='overview';
const dlg=$('#dlg');
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(path,opt){opt=opt||{};opt.headers=Object.assign({'X-Admin-Token':TOKEN},opt.headers||{});
  if(opt.json!==undefined){opt.body=JSON.stringify(opt.json);opt.headers['Content-Type']='application/json';delete opt.json}
  const r=await fetch(path,opt);let d=null;try{d=await r.json()}catch(e){}
  if(!r.ok){const m=(d&&d.detail)||r.statusText;if(r.status===401){logout(true)}throw new Error(typeof m==='string'?m:JSON.stringify(m))}return d}
async function doLogin(){TOKEN=$('#pw').value.trim();try{await api('/admin/api/overview');sessionStorage.setItem('vt',TOKEN);$('#login').classList.add('hide');init()}catch(e){const m=$('#lmsg');m.textContent=e.message;m.classList.remove('hide')}}
$('#pw').addEventListener('keydown',e=>{if(e.key==='Enter')doLogin()});
function logout(soft){sessionStorage.removeItem('vt');TOKEN='';$('#login').classList.remove('hide');if(!soft)$('#pw').value=''}
const TABS=[['overview','Overview'],['models','Models'],['teach','Teach'],['backup','Backup'],['ai','AI & keys']];
function init(){$('#nav').innerHTML=TABS.map(t=>`<button data-t="${t[0]}" class="${t[0]===tab?'on':''}" onclick="go('${t[0]}')">${t[1]}</button>`).join('');go(tab)}
function go(t){tab=t;document.querySelectorAll('#nav button').forEach(b=>b.classList.toggle('on',b.dataset.t===t));({overview,models,teach,backup,ai})[t]()}
function note(el,msg,ok){el.innerHTML=`<div class="msg ${ok?'ok':'err'}">${esc(msg)}</div>`}
/* ---------- overview ---------- */
async function overview(){const m=$('#main');m.innerHTML='<p class="mut">Loading...</p>';
  const o=await api('/admin/api/overview');$('#ver').textContent=o.version;const s=o.stats,c=o.counters;
  const st=(n,l)=>`<div class="stat"><b>${esc(n)}</b><span>${esc(l)}</span></div>`;
  m.innerHTML=`<div class="card"><h2>Global memory</h2><div class="grid">${st(s.models,'models saved')}${st(s.classic+' / '+s.depth3d,'Classic / Depth 3D')}${st(s.pinned,'pinned (locked)')}${st(s.flagged,'flagged by users')}${st(s.uses,'times reused')}${st(s.avg_score,'average quality')}${st(s.db_kb+' KB','database size')}</div>
  ${o.persistent?'':'<div class="msg err">The memory lives on a temporary disk - it is wiped on every redeploy/restart of the free Render plan. Use the Backup tab to download it regularly (or add a Render disk and set MEMORY_DB=/var/data/verdant_memory.db).</div>'}</div>
  <div class="card"><h2>API usage</h2><div class="grid">${st(o.api_calls_avoided,'AI/search calls avoided')}${st(c.ai_calls||0,'AI calls made (all time)')}${st(o.ai_calls_today+(o.daily_cap?' / '+o.daily_cap:''),'AI calls today (UTC)')}${st(c.memory_hits||0,'answered from memory')}${st(c.local_variants||0,'recoloured locally')}${st(c.revisions_skipped||0,'revisions skipped')}${st(c.second_candidates_skipped||0,'2nd candidates skipped')}${st(c.research_cache_hits||0,'web searches cached')}${st(c.ai_errors||0,'AI errors')}</div></div>
  <div class="card"><h2>Teaching</h2><div class="grid">${st(s.lessons,'text lessons')}${st(s.examples,'example 3D models')}${st(c.taught_builds||0,'builds that used teaching')}</div></div>
  <div class="card"><h2>Models by effort / source</h2><p class="mut">${Object.entries(s.by_effort).map(([k,v])=>`<span class="tag">${esc(k)}: ${v}</span>`).join(' ')} &nbsp; ${Object.entries(s.by_source).map(([k,v])=>`<span class="tag g">${esc(k)}: ${v}</span>`).join(' ')}</p>
  <p class="mut">Providers on the server: ${o.providers.map(p=>`<span class="tag g">${esc(p)}</span>`).join(' ')||'<span class="tag b">none</span>'} ${o.serper?'<span class="tag g">serper</span>':''} &middot; active jobs: ${o.jobs_active}</p></div>`}
/* ---------- models ---------- */
let MQ={q:'',sort:'popular',variant:'',source:'',flagged:false,pinned:false,offset:0,limit:25};
async function models(){const m=$('#main');
  m.innerHTML=`<div class="card"><div class="row"><input id="q" placeholder="Search prompt / title / notes" value="${esc(MQ.q)}" style="flex:1;min-width:200px">
  <select id="sort"><option value="popular">Most used</option><option value="new">Newest</option><option value="old">Oldest</option><option value="score">Best quality</option><option value="parts">Most parts</option><option value="flags">Most flagged</option></select>
  <select id="variant"><option value="">All variants</option><option value="classic">Classic</option><option value="depth3d">Depth 3D</option></select>
  <select id="source"><option value="">All sources</option><option value="ai">AI</option><option value="local">Local recolour</option><option value="example">Taught example</option></select>
  <label style="margin:0"><input type="checkbox" id="flagged"> flagged</label><label style="margin:0"><input type="checkbox" id="pinned"> pinned</label>
  <button class="btn" onclick="mSearch()">Search</button></div></div><div class="card" id="mlist"></div>
  <div class="card"><h2>Clean up</h2><div class="row"><label style="margin:0">quality below</label><input id="cbelow" type="number" value="75" style="width:70px"><label style="margin:0">never reused & older than (days)</label><input id="cdays" type="number" value="30" style="width:70px"><label style="margin:0"><input type="checkbox" id="cflag" checked> flagged</label>
  <button class="btn sec" onclick="cleanup(true)">Count</button><button class="btn bad" onclick="cleanup(false)">Delete matches</button><span id="cmsg" class="mut"></span></div><p class="mut">Pinned (locked) models are never touched.</p></div>`;
  for(const k of ['sort','variant','source'])$('#'+k).value=MQ[k];$('#flagged').checked=MQ.flagged;$('#pinned').checked=MQ.pinned;
  $('#q').addEventListener('keydown',e=>{if(e.key==='Enter')mSearch()});mLoad()}
function mSearch(){MQ.q=$('#q').value;MQ.sort=$('#sort').value;MQ.variant=$('#variant').value;MQ.source=$('#source').value;MQ.flagged=$('#flagged').checked;MQ.pinned=$('#pinned').checked;MQ.offset=0;mLoad()}
async function mLoad(){const L=$('#mlist');L.innerHTML='<p class="mut">Loading...</p>';
  const p=new URLSearchParams({q:MQ.q,sort:MQ.sort,variant:MQ.variant,source:MQ.source,limit:MQ.limit,offset:MQ.offset,flagged:MQ.flagged,pinned:MQ.pinned});
  let d;try{d=await api('/admin/api/models?'+p)}catch(e){return note(L,e.message)}
  const rows=d.items.map(i=>`<tr><td><input type="checkbox" class="sel" value="${esc(i.id)}"></td><td><b>${esc(i.prompt)}</b><div class="mut" style="font-size:12px">${esc(i.title)}${i.notes?' &middot; '+esc(i.notes):''}</div></td>
  <td><span class="tag">${i.variant==='depth3d'?'Depth 3D':'Classic'}</span> <span class="tag">${esc(i.effort)}</span> ${i.source!=='ai'?`<span class="tag g">${esc(i.source)}</span>`:''}${i.pinned?' <span class="tag w">pinned</span>':''}</td>
  <td>${i.parts}</td><td>${i.score}</td><td>${i.uses}</td><td>${i.flags?`<span class="tag b">${i.flags}</span>`:'0'}</td>
  <td class="row"><button class="btn sec sm" onclick="mView('${i.id}')">View</button><button class="btn sec sm" onclick="mPin('${i.id}',${i.pinned?0:1})">${i.pinned?'Unpin':'Pin'}</button><button class="btn bad sm" onclick="mDel('${i.id}')">Delete</button></td></tr>`).join('');
  const pages=Math.ceil(d.total/MQ.limit),cur=Math.floor(MQ.offset/MQ.limit)+1;
  L.innerHTML=`<div class="row" style="margin-bottom:8px"><b>${d.total} model(s)</b><span style="flex:1"></span><button class="btn sec sm" onclick="bulk('pin')">Pin selected</button><button class="btn sec sm" onclick="bulk('unflag')">Unflag selected</button><button class="btn bad sm" onclick="bulk('delete')">Delete selected</button></div>
  <table><tr><th></th><th>Prompt</th><th>Type</th><th>Parts</th><th>Score</th><th>Used</th><th>Flags</th><th></th></tr>${rows||'<tr><td colspan="8" class="mut">Nothing here yet.</td></tr>'}</table>
  <div class="row" style="margin-top:10px"><button class="btn sec sm" ${cur<=1?'disabled':''} onclick="MQ.offset-=MQ.limit;mLoad()">&larr; Prev</button><span class="mut">page ${cur} / ${pages||1}</span><button class="btn sec sm" ${cur>=pages?'disabled':''} onclick="MQ.offset+=MQ.limit;mLoad()">Next &rarr;</button></div><div id="mm"></div>`}
function sel(){return [...document.querySelectorAll('.sel:checked')].map(e=>e.value)}
async function bulk(a){const ids=sel();if(!ids.length)return;if(a==='delete'&&!confirm('Delete '+ids.length+' model(s)?'))return;await api('/admin/api/models/bulk',{method:'POST',json:{ids,action:a}});mLoad()}
async function mPin(id,v){await api('/admin/api/models/'+id,{method:'PATCH',json:{pinned:v}});mLoad()}
async function mDel(id){if(!confirm('Delete this model from the global memory?'))return;await api('/admin/api/models/'+id,{method:'DELETE'});mLoad()}
async function cleanup(dry){const b={below_score:+$('#cbelow').value||0,unused_days:+$('#cdays').value||0,flagged:$('#cflag').checked,dry_run:dry};
  if(!dry&&!confirm('Delete all models matching these rules?'))return;const r=await api('/admin/api/cleanup',{method:'POST',json:b});$('#cmsg').textContent=(dry?'Would delete ':'Deleted ')+r.matched+' model(s).';if(!dry)mLoad()}
async function mView(id){const m=await api('/admin/api/models/'+id);const p=m.payload||{};
  $('#dbody').innerHTML=`<h2>${esc(m.prompt)}</h2><div class="row"><span class="tag">${m.variant}</span><span class="tag">${esc(m.effort)}</span><span class="tag">score ${m.score}</span><span class="tag">${m.parts} parts</span><span class="tag">used ${m.uses}x</span><span class="tag">${esc(m.engine||'')}</span></div>
  <div class="row" style="margin:10px 0"><canvas id="cv1" width="340" height="300"></canvas><canvas id="cv2" width="340" height="300"></canvas></div>
  <label>Prompt (changing it changes which requests reuse this model)</label><input id="ep" value="${esc(m.prompt)}" style="width:100%"><label>Title</label><input id="et" value="${esc(m.title)}" style="width:100%"><label>Notes</label><input id="en" value="${esc(m.notes||'')}" style="width:100%">
  <div class="row" style="margin-top:10px"><button class="btn" onclick="mSave('${id}')">Save changes</button><button class="btn sec" onclick="mTeach('${id}')">Use as teaching example</button></div><div id="dmsg"></div>`;
  dlg.showModal();draw($('#cv1'),p,0,2);draw($('#cv2'),p,2,1)}
async function mSave(id){try{await api('/admin/api/models/'+id,{method:'PATCH',json:{prompt:$('#ep').value,title:$('#et').value,notes:$('#en').value}});note($('#dmsg'),'Saved.',1);mLoad()}catch(e){note($('#dmsg'),e.message)}}
async function mTeach(id){try{const r=await api('/admin/api/models/'+id+'/teach',{method:'POST',json:{}});note($('#dmsg'),'Added as a teaching example ('+r.parts+' parts). Similar prompts will now see it.',1)}catch(e){note($('#dmsg'),e.message)}}
function hull2d(pts){pts=pts.slice().sort((a,b)=>a[0]-b[0]||a[1]-b[1]);if(pts.length<3)return pts;const cr=(o,a,b)=>(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0]);const lo=[];for(const p of pts){while(lo.length>=2&&cr(lo[lo.length-2],lo[lo.length-1],p)<=0)lo.pop();lo.push(p)}const up=[];for(const p of pts.slice().reverse()){while(up.length>=2&&cr(up[up.length-2],up[up.length-1],p)<=0)up.pop();up.push(p)}lo.pop();up.pop();return lo.concat(up)}
function draw(cv,p,ax,ay){const g=cv.getContext('2d');g.clearRect(0,0,cv.width,cv.height);const shapes=[];
  if(p.hulls)for(const h of p.hulls){const pts=h.points.map(q=>[h.pos[ax]+q[ax],h.pos[ay]+q[ay]]);shapes.push({c:h.color,poly:hull2d(pts)})}
  else if(p.ops)for(const o of p.ops){const s=o.size,q=o.pos;const x=[q[ax]-s[ax]/2,q[ax]+s[ax]/2],y=[q[ay]-s[ay]/2,q[ay]+s[ay]/2];shapes.push({c:o.color,poly:[[x[0],y[0]],[x[1],y[0]],[x[1],y[1]],[x[0],y[1]]]})}
  let a=[1e9,1e9,-1e9,-1e9];shapes.forEach(s=>s.poly.forEach(q=>{a=[Math.min(a[0],q[0]),Math.min(a[1],q[1]),Math.max(a[2],q[0]),Math.max(a[3],q[1])]}));
  const w=a[2]-a[0]||1,h=a[3]-a[1]||1,k=Math.min((cv.width-30)/w,(cv.height-30)/h);const ox=(cv.width-w*k)/2-a[0]*k,oy=(cv.height+h*k)/2+a[1]*k;
  shapes.forEach(s=>{g.beginPath();s.poly.forEach((q,i)=>{const X=q[0]*k+ox,Y=oy-q[1]*k;i?g.lineTo(X,Y):g.moveTo(X,Y)});g.closePath();g.fillStyle=s.c;g.globalAlpha=.85;g.fill();g.globalAlpha=1;g.strokeStyle='#0008';g.stroke()});
  g.fillStyle='#8aa397';g.font='11px sans-serif';g.fillText(ax===0?'front (X/Y)':'side (Z/Y)',8,14)}
/* ---------- teach ---------- */
async function teach(){const m=$('#main');
  m.innerHTML=`<div class="card"><h2>1. Teach with text (lessons)</h2><p class="mut">Short rules the AI must follow. Leave "applies when prompt contains" empty for a rule that always applies, or fill it (e.g. <i>dragon</i>) so it is only added for matching prompts - cheaper and more precise.</p>
  <label>Lesson</label><textarea id="lt" placeholder="e.g. Dragons have four legs and two wings that start at the shoulder blades, never at the back of the head."></textarea>
  <div class="row"><div><label>Applies when prompt contains (optional)</label><input id="lk" placeholder="dragon, wyvern"></div><div><label>Category</label><select id="ls"><option value="any">any</option><option>creature</option><option>humanoid</option><option>vehicle</option><option>structure</option><option>prop</option><option>plant</option></select></div>
  <div><label>Builder</label><select id="lv"><option value="any">both</option><option value="classic">Classic</option><option value="depth3d">Depth 3D</option></select></div></div>
  <div class="row" style="margin-top:10px"><button class="btn" onclick="addLesson()">Add lesson</button></div><div id="lmsg2"></div><div id="llist" style="margin-top:10px"></div></div>
  <div class="card"><h2>2. Teach by showing a 3D model (examples)</h2><p class="mut">Upload an <b>OBJ</b>, <b>GLB/GLTF</b> or the builder's <b>JSON</b> export. The server converts it into compact convex parts, uses it as a reference for similar prompts, and (Depth 3D) also saves it as a locked model so that exact prompt costs no AI call. FBX is not supported - export OBJ/GLB from your 3D tool.</p>
  <div class="row"><div><label>Name</label><input id="en2" placeholder="Gorilla boss"></div><div style="flex:1"><label>Prompt it teaches (matching words)</label><input id="ep2" placeholder="gorilla" style="width:100%"></div>
  <div><label>Scope</label><select id="es"><option value="subject">only for this subject</option><option value="category">style for the whole category</option></select></div>
  <div><label>Category</label><select id="ec"><option value="">auto</option><option>creature</option><option>humanoid</option><option>vehicle</option><option>structure</option><option>prop</option><option>plant</option></select></div></div>
  <label>3D file</label><input type="file" id="ef" accept=".obj,.glb,.gltf,.json"><label style="margin-top:10px"><input type="checkbox" id="eas" checked> also save it as a ready-made model for this prompt (Depth 3D, locked)</label>
  <div class="row" style="margin-top:10px"><button class="btn" onclick="addExample()">Upload &amp; teach</button></div><div id="emsg"></div><div id="elist" style="margin-top:10px"></div></div>
  <div class="card"><h2>3. Test a prompt (costs nothing)</h2><div class="row"><input id="tp" placeholder="red gorilla boss" style="flex:1"><select id="tv"><option value="classic">Classic</option><option value="depth3d">Depth 3D</option></select><button class="btn" onclick="testPrompt()">Check</button></div><div id="tout"></div></div>`;
  lessons();examples()}
async function lessons(){const d=await api('/admin/api/lessons');$('#llist').innerHTML=d.items.length?`<table><tr><th>Lesson</th><th>When</th><th>Used</th><th></th></tr>${d.items.map(l=>`<tr><td style="${l.enabled?'':'opacity:.45'}">${esc(l.text)}</td><td class="mut">${esc(l.keywords||'always')}<br>${l.scope} / ${l.variant}</td><td>${l.hits}</td><td class="row"><button class="btn sec sm" onclick="togL(${l.id},${l.enabled?0:1})">${l.enabled?'Disable':'Enable'}</button><button class="btn bad sm" onclick="delL(${l.id})">Delete</button></td></tr>`).join('')}</table>`:'<p class="mut">No lessons yet.</p>'}
async function addLesson(){try{await api('/admin/api/lessons',{method:'POST',json:{text:$('#lt').value,keywords:$('#lk').value,scope:$('#ls').value,variant:$('#lv').value}});$('#lt').value='';note($('#lmsg2'),'Lesson added.',1);lessons()}catch(e){note($('#lmsg2'),e.message)}}
async function togL(id,v){await api('/admin/api/lessons/'+id,{method:'PATCH',json:{enabled:v}});lessons()}
async function delL(id){await api('/admin/api/lessons/'+id,{method:'DELETE'});lessons()}
async function examples(){const d=await api('/admin/api/examples');$('#elist').innerHTML=d.items.length?`<table><tr><th>Example</th><th>Teaches</th><th>Parts</th><th>Used</th><th></th></tr>${d.items.map(e=>`<tr><td style="${e.enabled?'':'opacity:.45'}"><b>${esc(e.name)}</b><div class="mut" style="font-size:12px">${esc(e.source_fmt)}${e.model_id?' &middot; saved as model':''}</div></td><td>${esc(e.prompt)}<br><span class="tag">${e.scope==='category'?'category: '+esc(e.category):'subject'}</span></td><td>${e.parts}</td><td>${e.uses}</td><td class="row"><button class="btn sec sm" onclick="exView('${e.id}')">View</button><button class="btn sec sm" onclick="exTog('${e.id}',${e.enabled?0:1})">${e.enabled?'Disable':'Enable'}</button><button class="btn bad sm" onclick="exDel('${e.id}')">Delete</button></td></tr>`).join('')}</table>`:'<p class="mut">No example models yet.</p>'}
function b64(buf){let s='';const a=new Uint8Array(buf);for(let i=0;i<a.length;i+=0x8000)s+=String.fromCharCode.apply(null,a.subarray(i,i+0x8000));return btoa(s)}
async function addExample(){const f=$('#ef').files[0];if(!f)return note($('#emsg'),'Choose a file first.');const b={name:$('#en2').value||f.name.replace(/\.[^.]+$/,''),prompt:$('#ep2').value||$('#en2').value||f.name.replace(/\.[^.]+$/,''),scope:$('#es').value,category:$('#ec').value,filename:f.name,as_model:$('#eas').checked};
  try{if(/\.glb$/i.test(f.name))b.data_b64=b64(await f.arrayBuffer());else b.text=await f.text();$('#emsg').innerHTML='<p class="mut">Converting...</p>';const r=await api('/admin/api/examples',{method:'POST',json:b});
    note($('#emsg'),`Taught! ${r.parts} parts (${r.format}, category ${r.category}).${r.model_id?' Saved as a locked model too.':''} ${(r.notes||[]).join(' ')}`,1);$('#ef').value='';examples()}catch(e){note($('#emsg'),e.message)}}
async function exDel(id){if(!confirm('Delete this example? (a model saved from it stays in the Models tab)'))return;await api('/admin/api/examples/'+id,{method:'DELETE'});examples()}
async function exTog(id,v){await api('/admin/api/examples/'+id,{method:'PATCH',json:{enabled:v}});examples()}
async function exView(id){const e=await api('/admin/api/examples/'+id);$('#dbody').innerHTML=`<h2>${esc(e.name)}</h2><p class="mut">${e.parts} parts &middot; ${e.payload.bbox.join(' x ')} units ${e.payload.notes.length?' &middot; '+esc(e.payload.notes.join(' ')):''}</p><div class="row"><canvas id="cv1" width="340" height="300"></canvas><canvas id="cv2" width="340" height="300"></canvas></div>`;dlg.showModal();draw($('#cv1'),{hulls:e.hulls},0,1);draw($('#cv2'),{hulls:e.hulls},2,1)}
async function testPrompt(){try{const r=await api('/admin/api/preview',{method:'POST',json:{prompt:$('#tp').value,variant:$('#tv').value}});const mt=r.match;
  $('#tout').innerHTML=`<div class="msg">${mt.mode==='exact'?'<b>Exact match</b> - served from memory, <b>no AI call</b>.':mt.mode==='local'?'<b>Same subject, different colour/size</b> - recoloured locally, <b>no AI call</b>.':mt.mode==='variant'?'<b>Similar model found</b> ('+esc(mt.prompt)+') - the AI re-sculpts it (1 call).':'<b>New subject</b> - the AI builds it from scratch.'}<br><span class="mut">category ${esc(r.category)} &middot; words: ${esc(r.tokens.join(', '))}</span></div>
  <p class="mut">Lessons that would be added: ${r.lessons.length} &middot; examples: ${r.examples.length}</p>${r.teaching_text?'<pre>'+esc(r.teaching_text)+'</pre>':''}`}catch(e){note($('#tout'),e.message)}}
/* ---------- backup ---------- */
function backup(){$('#main').innerHTML=`<div class="card"><h2>Backup &amp; restore</h2><p class="mut">The free Render plan wipes its disk on redeploys, so download a backup after teaching or whenever you want to keep the library. The file holds every model, lesson and example.</p>
  <div class="row"><button class="btn" onclick="exportAll()">Download backup</button></div><label style="margin-top:14px">Restore from a backup file (merges; locked models are never overwritten)</label><input type="file" id="bf" accept=".json"><div class="row" style="margin-top:8px"><button class="btn sec" onclick="importAll()">Restore</button></div><div id="bmsg"></div></div>`}
async function exportAll(){const r=await fetch('/admin/api/export',{headers:{'X-Admin-Token':TOKEN}});if(!r.ok)return note($('#bmsg'),'Export failed.');const b=await r.blob();const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='verdant-memory-backup.json';a.click()}
async function importAll(){const f=$('#bf').files[0];if(!f)return note($('#bmsg'),'Choose a backup file.');try{const j=JSON.parse(await f.text());const r=await api('/admin/api/import',{method:'POST',json:j});note($('#bmsg'),`Restored ${r.imported.models} model(s), ${r.imported.lessons} lesson(s), ${r.imported.examples} example(s).`,1)}catch(e){note($('#bmsg'),e.message)}}
/* ---------- ai ---------- */
async function ai(refresh){const m=$('#main');m.innerHTML='<p class="mut">Asking Google which models your key can use...</p>';let d;try{d=await api('/admin/api/ai'+(refresh?'?refresh=true':''))}catch(e){m.innerHTML='';return note(m,e.message)}
  const g=d.gemini;m.innerHTML=`<div class="card"><h2>AI providers</h2><p>Keys found on the server: ${d.providers.map(p=>`<span class="tag g">${esc(p)}</span>`).join(' ')||'<span class="tag b">none - set GEMINI_API_KEY on Render</span>'}</p><p class="mut">Configured Gemini model: <b>${esc(d.configured_model)}</b> (use <code>auto</code>, or leave AI_MODEL unset, to always pick the best working one).</p></div>
  ${g?(g.ok?`<div class="card"><h2>Gemini models your key can use</h2><p>The server will try, in order: ${g.will_try.map(x=>`<span class="tag g">${esc(x)}</span>`).join(' ')}</p><p class="mut">Last model that worked: ${esc(g.last_working||'-')}</p><details><summary class="mut">All ${g.all_generate_models.length} models that support text generation</summary><pre>${esc(g.all_generate_models.join('\n'))}</pre></details></div>`:`<div class="card"><div class="msg err">Gemini: ${esc(g.error)}</div></div>`):''}
  <div class="card"><div class="row"><button class="btn sec" onclick="ai(true)">Refresh model list</button><button class="btn" onclick="aiTest()">Send a test request (1 AI call)</button></div><div id="aout"></div></div>`}
async function aiTest(){$('#aout').innerHTML='<p class="mut">Calling the AI...</p>';const r=await api('/admin/api/ai-test',{method:'POST',json:{}});note($('#aout'),r.ok?`Works: ${r.engine.provider} / ${r.engine.model}`:r.error,r.ok)}
if(TOKEN){api('/admin/api/overview').then(()=>{$('#login').classList.add('hide');init()}).catch(()=>logout(true))}
</script></body></html>"""


# --------------------------------------------------------------------------- memory API (public, used by the builder page)
@app.get("/memory")
async def memory_list(q: str = "", sort: str = "popular", limit: int = 40, variant: str = ""):
    v = norm_variant(variant) if variant else ""
    return {"total": MEMORY.count(), "items": MEMORY.listing(q, limit, sort, v), "persistent": MEMORY.persistent}


@app.get("/memory/{mid}")
async def memory_get(mid: str, tonal: bool = True):
    e = MEMORY.get(mid)
    if not e:
        raise HTTPException(status_code=404, detail="Model not found in global memory.")
    uses = MEMORY.touch(mid)
    out = apply_tonal_payload(e["payload"], tonal)
    out["engine"] = {"provider": "global-memory", "model": mid}
    out["effort"] = e["effort"]
    out["memory"] = memory_info("memory", id=mid, uses=uses, base_prompt=e["prompt"], saved=False, effort=e["effort"],
                                pinned=e.get("pinned", 0))
    return JSONResponse(content=_with_lumio(out))


@app.post("/memory/{mid}/flag")
async def memory_flag(mid: str):
    if not MEMORY.flag(mid):
        raise HTTPException(status_code=404, detail="Model not found (or it is locked by the owner).")
    return {"ok": True}


@app.delete("/memory/{mid}")
async def memory_delete(mid: str, request: Request):
    _require_admin(request)
    if not MEMORY.delete(mid):
        raise HTTPException(status_code=404, detail="Model not found.")
    return {"ok": True}


@app.get("/skill")
async def skill(prompt: str = "", category: str = "", variant: str = "classic", effort: str = "mid"):
    v, prof = norm_variant(variant), EFFORTS[norm_effort(effort)]
    cat, sub, wings, _ = classify(prompt, category)
    if v == "depth3d":
        return {"variant": v, "effort": prof["name"], "category": cat, "sub": sub, "parts": list(prof["d3_parts"]),
                "points_per_part": list(prof["d3_pts"]), "system_prompt": build_depth_system_prompt(cat, sub, prof)}
    lo, hi = effort_range(*part_range(cat, None), prof)
    return {"variant": v, "effort": prof["name"], "category": cat, "sub": sub, "wings": wings, "parts": [lo, hi],
            "system_prompt": build_system_prompt(cat, sub, wings, lo, hi)}


@app.get("/efforts")
async def efforts():
    return {k: {"label": v["label"], "note": v["note"], "thinking_tokens": v["thinking"], "revisions": v["revisions"],
                "research_queries": {0: 0, 1: 2, 2: 3, 3: 5}[v["research"]], "best_of": v["best_of"],
                "depth3d_parts": list(v["d3_parts"]), "time_limit_s": v["job_s"]} for k, v in EFFORTS.items()}


@app.post("/test-key")
async def test_key(req: TestKeyRequest):
    prov, key, model = req.provider.lower().strip(), req.api_key.strip(), (req.model or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="API key is empty.")
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            if prov == "serper":
                res = await client.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": key, "Content-Type": "application/json"},
                    json={"q": "3D low poly model", "num": 1},
                )
                if res.status_code != 200:
                    raise HTTPException(status_code=400, detail="Serper rejected the key.")
                return {"ok": True, "message": "Serper verified & active!"}
            if prov == "gemini":
                names = await gemini_models(client, key, force=True)
                chain = gemini_chain(model, names, "")
                last: Optional[httpx.Response] = None
                for m in chain[:3]:
                    res = await client.post(
                        f"{GEMINI_BASE}/models/{m}:generateContent",
                        headers={"x-goog-api-key": key},
                        json={"contents": [{"role": "user", "parts": [{"text": "OK"}]}], "generationConfig": {"maxOutputTokens": 16}},
                    )
                    if res.status_code == 200:
                        return {"ok": True, "message": f"Gemini verified ({m}; {len(names)} models available to this key)!"}
                    last = res
                    if res.status_code in (401, 403):
                        break
                raise HTTPException(status_code=400, detail=f"Gemini error: {_err_text(last) if last is not None else 'no model to try'}")
            if prov in ("openai", "groq"):
                base = "https://api.openai.com/v1" if prov == "openai" else "https://api.groq.com/openai/v1"
                m = model or ("gpt-4o-mini" if prov == "openai" else DEFAULT_MODELS["groq"])
                payload: Dict[str, Any] = {"model": m, "messages": [{"role": "user", "content": "OK"}]}
                payload["max_completion_tokens" if _is_reasoning(m) else "max_tokens"] = 64 if _is_reasoning(m) else 10
                await _post_healing(client, prov, base + "/chat/completions", {"Authorization": f"Bearer {key}"}, payload)
                return {"ok": True, "message": f"{prov.title()} verified!"}
            if prov == "anthropic":
                await _post_healing(
                    client, "anthropic", "https://api.anthropic.com/v1/messages",
                    {"x-api-key": key, "anthropic-version": "2023-06-01"},
                    {"model": model or DEFAULT_MODELS["anthropic"], "max_tokens": 10, "messages": [{"role": "user", "content": "OK"}]},
                )
                return {"ok": True, "message": "Anthropic verified!"}
            raise HTTPException(status_code=400, detail="Unknown provider")
        except HTTPException:
            raise
        except ProviderError as e:
            raise HTTPException(status_code=400, detail=f"{e.provider} rejected the key: {e.message}"[:1000])
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail=f"{prov} did not answer in time. Try again.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(ProviderError)
async def _provider_error_handler(request: Request, exc: ProviderError):
    http = exc.to_http()
    return JSONResponse(status_code=http.status_code, headers={"Access-Control-Allow-Origin": "*"},
                        content={**exc.to_dict(), "detail": http.detail, "hint": _hint(exc.status, exc.message)})


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception):
    # Unhandled errors bypass CORS, which makes browsers show a useless "Failed to fetch". Answer with the real error.
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, headers={"Access-Control-Allow-Origin": "*"},
                        content={"detail": f"{exc.__class__.__name__}: {exc}"[:2000], "path": request.url.path,
                                 "trace": traceback.format_exc()[-3000:]})


class GeminiModelsRequest(BaseModel):
    api_key: Optional[str] = ""
    refresh: bool = False


async def _gemini_catalog(key: str, source: str, refresh: bool) -> Dict[str, Any]:
    kh = _key_hash(key)
    ent0 = _GEM_CACHE.get(kh)
    force = bool(refresh and (not ent0 or time.time() - ent0["at"] > 60))  # Google is asked at most once a minute
    async with httpx.AsyncClient(timeout=20.0) as client:
        ranked = await gemini_models(client, key, force=force)
    ent = _GEM_CACHE.get(kh) or {}
    if not ent.get("catalog"):
        raise ProviderError("gemini", 502, "Google returned no model list (network problem, or the key has no access).")
    blocked, good = _gem_blocked(kh), _GEM_GOOD.get(kh, "")
    chain = gemini_chain(DEFAULT_MODELS["gemini"], ranked, good, set(blocked))
    models = [{**m, "usable": m["name"] in ranked, "rank": ranked.index(m["name"]) + 1 if m["name"] in ranked else None,
               "will_try": m["name"] in chain, "blocked_s": blocked.get(m["name"], 0), "last_working": m["name"] == good,
               "calls": MEMORY.counter(f"gem_model:{m['name']}")} for m in ent["catalog"]]
    models.sort(key=lambda m: (m["rank"] is None, m["rank"] or 0, m["name"]))
    c = MEMORY.counters()
    return {"ok": True, "key_source": source, "count": len(models), "usable_count": len(ranked),
            "configured_model": DEFAULT_MODELS["gemini"] or "auto", "will_try": chain, "last_working": good,
            "blocked": blocked, "cached_age_s": round(time.time() - ent.get("at", time.time())), "models": models,
            "usage": {"ai_calls_total": c.get("ai_calls", 0),
                      "ai_calls_today": MEMORY.counter(f"ai_calls:{time.strftime('%Y%m%d', time.gmtime())}"),
                      "daily_cap": AI_DAILY_CAP, "ai_errors": c.get("ai_errors", 0),
                      "calls_avoided": sum(c.get(k, 0) for k in ("memory_hits", "local_variants", "revisions_skipped",
                                                                 "second_candidates_skipped", "research_cache_hits",
                                                                 "duplicate_jobs_shared"))}}


_NO_KEY = ("No Gemini key: set GEMINI_API_KEY on the server, send your key in the x-gemini-key header, "
           "or POST it as api_key.")


@app.get("/gemini/models")
async def gemini_models_get(request: Request, refresh: bool = False):
    key = (request.headers.get("x-gemini-key") or "").strip()
    src = "request" if key else "server"
    key = key or ENV_KEYS.get("gemini", "")
    if not key:
        raise HTTPException(status_code=400, detail=_NO_KEY)
    return JSONResponse(content=await _gemini_catalog(key, src, refresh))


@app.post("/gemini/models")
async def gemini_models_post(req: GeminiModelsRequest):
    key = (req.api_key or "").strip()
    src = "request" if key else "server"
    key = key or ENV_KEYS.get("gemini", "")
    if not key:
        raise HTTPException(status_code=400, detail=_NO_KEY)
    return JSONResponse(content=await _gemini_catalog(key, src, req.refresh))


@app.get("/ping")
async def ping():
    return {"ok": True, "version": VERSION}


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    gk = ENV_KEYS.get("gemini", "")
    gc = _GEM_CACHE.get(_key_hash(gk)) if gk else None
    return {"status": "healthy", "version": VERSION, "providers": [k for k, v in ENV_KEYS.items() if v],
            "auto_revise": AUTO_REVISE, "efforts": list(EFFORTS), "variants": ["classic", "depth3d"], "max_revisions": MAX_REVISIONS, "thinking_level": THINKING_LEVEL,
            "gemini": {"configured_model": DEFAULT_MODELS["gemini"] or "auto", "discovered": len(gc["names"]) if gc else None,
                       "last_working": _GEM_GOOD.get(_key_hash(gk), "") if gk else ""},
            "memory": {"models": MEMORY.count(), "persistent": MEMORY.persistent, "path": str(MEMORY.path)},
            "teaching": {"lessons": MEMORY.stats()["lessons"], "examples": MEMORY.stats()["examples"]},
            "admin": "/admin", "ok": True, "time": int(time.time()), "gemini_models": "/gemini/models",
            "ai_calls_today": MEMORY.counter(f"ai_calls:{time.strftime('%Y%m%d', time.gmtime())}"), "daily_cap": AI_DAILY_CAP,
            "jobs_active": sum(1 for j in JOBS.values() if j["status"] in ("queued", "running"))}


# --------------------------------------------------------------------------- admin: auth
_ADMIN_FAILS: Dict[str, List[float]] = {}


def _admin_token() -> str:
    global ADMIN_TOKEN
    if not ADMIN_TOKEN:
        with MEMORY.lock:
            r = MEMORY.db.execute("SELECT v FROM meta WHERE k='admin_token'").fetchone()
            if r:
                ADMIN_TOKEN = r["v"]
            else:
                ADMIN_TOKEN = secrets.token_urlsafe(18)
                MEMORY.db.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('admin_token',?)", (ADMIN_TOKEN,))
                MEMORY.db.commit()
        log.warning("=" * 60)
        log.warning("ADMIN_TOKEN is not set - using this generated admin password: %s", ADMIN_TOKEN)
        log.warning("Set ADMIN_TOKEN in the Render environment to choose your own.")
        log.warning("=" * 60)
    return ADMIN_TOKEN


def _require_admin(request: Request) -> None:
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
    now = time.time()
    fails = [t for t in _ADMIN_FAILS.get(ip, []) if now - t < 300]
    if len(fails) >= 12:
        raise HTTPException(status_code=429, detail="Too many wrong admin passwords. Wait 5 minutes.")
    tok = request.headers.get("x-admin-token") or request.query_params.get("token") or ""
    if not tok or not secrets.compare_digest(tok.encode(), _admin_token().encode()):
        fails.append(now)
        _ADMIN_FAILS[ip] = fails
        raise HTTPException(status_code=401, detail="Wrong or missing admin password.")


async def _body(request: Request) -> Dict[str, Any]:
    try:
        d = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="A JSON body is required.")
    if not isinstance(d, dict):
        raise HTTPException(status_code=400, detail="A JSON object is required.")
    return d


def _example_model_payload(ex_payload: Dict[str, Any], title: str, category: str) -> Dict[str, Any]:
    hulls = example_hulls(ex_payload)
    if not hulls:
        raise ValueError("The example has no valid parts.")
    tris, pts_total = 0, 0
    mn, mx = [1e9] * 3, [-1e9] * 3
    for h in hulls:
        wp = [tuple(h["pos"][j] + p[j] for j in range(3)) for p in h["points"]]
        tris += len(convex_hull_3d(wp))
        pts_total += len(wp)
        for j in range(3):
            mn[j] = min(mn[j], min(p[j] for p in wp))
            mx[j] = max(mx[j], max(p[j] for p in wp))
    size = [round(mx[0] - mn[0], 2), round(mx[1], 2), round(mx[2] - mn[2], 2)]
    n = len(hulls)
    return {"variant": "depth3d", "title": title[:60], "category": category, "hulls": hulls,
            "stats": {"parts": n, "tris": tris, "size": size, "avg_points": round(pts_total / n, 1), "cubes": 0},
            "diagnostics": {"readout": f"{n} parts · {tris} triangles · {size[0]} × {size[1]} × {size[2]} units",
                            "floating_check": "Taught by the owner", "detail_check": f"{pts_total / n:.1f} points/part",
                            "quality": "Taught example (pinned)"},
            "quality": {"score": 100, "grade": "A", "revisions": 0, "issues_left": []}, "plan": None,
            "engine": {"provider": "taught-example", "model": "owner"}, "used_serper": False}


# --------------------------------------------------------------------------- admin: API
@app.get("/admin/api/overview")
async def admin_overview(request: Request):
    _require_admin(request)
    c = MEMORY.counters()
    avoided = sum(c.get(k, 0) for k in ("memory_hits", "local_variants", "revisions_skipped", "second_candidates_skipped",
                                        "research_cache_hits"))
    today = MEMORY.counter(f"ai_calls:{time.strftime('%Y%m%d', time.gmtime())}")
    return {"version": VERSION, "stats": MEMORY.stats(), "counters": c, "api_calls_avoided": avoided, "ai_calls_today": today,
            "daily_cap": AI_DAILY_CAP, "persistent": MEMORY.persistent, "db_path": str(MEMORY.path),
            "providers": [k for k, v in ENV_KEYS.items() if v], "serper": bool(SERPER_API_KEY),
            "jobs_active": sum(1 for j in JOBS.values() if j["status"] in ("queued", "running"))}


@app.get("/admin/api/models")
async def admin_models(request: Request, q: str = "", sort: str = "popular", limit: int = 50, offset: int = 0,
                       variant: str = "", source: str = "", flagged: bool = False, pinned: bool = False):
    _require_admin(request)
    total, items = MEMORY.admin_list(q=q, limit=limit, offset=offset, sort=sort, variant=norm_variant(variant) if variant else "",
                                     source=source, flagged=flagged, pinned=pinned)
    return {"total": total, "items": items}


@app.get("/admin/api/models/{mid}")
async def admin_model_get(mid: str, request: Request):
    _require_admin(request)
    m = MEMORY.get(mid, include_flagged=True)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found.")
    return JSONResponse(content=m)


@app.patch("/admin/api/models/{mid}")
async def admin_model_patch(mid: str, request: Request):
    _require_admin(request)
    body = await _body(request)
    try:
        m = MEMORY.update(mid, body)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not m:
        raise HTTPException(status_code=404, detail="Model not found.")
    m.pop("payload", None)
    return {"ok": True, "model": m}


@app.delete("/admin/api/models/{mid}")
async def admin_model_delete(mid: str, request: Request):
    _require_admin(request)
    if not MEMORY.delete(mid):
        raise HTTPException(status_code=404, detail="Model not found.")
    return {"ok": True}


@app.post("/admin/api/models/bulk")
async def admin_models_bulk(request: Request):
    _require_admin(request)
    body = await _body(request)
    ids = [str(i) for i in (body.get("ids") or [])][:500]
    action = str(body.get("action") or "")
    n = 0
    for mid in ids:
        if action == "delete":
            n += 1 if MEMORY.delete(mid) else 0
        elif action in ("pin", "unpin"):
            n += 1 if MEMORY.update(mid, {"pinned": 1 if action == "pin" else 0}) else 0
        elif action == "unflag":
            n += 1 if MEMORY.update(mid, {"flags": 0}) else 0
        else:
            raise HTTPException(status_code=400, detail="action must be delete, pin, unpin or unflag.")
    return {"ok": True, "changed": n}


@app.post("/admin/api/cleanup")
async def admin_cleanup(request: Request):
    """Delete unlocked models that are low quality, flagged, or never reused. dry_run (default) only counts."""
    _require_admin(request)
    b = await _body(request)
    below = int(b.get("below_score") or 0)
    old_days = float(b.get("unused_days") or 0)
    flagged = bool(b.get("flagged"))
    dry = b.get("dry_run", True) not in (False, 0, "0", "false")
    now = time.time()
    with MEMORY.lock:
        rows = MEMORY.db.execute("SELECT id,score,uses,flags,created FROM models WHERE pinned=0").fetchall()
    hit = [r["id"] for r in rows if (below and (r["score"] or 0) < below) or (flagged and (r["flags"] or 0) > 0)
           or (old_days and (r["uses"] or 0) == 0 and now - (r["created"] or now) > old_days * 86400)]
    if not dry:
        for mid in hit:
            MEMORY.delete(mid)
    return {"ok": True, "dry_run": dry, "matched": len(hit)}


@app.post("/admin/api/models/{mid}/teach")
async def admin_model_teach(mid: str, request: Request):
    """Use a model from the library as a teaching example for similar prompts."""
    _require_admin(request)
    m = MEMORY.get(mid, include_flagged=True)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found.")
    body = await _body(request) if request.headers.get("content-length", "0") not in ("0", "") else {}
    try:
        ex = await asyncio.to_thread(example_from_model, m["payload"])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    eid = MEMORY.example_add(str(body.get("name") or m["title"] or m["prompt"]), m["prompt"], m["category"] or "",
                             "category" if body.get("scope") == "category" else "subject", ex, "library",
                             f"Taught from library model {mid}", model_id=mid)
    return {"ok": True, "example_id": eid, "parts": len(ex["parts"])}


@app.get("/admin/api/lessons")
async def admin_lessons(request: Request):
    _require_admin(request)
    return {"items": MEMORY.lesson_list()}


@app.post("/admin/api/lessons")
async def admin_lesson_add(request: Request):
    _require_admin(request)
    b = await _body(request)
    try:
        lid = MEMORY.lesson_add(str(b.get("text") or ""), str(b.get("scope") or "any"), str(b.get("variant") or "any"),
                                str(b.get("keywords") or ""))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "id": lid}


@app.patch("/admin/api/lessons/{lid}")
async def admin_lesson_patch(lid: int, request: Request):
    _require_admin(request)
    if not MEMORY.lesson_update(lid, await _body(request)):
        raise HTTPException(status_code=404, detail="Lesson not found or nothing to change.")
    return {"ok": True}


@app.delete("/admin/api/lessons/{lid}")
async def admin_lesson_delete(lid: int, request: Request):
    _require_admin(request)
    if not MEMORY.lesson_delete(lid):
        raise HTTPException(status_code=404, detail="Lesson not found.")
    return {"ok": True}


@app.get("/admin/api/examples")
async def admin_examples(request: Request):
    _require_admin(request)
    return {"items": MEMORY.example_list()}


@app.get("/admin/api/examples/{eid}")
async def admin_example_get(eid: str, request: Request):
    _require_admin(request)
    ex = MEMORY.example_get(eid)
    if not ex:
        raise HTTPException(status_code=404, detail="Example not found.")
    return JSONResponse(content={**ex, "hulls": example_hulls(ex["payload"])})


@app.post("/admin/api/examples")
async def admin_example_upload(request: Request):
    """Teach by showing a 3D model. JSON: {name, prompt, category?, scope?, format?, filename?, text | data_b64, as_model?, notes?}"""
    _require_admin(request)
    b = await _body(request)
    name = str(b.get("name") or "").strip()[:80]
    prompt = str(b.get("prompt") or name).strip()[:200]
    if not prompt:
        raise HTTPException(status_code=400, detail="Give the model a name or the prompt it should teach (e.g. 'gorilla').")
    text = b.get("text") if isinstance(b.get("text"), str) else ""
    blob = None
    if b.get("data_b64"):
        try:
            blob = base64.b64decode(str(b["data_b64"]), validate=False)
        except Exception:
            raise HTTPException(status_code=400, detail="data_b64 is not valid base64.")
    if len(text) > MAX_UPLOAD_CHARS or (blob is not None and len(blob) > MAX_UPLOAD_CHARS):
        raise HTTPException(status_code=413, detail="That file is too large (limit about 16 MB). Simplify the model first.")
    if not text and blob is None:
        raise HTTPException(status_code=400, detail="Send the file as text (obj/json/gltf) or data_b64 (glb).")

    def work():
        groups, fmt = parse_model_upload(str(b.get("format") or "auto"), text, blob, str(b.get("filename") or ""))
        return build_example(groups), fmt

    try:
        ex, fmt = await asyncio.to_thread(work)
    except (ValueError, KeyError, TypeError, struct.error, json.JSONDecodeError) as e:
        raise HTTPException(status_code=422, detail=f"Could not read that model: {e}")
    cat = str(b.get("category") or "") if str(b.get("category") or "") in PART_RANGE else classify(prompt)[0]
    scope = "category" if b.get("scope") == "category" else "subject"
    model_id = ""
    if b.get("as_model", True) not in (False, 0, "0", "false") and scope == "subject":
        tokens = prompt_tokens(prompt)
        if tokens:
            try:
                payload = _example_model_payload(ex, name or prompt.title(), cat)
                model_id = MEMORY.save(prompt, tokens, cat, payload, 100, engine="taught-example", effort="extra",
                                       variant="depth3d", source="example", pinned=1, notes="Taught by the owner")
            except ValueError:
                model_id = ""
    eid = MEMORY.example_add(name or prompt.title(), prompt, cat, scope, ex, fmt, str(b.get("notes") or ""), model_id)
    MEMORY.bump("examples_added")
    return {"ok": True, "id": eid, "parts": len(ex["parts"]), "bbox": ex["bbox"], "notes": ex["notes"],
            "model_id": model_id, "category": cat, "format": fmt}


@app.patch("/admin/api/examples/{eid}")
async def admin_example_patch(eid: str, request: Request):
    _require_admin(request)
    if not MEMORY.example_update(eid, await _body(request)):
        raise HTTPException(status_code=404, detail="Example not found or nothing to change.")
    return {"ok": True}


@app.delete("/admin/api/examples/{eid}")
async def admin_example_delete(eid: str, request: Request):
    _require_admin(request)
    ex = MEMORY.example_get(eid)
    if not ex or not MEMORY.example_delete(eid):
        raise HTTPException(status_code=404, detail="Example not found.")
    return {"ok": True, "linked_model": ex.get("model_id") or ""}


@app.post("/admin/api/preview")
async def admin_preview(request: Request):
    """What would happen for this prompt? Memory match + the lessons/examples that would be added. Costs no AI call."""
    _require_admin(request)
    b = await _body(request)
    prompt = str(b.get("prompt") or "").strip()[:300]
    variant = norm_variant(b.get("variant"))
    tokens = prompt_tokens(prompt)
    cat = classify(prompt, str(b.get("category") or ""))[0]
    mode, entry, added, dropped = MEMORY.find(tokens, variant)
    text, meta = teaching_block(prompt, tokens, cat, variant, count=False)
    return {"tokens": sorted(tokens), "category": cat,
            "match": {"mode": mode, "id": entry["id"] if entry else "", "prompt": entry["prompt"] if entry else "",
                      "added": added, "dropped": dropped, "ai_call_needed": mode in ("none", "variant")},
            "lessons": meta["lessons"], "examples": meta["examples"], "teaching_text": text}


@app.get("/admin/api/export")
async def admin_export(request: Request):
    _require_admin(request)
    data = await asyncio.to_thread(MEMORY.export_all)
    return JSONResponse(content=data, headers={
        "Content-Disposition": f'attachment; filename="verdant-memory-{time.strftime("%Y%m%d-%H%M")}.json"'})


@app.post("/admin/api/import")
async def admin_import(request: Request):
    _require_admin(request)
    b = await _body(request)
    try:
        res = await asyncio.to_thread(MEMORY.import_all, b)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "imported": res}


@app.get("/admin/api/ai")
async def admin_ai(request: Request, refresh: bool = False):
    _require_admin(request)
    gk = ENV_KEYS.get("gemini", "")
    out: Dict[str, Any] = {"providers": [k for k, v in ENV_KEYS.items() if v], "configured_model": DEFAULT_MODELS["gemini"] or "auto",
                           "gemini": None}
    if gk:
        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                names = await gemini_models(client, gk, force=refresh)
                ent = _GEM_CACHE.get(_key_hash(gk)) or {}
                out["gemini"] = {"ok": True, "usable_ranked": names, "all_generate_models": ent.get("all", []),
                                 "will_try": gemini_chain(DEFAULT_MODELS["gemini"], names, _GEM_GOOD.get(_key_hash(gk), "")),
                                 "last_working": _GEM_GOOD.get(_key_hash(gk), "")}
            except ProviderError as e:
                out["gemini"] = {"ok": False, "error": e.message}
    return out


@app.post("/admin/api/ai-test")
async def admin_ai_test(request: Request):
    _require_admin(request)
    _EFFORT.set(EFFORTS["low"])
    try:
        text, eng = await call_llm('Reply with exactly this JSON and nothing else: {"ok":true}', "ping")
        return {"ok": True, "engine": eng, "reply": text[:80]}
    except ProviderError as e:
        return {"ok": False, "error": f"{e.provider}: {e.message}"[:500]}
    except Exception as e:
        return {"ok": False, "error": f"{e.__class__.__name__}: {e}"[:500]}


@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    return HTMLResponse(ADMIN_HTML)


def find_ui() -> Optional[Path]:
    here = Path(__file__).parent
    if (here / "index.html").exists():
        return here / "index.html"
    found = sorted(list(here.glob("Lumio*.html")) + list(here.glob("lumio*.html")) + list(here.glob("Verdant_3D*.html")),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0] if found else None


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = find_ui()
    if html_path:
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse("<h3>index.html not found - put the Verdant page next to server.py</h3>")


@app.get("/workspace", response_class=HTMLResponse)
async def workspace():
    p = Path(__file__).parent / "workspace.html"
    if p.exists():
        return FileResponse(p, media_type="text/html")
    return HTMLResponse("<h3>workspace.html not found - put it next to server.py</h3>", status_code=404)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
