"""Render the SoundLab sound set used by the Godot demo (demos/godot_walk) as Ogg Vorbis files plus a manifest.

    python tools/godot_export.py demos/godot_walk/sounds

Beds that a game plays in 2D (wind, rain) are stereo seamless loops; sounds placed in the world (fire, the
stream, the pipe howling in the wind) are mono seamless loops, because Godot's 3D players pan a mono source.
One-shots are mono variation banks (v01 as written, v02.. seeded variations), trimmed and loudness-normalised
by `soundlab.finish`. Every file is levelled to its category target, so the manifest carries the designer gain
that sets intensity layers apart (a calm wind and a gale are not equally loud in a scene).
Needs ffmpeg (libvorbis) on PATH.
"""
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import soundlab  # noqa: E402

R = ROOT / "recipes"
LOOP_S = 20.0

# name: (recipe, macros, stereo, designer gain dB[, layer-0 parameter overrides])
# v0.25.2: the gale, the rain and the rain on the roof come from presets fitted to field recordings
# (wind_gale, rain_steady, rain_roof); the wind/rain recipes' macros at their extremes gave crinkling leaves
# (gale) and a scratchy 2-5 kHz hump (rain), and the roof was as bright as open ground.
LOOPS = {
    "wind_calm": ("wind", {"strength": 0.15, "gusts": 0.3}, True, -12.0),
    "wind_breeze": ("wind", {"strength": 0.5}, True, -5.0),
    "wind_gale": ("wind_gale", {}, True, 0.0),
    "rain_light": ("rain_steady", {}, True, -6.0, {"rate_mm_h": 8.0, "near": 0.3}),
    "rain_heavy": ("rain_steady", {}, True, 0.0),
    "rain_roof": ("rain_roof", {}, True, 0.0),
    "fire": ("fire", {"size": 0.45}, False, 0.0),
    "stream": ("water", {"size": 0.5}, False, 0.0),
    "pipe_howl": ("wind_howl", {}, False, -4.0),
}

# name: (recipe, macros, variations)
# Footsteps get a listener-perspective eq per surface (v0.25.3), fitted to 14 BigSoundBank footstep recordings
# (gravel #0085 #0510 #1117 #3202 #3216, concrete/stone #0514 #0606 #1318 #1319, wood #1515-#1518 #1845): heard
# from ear height the floor's 63-125 Hz thump is 10-20 dB under the recipe's contact-mic-like body, and gravel's
# 4 kHz crunch was 7 dB too bright (the crackle in a walk).
STEP_EQ = {"gravel": [[63, -12.0], [125, -12.0], [250, -1.8], [500, 1.7], [1000, 2.6], [2000, -3.4], [4000, -10.1], [8000, -3.0], [16000, 7.9]], "stone": [[63, -12.0], [125, -12.0], [250, 3.9], [500, 7.3], [1000, 4.1], [2000, -2.8], [4000, -5.6], [8000, -7.8], [16000, -2.0]], "wood": [[63, -12.0], [125, -5.8], [250, 6.5], [500, 4.1], [1000, -1.6], [2000, -1.9], [4000, -7.8], [8000, -7.1], [16000, 5.0]]}

ONESHOTS = {
    "step_wood": ("footstep", {"surface": "wood"}, 8, STEP_EQ["wood"]),
    "step_stone": ("footstep", {"surface": "stone"}, 8, STEP_EQ["stone"]),
    "step_gravel": ("footstep", {"surface": "gravel"}, 8, STEP_EQ["gravel"]),
    "door_open": ("creak_model/door_creak", {"open_s": 1.4}, 3),     # the creak lasts exactly as long as the swing:
    "door_close": ("creak_model/door_creak", {"open_s": 0.55, "pitch": 0.6}, 3),   # the game reads open_s to time it
    "door_slam": ("door_slam", {}, 4),
    "knock": ("hc_knock", {}, 6),
    "clang": ("metal_clang", {}, 6),
    "bounce": ("hc_bounce", {}, 4),
    "splash": ("splash", {"size": 0.35}, 4),
    "drumstick": ("drumstick", {}, 4),
}


def gusty_howl(raw: dict, seconds: float) -> dict:
    """wind_howl's wind_speed curve is 6 s long; repeat its shape so a long loop keeps gusting."""
    r = copy.deepcopy(raw)
    curve = r["layers"][0]["params"]["wind_speed"]
    period = curve[-1][0]
    out, k = [], 0
    while k * period < seconds + 3:
        out += [[t + k * period, v] for t, v in curve[(1 if k else 0):]]
        k += 1
    r["layers"][0]["params"]["wind_speed"] = out
    r["loop"] = {"lead_s": 1.0, "xfade_s": 1.5}
    return r


def to_ogg(wav: Path, ogg: Path):
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(wav), "-c:a", "libvorbis",
                    "-q:a", "6", str(ogg)], check=True)


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    man = {"soundlab": soundlab.__version__, "loops": {}, "oneshots": {}}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, (rec, mac, st, gain, *over) in LOOPS.items():
            raw = json.loads((R / f"{rec}.json").read_text())
            if over:
                raw["layers"][0]["params"].update(over[0])
            if rec == "wind_howl":
                raw = gusty_howl(raw, LOOP_S)
            info = soundlab.render_to_file(raw, tmp / f"{name}.wav", macros=mac, loop=LOOP_S, stereo=st)
            to_ogg(tmp / f"{name}.wav", out / f"{name}.ogg")
            man["loops"][name] = {"file": f"{name}.ogg", "recipe": rec, "macros": mac, "stereo": st,
                                  "gain_db": gain, "seconds": round(info["seconds"], 3)}
            print(f"loop    {name:12s} {rec:10s} {info['seconds']:.1f} s {'stereo' if st else 'mono'}")
        for name, (rec, mac, n, *eq) in ONESHOTS.items():
            files = []
            raw = json.loads((R / f"{rec}.json").read_text())
            if eq:
                raw["post"] = raw.get("post", []) + [{"model": "eq", "params": {"curve": eq[0]}}]
            mac = dict(mac)
            swing = mac.pop("open_s", None)
            if swing:
                # a door's creak follows its motion: set the swing time and fit the recipe length to it
                lp = raw["layers"][0]["params"]
                lp["length_s"] = swing          # creak model (the exact-length fallback): length of the motion
                raw["duration"] = lp.get("at", 0.0) + swing + 0.35
            for v in range(n):
                f = f"{name}_v{v + 1:02d}"
                info = soundlab.render_to_file(raw, tmp / f"{f}.wav", macros=mac, variation=v)
                to_ogg(tmp / f"{f}.wav", out / f"{f}.ogg")
                files.append(f"{f}.ogg")
            man["oneshots"][name] = {"files": files, "recipe": rec, "macros": mac}
            if swing:
                man["oneshots"][name]["open_s"] = swing
            print(f"oneshot {name:12s} {rec:10s} x{n}")
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "demos" / "godot_walk" / "sounds")
