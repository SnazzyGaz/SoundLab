"""soundlab command line: render, models, describe, materials, analyze, validate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundlab


def _print(obj, as_json):
    if as_json:
        print(json.dumps(obj, indent=1))
        return True
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="soundlab", description="Physical-modelling sound engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="render a recipe JSON to WAV (and MP3)")
    r.add_argument("recipe")
    r.add_argument("-o", "--out", help="output .wav (default: recipe name next to it)")
    r.add_argument("--mp3", action="store_true", help="also write an MP3 (needs ffmpeg)")
    r.add_argument("--sr", type=int, help="override sample rate")
    r.add_argument("--seed", type=int, help="override seed (variations)")
    r.add_argument("--set", action="append", default=[], metavar="LAYER.PARAM=VALUE",
                   help="override a parameter of a single-model layer, e.g. --set 0.rpm=3000")

    m = sub.add_parser("models", help="list models")
    m.add_argument("--stage")
    m.add_argument("--json", action="store_true")

    d = sub.add_parser("describe", help="show a model's controls, units, ranges and sources")
    d.add_argument("name")
    d.add_argument("--json", action="store_true")

    mt = sub.add_parser("materials", help="material table")
    mt.add_argument("--json", action="store_true")

    a = sub.add_parser("analyze", help="split a sample into harmonic/noise/transient/sines stems + features")
    a.add_argument("audio")
    a.add_argument("-o", "--out", help="output folder (default: <name>_analysis)")
    a.add_argument("--no-plot", action="store_true")

    v = sub.add_parser("validate", help="check a recipe")
    v.add_argument("recipe")

    args = ap.parse_args(argv)
    try:
        if args.cmd == "render":
            rec = soundlab.load_recipe(args.recipe)
            for s in args.set:
                key, val = s.split("=", 1)
                li, pname = key.split(".", 1)
                layer = rec["layers"][int(li)]
                tgt = layer if "model" in layer else layer["chain"][0]
                tgt.setdefault("params", {})[pname] = json.loads(val) if val[:1] in "[{-0123456789tfn\"" else val
            out = args.out or str(Path(args.recipe).with_suffix(".wav"))
            info = soundlab.render_to_file(rec, out, mp3=args.mp3, sample_rate=args.sr, seed=args.seed)
            print(json.dumps(info))
        elif args.cmd == "models":
            ms = soundlab.list_models(args.stage)
            if not _print(ms, args.json):
                for st in ("sound", "excitation", "resonator", "radiation", "propagation", "room"):
                    rows = [x for x in ms if x["stage"] == st]
                    if rows:
                        print(f"\n[{st}]")
                        for x in rows:
                            print(f"  {x['name']:<12} {x['doc'][:100]}")
        elif args.cmd == "describe":
            dd = soundlab.describe(args.name)
            if not _print(dd, args.json):
                print(f"{dd['name']} ({dd['stage']})\n  {dd['doc']}\n")
                for p in dd["params"]:
                    rng = p.get("range")
                    rs = f"[{rng[0]} .. {rng[1]}]" if rng else (str(p.get("choices", "")) if p.get("choices") else "")
                    auto = " (automatable)" if p.get("automatable") else ""
                    print(f"  {p['name']:<16} default={p['default']!r:<12} {p['unit']:<10} {rs}{auto}")
                    if p["doc"]:
                        print(f"  {'':<16} {p['doc']}")
                if dd["sources"]:
                    print("\n  sources: " + "; ".join(dd["sources"]))
        elif args.cmd == "materials":
            t = soundlab.materials.table()
            if not _print(t, args.json):
                print(f"{'material':<11}{'rho':>7}{'E GPa':>8}  {'eta range':<20}{'T60@1kHz s':>11}")
                for row in t:
                    print(f"{row['material']:<11}{row['density_kg_m3']:>7}{row['E_GPa']:>8.2f}  "
                          f"{str(row['eta']):<20}{row['t60_at_1kHz_s']:>11}")
        elif args.cmd == "analyze":
            out = args.out or str(Path(args.audio).with_suffix("")) + "_analysis"
            feats = soundlab.analyze_file(args.audio, out, plot=not args.no_plot)
            short = {k: v for k, v in feats.items() if k not in ("envelope_db_10ms", "partials", "onsets_s")}
            short["top_partials"] = feats["partials"][:6]
            short["n_onsets"] = len(feats["onsets_s"])
            short["stems_dir"] = out
            print(json.dumps(short, indent=1))
        elif args.cmd == "validate":
            errs = soundlab.validate(soundlab.load_recipe(args.recipe))
            print("ok" if not errs else "\n".join(errs))
            return 1 if errs else 0
    except soundlab.RecipeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
