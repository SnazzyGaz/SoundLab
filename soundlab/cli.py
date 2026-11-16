"""soundlab command line: render, models, describe, materials, analyze, match, compare, validate."""
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

    mm = sub.add_parser("match", help="fit a recipe's controls to a reference recording (analysis-by-synthesis)")
    mm.add_argument("ref", help="reference audio (wav/mp3/ogg/flac)")
    mm.add_argument("--recipe", required=True, help="base recipe JSON to start from")
    mm.add_argument("--param", "-p", action="append", default=[], metavar="L.NAME[:LO:HI]",
                    help="control to fit, e.g. 0.f0 or 0.f0:120:300 or 0.2.freq (layer.block.name); "
                         "default: every scalar numeric control with a range")
    mm.add_argument("--start", type=float, help="reference crop start (s)")
    mm.add_argument("--end", type=float, help="reference crop end (s)")
    mm.add_argument("--method", choices=("cmaes", "nelder-mead"), default="cmaes")
    mm.add_argument("--budget", type=int, default=150, help="number of renders (default 150)")
    mm.add_argument("--weights", default="", help="loss weights, e.g. spectral=1,stft=1,envelope=0.7,texture=0.7")
    mm.add_argument("--duration", type=float, help="render length while fitting (default: reference length)")
    mm.add_argument("-o", "--out", help="output folder (default: <recipe>_fit)")
    mm.add_argument("--no-pitch-first", action="store_true",
                    help="fit pitch controls together with the rest (default: measure + lock pitch first)")
    mm.add_argument("--holdout", action="append", default=[], metavar="AUDIO[:START[:END]]",
                    help="second recording of the same sound to score the fit on (repeatable; strongly advised)")
    mm.add_argument("--no-ear", action="store_true", help="skip the listening renders")

    c = sub.add_parser("compare", help="distance between a reference and one or more renders")
    c.add_argument("ref")
    c.add_argument("candidates", nargs="+")
    c.add_argument("--start", type=float)
    c.add_argument("--end", type=float)
    c.add_argument("--json", action="store_true")

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
                for st in ("sound", "excitation", "resonator", "radiation", "propagation", "room", "synth", "fx"):
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
        elif args.cmd == "match":
            from . import fit as F
            from .audio_io import read_audio
            rec = soundlab.load_recipe(args.recipe)
            ref, rsr = read_audio(args.ref)
            a = args.start or 0.0
            b = args.end or len(ref) / rsr
            rec["duration"] = float(args.duration or (b - a))
            w = {k: float(v) for k, v in (kv.split("=") for kv in args.weights.split(",") if kv)}
            res = F.fit(rec, ref, rsr, args.param or None, args.start, args.end, args.method, args.budget, w,
                        log=lambda m: print(m, file=sys.stderr), pitch_first=not args.no_pitch_first)
            out = args.out or str(Path(args.recipe).with_suffix("")) + "_fit"
            res["holdouts"] = []
            for h in args.holdout:
                hp, *ab = h.split(":")
                hx, hsr = read_audio(hp)
                hs = float(ab[0]) if len(ab) > 0 and ab[0] else None
                he = float(ab[1]) if len(ab) > 1 and ab[1] else None
                res["holdouts"].append(F.holdout(res, hx, hsr, hs, he, w, name=Path(h).name))
            if not args.holdout:
                print("warning: no --holdout recording; the fit is unchecked on a second recording", file=sys.stderr)
            if not args.no_ear:
                res["ear"] = F.ear_renders(res, out)
            F.report(res, ref, out, Path(args.ref).name, Path(args.recipe).name)
            print(json.dumps({"out": out, "before": res["before"], "after": res["after"],
                              "locked": res.get("locked", []),
                              "holdouts": [{"name": h["name"], "before": h["before"]["total"],
                                            "after": h["after"]["total"]} for h in res["holdouts"]],
                              "params": [{k: p[k] for k in ("key", "before", "after")} for p in res["params"]]},
                             indent=1, default=float))
        elif args.cmd == "compare":
            from . import fit as F
            from .audio_io import read_audio
            ref, rsr = read_audio(args.ref)
            rows = {}
            for cpath in args.candidates:
                y, ysr = read_audio(cpath)
                rows[cpath] = F.compare_signals(ref, rsr, y, ysr, args.start, args.end)
            if not _print(rows, args.json):
                print(f"{'candidate':<40}" + "".join(f"{k:>10}" for k in list(F.DEFAULT_WEIGHTS) + ["total"]))
                for k, d in rows.items():
                    print(f"{Path(k).name[:39]:<40}" + "".join(f"{d[c]:>10.3f}" for c in list(F.DEFAULT_WEIGHTS) + ["total"]))
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
