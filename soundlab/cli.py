"""soundlab command line: render, sequence, batch, audition, models, describe, materials, analyze, match, resynth, compare, validate, gui, blindtest."""
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


def _sequence_cmd(args) -> None:
    from . import sequence as S
    from .macros import parse_macro_args
    if args.list or not args.sequence:
        for d in S.presets():
            print(f"{d['id']:<22} {d['title']}: {d['doc']}")
        return
    src = args.sequence
    seq = S.load(src) if src.endswith(".json") or Path(src).is_file() else S.preset(src)
    seq = json.loads(json.dumps(seq))
    if args.seed is not None:
        seq["seed"] = args.seed
    if args.spm:
        seq["rate"] = {"spm": args.spm}
    elif args.interval:
        seq["rate"] = {"interval": args.interval}
    hum = seq.setdefault("humanise", {})
    if args.jitter_ms is not None:
        hum["timing_ms"] = args.jitter_ms
    if args.level_jitter_db is not None:
        hum["level_db"] = args.level_jitter_db
    mv = parse_macro_args(args.macro)
    for st in seq["steps"]:
        if args.repeat and int(st.get("repeat", 1) or 1) > 1:
            st["repeat"] = args.repeat
        if mv:
            raw = soundlab.load_recipe(S.default_resolver(st["entry"]))
            have = {k: v for k, v in mv.items() if k in (raw.get("macros") or {})}
            if have:
                st["macros"] = {**(st.get("macros") or {}), **have}
    out = args.out or f"{seq.get('name') or Path(src).stem}.wav"
    info = S.export(seq, out, mp3=args.mp3, oneshots=args.oneshots, level=args.level)
    for row in info.pop("cue_list"):
        print(f"  {row['time']:8.3f} s  {row['title']:<28} take {row['take']:<6} {row['gain_db']:+.1f} dB  pan {row['pan']:+.2f}")
    print(json.dumps(info))


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
    r.add_argument("--macro", "-m", action="append", default=[], metavar="NAME=VALUE",
                   help="set a macro slider, e.g. -m size=0.8 -m material=metal (see: soundlab describe RECIPE)")
    r.add_argument("--variations", type=int, default=0, help="write N files: v01 as written, v02.. seeded variations")
    r.add_argument("--variation", type=int, default=0, help="render only variation k (0 = as written)")

    def _finish_args(p):
        p.add_argument("--level", choices=["lufs", "peak"], default="lufs",
                       help="lufs (default): loudness per category with a -1 dBTP ceiling; peak: raw -1 dBFS peak")
        p.add_argument("--no-trim", action="store_true", help="keep the full recipe length (no tail trim)")
        p.add_argument("--loop", type=float, metavar="SECONDS",
                       help="write a seamless loop of this length (for loopable recipes: rain, wind, fire, ...)")
        sg = p.add_mutually_exclusive_group()
        sg.add_argument("--stereo", dest="stereo", action="store_true", default=None,
                        help="stereo file (default: stereo only for loops of recipes with a stereo bed config)")
        sg.add_argument("--mono", dest="stereo", action="store_false", help="always mono")
    _finish_args(r)

    sq = sub.add_parser("sequence", help="render a sequence (library entries on a timeline) to one file, "
                                          "optionally plus numbered one-shots and a cue CSV")
    sq.add_argument("sequence", nargs="?", help="a preset name (see --list) or a sequence .json")
    sq.add_argument("--list", action="store_true", help="list the presets")
    sq.add_argument("-o", "--out", help="output .wav (default: <name>.wav in the current folder)")
    sq.add_argument("--mp3", action="store_true", help="also write an MP3 (needs ffmpeg)")
    sq.add_argument("--oneshots", action="store_true",
                    help="also write <out>_oneshots/: numbered event files and cues.csv (time, entry, take, gain)")
    sq.add_argument("--seed", type=int, help="sequence seed (which takes the events draw, and the jitter)")
    sq.add_argument("--repeat", type=int, help="events for every repeating step (walk steps, knocks, shots)")
    rg = sq.add_mutually_exclusive_group()
    rg.add_argument("--spm", type=float, help="base rate in steps per minute")
    rg.add_argument("--interval", type=float, help="base interval in seconds")
    sq.add_argument("--jitter-ms", type=float, help="timing jitter, +- milliseconds")
    sq.add_argument("--level-jitter-db", type=float, help="level jitter, +- dB")
    sq.add_argument("--macro", "-m", action="append", default=[], metavar="NAME=VALUE",
                    help="set a control on every step whose entry has it, e.g. -m surface=snow -m shoe=boot")
    sq.add_argument("--level", choices=["lufs", "peak"], help="the mix level (default: the sequence's, else lufs)")

    b = sub.add_parser("batch", help="render many recipes (folders, files, .txt lists) with variations and macro sweeps")
    b.add_argument("inputs", nargs="+")
    b.add_argument("-o", "--out", required=True, help="output folder")
    b.add_argument("--variations", type=int, default=1)
    b.add_argument("--macro", "-m", action="append", default=[], metavar="NAME=VALUE")
    b.add_argument("--sweep", action="append", default=[], metavar="NAME=V1,V2,..",
                   help="render every listed value of a macro (combinations multiply), e.g. --sweep size=0,0.5,1")
    b.add_argument("--mp3", action="store_true")
    b.add_argument("--sr", type=int)
    b.add_argument("--recursive", action="store_true")
    b.add_argument("--zip", action="store_true", help="also zip the output folder")
    _finish_args(b)

    au = sub.add_parser("audition", help="quick preview: render at 24 kHz and play; --watch re-renders on save")
    au.add_argument("recipe")
    au.add_argument("--macro", "-m", action="append", default=[], metavar="NAME=VALUE")
    au.add_argument("--variation", type=int, default=0)
    au.add_argument("--sr", type=int, default=24000)
    au.add_argument("--seconds", type=float, help="cut the preview to this length")
    au.add_argument("--no-play", action="store_true")
    au.add_argument("--watch", action="store_true")
    au.add_argument("-o", "--out", default=None, help="scratch folder (default: <system temp>/soundlab-audition)")

    m = sub.add_parser("models", help="list models")
    m.add_argument("--stage")
    m.add_argument("--json", action="store_true")

    d = sub.add_parser("describe", help="show a recipe's macros (path to .json) or a model's controls")
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

    rs = sub.add_parser("resynth", help="analyse a recording and build + fit an editable layered recipe for it")
    rs.add_argument("ref", help="reference audio (kept outside the repo)")
    rs.add_argument("--kind", choices=("auto", "impact", "texture", "voice", "choir", "growl"), default="auto")
    rs.add_argument("--fit-start", type=float, help="start of the section to fit on (s)")
    rs.add_argument("--fit-end", type=float, help="end of the section to fit on (s)")
    rs.add_argument("--holdout-section", metavar="START:END", help="section of the same clip to check the fit on")
    rs.add_argument("--holdout", action="append", default=[], metavar="AUDIO[:START[:END]]",
                    help="another recording of the same sound to check the recipe on (repeatable)")
    rs.add_argument("--clean-tonal", action="store_true",
                    help="remove narrow tonal peaks (birdsong, whistles) above 1.5 kHz before analysing")
    rs.add_argument("--budget", type=int, default=80, help="renders for the fit (default 80)")
    rs.add_argument("-o", "--out", required=True, help="output folder (outside the repo)")

    c = sub.add_parser("compare", help="distance between a reference and one or more renders")
    c.add_argument("ref")
    c.add_argument("candidates", nargs="+")
    c.add_argument("--start", type=float)
    c.add_argument("--end", type=float)
    c.add_argument("--json", action="store_true")

    g = sub.add_parser("gui", help="open the slider app in your browser (local only, 127.0.0.1)")
    g.add_argument("--port", type=int, default=8766, help="port (default 8766; AIDAW uses 8765)")
    g.add_argument("--out", help="folder for exports, saved recipes and the blind-test pack (default ~/SoundLab)")
    g.add_argument("--pack", help="blind-test pack folder (default <out>/blindtest)")
    g.add_argument("--recipes", action="append", help="extra recipe folder (repeatable)")
    g.add_argument("--no-browser", action="store_true", help="do not open the browser")
    bt = sub.add_parser("blindtest", help="blind listening test: build a pack from your library, or score it")
    btsub = bt.add_subparsers(dest="btcmd", required=True)
    btb = btsub.add_parser("build", help="build a listening pack from library/<sound>/*.wav + engine renders")
    btb.add_argument("--library", required=True, help="folder with one sub-folder of real recordings per sound")
    btb.add_argument("--out", help="pack folder (default ~/SoundLab/blindtest)")
    btb.add_argument("--per-sound", type=int, default=2, help="real takes per sound (same number of engine clips)")
    btb.add_argument("--max-s", type=float, default=4.0, help="cap clip length (s)")
    btb.add_argument("--seed", type=int, default=0)
    btb.add_argument("--sounds", nargs="*", help="only these sounds")
    btb.add_argument("--exempt", nargs="*", default=[], help="sounds outside the pass mark (stylised)")
    bts = btsub.add_parser("score", help="score the answers saved in a pack")
    bts.add_argument("pack", nargs="?", help="pack folder (default ~/SoundLab/blindtest)")
    bts.add_argument("--pass-mark", type=float, default=0.60,
                     help="upper end of the balanced-accuracy interval for a clear pass (default 0.60)")
    bts.add_argument("--codes", action="append", default=[],
                     help="text file with answer codes (SL1.…) from the static web page, or - for stdin; "
                          "they are saved into <pack>/answers before scoring (repeatable)")
    bts.add_argument("--json", action="store_true")
    v = sub.add_parser("validate", help="check a recipe")
    v.add_argument("recipe")

    args = ap.parse_args(argv)
    try:
        if args.cmd == "render" and soundlab.sequence.is_sequence(soundlab.load_recipe(args.recipe)):
            info = soundlab.sequence.export(args.recipe, args.out or str(Path(args.recipe).with_suffix(".wav")),
                                            mp3=args.mp3, sample_rate=args.sr)
            info.pop("cue_list", None)
            print(json.dumps(info))
        elif args.cmd == "sequence":
            _sequence_cmd(args)
        elif args.cmd == "render":
            rec = soundlab.load_recipe(args.recipe)
            for s in args.set:
                key, val = s.split("=", 1)
                li, pname = key.split(".", 1)
                layer = rec["layers"][int(li)]
                tgt = layer if "model" in layer else layer["chain"][0]
                try:  # numbers, lists, true/false/null; anything else is a plain string (e.g. "trainer")
                    parsed = json.loads(val)
                except json.JSONDecodeError:
                    parsed = val
                tgt.setdefault("params", {})[pname] = parsed
            from .macros import parse_macro_args
            mv = parse_macro_args(args.macro)
            out = args.out or str(Path(args.recipe).with_suffix(".wav"))
            fk = {"level": args.level, "trim": not args.no_trim, "loop": args.loop, "stereo": args.stereo}
            if args.loop and not rec.get("loop"):
                print("note: this recipe is not tagged as loopable; the loop may not sound seamless", file=sys.stderr)
            if args.variations and args.variations > 1:
                o = Path(out)
                for k in range(args.variations):
                    info = soundlab.render_to_file(rec, o.with_name(f"{o.stem}_v{k + 1:02d}{o.suffix}"), mp3=args.mp3,
                                                   sample_rate=args.sr, macros=mv, variation=k, **fk)
                    print(json.dumps(info))
            else:
                info = soundlab.render_to_file(rec, out, mp3=args.mp3, sample_rate=args.sr, seed=args.seed,
                                               macros=mv, variation=args.variation, **fk)
                print(json.dumps(info))
        elif args.cmd == "batch":
            from .batch import batch_export, parse_sweep
            from .macros import parse_macro_args
            man = batch_export(args.inputs, args.out, args.variations, parse_macro_args(args.macro),
                               parse_sweep(args.sweep), args.mp3, args.sr, args.recursive,
                               level=args.level, trim=not args.no_trim, loop=args.loop, stereo=args.stereo)
            if args.zip:
                import shutil
                z = shutil.make_archive(str(Path(args.out).resolve()), "zip", root_dir=Path(args.out).resolve().parent,
                                        base_dir=Path(args.out).resolve().name)
                print(f"zip: {z}")
            print(f"{len(man)} files in {args.out}")
        elif args.cmd == "audition":
            from .batch import audition
            from .macros import parse_macro_args
            audition(args.recipe, parse_macro_args(args.macro), args.variation, args.sr, args.seconds, args.out,
                     not args.no_play, args.watch)
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
            if "macros" in dd and not args.json:
                print(f"{dd['name']}  [category: {dd['category']}, {dd['duration']} s]\n")
                print("MACROS (primary controls; set with -m NAME=VALUE)")
                for m in dd["macros"]:
                    if "choices" in m:
                        print(f"  {m['name']:<12} default={m['default']!r:<8} choices: {', '.join(m['choices'])}")
                    else:
                        print(f"  {m['name']:<12} default={m['default']:<8} 0..1   jitter +-{m['jitter']}")
                    if m["doc"]:
                        print(f"  {'':<12} {m['doc']}")
                    for mv in m.get("moves", []):
                        print(f"  {'':<14}-> {mv}")
                if dd["vary"]:
                    print("\nVARIATIONS jitter: " + ", ".join(f"{k} {v}" for k, v in dd["vary"].items()))
                print("\nLAYERS")
                for L in dd["layers"]:
                    print(f"  {L['index']}: {L['name']:<10} " + " -> ".join(L["blocks"]))
                if dd["post"]:
                    print("  post: " + " -> ".join(dd["post"]))
            elif not _print(dd, args.json):
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
        elif args.cmd == "resynth":
            from . import resynth as RS
            from . import fit as F
            from .audio_io import read_audio, write_wav
            x, sr = read_audio(args.ref)
            hold = []
            for h in args.holdout:
                parts = h.split(":")
                hx, hsr = read_audio(parts[0])
                a = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
                b = float(parts[2]) if len(parts) > 2 and parts[2] else len(hx) / hsr
                hold.append((Path(parts[0]).name, hx, hsr, a, b, "fresh"))
            hs = tuple(float(v) for v in args.holdout_section.split(":")) if args.holdout_section else None
            fs = None
            if args.fit_start is not None or args.fit_end is not None:
                fs = (args.fit_start or 0.0, args.fit_end or len(x) / sr)
            res = RS.run(x, sr, args.kind, Path(args.ref).stem, fs, hold, hs, args.clean_tonal, args.budget)
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / "recipe.json").write_text(json.dumps(res["recipe"], indent=1))
            y, ysr = soundlab.render(res["recipe"])
            write_wav(out / "resynth.wav", y, ysr)
            rep = {k: res[k] for k in ("kind", "fit_section", "fit_before", "fit_after", "full", "full_other_seed",
                                       "holdout", "pitch_error_cents", "params", "tonal_removed_fraction",
                                       "features")}
            (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
            print(json.dumps({"out": str(out), "kind": res["kind"], "fit_after": res["fit_after"]["total"],
                              "full": res["full"]["total"], "other_seed": res["full_other_seed"]["total"],
                              "holdout": [(h["name"], h["total"]) for h in res["holdout"]],
                              "pitch_error_cents": res["pitch_error_cents"]}, indent=1, default=float))
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
        elif args.cmd == "gui":
            from .gui import serve
            from .gui.server import REPO_RECIPES
            dirs = [REPO_RECIPES] + [Path(d) for d in (args.recipes or [])]
            serve(port=args.port, open_browser=not args.no_browser, recipe_dirs=dirs, out_dir=args.out,
                  pack_dir=args.pack)
        elif args.cmd == "blindtest":
            from .checkpoint import build_pack, report_pack, score_pack
            from .gui.server import default_out_dir
            if args.btcmd == "build":
                out = Path(args.out) if args.out else default_out_dir() / "blindtest"
                info = build_pack(Path(args.library), out, per_sound=args.per_sound, max_s=args.max_s,
                                  seed=args.seed, sounds=args.sounds, exempt=args.exempt)
                print(f"{info['clips']} clips from {len(info['sounds'])} sounds -> {out}")
                print("sounds: " + ", ".join(info["sounds"]))
                if info["skipped_no_recipe"]:
                    print("skipped (no recipe of that name): " + ", ".join(info["skipped_no_recipe"]))
                print(f"run it: soundlab gui --pack \"{out}\"   then open http://127.0.0.1:8766/blindtest")
            else:
                pack = Path(args.pack) if args.pack else default_out_dir() / "blindtest"
                if args.codes:
                    from .checkpoint import import_codes
                    text = " ".join(sys.stdin.read() if c == "-" else Path(c).read_text() for c in args.codes)
                    import_codes(pack, text, log=lambda m: print(m, file=sys.stderr))
                res = score_pack(pack, args.pass_mark)
                print(json.dumps(res, indent=1, default=str) if args.json else report_pack(res))
    except soundlab.RecipeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
