# Research notes

Background research behind SoundLab's models. These are working notes, not user documentation. Licences are
listed so that anything we port follows the rule in [../licensing.md](../licensing.md) and
[../../THIRD_PARTY.md](../../THIRD_PARTY.md): permissive code may be ported with attribution; GPL, AGPL and
non-commercial work is used as a method only.

| File | What it covers |
|---|---|
| [oss-survey.md](oss-survey.md) | Open-source sound generation knowledge base (snapshot at v0.19), by sound domain: contact, friction, liquids, air, machines, voice, toolkits, analysis. Method, where it lands in SoundLab, licence, status. |
| [creature-research.md](creature-research.md) | Why the v0.21 creature growl falls short, practitioner and bioacoustics evidence, resynthesis methods (WORLD, PSOLA, DDSP-style harmonic + noise, neural vocoders), a survey of CHIMERA forks with animal or creature work, the proposed frame-wise analysis-resynthesis rebuild, and a licence table. |

Tool: [`tools/world_probe.py`](../../tools/world_probe.py) reproduces the measurements in creature-research.md
section 2 on any recording (optional dependency: `pip install "soundlab[world]"`). It writes renders to a folder
you choose; keep that folder outside the repository.

Credit: Garyck Arntzen (https://x.com/_mintydoge).
