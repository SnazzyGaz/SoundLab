"""Importing this package registers every block and sound model."""
from . import excitation, resonator, radiation, propagation, room, sounds, synth, foley, library, resynth_blocks, sdt_methods, creature_voice, impacts, friction, creak, squeal_stems, air_stems, liquid_stems  # noqa: F401
from . import tails  # noqa: F401,E402  (after the blocks: sets their ring-out tails)
