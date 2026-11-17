# SoundLab Walk (Godot 4.7)

A small first-person yard where every sound comes from SoundLab recipes: walk on gravel, a stone path and a
wooden footbridge, wade through the stream, open the shed door, knock, strike a metal pipe, throw stones, and
turn the wind and rain up and down.

## Run

1. Render the sounds (once, or after changing recipes):
   `python tools/godot_export.py demos/godot_walk/sounds`
2. Open `demos/godot_walk/project.godot` in Godot 4.7 (Compatibility renderer) and press Play.

| key | action |
|---|---|
| WASD, mouse | walk, look (Shift runs, Space jumps) |
| 1 / 2 | wind down / up |
| 3 / 4 | rain down / up |
| F | throw a stone (pipe: clang, wood: knock, stream: splash) |
| E / Q | open or close the shed door / knock on it |
| left click | strike the pipe |
| Esc | free the mouse (click to capture again) |

## How the sound works

- **Beds** (`scripts/sound_bank.gd`, `scripts/main.gd`): wind is three seamless stereo loops (calm, breeze,
  gale) and rain two (light, heavy), rendered from the `wind` and `rain` recipes with different macro settings.
  The 0..1 wind and rain parameters crossfade them with equal-power curves; the manifest's `gain_db` sets the
  layers apart in level (every file is loudness-normalised on export).
- **Indoors**: inside the shed the outdoor beds go through a low-pass (20 kHz -> 900 Hz) and drop 9 dB, and
  the rain crossfades to the `rain` recipe's `roof` surface.
- **3D emitters**: the fire, four points along the stream, and a pipe that howls with the wind (`wind_howl`,
  louder and higher as the wind rises), each a mono seamless loop started at a random point.
- **One-shots**: footsteps (`footstep` with surface wood / stone / gravel), door creak and slam, knocks, pipe
  clangs, splashes: banks of 3-8 SoundLab variations, picked at random but never the same take twice in a row,
  with +-1.5 dB and +-3 % pitch jitter. Footsteps follow stride length, surface comes from the floor's
  `surface` metadata.

This first version plays SoundLab renders (the way most games ship sound) and makes them respond live through
mixing. Rendering the cheap models live inside Godot (a GDExtension) is the next step.

## Check

`godot --headless --path . --script res://dev/autotest.gd` walks the scene, uses every action and checks what
played and how the beds moved (19 checks).
