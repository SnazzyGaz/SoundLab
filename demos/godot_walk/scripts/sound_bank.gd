extends Node
## Loads the SoundLab sound set (sounds/manifest.json) and plays it the way a game does:
## one-shots from variation banks (random, never the same take twice in a row, small pitch/level jitter),
## and ambience beds as layered seamless loops whose levels follow game parameters (wind, rain, indoors).

signal played(bank: String, take: int)

const SOUND_DIR := "res://sounds/"

var manifest: Dictionary = {}
var banks: Dictionary = {}        # name -> Array[AudioStream]
var last_pick: Dictionary = {}    # name -> index of the last take played
var loops: Dictionary = {}        # name -> AudioStream (looping)
var beds: Dictionary = {}         # name -> AudioStreamPlayer (2D beds)
var rng := RandomNumberGenerator.new()


func _ready() -> void:
	rng.randomize()
	_make_buses()
	var f := FileAccess.open(SOUND_DIR + "manifest.json", FileAccess.READ)
	if f == null:
		push_error("SoundLab Walk: sounds/manifest.json missing. Render it with: python tools/godot_export.py demos/godot_walk/sounds")
		return
	manifest = JSON.parse_string(f.get_as_text())
	for key in manifest["oneshots"]:
		var takes: Array[AudioStream] = []
		for file in manifest["oneshots"][key]["files"]:
			var s: AudioStream = load(SOUND_DIR + file)
			if s:
				takes.append(s)
		banks[key] = takes
	for key in manifest["loops"]:
		var s: AudioStream = load(SOUND_DIR + manifest["loops"][key]["file"])
		if s is AudioStreamOggVorbis:
			s.loop = true
		loops[key] = s


func _make_buses() -> void:
	# Master <- Ambience (2D beds, with an indoor low-pass), World (3D loops), SFX (one-shots)
	for bus in ["Ambience", "World", "SFX"]:
		if AudioServer.get_bus_index(bus) == -1:
			AudioServer.add_bus()
			var i := AudioServer.bus_count - 1
			AudioServer.set_bus_name(i, bus)
			AudioServer.set_bus_send(i, "Master")
	# SFX sit a little under the beds' headroom; a limiter on the master keeps overlapping hits (a clang over
	# running footsteps over a gale) from clipping
	AudioServer.set_bus_volume_db(AudioServer.get_bus_index("SFX"), -3.0)
	if AudioServer.get_bus_effect_count(0) == 0:
		var lim := AudioEffectHardLimiter.new()
		lim.ceiling_db = -1.0
		AudioServer.add_bus_effect(0, lim)
	var amb := AudioServer.get_bus_index("Ambience")
	if AudioServer.get_bus_effect_count(amb) == 0:
		var lp := AudioEffectLowPassFilter.new()
		lp.cutoff_hz = 20000.0
		lp.resonance = 0.5
		AudioServer.add_bus_effect(amb, lp)


func set_indoor_filter(cutoff_hz: float) -> void:
	var lp := AudioServer.get_bus_effect(AudioServer.get_bus_index("Ambience"), 0) as AudioEffectLowPassFilter
	if lp:
		lp.cutoff_hz = cutoff_hz


func gain_db(loop_name: String) -> float:
	return float(manifest["loops"][loop_name]["gain_db"]) if manifest.has("loops") and manifest["loops"].has(loop_name) else 0.0


## How long a door swing lasts, as rendered (manifest "open_s"), so the animation matches the creak.
func swing_s(bank: String, fallback: float) -> float:
	if manifest.has("oneshots") and manifest["oneshots"].has(bank):
		return float(manifest["oneshots"][bank].get("open_s", fallback))
	return fallback


## Pick a take: random, but never the one just played (the "machine-gun" effect).
func pick(bank: String) -> AudioStream:
	var takes: Array = banks.get(bank, [])
	if takes.is_empty():
		return null
	var i := rng.randi_range(0, takes.size() - 1)
	if takes.size() > 1 and i == last_pick.get(bank, -1):
		i = (i + 1 + rng.randi_range(0, takes.size() - 2)) % takes.size()
	last_pick[bank] = i
	return takes[i]


## 2D one-shot (the listener's own footsteps).
func play(bank: String, volume_db := 0.0, pitch := 1.0) -> void:
	var s := pick(bank)
	if s == null:
		return
	played.emit(bank, last_pick[bank])
	var p := AudioStreamPlayer.new()
	p.stream = s
	p.bus = "SFX"
	p.volume_db = volume_db + rng.randf_range(-1.5, 1.5)
	p.pitch_scale = pitch * rng.randf_range(0.97, 1.03)
	add_child(p)
	p.finished.connect(p.queue_free)
	p.play()


## 3D one-shot at a world position (door, knock, clang, splash, a thrown stone).
func play_at(bank: String, pos: Vector3, volume_db := 0.0, pitch := 1.0) -> void:
	var s := pick(bank)
	if s == null:
		return
	played.emit(bank, last_pick[bank])
	var p := AudioStreamPlayer3D.new()
	p.stream = s
	p.bus = "SFX"
	p.unit_size = 4.0
	p.max_db = 3.0
	p.volume_db = volume_db + rng.randf_range(-1.5, 1.5)
	p.pitch_scale = pitch * rng.randf_range(0.97, 1.03)
	p.attenuation_filter_cutoff_hz = 9000.0   # air absorption with distance
	get_tree().current_scene.add_child(p)
	p.global_position = pos
	p.finished.connect(p.queue_free)
	p.play()


## A looping 3D emitter placed in the world (fire, stream, howling pipe).
func emitter(loop_name: String, parent: Node3D, pos: Vector3, unit_size := 3.0, volume_db := 0.0) -> AudioStreamPlayer3D:
	var p := AudioStreamPlayer3D.new()
	p.stream = loops.get(loop_name)
	p.bus = "World"
	p.unit_size = unit_size
	p.volume_db = volume_db + gain_db(loop_name)
	p.attenuation_filter_cutoff_hz = 7000.0
	parent.add_child(p)
	p.position = pos
	if p.stream:
		p.play(rng.randf_range(0.0, 19.0))    # start each copy at its own point in the loop
	return p


## A 2D ambience bed (silent until set_bed is called).
func bed(loop_name: String) -> AudioStreamPlayer:
	var p := AudioStreamPlayer.new()
	p.stream = loops.get(loop_name)
	p.bus = "Ambience"
	p.volume_db = -80.0
	add_child(p)
	if p.stream:
		p.play(rng.randf_range(0.0, 19.0))
	beds[loop_name] = p
	return p


## Set a bed's level from a 0..1 weight (equal-power) plus its designer gain.
func set_bed(loop_name: String, weight: float, extra_db := 0.0) -> void:
	var p: AudioStreamPlayer = beds.get(loop_name)
	if p == null:
		return
	p.volume_db = -80.0 if weight <= 0.001 else linear_to_db(weight) + gain_db(loop_name) + extra_db
