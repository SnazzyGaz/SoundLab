extends SceneTree
## Headless smoke test: godot --headless --path . --script res://dev/autotest.gd
## Walks across gravel, stone, the wooden bridge and through the stream, opens and closes the door, knocks,
## strikes the pipe, throws stones, steps inside the shed in the rain, and checks what played and how the beds moved.

var main: Node
var log: Array = []
var frame := 0
var fails := 0


func _initialize() -> void:
	main = load("res://main.tscn").instantiate()
	root.add_child(main)
	current_scene = main


func check(ok: bool, what: String) -> void:
	print(("PASS  " if ok else "FAIL  ") + what)
	if not ok:
		fails += 1


func count(bank: String) -> int:
	return log.filter(func(e): return e[0] == bank).size()


func walk(to: Vector3, frames: int) -> void:
	var p: CharacterBody3D = main.player
	var from := p.global_position
	for i in frames:
		var target := from.lerp(to, float(i + 1) / frames)
		var v := (target - p.global_position) * 60.0
		p.velocity = Vector3(v.x, p.velocity.y, v.z)
		Input.action_press("move_forward")   # so the controller does not brake
		p.look_at(Vector3(to.x, p.global_position.y, to.z) + Vector3(0.001, 0, 0.001))
		await physics_frame
	Input.action_release("move_forward")


func bed_db(name: String) -> float:
	return main.sounds.beds[name].volume_db


func _process(_delta: float) -> bool:
	frame += 1
	if frame == 2:
		main.sounds.played.connect(func(b, k): log.append([b, k]))
		run()
	return false


func run() -> void:
	await process_frame
	var s = main.sounds
	check(s.banks.size() == 11, "11 one-shot banks loaded (%d)" % s.banks.size())
	check(s.loops.size() == 9, "9 loops loaded (%d)" % s.loops.size())
	check(s.loops["wind_gale"] is AudioStreamOggVorbis and s.loops["wind_gale"].loop, "beds loop")
	# gravel -> stone path -> bridge -> into the stream
	await walk(Vector3(0, 0.1, 1), 120)
	await walk(Vector3(0, 0.1, -9.5), 200)
	await walk(Vector3(0, 0.1, -13), 80)
	main.player.global_position = Vector3(5, 0.0, -12)
	await walk(Vector3(8, -0.5, -12), 90)
	var steps := count("step_gravel") + count("step_stone") + count("step_wood")
	check(count("step_stone") >= 3, "stone steps on the path (%d)" % count("step_stone"))
	check(count("step_wood") >= 2, "wood steps on the bridge (%d)" % count("step_wood"))
	check(count("splash") >= 1, "wading splashes in the stream (%d)" % count("splash"))
	var takes := log.filter(func(e): return e[0] == "step_stone").map(func(e): return e[1])
	var repeats := 0
	for i in range(1, takes.size()):
		if takes[i] == takes[i - 1]:
			repeats += 1
	check(repeats == 0, "no take played twice in a row (%d repeats over %d steps)" % [repeats, steps])
	# door, knock
	main.player.global_position = Vector3(7.5, 0.1, 0)
	await physics_frame
	main._knock()
	await create_timer(0.8).timeout
	check(count("knock") == 3, "three knocks (%d)" % count("knock"))
	main._toggle_door()
	await create_timer(1.7).timeout
	check(main.door_open and count("door_open") == 1, "door opens with a creak")
	main._toggle_door()
	await create_timer(0.9).timeout
	check(not main.door_open and count("door_close") == 1 and count("door_slam") == 1, "door closes with a short creak and a slam")
	# pipe strike and thrown stones
	main.player.global_position = Vector3(-3, 0.1, -3)
	main.player.rotation = Vector3.ZERO
	main.player.camera.rotation.x = deg_to_rad(-20)
	await physics_frame
	main.strike_ahead()
	check(count("clang") >= 1, "striking the pipe clangs (%d)" % count("clang"))
	main.player.global_position = Vector3(5, 0.1, -5)
	main.player.rotation = Vector3.ZERO
	main.player.camera.rotation.x = deg_to_rad(-15)
	await physics_frame
	var splashes_before := count("splash")
	for i in 3:
		main._throw_stone()
		await create_timer(0.3).timeout
	await create_timer(2.0).timeout
	check(count("splash") - splashes_before >= 2, "stones thrown into the stream splash (%d of 3)" % (count("splash") - splashes_before))
	# weather parameters drive the beds
	main.wind = 0.0
	main.rain = 0.0
	await create_timer(0.2).timeout
	var calm := bed_db("wind_calm")
	var gale := bed_db("wind_gale")
	check(calm > -40 and gale <= -79, "calm wind: calm bed on (%.1f dB), gale off (%.1f)" % [calm, gale])
	check(bed_db("rain_light") <= -79 and bed_db("rain_heavy") <= -79, "no rain: rain beds silent")
	main.wind = 1.0
	main.rain = 1.0
	await create_timer(0.2).timeout
	check(bed_db("wind_gale") > -10 and bed_db("wind_calm") <= -79, "gale: gale bed on (%.1f dB)" % bed_db("wind_gale"))
	check(bed_db("rain_heavy") > -10, "heavy rain on (%.1f dB)" % bed_db("rain_heavy"))
	var out_heavy := bed_db("rain_heavy")
	main.player.global_position = Vector3(12, 0.1, 0)
	await create_timer(1.0).timeout
	var lp := AudioServer.get_bus_effect(AudioServer.get_bus_index("Ambience"), 0) as AudioEffectLowPassFilter
	check(main.indoor and bed_db("rain_roof") > -12, "inside the shed: rain on the roof (%.1f dB)" % bed_db("rain_roof"))
	check(bed_db("rain_heavy") < out_heavy - 6 and lp.cutoff_hz < 1500, "inside: outdoor rain down %.1f dB, low-pass %.0f Hz" % [out_heavy - bed_db("rain_heavy"), lp.cutoff_hz])
	check(main.pipe_howl.volume_db > -20, "the pipe howls in a gale (%.1f dB)" % main.pipe_howl.volume_db)
	print("played: ", log.size(), " one-shots; ", fails, " failures")
	quit(1 if fails else 0)
