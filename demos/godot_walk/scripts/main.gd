extends Node3D
## SoundLab Walk: a small yard built in code (no imported art), with every sound rendered by SoundLab.
##
## Live control the way a game mixes: wind and rain are 0..1 parameters that crossfade layered seamless loops
## (calm / breeze / gale wind, light / heavy rain) with equal-power curves; stepping into the shed crossfades the
## rain to rain-on-the-roof and low-passes the outdoor beds; the fire, the stream and a pipe that howls with the
## wind are 3D emitters; footsteps, the door, knocks, pipe strikes, stone impacts and splashes are variation
## banks played with no immediate repeats.

const SoundBank := preload("res://scripts/sound_bank.gd")
const Player := preload("res://scripts/player.gd")
const Stone := preload("res://scripts/stone.gd")

var sounds: SoundBank
var player: Player
var wind := 0.35
var rain := 0.0
var indoor := false
var indoor_mix := 0.0
var door_hinge: Node3D
var door_open := false
var door_busy := false
var pipe_body: StaticBody3D
var pipe_howl: AudioStreamPlayer3D
var fire_light: OmniLight3D
var rain_fx: CPUParticles3D
var rain_amount_step := -1
var env: Environment
var sky_mat: ProceduralSkyMaterial
var sun: DirectionalLight3D
var hud_wind: ProgressBar
var hud_rain: ProgressBar
var hud_prompt: Label
var t := 0.0

const SHED := Vector3(12, 0, 0)
const DOOR_POS := Vector3(9, 0, 0)
const STREAM_Z := -12.0


func _ready() -> void:
	_inputs()
	sounds = SoundBank.new()
	sounds.name = "Sounds"
	add_child(sounds)
	_environment()
	_ground()
	_stream()
	_shed()
	_campfire(Vector3(-6, 0, 4))
	_pipe(Vector3(-3, 0, -5))
	for p in [Vector3(-14, 0, -4), Vector3(-10, 0, 10), Vector3(5, 0, 12), Vector3(18, 0, 8), Vector3(-18, 0, 6),
			Vector3(20, 0, -6), Vector3(-6, 0, -20), Vector3(8, 0, -19)]:
		_tree(p)
	for b in ["wind_calm", "wind_breeze", "wind_gale", "rain_light", "rain_heavy", "rain_roof"]:
		sounds.bed(b)
	if sounds.beds.has("rain_roof"):
		sounds.beds["rain_roof"].bus = "World"   # the roof is heard unfiltered from inside
	player = Player.new()
	player.name = "Player"
	add_child(player)
	player.setup(sounds)
	player.global_position = Vector3(0, 0.1, 6)
	_rain_fx()
	_hud()
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED


# ------------------------------------------------------------------ input
func _inputs() -> void:
	var keys := {
		"move_forward": [KEY_W, KEY_UP], "move_back": [KEY_S, KEY_DOWN], "move_left": [KEY_A, KEY_LEFT],
		"move_right": [KEY_D, KEY_RIGHT], "sprint": [KEY_SHIFT], "jump": [KEY_SPACE], "interact": [KEY_E],
		"knock": [KEY_Q], "throw": [KEY_F], "wind_down": [KEY_1], "wind_up": [KEY_2], "rain_down": [KEY_3],
		"rain_up": [KEY_4], "release_mouse": [KEY_ESCAPE],
	}
	for action in keys:
		if not InputMap.has_action(action):
			InputMap.add_action(action)
		for k in keys[action]:
			var ev := InputEventKey.new()
			ev.physical_keycode = k
			InputMap.action_add_event(action, ev)
	if not InputMap.has_action("strike"):
		InputMap.add_action("strike")
		var mb := InputEventMouseButton.new()
		mb.button_index = MOUSE_BUTTON_LEFT
		InputMap.action_add_event("strike", mb)


func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("release_mouse"):
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
	elif event is InputEventMouseButton and event.pressed and Input.mouse_mode != Input.MOUSE_MODE_CAPTURED:
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
		get_viewport().set_input_as_handled()
	elif event.is_action_pressed("interact"):
		_toggle_door()
	elif event.is_action_pressed("knock"):
		_knock()
	elif event.is_action_pressed("throw"):
		_throw_stone()
	elif event.is_action_pressed("strike"):
		_strike()


# ------------------------------------------------------------------ frame
func _process(delta: float) -> void:
	t += delta
	if Input.is_action_pressed("wind_up"):
		wind = minf(1.0, wind + 0.35 * delta)
	if Input.is_action_pressed("wind_down"):
		wind = maxf(0.0, wind - 0.35 * delta)
	if Input.is_action_pressed("rain_up"):
		rain = minf(1.0, rain + 0.35 * delta)
	if Input.is_action_pressed("rain_down"):
		rain = maxf(0.0, rain - 0.35 * delta)
	if player:
		indoor = _inside_shed(player.global_position)
	indoor_mix = move_toward(indoor_mix, 1.0 if indoor else 0.0, delta * 2.5)
	_mix()
	_visuals(delta)
	_prompt()


static func xfade(x: float, centre: float, half_width: float) -> float:
	## equal-power triangular weight: 1 at centre, 0 at +-half_width
	var d := clampf(absf(x - centre) / half_width, 0.0, 1.0)
	return cos(d * PI * 0.5)


func _mix() -> void:
	var out := 1.0 - indoor_mix
	var inside_db := -9.0 * indoor_mix
	sounds.set_bed("wind_calm", xfade(wind, 0.0, 0.5), inside_db)
	sounds.set_bed("wind_breeze", xfade(wind, 0.5, 0.5), inside_db)
	sounds.set_bed("wind_gale", xfade(wind, 1.0, 0.5), inside_db)
	var light := sin(clampf(rain / 0.33, 0.0, 1.0) * PI * 0.5) if rain < 0.33 else xfade(rain, 0.33, 0.67)
	var heavy := 0.0 if rain < 0.33 else sin(clampf((rain - 0.33) / 0.67, 0.0, 1.0) * PI * 0.5)
	sounds.set_bed("rain_light", light * sqrt(maxf(out, 0.0)) + light * 0.3 * indoor_mix, inside_db)
	sounds.set_bed("rain_heavy", heavy * sqrt(maxf(out, 0.0)) + heavy * 0.3 * indoor_mix, inside_db)
	sounds.set_bed("rain_roof", sqrt(rain) * sqrt(indoor_mix), -6.0 + 6.0 * rain)
	sounds.set_indoor_filter(lerpf(20000.0, 900.0, indoor_mix))
	if pipe_howl:
		# the pipe only sings once the wind is up
		var w := clampf((wind - 0.25) / 0.75, 0.0, 1.0)
		pipe_howl.volume_db = -80.0 if w <= 0.0 else linear_to_db(w * w) + sounds.gain_db("pipe_howl") + 2.0
		pipe_howl.pitch_scale = 0.85 + 0.3 * wind


func _visuals(_delta: float) -> void:
	if fire_light:
		fire_light.light_energy = 2.2 + 0.5 * sin(t * 13.0) * sin(t * 7.3) + 0.25 * sin(t * 23.0)
	if sky_mat:
		var dark := rain * 0.75
		sky_mat.sky_top_color = Color(0.32, 0.5, 0.78).lerp(Color(0.25, 0.27, 0.3), dark)
		sky_mat.sky_horizon_color = Color(0.7, 0.75, 0.8).lerp(Color(0.42, 0.44, 0.46), dark)
		sky_mat.ground_horizon_color = sky_mat.sky_horizon_color
		sky_mat.ground_bottom_color = Color(0.2, 0.19, 0.17)
	if sun:
		sun.light_energy = lerpf(0.6, 0.2, rain)
	if env:
		env.ambient_light_energy = lerpf(0.55, 0.4, rain)
	if rain_fx and player:
		rain_fx.global_position = player.global_position + Vector3(0, 10, 0)
		var step := int(round(rain * 8.0))
		rain_fx.emitting = step > 0
		rain_fx.visible = not indoor and step > 0
		if step != rain_amount_step and step > 0:
			rain_amount_step = step
			rain_fx.amount = 120 * step
	for tree in get_tree().get_nodes_in_group("sway"):
		tree.rotation.z = (0.01 + 0.06 * wind * wind) * sin(t * (1.2 + wind) + tree.position.x)


func _prompt() -> void:
	if hud_prompt == null or player == null:
		return
	var msg := ""
	if player.global_position.distance_to(DOOR_POS) < 2.6:
		msg = "E  close the door" if door_open else "E  open the door      Q  knock"
	elif player.global_position.distance_to(pipe_body.global_position) < 3.0:
		msg = "Left click  strike the pipe"
	hud_wind.value = wind
	hud_rain.value = rain
	hud_prompt.text = msg


# ------------------------------------------------------------------ actions
func _toggle_door() -> void:
	if door_busy or player.global_position.distance_to(DOOR_POS) > 2.6:
		return
	door_busy = true
	var tw := create_tween()
	# the swing takes as long as the rendered creak (the export sets the recipe's open_s to the animation time)
	if door_open:
		sounds.play_at("door_close", DOOR_POS + Vector3(0, 1.2, 0), -4.0)
		tw.tween_property(door_hinge, "rotation:y", 0.0, sounds.swing_s("door_close", 0.55)).set_trans(Tween.TRANS_QUAD).set_ease(Tween.EASE_IN)
		tw.tween_callback(func(): sounds.play_at("door_slam", DOOR_POS + Vector3(0, 1.0, 0), 0.0))
	else:
		sounds.play_at("door_open", DOOR_POS + Vector3(0, 1.2, 0), 0.0)
		tw.tween_property(door_hinge, "rotation:y", deg_to_rad(-105.0), sounds.swing_s("door_open", 1.4)).set_trans(Tween.TRANS_SINE).set_ease(Tween.EASE_IN_OUT)
	tw.tween_callback(func():
		door_open = not door_open
		door_busy = false)


func _knock() -> void:
	if door_open or player.global_position.distance_to(DOOR_POS) > 2.6:
		return
	for i in 3:
		get_tree().create_timer(0.22 * i).timeout.connect(
			func(): sounds.play_at("knock", DOOR_POS + Vector3(-0.05, 1.3, 0.2), -2.0 if i < 2 else 0.0))


func _strike() -> void:
	if Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		strike_ahead()


## Strike whatever metal is under the crosshair within arm's reach.
func strike_ahead() -> void:
	var cam := player.camera
	var q := PhysicsRayQueryParameters3D.create(cam.global_position, cam.global_position - cam.global_basis.z * 3.0)
	var ex: Array[RID] = [player.get_rid()]
	q.exclude = ex
	var hit := get_world_3d().direct_space_state.intersect_ray(q)
	if hit and hit.collider == pipe_body:
		sounds.play_at("clang", hit.position, 0.0)


func _throw_stone() -> void:
	var cam := player.camera
	var s := Stone.new()
	add_child(s)
	s.setup(sounds)
	s.global_position = cam.global_position - cam.global_basis.z * 0.6 + Vector3(0, -0.15, 0)
	s.linear_velocity = -cam.global_basis.z * 11.0 + Vector3(0, 2.5, 0) + player.velocity * 0.5
	s.angular_velocity = Vector3(randf() - 0.5, randf() - 0.5, randf() - 0.5) * 10.0


func _inside_shed(p: Vector3) -> bool:
	return absf(p.x - SHED.x) < 2.9 and absf(p.z - SHED.z) < 2.4 and p.y < 2.8


# ------------------------------------------------------------------ world building
func _mat(c: Color, rough := 0.9) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = c
	m.roughness = rough
	return m


func _box(size: Vector3, pos: Vector3, color: Color, surface := "", solid := true, parent: Node = null) -> Node3D:
	var root: Node3D = StaticBody3D.new() if solid else Node3D.new()
	if surface != "":
		root.set_meta("surface", surface)
	var mi := MeshInstance3D.new()
	var bm := BoxMesh.new()
	bm.size = size
	mi.mesh = bm
	mi.material_override = _mat(color)
	root.add_child(mi)
	if solid:
		var cs := CollisionShape3D.new()
		var sh := BoxShape3D.new()
		sh.size = size
		cs.shape = sh
		root.add_child(cs)
	(parent if parent else self).add_child(root)
	root.position = pos
	return root


func _environment() -> void:
	var we := WorldEnvironment.new()
	env = Environment.new()
	env.background_mode = Environment.BG_SKY
	var sky := Sky.new()
	sky_mat = ProceduralSkyMaterial.new()
	sky.sky_material = sky_mat
	env.sky = sky
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.62, 0.64, 0.68)
	env.ambient_light_energy = 0.55
	env.reflected_light_source = Environment.REFLECTION_SOURCE_DISABLED
	env.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	env.fog_enabled = false
	env.fog_light_color = Color(0.5, 0.54, 0.58)
	env.fog_density = 0.0005
	we.environment = env
	add_child(we)
	sun = DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-50, -35, 0)
	sun.shadow_enabled = true
	add_child(sun)


func _ground() -> void:
	# gravel yard in two halves either side of the stream channel, a stone path to the shed
	var gravel := Color(0.36, 0.33, 0.29)
	_box(Vector3(80, 1, 70), Vector3(0, -0.5, -10 + 35 + 0), gravel, "gravel")          # z from -10 to +60
	_box(Vector3(80, 1, 40), Vector3(0, -0.5, -14 - 20), gravel, "gravel")             # z from -14 to -54
	_box(Vector3(10.5, 0.04, 1.8), Vector3(3.75, 0.0, 0), Color(0.42, 0.42, 0.42), "stone")   # path to the door
	_box(Vector3(1.8, 0.04, 6.0), Vector3(0, 0.0, 3.0), Color(0.42, 0.42, 0.42), "stone")     # path from the start


func _stream() -> void:
	# channel bed (wading in it splashes), water surface, a wooden footbridge across
	_box(Vector3(80, 0.4, 4), Vector3(0, -0.8, STREAM_Z), Color(0.35, 0.33, 0.3), "water")
	var water := MeshInstance3D.new()
	var pm := PlaneMesh.new()
	pm.size = Vector2(80, 4)
	water.mesh = pm
	var wm := _mat(Color(0.16, 0.32, 0.38, 0.75), 0.1)
	wm.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	water.material_override = wm
	add_child(water)
	water.position = Vector3(0, -0.25, STREAM_Z)
	var area := Area3D.new()
	var cs := CollisionShape3D.new()
	var sh := BoxShape3D.new()
	sh.size = Vector3(80, 0.5, 4)
	cs.shape = sh
	area.add_child(cs)
	add_child(area)
	area.position = Vector3(0, -0.45, STREAM_Z)
	area.body_entered.connect(func(b):
		if b.has_method("splash"):
			b.splash())
	_box(Vector3(2.4, 0.12, 6.0), Vector3(0, -0.04, STREAM_Z), Color(0.48, 0.34, 0.2), "wood")    # footbridge
	for x in [-1.15, 1.15]:
		_box(Vector3(0.08, 0.9, 6.0), Vector3(x, 0.5, STREAM_Z), Color(0.42, 0.3, 0.18), "wood")  # rails
	for x in [-20.0, -6.0, 6.0, 20.0]:
		sounds.emitter("stream", self, Vector3(x, -0.2, STREAM_Z), 3.5, -2.0)


func _shed() -> void:
	var wood := Color(0.5, 0.36, 0.22)
	var c := SHED
	_box(Vector3(6, 0.1, 5), c + Vector3(0, -0.03, 0), Color(0.55, 0.4, 0.25), "wood")            # floor
	_box(Vector3(0.15, 3, 5), c + Vector3(3, 1.5, 0), wood, "wood")                              # east wall
	_box(Vector3(6, 3, 0.15), c + Vector3(0, 1.5, -2.5), wood, "wood")                           # north wall
	_box(Vector3(6, 3, 0.15), c + Vector3(0, 1.5, 2.5), wood, "wood")                            # south wall
	_box(Vector3(0.15, 3, 1.95), c + Vector3(-3, 1.5, -1.525), wood, "wood")                     # west wall, left
	_box(Vector3(0.15, 3, 1.95), c + Vector3(-3, 1.5, 1.525), wood, "wood")                      # west wall, right
	_box(Vector3(0.15, 0.85, 1.1), c + Vector3(-3, 2.575, 0), wood, "wood")                      # lintel
	_box(Vector3(6.6, 0.15, 5.6), c + Vector3(0, 3.07, 0), Color(0.3, 0.22, 0.16), "wood")       # roof
	door_hinge = Node3D.new()
	add_child(door_hinge)
	door_hinge.position = DOOR_POS + Vector3(0, 0, 0.55)
	_box(Vector3(0.06, 2.1, 1.08), Vector3(0, 1.08, -0.55), Color(0.4, 0.27, 0.16), "wood", true, door_hinge)
	var lamp := OmniLight3D.new()
	lamp.light_color = Color(1.0, 0.85, 0.6)
	lamp.light_energy = 0.8
	lamp.omni_range = 5.0
	add_child(lamp)
	lamp.position = c + Vector3(0, 2.6, 0)


func _campfire(p: Vector3) -> void:
	for i in 8:
		var a := TAU * i / 8.0
		_box(Vector3(0.3, 0.2, 0.3), p + Vector3(cos(a) * 0.75, 0.1, sin(a) * 0.75), Color(0.45, 0.44, 0.42), "stone")
	for i in 4:
		var log_node := _box(Vector3(0.9, 0.12, 0.12), p + Vector3(0, 0.12, 0), Color(0.3, 0.2, 0.12), "wood", false)
		log_node.rotation.y = PI * i / 4.0
	var flame := MeshInstance3D.new()
	var fm := SphereMesh.new()
	fm.radius = 0.25
	fm.height = 0.7
	flame.mesh = fm
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.albedo_color = Color(1.0, 0.55, 0.15)
	flame.material_override = mat
	add_child(flame)
	flame.position = p + Vector3(0, 0.45, 0)
	fire_light = OmniLight3D.new()
	fire_light.light_color = Color(1.0, 0.6, 0.25)
	fire_light.omni_range = 9.0
	add_child(fire_light)
	fire_light.position = p + Vector3(0, 0.8, 0)
	sounds.emitter("fire", self, p + Vector3(0, 0.4, 0), 2.5, 0.0)


func _pipe(p: Vector3) -> void:
	pipe_body = StaticBody3D.new()
	pipe_body.set_meta("surface", "metal")
	var mi := MeshInstance3D.new()
	var cm := CylinderMesh.new()
	cm.top_radius = 0.06
	cm.bottom_radius = 0.06
	cm.height = 3.2
	mi.mesh = cm
	mi.material_override = _mat(Color(0.55, 0.57, 0.6), 0.35)
	pipe_body.add_child(mi)
	var cs := CollisionShape3D.new()
	var sh := CylinderShape3D.new()
	sh.radius = 0.08
	sh.height = 3.2
	cs.shape = sh
	pipe_body.add_child(cs)
	add_child(pipe_body)
	pipe_body.position = p + Vector3(0, 1.6, 0)
	pipe_howl = sounds.emitter("pipe_howl", self, p + Vector3(0, 3.1, 0), 4.0, 0.0)


func _tree(p: Vector3) -> void:
	var tree := Node3D.new()
	add_child(tree)
	tree.position = p
	tree.add_to_group("sway")
	var trunk := MeshInstance3D.new()
	var cm := CylinderMesh.new()
	cm.top_radius = 0.15
	cm.bottom_radius = 0.22
	cm.height = 2.5
	trunk.mesh = cm
	trunk.material_override = _mat(Color(0.33, 0.24, 0.16))
	tree.add_child(trunk)
	trunk.position.y = 1.25
	var crown := MeshInstance3D.new()
	var cone := CylinderMesh.new()
	cone.top_radius = 0.0
	cone.bottom_radius = 1.6
	cone.height = 4.0
	crown.mesh = cone
	crown.material_override = _mat(Color(0.2, 0.36, 0.2))
	tree.add_child(crown)
	crown.position.y = 4.2
	var body := StaticBody3D.new()
	body.set_meta("surface", "wood")
	var cs := CollisionShape3D.new()
	var sh := CylinderShape3D.new()
	sh.radius = 0.22
	sh.height = 2.5
	cs.shape = sh
	body.add_child(cs)
	tree.add_child(body)
	body.position.y = 1.25


func _rain_fx() -> void:
	rain_fx = CPUParticles3D.new()
	rain_fx.amount = 120
	rain_fx.lifetime = 1.0
	rain_fx.emission_shape = CPUParticles3D.EMISSION_SHAPE_BOX
	rain_fx.emission_box_extents = Vector3(14, 0.5, 14)
	rain_fx.direction = Vector3(0, -1, 0)
	rain_fx.spread = 3.0
	rain_fx.initial_velocity_min = 12.0
	rain_fx.initial_velocity_max = 14.0
	rain_fx.gravity = Vector3(0, -9.8, 0)
	var m := BoxMesh.new()
	m.size = Vector3(0.012, 0.35, 0.012)
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.albedo_color = Color(0.75, 0.8, 0.9, 0.45)
	m.material = mat
	rain_fx.mesh = m
	rain_fx.emitting = false
	add_child(rain_fx)


func _hud() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	var help := Label.new()
	help.text = ("SoundLab Walk: every sound here is rendered by SoundLab\n" +
		"WASD move   Shift run   Space jump   mouse look   Esc free the mouse\n" +
		"1 / 2  wind down / up      3 / 4  rain down / up\n" +
		"F  throw a stone (try the pipe, the bridge, the stream)   E  door   Q  knock   click  strike the pipe")
	help.position = Vector2(16, 12)
	help.add_theme_color_override("font_outline_color", Color.BLACK)
	help.add_theme_constant_override("outline_size", 4)
	layer.add_child(help)
	var box := VBoxContainer.new()
	box.position = Vector2(16, 600)
	box.custom_minimum_size = Vector2(260, 0)
	layer.add_child(box)
	for spec in [["Wind", "w"], ["Rain", "r"]]:
		var row := HBoxContainer.new()
		var l := Label.new()
		l.text = spec[0]
		l.custom_minimum_size = Vector2(50, 0)
		l.add_theme_color_override("font_outline_color", Color.BLACK)
		l.add_theme_constant_override("outline_size", 4)
		var bar := ProgressBar.new()
		bar.min_value = 0.0
		bar.max_value = 1.0
		bar.step = 0.01
		bar.show_percentage = false
		bar.custom_minimum_size = Vector2(200, 16)
		row.add_child(l)
		row.add_child(bar)
		box.add_child(row)
		if spec[1] == "w":
			hud_wind = bar
		else:
			hud_rain = bar
	hud_prompt = Label.new()
	hud_prompt.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	hud_prompt.anchor_left = 0.0
	hud_prompt.anchor_right = 1.0
	hud_prompt.anchor_top = 0.62
	hud_prompt.anchor_bottom = 0.62
	hud_prompt.add_theme_color_override("font_outline_color", Color.BLACK)
	hud_prompt.add_theme_constant_override("outline_size", 5)
	layer.add_child(hud_prompt)
	var dot := ColorRect.new()
	dot.color = Color(1, 1, 1, 0.8)
	dot.size = Vector2(4, 4)
	dot.anchor_left = 0.5
	dot.anchor_top = 0.5
	dot.anchor_right = 0.5
	dot.anchor_bottom = 0.5
	dot.offset_left = -2
	dot.offset_top = -2
	dot.offset_right = 2
	dot.offset_bottom = 2
	layer.add_child(dot)
	# the HUD must never take the mouse: with the mouse captured the cursor sits at the centre of the screen,
	# on the crosshair dot, and a control there would swallow every mouse-look motion event
	_ignore_mouse(layer)


func _ignore_mouse(n: Node) -> void:
	for c in n.get_children():
		if c is Control:
			c.mouse_filter = Control.MOUSE_FILTER_IGNORE
		_ignore_mouse(c)
