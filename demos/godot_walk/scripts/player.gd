extends CharacterBody3D
## First-person walker. Footsteps come from SoundLab variation banks, chosen by the surface underfoot
## (each floor body carries metadata "surface": wood / stone / gravel), one step per stride length.

const WALK_SPEED := 3.2
const SPRINT_SPEED := 6.0
const JUMP_VELOCITY := 4.2
const MOUSE_SENS := 0.0025
# Step cadence, not a fixed stride: the controller moves faster than a real walker (3.2 m/s), so a 0.75 m
# stride gave 4-5 steps a second, a machine-gun patter. People walk at ~1.9 steps/s and run at ~2.7.
const WALK_CADENCE := 1.9
const SPRINT_CADENCE := 2.7
const STRIDE := WALK_SPEED / WALK_CADENCE
const SPRINT_STRIDE := SPRINT_SPEED / SPRINT_CADENCE

var sounds: Node
var camera: Camera3D
var feet: RayCast3D
var stride_left := 0.0
var was_on_floor := true
var gravity: float = ProjectSettings.get_setting("physics/3d/default_gravity", 9.8)


func setup(sound_bank: Node) -> void:
	sounds = sound_bank
	var shape := CollisionShape3D.new()
	var cap := CapsuleShape3D.new()
	cap.radius = 0.35
	cap.height = 1.75
	shape.shape = cap
	shape.position.y = 0.875
	add_child(shape)
	camera = Camera3D.new()
	camera.position.y = 1.62
	camera.fov = 75.0
	camera.current = true
	add_child(camera)
	feet = RayCast3D.new()
	feet.position.y = 0.2
	feet.target_position = Vector3(0, -0.6, 0)
	add_child(feet)


func _input(event: InputEvent) -> void:
	# _input, not _unhandled_input: mouse look must not depend on what the GUI does with the event
	if event is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		rotate_y(-event.relative.x * MOUSE_SENS)
		camera.rotate_x(-event.relative.y * MOUSE_SENS)
		camera.rotation.x = clamp(camera.rotation.x, deg_to_rad(-85), deg_to_rad(85))


func surface() -> String:
	if feet and feet.is_colliding():
		var c := feet.get_collider()
		if c and c.has_meta("surface"):
			return String(c.get_meta("surface"))
	return "gravel"


func _physics_process(delta: float) -> void:
	if not is_on_floor():
		velocity.y -= gravity * delta
	if Input.is_action_just_pressed("jump") and is_on_floor():
		velocity.y = JUMP_VELOCITY
	var sprint := Input.is_action_pressed("sprint")
	var speed := SPRINT_SPEED if sprint else WALK_SPEED
	var dir2 := Input.get_vector("move_left", "move_right", "move_forward", "move_back")
	var dir := (transform.basis * Vector3(dir2.x, 0, dir2.y)).normalized()
	if dir:
		velocity.x = dir.x * speed
		velocity.z = dir.z * speed
	else:
		velocity.x = move_toward(velocity.x, 0, speed * 8 * delta)
		velocity.z = move_toward(velocity.z, 0, speed * 8 * delta)
	var before := global_position
	move_and_slide()
	_footsteps(before, sprint)


func _footsteps(before: Vector3, sprint: bool) -> void:
	if sounds == null:
		return
	var on_floor := is_on_floor()
	if on_floor and not was_on_floor:
		# landing: a heavier step
		_step(-2.0, 0.92)
		stride_left = STRIDE * 0.5
	was_on_floor = on_floor
	if not on_floor:
		return
	var moved := Vector2(global_position.x - before.x, global_position.z - before.z).length()
	if moved < 0.0005:
		stride_left = min(stride_left, STRIDE * 0.35)   # the first step comes quickly after starting off
		return
	stride_left -= moved
	if stride_left <= 0.0:
		stride_left += SPRINT_STRIDE if sprint else STRIDE
		# running steps are louder and a touch brighter; the bank's own variations do the rest
		_step(-3.0 if sprint else -6.0, 1.04 if sprint else 1.0)


func _step(db: float, pitch: float) -> void:
	var s := surface()
	if s == "water":
		sounds.play("splash", db - 9.0, pitch * 1.35)   # wading in the stream
	else:
		sounds.play("step_" + s, db, pitch)
