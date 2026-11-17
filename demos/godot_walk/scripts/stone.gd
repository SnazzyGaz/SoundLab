extends RigidBody3D
## A thrown stone. Every hard contact plays a SoundLab one-shot chosen by what it hit (metal pipe -> clang,
## wood -> knock, stone/gravel -> a small step-like thud), louder for faster hits. Falling in the stream
## plays a splash (the Water area calls splash()).

var sounds: Node
var cooldown := 0.0
var sunk := false


func setup(sound_bank: Node) -> void:
	sounds = sound_bank
	contact_monitor = true
	max_contacts_reported = 2
	continuous_cd = true
	var shape := CollisionShape3D.new()
	var sph := SphereShape3D.new()
	sph.radius = 0.07
	shape.shape = sph
	add_child(shape)
	var mesh := MeshInstance3D.new()
	var sm := SphereMesh.new()
	sm.radius = 0.07
	sm.height = 0.12
	mesh.mesh = sm
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(0.45, 0.43, 0.4)
	mat.roughness = 0.9
	mesh.material_override = mat
	add_child(mesh)
	mass = 0.25
	var pm := PhysicsMaterial.new()
	pm.bounce = 0.35
	pm.friction = 0.8
	physics_material_override = pm
	body_entered.connect(_on_hit)
	get_tree().create_timer(20.0).timeout.connect(queue_free)


func _physics_process(delta: float) -> void:
	cooldown = max(0.0, cooldown - delta)


func _on_hit(body: Node) -> void:
	if sounds == null or sunk or cooldown > 0.0:
		return
	var speed := linear_velocity.length()
	if speed < 1.0:
		return
	cooldown = 0.08
	var db := clampf(linear_to_db(speed / 8.0), -24.0, 0.0)
	var mat := String(body.get_meta("surface", "gravel"))
	match mat:
		"metal":
			sounds.play_at("clang", global_position, db - 2.0, 1.15)
		"wood":
			sounds.play_at("knock", global_position, db, 1.25)
		_:
			sounds.play_at("step_" + mat, global_position, db - 8.0, 1.6)


func splash() -> void:
	if sunk:
		return
	sunk = true
	if sounds:
		sounds.play_at("splash", global_position, clampf(linear_to_db(linear_velocity.length() / 6.0), -18.0, 0.0))
	linear_velocity *= 0.1
	gravity_scale = 0.15
	get_tree().create_timer(1.5).timeout.connect(queue_free)
