extends SceneTree
## Render a few screenshots (needs a display; e.g. xvfb-run). godot --path . --script res://dev/shot.gd
var main: Node
var f := 0
func _initialize() -> void:
	main = load("res://main.tscn").instantiate()
	root.add_child(main)
	current_scene = main
func _process(_d: float) -> bool:
	f += 1
	if f == 5:
		main.player.global_position = Vector3(-1, 0.1, 9)
		main.player.rotation.y = deg_to_rad(-20)
	if f == 40:
		root.get_texture().get_image().save_png("/tmp/shot1.png")
		main.rain = 0.8
		main.wind = 0.9
		main.player.global_position = Vector3(2, 0.1, -6)
		main.player.rotation.y = deg_to_rad(30)
	if f == 90:
		root.get_texture().get_image().save_png("/tmp/shot2.png")
		quit()
	return false
