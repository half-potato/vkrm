# vkrm
This is the Vulkan renderer for Radiance Meshes. The training code can be found [here](https://github.com/half-potato/radiance_meshes).
The training code emits `ckpt.ply` files that can be rendered using this program. 

Sorting in Vulkan is based on the work done by bones164 [here](https://github.com/b0nes164/GPUSorting).

# Controls
- WASD to move
- Q to go down, E to go up
- Shift to go faster
- Left click+drag to move the viewpoint
- Right click to select points, scroll to adjust selection depth
- Press `g` to initiate a move points, Press `g` to end the move.

# Install Instructions

## Windows:

## Ubuntu:
Install dependencies for Ubuntu:
```
sudo apt install build-essentials libxrandr-dev libx11-dev libvulkan-dev libxinerama-dev libxcursor-dev libxi-dev libxcb-keysyms1-dev
```
This could be necessary to execute:
```
echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/local-slanguages.conf
```
Then, finally, just run `cmake` and `make`.
```
cmake -S . -B build
cmake --build build -j8
```
