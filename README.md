# XRPlay: Python/CUDA Video Player with OpenXR Support

XRPlay is a proof-of-concept, high-performance, command-line video player for desktop and VR, written entirely in Python (plus some Cuda kernels).

It leverages NVIDIA CUDA and Cupy for fast video decoding and OpenXR for optional VR headset support.  Currently it requires an NVIDIA GPU, but future AMD support is theoretically possible (especially if Cupy supports it).

I've tried to keep it as simple and modular as possible, without sacrificing speed.

Audio currently works but is a big hack (it pre-loads the entire audio track into memory).

Multiple VR projections are supported (SBS, Top-Bottom, Fisheye, all at 180, 360, or arbitrary).

If you launch it on a folder, it will index the folder (recursively) for videos and bring up a video navigator.  It (simply) expects .jpg files with the same root path as the video file for the thumbnails (hit T in the desktop viewer while paused to save the current frame as a thumbnail).  The navigator works in both desktop and VR modes.  The navigator supports tagging, ratings, filtering and sorting.

VR rendering goes from the video decode buffer to the OpenXR swapchain image via a single Cuda kernel--no needless frame copies.  It achieves smooth playback, with head tracking, of high-resolution 180 SBS VR videos (e.g., 6Kx3K@60fps to a Quest 3 via WiVRn without breaking a sweat) which other OpenXR players I found, even though written in faster languages, couldn't keep up with.

Despite all the features, it was thrown together rather quickly, so expect rough edges:  It's a proof-of-concept...

Ping me if you try it.  As far as I'm aware, nobody has yet.

### System Dependencies
- **Hardware**: NVIDIA GPU (required for CUDA).
- **NVIDIA CUDA Toolkit**: Install from [NVIDIA’s website](https://developer.nvidia.com/cuda-downloads).
- **OpenXR Runtime**: For VR mode, install an OpenXR runtime like [WiVRn](https://github.com/WiVRn/WiVRn) or another compatible runtime (SteamVR, etc).
- **FFmpeg**: Required for video demuxing (install via system package manager, e.g., `sudo apt install ffmpeg` on Ubuntu).
- **Python**: 3.8+
- **OS**: Tested on Linux; Windows will need some tweaks.
- **VR**: OpenXR-compatible headset (e.g., Quest 3 via Wivrn, optional).

### Intalling
1. Clone the repo:
```bash
   git clone https://github.com/your-username/xrplay.git
   cd xrplay
```
2. Install/build `pycuda` with OpenGL support, if needed:
```bash
   git clone --recursive https://github.com/inducer/pycuda.git
   cd pycuda
   ./configure.py --cuda-enable-gl
   pip install -e .
   cd ..
```
3. Install remaining dependencies (these are just what I happen to have--it's probably not this particular):
```bash
   pip install -r requirements.txt
```
   Current dependencies:
   - `glfw>=2.10.0`
   - `numpy>=2.3.4`
   - `pycuda>=2025.1.2`
   - `PyNvVideoCodec>=2.0.2`
   - `PyOpenGL>=3.1.10`
   - `pyopenxr>=1.1.5201`
   - `pyaudio>=0.2.13`
   - `imgui[glfw]==2.0.0`

## Usage
Run xrplay or vplay (same script, different name, slightly different default behavior) from the command line:
```bash
vplay video.mp4         # Desktop playback
vplay video.mp4 -a      # Desktop playback with audio (pre-loads entire audio stream into memory)
vplay video.mp4 -p 190  # Desktop playback of a VR 190-degree SBS video (no view angle controls yet)
xrplay video.mp4 -a     # VR playback (180° SBS) with audio
vplay -s 1920,1080      # Desktop with max width,height
vplay -f                # Full-screen desktop playback
vplay -xr video.mp4     # Desktop and VR playback at the same time
xrplay video.mp4 -f     # Fullscreen desktop and VR playback at the same time
xrplay -h               # Show help
```

- **Controls**: Press escape to quit. Arrows control speed.  Space bar pauses.  VR: Right controller stick controls speed; press to reset.  Buttons are pause and quit.  When paused, right stick controls pan/tilt.

## License
MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments
Built with [pyopenxr](https://github.com/cmbruns/pyopenxr), [pycuda](https://github.com/inducer/pycuda), [cupy](https://cupy.dev/), [PyNvVideoCodec](https://github.com/NVIDIA/PyNvVideoCodec), and [pyimgui](https://github.com/pyimgui/pyimgui).

