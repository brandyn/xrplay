# XRPlay: Python/CUDA Video Player with OpenXR Support

XRPlay is, for the moment, a proof-of-concept for a high-performance, command-line video player written in Python, leveraging NVIDIA CUDA for fast video decoding and OpenXR for optional VR headset support.

I've tried to keep it as simple and modular as possible, without sacrificing speed.  As of the first release, it's only 1300 lines including comments, covering two distinct rendering pipelines (ordinary video, and VR SBS 180).

It achieves smooth playback of high-resolution 180 SBS VR videos (e.g., 6Kx3K@60fps to a Quest 3 via WiVRn without breaking a sweat) which other OpenXR players I found couldn't keep up with.

Currently, it supports video playback only (no audio, and no play controls yet) and requires an NVIDIA GPU.
The only VR video format supported right now is 180 SBS, but it would be pretty easy to add others.  It
will also play ordinary videos to the desktop just fine (still no audio though).

### System Dependencies
1. **NVIDIA CUDA Toolkit**: Install from [NVIDIA’s website](https://developer.nvidia.com/cuda-downloads).
2. **OpenXR Runtime**: For VR mode, install an OpenXR runtime like [WiVRn](https://github.com/WiVRn/WiVRn) or another compatible runtime (SteamVR, etc).
3. **FFmpeg**: Required for video decoding (install via system package manager, e.g., `sudo apt install ffmpeg` on Ubuntu).

### Python Dependencies
1. Clone the repo:
```bash
   git clone https://github.com/your-username/xrplay.git
   cd xrplay
```
2. Install/build `pycuda` with OpenGL support:
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

## Usage
Run xrplay or cuda-play (same script, different name, slightly different default behavior) from the command line:
```bash
./cuda-play video.mp4       # Desktop playback
./xrplay video.mp4          # VR playback (180° SBS)
./cuda-play -s 1920,1080    # Desktop with max width,height
./cuda-play -f              # Full-screen desktop playback
./cuda-play -xr video.mp4   # Desktop AND VR playback at the same time
./xrplay video.mp4 -f       # Fullscreen desktop AND VR playback at the same time
./xrplay -h                 # Show help
```

- **Controls**: Press `q` to quit. Future updates will add pause, jump, and OpenXR controller support.
- **Note**: No audio support yet (planned).

## Performance
XRPlay is optimized for speed:
- Easily achieves 6Kx3K@60fps playback on a Quest 3 via WiVRn, outperforming other XR players I tried.
- Uses a single CUDA kernel pass for VR rendering, avoiding extra image copies.
- "Pure" Python (with embedded CUDA kernel) for clean, maintainable code.  All the heavy lifting is done by the GPU, and python just holds the pieces together.

## Requirements
- **Python**: 3.8+
- **Hardware**: NVIDIA GPU (required for CUDA).
- **VR**: OpenXR-compatible headset (e.g., Quest 3, optional).
- **OS**: Tested on Linux; Windows will need some tweaks.

## Notes
- **Windows Users**: I haven't tried it on Windows.  It will require a few tweaks, and I'm not sure how to get/compile pycuda with GL support.
- **Contributing**: Audio support and play controls are planned. Contributions welcome!
- **Limitations**: Requires NVIDIA GPU. Non-CUDA support is theoretically possible but not implemented.

## License
MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments
Built with [pyopenxr](https://github.com/cmbruns/pyopenxr), [pycuda](https://github.com/inducer/pycuda), and [PyNvVideoCodec](https://github.com/NVIDIA/PyNvVideoCodec).

