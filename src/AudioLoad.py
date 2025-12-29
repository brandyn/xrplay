
import numpy as np
import subprocess as sp

"""Method to Load entire audio stream into numpy array.
"""

PYLIB = False

def load_audio(video_path, sr=48000, channels=2):
    command = [
        'ffmpeg',
        '-i', video_path,
        '-f', 'f32le',
        '-acodec', 'pcm_f32le',
        '-ar', str(sr),
        '-ac', str(channels),
        'pipe:1'
    ]
    pipe  = sp.Popen(command, stdout=sp.PIPE, stderr=sp.DEVNULL, bufsize=10**8)
    raw   = pipe.stdout.read()
    audio = np.frombuffer(raw, dtype=np.float32)
    return audio.reshape(-1, channels)

if PYLIB:

    import ffmpeg
    def load_audio_fast(video_path, sr=48000):
        out, _ = (
            ffmpeg
            .input(video_path)
            .output('pipe:', format='f32le', acodec='pcm_f32le', ac=2, ar=sr)
            .run(capture_stdout=True, capture_stderr=True)
        )
        return np.frombuffer(out, np.float32).reshape(-1, 2)

if __name__ == "__main__":

    from time import time
    t0 = time()
    for i in range(3):

        if PYLIB:
            audio = load_audio_fast('video.mp4')
            t1 = time()
            print(f"FFmpeg: {audio.shape} - {t1-t0:.3f} seconds")
            t0 = t1

        audio = load_audio('video.mp4')
        t1 = time()
        print(f" Popen: {audio.shape} - {t1-t0:.3f} seconds")
        t0 = t1

