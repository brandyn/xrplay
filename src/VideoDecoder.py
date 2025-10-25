
from typing      import Optional
from dataclasses import dataclass

@dataclass
class VideoFrame(object):
    """Abstract base class for frame representation. Always RGB format."""
    __slots__ = ('width', 'height', 'pitch', 'on_gpu')
    
    width: int
    height: int
    pitch: int          # bytes per row
    on_gpu: bool        # True if GPU, False if CPU
    
    @property
    def on_cpu(self) -> bool:
        return not self.on_gpu
    
    def get_data(self):
        """Returns int (CUDA ptr) if on_gpu, np.ndarray if on_cpu"""
        raise NotImplementedError("Subclass must implement get_data()")

class VideoDecoder(object):
    """Minimal, stable API for video decoding. Returns frames in standard format.
    """
    def __init__(self, video_path):
        self.video_path = video_path
    
    def get_next_frame(self) -> Optional[VideoFrame]:
        """Decode next frame. Returns VideoFrame or None on end of video."""
        raise NotImplementedError("Subclass must implement get_next_frame()")
    
    def close(self):
        """Clean up decoder resources."""
        pass

