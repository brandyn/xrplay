
import os
import cupy as cp
from typing import Optional

class VideoDecoder(object):
    """Minimal, stable API for video decoding. Returns frames in standard format.
    """
    # Instances should populate these when the video is opened:
    width       = None  # Size of the image (int)
    height      = None
    framerate   = None  # Frames per second (float)
    duration    = None  # Expected duration in seconds (float)
    frame_count = None  # Expected frame count (int)

    def __init__(self, video_path):

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} not found")
 
        self.video_path = video_path
    
    def get_frame(self, frame_number) -> Optional[cp.ndarray]:
        """Decode next frame. Returns cupy ndarray or None on end of video.
        Note the returned shape should be (height, width, 3) [or maybe 4 if ever RGBA?]
        And strides will be (pitch, 3, 1) [or maybe pitch, 4, 1]
        """
        raise NotImplementedError("Subclass must implement get_next_frame()")
    
    def close(self):
        """Clean up decoder resources."""
        pass

