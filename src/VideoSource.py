
import cupy as cp
from typing import Optional

class VideoSource(object):
    """Minimal, stable API for video sources. Returns frames in standard format.
    """
    # Instances should populate these when the video is opened:
    width       = None  # Size of the image (int)
    height      = None
    framerate   = None  # Frames per second (float)
    duration    = None  # Expected duration in seconds (float)  ; May be None if N/A
    frame_count = None  # Expected frame count (int)            ; May be None if N/A

    def get_frame(self, frame_number, io=None) -> Optional[cp.ndarray]:
        """Fetch the given frame. Returns cupy ndarray or None if no such frame (end of video, etc).
        Note the returned shape should be (height, width, 3) [or maybe 4 if ever RGBA?]
        And strides will be (pitch, 3, 1) [or maybe pitch, 4, 1]

        io is an IO object which may be used by some providers (e.g. interactive UI)
        """
        raise NotImplementedError("Subclass must implement get_next_frame()")
    
    def handle_events(self, io):
        # TBD
        pass

    def close(self):
        """Clean up decoder resources."""
        pass

