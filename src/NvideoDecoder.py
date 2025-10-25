
import os

import PyNvVideoCodec as nvc
import pycuda.driver  as cuda

from VideoDecoder import VideoDecoder, VideoFrame
from typing       import Optional
from dataclasses  import dataclass

@dataclass
class NvideoFrame(VideoFrame):
    """NVC-specific frame implementation.
    """
    __slots__ = ('_nvc_frame',)
    
    _nvc_frame: object  # Keep NVC frame alive
    
    def get_data(self):
        """Returns int (CUDA ptr) if on_gpu, np.ndarray if on_cpu"""
        if self.on_gpu:
            return self._nvc_frame.cuda()[0].dataptr
        else:
            return self._nvc_frame.host()

class NvideoDecoder(VideoDecoder):
    """Minimal, stable API for video decoding. Returns frames in standard format.
    """
    def __init__(self, video_path, gpu_id=0, use_cpu=False):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} not found")
        
        self.use_cpu     = use_cpu
        self.cuda_stream = cuda.Stream()
        
        # Initialize decoder
        self.decoder = nvc.SimpleDecoder(
            video_path,
            use_device_memory=not use_cpu,
            output_color_type=nvc.OutputColorType.RGB
        )
        
        # Get video properties
        metadata         = self.decoder.get_stream_metadata()
        self.width       = metadata.width
        self.height      = metadata.height
        self.framerate   = metadata.average_fps
        self.duration    = metadata.duration
        self.frame_count = len(self.decoder)
    
        # State:
        self.frame_number = 0

    def get_next_frame(self, speed=1, loop=False) -> Optional[VideoFrame]:
        """Decode next frame. Returns VideoFrame or None on end of video.
        Speed is an integer which determines how many frames to move from
            the prior frame.  It may be negative.
        If loop is True the video will start over at the end.
        """
        try:
            # Decode frame
            if True:
                self.frame_number += speed
                if self.frame_number < 0:
                    self.frame_number = 0
                try:
                    nvc_frame = self.decoder[self.frame_number]
                except IndexError:  # EOF?
                    if loop and self.frame_number:
                        self.frame_number = 0
                        nvc_frame         = self.decoder[self.frame_number] # If we get another IndexError here, something's very wrong
                    else:
                        return None
            else:
                frames = self.decoder.get_batch_frames(1)
                if not frames:  # Empty list on EOF
                    return None
                nvc_frame = frames[0]
            
            # Extract frame properties
            shape   = nvc_frame.shape
            strides = nvc_frame.strides
            height  = shape[0]
            width   = shape[1]
            pitch   = strides[0] if strides else width * 3
            
            return NvideoFrame(
                _nvc_frame=nvc_frame,
                width=width,
                height=height,
                pitch=pitch,
                on_gpu=not self.use_cpu
            )
            
        except Exception as e:
            print(f"Decode error: {type(e).__name__}: {e}")
            return None
    
    def close(self):
        """Clean up decoder resources."""
        pass  # NVC handles cleanup automatically

