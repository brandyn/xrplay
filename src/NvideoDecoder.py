
import PyNvVideoCodec as nvc
import cupy           as cp
import os

from VideoSource import VideoSource

class NvideoDecoder(VideoSource):

    def __init__(self, video_path, gpu_id=0, use_cpu=False):
        VideoSource.__init__(self)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} not found")
 
        self.video_path = video_path
        self.use_cpu    = use_cpu
        
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
        self.current_frame     = None  # reused sometimes when speed < 1
        self.current_frame_num = -1

    def get_frame(self, frame_number, io=None):

        if frame_number == self.current_frame_num:
            return self.current_frame

        try:
            try:
                nvc_frame = self.decoder[frame_number]
            except IndexError:  # EOF?
                return None
            self.current_frame     = cp.from_dlpack(nvc_frame)
            self.current_frame_num = frame_number
            return self.current_frame
            
        except Exception as e:
            print(f"Decode error: {type(e).__name__}: {e}")
            return None
    
    def close(self):
        """Clean up decoder resources."""
        pass  # NVC handles cleanup automatically

