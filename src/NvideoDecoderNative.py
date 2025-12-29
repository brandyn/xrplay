
#
# THIS file is currently just a starting point for doing work in the native color
#   space before the RGB conversion.  Currently it behaves at best the same as
#   NvideoDecoder, but less robust because it makes assumptions about the native
#   video representation.
#
# A likely use of this would be to add chromakeying Before the conversion to RGB,
#   and then change the API to output RGBA (which means everything downstream
#   has to accept RGBA).
#

import PyNvVideoCodec as nvc
import cupy           as cp
from cupyx import jit

from VideoDecoder import VideoDecoder

class NvideoDecoderNV12(VideoDecoder):
    """Video decoder that gets NV12 from hardware and converts to RGB in CUDA.
    Identical API to NvideoDecoder but with custom NV12->RGB conversion kernel.

    NOTE this ASSUMES the video's native format is NV12.  May break non-gracefully
    on some codecs or colorspaces.
    """
    def __init__(self, video_path, gpu_id=0, use_cpu=False):
        VideoDecoder.__init__(self, video_path)

        self.use_cpu = use_cpu
        
        # Initialize decoder - request NV12 format
        self.decoder = nvc.SimpleDecoder(
            video_path,
            use_device_memory=not use_cpu,
            output_color_type=nvc.OutputColorType.NATIVE
        )
        
        # Get video properties
        metadata         = self.decoder.get_stream_metadata()
        self.width       = metadata.width
        self.height      = metadata.height
        self.framerate   = metadata.average_fps
        self.duration    = metadata.duration
        self.frame_count = len(self.decoder)
    
        # Allocate output buffer for RGB
        self.rgb_buffer = cp.empty((self.height, self.width, 3), dtype=cp.uint8)
        
        # Compile CUDA kernel
        self._compile_kernel()
        
        # State:
        self.current_frame     = None
        self.current_frame_num = -1

    def _compile_kernel(self):
        """Compile the NV12 to RGB conversion kernel."""
        kernel_code = '''
        extern "C" __global__
        void nv12_to_rgb(
            const unsigned char* y_plane,
            const unsigned char* cbcr_plane,
            unsigned char* rgb_out,
            int width,
            int height,
            int y_pitch,
            int cbcr_pitch,
            int rgb_pitch
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (x >= width || y >= height) return;
            
            // Read Y (full resolution)
            float Y_raw = (float)y_plane[y * y_pitch + x];
            
            // Read CbCr (half resolution, interleaved)
            int cbcr_x = (x >> 1) * 2;  // x/2 * 2 for interleaved access
            int cbcr_y = y >> 1;         // y/2
            int cbcr_idx = cbcr_y * cbcr_pitch + cbcr_x;
            float Cb_raw = (float)cbcr_plane[cbcr_idx];
            float Cr_raw = (float)cbcr_plane[cbcr_idx + 1];
            
            // Convert from limited range (16-235 for Y, 16-240 for CbCr) to full range
            float Y = (Y_raw - 16.0f) * (255.0f / 219.0f);
            float Cb = (Cb_raw - 128.0f) * (255.0f / 224.0f);
            float Cr = (Cr_raw - 128.0f) * (255.0f / 224.0f);
            
            // BT.709 YCbCr to RGB conversion (HD video standard)
            float r = Y + 1.5748f * Cr;
            float g = Y - 0.1873f * Cb - 0.4681f * Cr;
            float b = Y + 1.8556f * Cb;
            
            // Clamp to [0, 255]
            r = fminf(fmaxf(r, 0.0f), 255.0f);
            g = fminf(fmaxf(g, 0.0f), 255.0f);
            b = fminf(fmaxf(b, 0.0f), 255.0f);
            
            // Write RGB
            int rgb_idx = y * rgb_pitch + x * 3;
            rgb_out[rgb_idx]     = (unsigned char)r;
            rgb_out[rgb_idx + 1] = (unsigned char)g;
            rgb_out[rgb_idx + 2] = (unsigned char)b;
        }
        '''
        
        self.kernel = cp.RawKernel(kernel_code, 'nv12_to_rgb')
        self.block_size = (16, 16, 1)
        self.grid_size = (
            (self.width + self.block_size[0] - 1) // self.block_size[0],
            (self.height + self.block_size[1] - 1) // self.block_size[1],
            1
        )

    def get_frame(self, frame_number):
        if frame_number == self.current_frame_num:
            return self.current_frame

        try:
            try:
                nvc_frame = self.decoder[frame_number]
            except IndexError:  # EOF?
                return None
            
            # Convert DLPack to CuPy array (NV12 format)
            nv12_frame = cp.from_dlpack(nvc_frame)
            
            # NV12 layout: Y plane (height rows), then CbCr plane (height/2 rows)
            # Both planes have width stride
            y_plane_height = self.height
            cbcr_plane_height = self.height // 2
            
            y_plane = nv12_frame[:y_plane_height, :]
            cbcr_plane = nv12_frame[y_plane_height:, :]
            
            # Get pitches (stride in bytes)
            y_pitch = y_plane.strides[0]
            cbcr_pitch = cbcr_plane.strides[0]
            rgb_pitch = self.rgb_buffer.strides[0]
            
            # Launch kernel
            self.kernel(
                self.grid_size,
                self.block_size,
                (
                    y_plane.data.ptr,
                    cbcr_plane.data.ptr,
                    self.rgb_buffer.data.ptr,
                    self.width,
                    self.height,
                    y_pitch,
                    cbcr_pitch,
                    rgb_pitch
                )
            )
            
            self.current_frame = self.rgb_buffer
            self.current_frame_num = frame_number
            return self.current_frame
            
        except Exception as e:
            print(f"Decode error: {type(e).__name__}: {e}")
            return None
    
    def close(self):
        """Clean up decoder resources."""
        pass  # NVC handles cleanup automatically

