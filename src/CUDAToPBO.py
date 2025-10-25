
import pycuda.driver as cuda
import pycuda.gl     as cuda_gl

from OpenGL.GL import *

class CUDAToPBO(object):
    """
    Copies CUDA device memory to a PBO.
    
    ALLOCATION: Creates and owns a PBO (must register with CUDA).
    
    Init chain: Call allocate_pbo() after construction.
                Returns the PBO handle for downstream use.
    
    Runtime: invoke(cuda_ptr) -> None (writes to internal PBO)
    """
    
    def __init__(self, width, height, init=False):
        self.width    = width
        self.height   = height
        self.pbo      = None
        self.cuda_pbo = None
    
        if init:
            self.init()

    def init(self):
        """Create PBO and register with CUDA. Returns PBO handle."""
        # Create GL PBO
        self.pbo = glGenBuffers(1)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, self.width * self.height * 3, None, GL_STREAM_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        
        # Register with CUDA
        self.cuda_pbo = cuda_gl.RegisteredBuffer(
            int(self.pbo),
            cuda_gl.graphics_map_flags.WRITE_DISCARD
        )
        self.cuda_stream = cuda.Stream()

        return self.pbo
    
    def invoke(self, cuda_ptr, pitch):
        """Copy from CUDA pointer to internal PBO."""
        mapping = self.cuda_pbo.map(stream=self.cuda_stream)
        try:
            dest_ptr, size = mapping.device_ptr_and_size()
            expected_pitch = self.width * 3
            
            if pitch == expected_pitch:
                cuda.memcpy_dtod_async(dest_ptr, cuda_ptr, self.height * pitch, self.cuda_stream)
            else:
                copy = cuda.Memcpy2D()
                copy.set_src_device(cuda_ptr)
                copy.set_dst_device(dest_ptr)
                copy.width_in_bytes = self.width * 3
                copy.src_pitch = pitch
                copy.dst_pitch = expected_pitch
                copy.height = self.height
                copy(self.cuda_stream)
            
            self.cuda_stream.synchronize()
        finally:
            mapping.unmap()

    def close(self):
        if self.pbo is not None:
            glDeleteBuffers(1, [self.pbo])
        if self.cuda_pbo is not None:
            self.cuda_pbo.unregister()

