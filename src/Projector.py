
import ctypes
from CudaRes import cudaResourceDesc, libcuda

#
# Most of these aren't actually needed here, but subclasses use them and
#  this enables them to just do "from Projector import *"
#
import pycuda.driver as cuda
import cupy          as cp
import numpy         as np
import pycuda.gl     as cuda_gl

from OpenGL          import GL
from pycuda.compiler import SourceModule

class Projector(object):
    """Implements various direct projections from an input cupy image to an output cuda.gl registered image
        (cuda registered GL texture) via cuda kernels (with no intermediate buffers or extra copying).
    
    Uses CUDA surface objects to write directly to texture memory.
    """
    cudaCreateSurfaceObject           = libcuda.cudaCreateSurfaceObject
    cudaCreateSurfaceObject.argtypes  = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(cudaResourceDesc)]
    cudaCreateSurfaceObject.restype   = ctypes.c_int

    cudaDestroySurfaceObject          = libcuda.cudaDestroySurfaceObject
    cudaDestroySurfaceObject.argtypes = [ctypes.c_ulonglong]
    cudaDestroySurfaceObject.restype  = ctypes.c_int

    def __init__(self):
        # Create CUDA stream
        self.cuda_stream = cuda.Stream()

    def register_texture(self, texture):
        "Here texture is a GL texture."
        return cuda_gl.RegisteredImage(
            int(texture),
            GL.GL_TEXTURE_2D,
            cuda_gl.graphics_map_flags.WRITE_DISCARD
        )

    def unregister_texture(self, cuda_img):
        cuda_img.unregister()

    def close(self):
        """Clean up CUDA and OpenGL resources."""
        
        if self.cuda_stream is not None:
            self.cuda_stream.synchronize()
            self.cuda_stream = None

