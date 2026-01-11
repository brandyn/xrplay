import ctypes
import cupy as cp
import numpy as np
import pycuda.driver as cuda
import pycuda.gl as cuda_gl
from OpenGL          import GL
from pycuda.compiler import SourceModule
from CudaRes         import cudaResourceDesc, libcuda

class cudaTextureDesc(ctypes.Structure):
    _fields_ = [
        ('addressMode', ctypes.c_int * 3),
        ('filterMode', ctypes.c_int),
        ('readMode', ctypes.c_int),
        ('sRGB', ctypes.c_int),
        ('borderColor', ctypes.c_float * 4),
        ('normalizedCoords', ctypes.c_int),
        ('maxAnisotropy', ctypes.c_uint),
        ('mipmapFilterMode', ctypes.c_int),
        ('mipmapLevelBias', ctypes.c_float),
        ('minMipmapLevelClamp', ctypes.c_float),
        ('maxMipmapLevelClamp', ctypes.c_float),
        ('disableTrilinearOptimization', ctypes.c_int),
        ('seamlessCubemap', ctypes.c_int)
    ]

class DualRenderer(object):
    """
    This is just a utility class for 2D image renderers (such as user interfaces)
        which allows them to be implemented in, and accessed by, either CuPy or GL,
        by providing cross-translators between those two.

    Subclasses should implement EITHER render_cupy() OR render_gl().

    The other method will be automatically implemented via zero-copy GPU conversion.
    
    Both methods accept arbitrary *args, **kwargs that are passed through.
    """
    
    # CUDA kernel to read RGBA texture surface and write RGB to linear memory
    # Also flip it in y while we're at it because the returned image is expected
    #  in scanline order like a video decoder output...
    _rgba_to_rgb_kernel_code = """
    extern "C" __global__
    void rgba_surface_to_rgb(cudaTextureObject_t tex, unsigned char* dst, 
                             int width, int height, int dst_pitch) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (x >= width || y >= height)
            return;
        
        uchar4 rgba = tex2D<uchar4>(tex, x, y);
        
        int dst_idx = (height-1-y) * dst_pitch + x * 3;
        dst[dst_idx + 0] = rgba.x;  // R
        dst[dst_idx + 1] = rgba.y;  // G
        dst[dst_idx + 2] = rgba.z;  // B
    }
    """
    _kernel_module = None
    _kernel_func   = None
    
    cudaCreateTextureObject = libcuda.cudaCreateTextureObject
    cudaCreateTextureObject.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    cudaCreateTextureObject.restype = ctypes.c_int

    cudaDestroyTextureObject = libcuda.cudaDestroyTextureObject
    cudaDestroyTextureObject.argtypes = [ctypes.c_ulonglong]
    cudaDestroyTextureObject.restype = ctypes.c_int

    @classmethod
    def _get_kernel(cls):
        """Lazy compile the RGBA->RGB kernel."""
        if cls._kernel_func is None:
            cls._kernel_module = SourceModule(cls._rgba_to_rgb_kernel_code)
            cls._kernel_func   = cls._kernel_module.get_function("rgba_surface_to_rgb")
        return cls._kernel_func
    
    def __init__(self):
        """Initialize with no cached resources."""
        # For render_cupy() calling render_gl()
        self._gl_texture         = None
        self._gl_fbo             = None
        self._cached_size        = None
        self._registered_texture = None
        self._cupy_buffer        = None
        self._cuda_stream        = cuda.Stream()
        self._tex_obj            = None
        self._mapping            = None
        
        # For render_gl() calling render_cupy()
        self._projector         = None
        self._upload_texture    = None
        self._upload_registered = None
        self._upload_size       = None
    
    def render_cupy(self, size, *args, **kwargs):
        """
        Render and return an RGB8 CuPy array of shape (height, width, 3).
        
        Default implementation: calls render_gl() to FBO, binds GL texture
        to CUDA texture reference, uses kernel to extract RGB to CuPy memory.
        
        Args:
            size: (width, height) tuple
            *args, **kwargs: Passed to render_gl()
            
        Returns:
            CuPy array of shape (height, width, 3), dtype=uint8
            Valid only until next call.
        """
        width, height = size
        
        # Create/update GL resources if needed
        if self._cached_size != size:
            #print(f"DEBUG: render_cupy() detected size change -> {size}")
            self._create_gl_resources(width, height)
            self._cached_size = size
        
        # Render to FBO
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._gl_fbo)
        GL.glViewport(0, 0, width, height)

        self.render_gl(size, *args, **kwargs)
        
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        
        GL.glFinish()   # Wait for GL to finish the render before we try to read the resulting image.

        # Allocate output buffer if needed
        if self._cupy_buffer is None or self._cupy_buffer.shape != (height, width, 3):
            self._cupy_buffer = cp.empty((height, width, 3), dtype=cp.uint8)
        
        cuda_array = self._mapping.array(0, 0)

        # Launch kernel
        block_size = (16, 16, 1)
        grid_size = (
            (width + block_size[0] - 1) // block_size[0],
            (height + block_size[1] - 1) // block_size[1],
            1
        )
        
        self._get_kernel()(
            np.uint64(self._tex_obj.value),
            np.intp(self._cupy_buffer.data.ptr),
            np.int32(width),
            np.int32(height),
            np.int32(self._cupy_buffer.strides[0]),
            block=block_size,
            grid=grid_size,
            stream=self._cuda_stream
        )
        
        self._cuda_stream.synchronize()

        return self._cupy_buffer

    # *** This is untested ***
    def render_gl(self, size, *args, **kwargs):
        """
        Render directly to current OpenGL context/viewport.
        
        Default implementation: calls render_cupy(), uses ResizeProjector
        to blit to viewport with zero GPU-CPU copies.
        
        Args:
            size: (width, height) tuple
            *args, **kwargs: Passed to render_cupy()
        """
        width, height = size
        
        # Get RGB8 CuPy array from subclass
        rgb_array = self.render_cupy(size, *args, **kwargs)
        
        # Create/update resources if needed
        if self._upload_size != size:
            self._create_upload_resources(width, height)
            self._upload_size = size
        
        # Use ResizeProjector for zero-copy blit from CuPy to GL texture
        self._projector.invoke(rgb_array, self._upload_registered)
        
        # Render texture to viewport
        from GLop import render_GL_texture_to_window
        render_GL_texture_to_window(self._upload_texture, size)
    
    def _create_gl_resources(self, width, height):
        """Create FBO and texture for render_cupy() default implementation."""
        # Clean up old resources in correct order
        # 1. Destroy texture object first (references the array from mapping)
        if self._tex_obj is not None:
            libcuda.cudaDestroyTextureObject(self._tex_obj)
            self._tex_obj = None
        
        # 2. Unmap before unregistering
        if self._mapping is not None:
            self._mapping.unmap(self._cuda_stream)
            self._mapping = None
        
        # 3. Now safe to unregister
        if self._registered_texture is not None:
            self._registered_texture.unregister()
            self._registered_texture = None
        
        # 4. Delete GL resources
        if self._gl_texture is not None:
            GL.glDeleteTextures([self._gl_texture])
            self._gl_texture = None
        
        if self._gl_fbo is not None:
            GL.glDeleteFramebuffers(1, [self._gl_fbo])
            self._gl_fbo = None
        
        # Create RGBA8 texture
        self._gl_texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._gl_texture)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0,
                       GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        
        # Register texture with CUDA
        self._registered_texture = cuda_gl.RegisteredImage(
            int(self._gl_texture),
            GL.GL_TEXTURE_2D,
            cuda_gl.graphics_map_flags.READ_ONLY
        )
        
        # Map once and keep alive
        self._mapping = self._registered_texture.map(self._cuda_stream)
        cuda_array = self._mapping.array(0, 0)

        # Create texture object (keep alive)
        res_desc = cudaResourceDesc()
        res_desc.resType = 0x00
        res_desc.res.array.array = ctypes.c_void_p(int(cuda_array.handle))

        tex_desc = cudaTextureDesc()
        ctypes.memset(ctypes.byref(tex_desc), 0, ctypes.sizeof(tex_desc))
        tex_desc.addressMode[0] = 1
        tex_desc.addressMode[1] = 1  
        tex_desc.filterMode = 0
        tex_desc.readMode = 0
        tex_desc.normalizedCoords = 0

        self._tex_obj = ctypes.c_ulonglong(0)
        err = libcuda.cudaCreateTextureObject(
            ctypes.byref(self._tex_obj),
            ctypes.byref(res_desc),
            ctypes.byref(tex_desc),
            None
        )
        if err != 0:
            raise RuntimeError(f"Failed to create texture object: error {err}")

        # Create FBO
        self._gl_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._gl_fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                 GL.GL_TEXTURE_2D, self._gl_texture, 0)
        
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Framebuffer incomplete: {status}")
        
        # After creating FBO, clear it once
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._gl_fbo)
        GL.glClearColor(0, 0, 0, 0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def _create_upload_resources(self, width, height):
        """Create texture and ResizeProjector for render_gl() default implementation."""
        if self._upload_registered is not None:
            self._upload_registered.unregister()
            self._upload_registered = None
        
        if self._upload_texture is not None:
            GL.glDeleteTextures([self._upload_texture])
        
        if self._projector is not None:
            self._projector.close()
            self._projector = None
        
        # Create RGBA8 texture (ResizeProjector outputs RGBA)
        self._upload_texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._upload_texture)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0,
                       GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        
        # Register with CUDA for ResizeProjector
        self._upload_registered = cuda_gl.RegisteredImage(
            int(self._upload_texture),
            GL.GL_TEXTURE_2D,
            cuda_gl.graphics_map_flags.WRITE_DISCARD
        )
        
        # Create ResizeProjector
        from ProjectorResize import ResizeProjector
        self._projector = ResizeProjector(
            input_size=(width, height),
            output_size=(width, height)
        )
    
    
    def close(self):
        """Clean up all allocated GL and CUDA resources."""
        if self._mapping is not None:
            self._mapping.unmap(self._cuda_stream)
            self._mapping = None

        if self._tex_obj is not None:
            libcuda.cudaDestroyTextureObject(self._tex_obj)
            self._tex_obj = None

        if self._registered_texture is not None:
            self._registered_texture.unregister()
            self._registered_texture = None
        
        if self._upload_registered is not None:
            self._upload_registered.unregister()
            self._upload_registered = None
        
        if self._gl_texture is not None:
            GL.glDeleteTextures([self._gl_texture])
            self._gl_texture = None
        
        if self._gl_fbo is not None:
            GL.glDeleteFramebuffers(1, [self._gl_fbo])
            self._gl_fbo = None
        
        if self._upload_texture is not None:
            GL.glDeleteTextures([self._upload_texture])
            self._upload_texture = None
        
        if self._projector is not None:
            self._projector.close()
            self._projector = None
        
        if self._cuda_stream is not None:
            self._cuda_stream.synchronize()
            self._cuda_stream = None
        
        self._cached_size = None
        self._upload_size = None
        self._cupy_buffer = None
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass


# Example subclass implementing render_cupy
class ExampleCuPyRenderer(DualRenderer):
    """Example: generates gradient in CuPy."""
    
    def render_cupy(self, size):
        width, height = size
        
        if self._cupy_buffer is None or self._cupy_buffer.shape != (height, width, 3):
            self._cupy_buffer = cp.empty((height, width, 3), dtype=cp.uint8)
        
        # Generate simple gradient
        x = cp.linspace(0, 255, width, dtype=cp.uint8)
        y = cp.linspace(0, 255, height, dtype=cp.uint8)
        
        self._cupy_buffer[:, :, 0] = x[None, :]  # Red gradient horizontal
        self._cupy_buffer[:, :, 1] = y[:, None]  # Green gradient vertical
        self._cupy_buffer[:, :, 2] = 128          # Blue constant
        
        return self._cupy_buffer


# Example subclass implementing render_gl
class ExampleGLRenderer(DualRenderer):
    """Example: renders checkerboard with GL."""
    
    def render_gl(self, size, *args, **kwargs):
        
        # --- Explicitly enforce a clean state for this 2D flat render ---
        GL.glDisable(GL.GL_LIGHTING)          # Crucial: turns off fixed-function lighting
        GL.glDisable(GL.GL_COLOR_MATERIAL)    # If enabled elsewhere
        GL.glDisable(GL.GL_DEPTH_TEST)        # Usually unnecessary for 2D ortho
        GL.glDisable(GL.GL_BLEND)             # If blending interferes
        GL.glDisable(GL.GL_TEXTURE_2D)        # If textures are enabled elsewhere
        GL.glShadeModel(GL.GL_FLAT)           # Prevents any smooth shading artifacts
        # Optional: reset current normal (defaults to (0,0,1), but lighting uses it)
        GL.glNormal3f(0.0, 0.0, 1.0)

        width, height = size
        
        GL.glClearColor(0.2, 0.2, 0.2, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(-1, 1, -1, 1, -1, 1)  # Normalized coords
        
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        
        GL.glColor3f(0.8, 0.8, 0.8)
        cell = 2.0 / 8.0  # Size of each cell in -1 to 1 space
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    x0 = -1.0 + i * cell
                    y0 = -1.0 + j * cell
                    x1 = x0 + cell
                    y1 = y0 + cell
                    GL.glBegin(GL.GL_QUADS)
                    GL.glVertex2f(x0, y0)
                    GL.glVertex2f(x1, y0)
                    GL.glVertex2f(x1, y1)
                    GL.glVertex2f(x0, y1)
                    GL.glEnd()

