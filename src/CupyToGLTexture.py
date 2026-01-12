
from OpenGL.GL import *

class CupyToGLTexture(object):
    """Projects a CuPy RGB8 array directly to an OpenGL RGBA texture via a cuda kernel.

    If the output size differs from the input size, the image
        will be scaled, otherwise just copied.  Unless vr is given, in
        which case the specified VR projection will be performed.

    The current texture is returned by invoke() and is only valid until the next call.
    """
    def __init__(self, vr=None):
        """vr indicates the source is a VR projection and needs to be non-linearly
            mapped to the output with the specified projection.
        """
        self.texture      = None
        self.texture_size = None    # So we know when it needs to be regenerated.
        self.cuda_img     = None

        self.projector    = None
        self.projection   = None

        self.set_projection(vr)

    def set_projection(self, projection):

        if self.projector is not None and projection == self.projection:
            return

        self.close()    # We'll just rebuild from scratch when we change projections...

        if projection and projection not in ("mono", "flat"):
            from ProjectorVR import VRProjector
            self.projector = VRProjector(projection=projection)
        else:
            from ProjectorResize import ResizeProjector
            self.projector = ResizeProjector()

        self.projection = projection

    def configure(self, output_size):
        """Initialize the OpenGL texture and register it with CUDA."""
        #print(f"DEBUG: CupyToGLTexture re-allocating texture at {output_size}")
        
        # Unregister from CUDA if needed
        if self.cuda_img is not None:
            self.projector.unregister_texture(self.cuda_img)
            self.cuda_img = None
        
        # Create texture if first time, otherwise reuse existing
        if self.texture is None:
            self.texture = glGenTextures(1)
        
        # (Re)allocate texture storage
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, output_size[0], output_size[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        
        # Set texture parameters (only needed first time, but harmless to repeat)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        glBindTexture(GL_TEXTURE_2D, 0)
        
        # Register the texture with CUDA
        self.cuda_img     = self.projector.register_texture(self.texture)
        self.texture_size = output_size
        
        return self.texture

    def invoke(self, input_image, dest_size, timers=None):
        """Copy CuPy array directly to the texture with optional scaling.
        
        Args:
            input_image: CuPy array of shape (input_height, input_width, 3) with dtype uint8 (RGB)
        """
        if dest_size != self.texture_size:
            self.configure(dest_size)

        if self.projection == 'flat':   # Special case, we're doing a simple resize on Half of a flat
            input_image = input_image[:, 0:input_image.shape[1]//2, :]

        self.projector.invoke(input_image, self.cuda_img, self.texture_size, timers=timers)

        return self.texture
    
    def close(self):
        """Clean up CUDA and OpenGL resources.
        """
        if self.projector is not None:

            if self.cuda_img is not None:
                self.projector.unregister_texture(self.cuda_img)
                self.cuda_img = None
            
            if self.texture is not None:
                glDeleteTextures(1, [self.texture])
                self.texture      = None
                self.texture_size = None

            self.projector.close()
            self.projector = None

