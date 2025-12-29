
from OpenGL.GL import *

class CupyToGLTexture(object):
    """Projects a CuPy RGB8 array directly to an OpenGL RGBA texture via a cuda kernel.

    If output_size is provided and differs from input_size, the image
        will be scaled, otherwise just copied.

    .init() returns the output texture, or pass init=True to the
        constructor and access .texture directly.
    """
    def __init__(self, input_size, output_size=None, init=False, vr=None):
        """vr indicates the source is a VR projection and needs to be non-linearly
            mapped to the output.
        """

        if output_size is None:
            output_size = input_size

        self.output_size = output_size
        self.texture     = None
        self.cuda_img    = None

        if vr:
            from ProjectorVR import VRProjector
            self.projector = VRProjector(input_size, output_size, projection=vr,
                            # TODO: add this as a parameter?
                            #chromakey = {'key_color': (128,128,128)}   # Key color is provided in RGB, but only the hue is used.
                            # NOTE this appears to work (see DEBUG_CHROMAKEY in VRProjector)
                            #  but the returned Alpha channel is being ignored by the rest of
                            #  the pipeline?
            )
        else:
            from ProjectorResize import ResizeProjector
            self.projector = ResizeProjector(input_size, output_size)

        if init:
            self.init()
    
    def init(self):
        """Initialize the OpenGL texture and register it with CUDA."""
        # Create OpenGL texture as RGBA (CUDA arrays only support 1, 2, or 4 channels)
        self.texture = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self.texture)
        
        # Set pixel store alignment
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        
        # Allocate texture storage (RGBA8 - required for CUDA 4-channel arrays)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.output_size[0], self.output_size[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        
        # Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        glBindTexture(GL_TEXTURE_2D, 0)
        
        # Register the texture with CUDA
        self.cuda_img = self.projector.register_texture(self.texture)

        return self.texture
    
    def invoke(self, image):
        """Copy CuPy array directly to the texture with optional scaling.
        
        Args:
            image: CuPy array of shape (input_height, input_width, 3) with dtype uint8 (RGB)
        """
        if self.cuda_img is None:
            raise RuntimeError("Must call init() before invoke()")
        self.projector.invoke(image, self.cuda_img)
    
    def close(self):
        """Clean up CUDA and OpenGL resources.
        """
        if self.cuda_img is not None:
            self.projector.unregister_texture(self.cuda_img)
            self.cuda_img = None
        
        if self.texture is not None:
            glDeleteTextures(1, [self.texture])
            self.texture = None

        self.projector.close()

