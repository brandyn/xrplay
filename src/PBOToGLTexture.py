
from OpenGL.GL import *

class PBOToGLTexture(object):
    """
    Wraps PBO contents as a GL texture.  (Generally zero-copy.)
    
    ALLOCATION: Creates and owns a GL texture.
    
    Init chain: Requires PBO handle from upstream.
                Call allocate_texture(pbo_handle) after upstream ready.
                Returns texture handle for downstream use.
    
    Runtime: invoke() -> None (uploads from PBO to internal texture)
            *** cuda context of the PBO must still be active when invoke is called ***
    """
    
    def __init__(self, width, height, pbo_handle=None):
        self.width      = width
        self.height     = height
        self.pbo_handle = None
        self.texture    = None

        if pbo_handle is not None:
            self.init(pbo_handle)
    
    def init(self, pbo_handle):
        """Create texture. Returns texture handle."""
        self.pbo_handle = pbo_handle
        
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8,
                     self.width, self.height,
                     0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        return self.texture
    
    def invoke(self):
        """Upload from PBO to texture."""
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo_handle)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                       self.width, self.height,
                       GL_RGB, GL_UNSIGNED_BYTE, None)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
    
    def close(self):
        if self.texture:
            glDeleteTextures([self.texture])

