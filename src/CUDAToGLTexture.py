
from CUDAToPBO      import CUDAToPBO
from PBOToGLTexture import PBOToGLTexture

class CUDAToGLTexture(object):
    """Copies a CUDA object to an OpenGL texture via PBO.
    """
    
    def __init__(self, width, height, init=False):

        self.cuda_to_pbo       = CUDAToPBO(width, height)
        self.pbo_to_gl_texture = PBOToGLTexture(width, height)
        self.texture           = None

        if init:
            self.init()

    def init(self):

        self.texture = self.pbo_to_gl_texture.init(self.cuda_to_pbo.init())

        return self.texture

    def invoke(self, cuda_ptr, pitch) -> bool:

        self.cuda_to_pbo      .invoke(cuda_ptr, pitch)
        self.pbo_to_gl_texture.invoke()

    def close(self):

        # Generally reverse order from create is safest:
        self.pbo_to_gl_texture.close()
        self.cuda_to_pbo      .close()

