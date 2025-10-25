
from OpenGL.GL import *

def render_GL_texture_to_window(texture):
    #glClear(GL_COLOR_BUFFER_BIT)   # Needless 100% overdraw.
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture)
    
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex2f(-1, -1)
    glTexCoord2f(1, 1); glVertex2f(1, -1)
    glTexCoord2f(1, 0); glVertex2f(1, 1)
    glTexCoord2f(0, 0); glVertex2f(-1, 1)
    glEnd()
    

