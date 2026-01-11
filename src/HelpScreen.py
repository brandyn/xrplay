
from PIL       import Image, ImageDraw, ImageFont
from GLop      import render_GL_texture_to_window
from OpenGL.GL import *
from pathlib import Path

def load_image_texture():

    if False:
        img  = Image.new('RGB', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        font = None # ImageFont.truetype("arial.ttf", 48)  # or None for default
        draw.text((10, 10), "Hello World", fill=(255, 255, 255, 255), font=font)
    else:
        # Get the directory of the current script
        script_dir = Path(__file__).parent.resolve()
        img = Image.open(f"{script_dir}/help_screen.png").convert('RGB')

    img_data = img.tobytes()
    
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 
                 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    return texture_id

texture_id = None

def render_help(window_size):

    global texture_id

    if texture_id is None:
        texture_id = load_image_texture()

    render_GL_texture_to_window(texture_id, window_size)

