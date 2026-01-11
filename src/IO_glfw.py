
from IO import IO
import glfw

glfw_key_map = {
    # Named keys
    glfw.KEY_SPACE:     ' ',
    glfw.KEY_BACKSPACE: 'backspace',
    glfw.KEY_DELETE:    'delete',
    glfw.KEY_ENTER:     'enter',
    glfw.KEY_TAB:       'tab',
    glfw.KEY_HOME:      'home',
    glfw.KEY_PAGE_UP:   'page-up',
    glfw.KEY_PAGE_DOWN: 'page-down',
    glfw.KEY_END:       'end',
    glfw.KEY_ESCAPE:    'escape', 

    # Arrows
    glfw.KEY_RIGHT:     'right',
    glfw.KEY_LEFT:      'left',
    glfw.KEY_UP:        'up',
    glfw.KEY_DOWN:      'down',

    # Common mods
    glfw.KEY_LEFT_CONTROL:  'control',
    glfw.KEY_RIGHT_CONTROL: 'control',
    glfw.KEY_LEFT_SHIFT:    'shift',
    glfw.KEY_RIGHT_SHIFT:   'shift',
}

class IO_glfw(IO):

    def __init__(self):

        if not glfw.init():
            raise Exception("GLFW init failed")

        monitor     = glfw.get_primary_monitor()
        mode        = glfw.get_video_mode(monitor)
        screen_size = (mode.size.width, mode.size.height)

        IO.__init__(self, screen_size)

        self.monitor = monitor
        self.vmode   = mode
        self.window  = None

    def set_size(self, size, aspect_ratio=True):

        osize = size

        if size is None:
            size = self.screen_size
        else:
            size = self.crop_size(size, self.screen_size, aspect_ratio)

        if size == self.screen_size:
            monitor = self.monitor
        else:
            monitor = None

        #print(f"SetWindowSize({osize}->{size}, aspect={aspect_ratio}) - fullscreen={monitor is not None}")

        if self.window is None:
            #
            # Create the GL window and make it the global rendering target:
            #
            self.window = glfw.create_window(*size, "XR Play", monitor, None)  # FIXME: title should be a parameter...
            if not self.window:
                raise Exception("Window creation failed")
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)

            if aspect_ratio:
                glfw.set_window_aspect_ratio(self.window, *size)

            glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)
            glfw.set_mouse_button_callback    (self.window, self.mouse_button_callback)
            glfw.set_cursor_pos_callback      (self.window, self.cursor_pos_callback)
            glfw.set_key_callback             (self.window, self.key_callback)
            glfw.set_char_callback            (self.window, self.char_callback)
            glfw.set_scroll_callback          (self.window, self.scroll_callback)
        else:
            glfw.set_window_aspect_ratio(self.window, glfw.DONT_CARE, glfw.DONT_CARE)
            glfw.set_window_size_limits(self.window, glfw.DONT_CARE, glfw.DONT_CARE, 
                                            glfw.DONT_CARE, glfw.DONT_CARE)

            if self.size == self.screen_size:
                if monitor is None:
                    x, y = 100, 100
                else:
                    x, y = 0, 0
            else:
                x, y = glfw.get_window_pos(self.window)

            glfw.set_window_monitor(self.window, monitor, x, y, *size, glfw.DONT_CARE)

            if False:
                # Debugging why fullscreen windows won't resize even though they do transition to non-fullscreen:
                print(f"glfw.set_window_monitor(win, {monitor and 'monitor'}, {x}, {y}, *{size}, glfw.DONT_CARE)")
                print(f"glfw.get_window_size(win) == {glfw.get_window_size(self.window)}")
                # glfw bug: sometimes the size isn't set during transitions away from fullscreen...
                if glfw.get_window_size(self.window) != size:
                    print(f"glfw.set_window_size(win, *{size})")
                    glfw.set_window_size(self.window, *size)
                print(f"glfw.get_window_size(win) == {glfw.get_window_size(self.window)}")

            if aspect_ratio:
                glfw.set_window_aspect_ratio(self.window, *size)

        self.size = size

    def swap_buffers(self):
        glfw.swap_buffers(self.window)

    def poll_events(self):
        glfw.poll_events()

        if glfw.window_should_close(self.window):
            self.quitted = True

    def close(self):
        glfw.terminate()

    def mouse_button_callback(self, window, button, action, mods):
        """Handle mouse button events."""
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_down = (action == glfw.PRESS)
    
    def cursor_pos_callback(self, window, x, y):
        """Handle mouse movement."""
        self.mouse_x = x
        self.mouse_y = y
    
    def key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard events."""
        if key in glfw_key_map or 64<key<91:
            # Map key from glfw standard to known_keys:
            if key in glfw_key_map:
                key = glfw_key_map[key]
            else:
                key = chr(key+32)
            # Update our set of down keys:
            if action == glfw.PRESS:
                self.keys_down.add(key)
                self.keystrokes.append(key)
            elif action == glfw.RELEASE:
                self.keys_down.discard(key)

    def char_callback(self, window, codepoint):
        self.text_input += chr(codepoint)

    def scroll_callback(self, window, dx, dy):
        self.mouse_scroll_x += dx
        self.mouse_scroll_y += dy

    def framebuffer_size_callback(self, window, width, height):
        """Handle window resize."""
        self.size = (width, height)

    def quit(self):
        if not self.quitted:
            IO.quit(self)
            if self.window is not None:
                glfw.set_window_should_close(self.window, True)

