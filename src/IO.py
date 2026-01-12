
#
# This module tracks mouse and keyboard with accumulators
#   which the various plugins and controlling objects can use.
#


#
# These are the standard names for keyboard keys for boolean key states.
#
# All single-letter key names correspond to that keyboard character
#   (lower-case version always).  Use ' ' for space.
#
# Key code from various front and back ends should be translated to and
#  from these standard names.
#
known_keys = {
        # Named keys
        'backspace',
        'delete',
        'enter',
        'tab',
        'home',
        'page-up',
        'page-down',
        'end',
        'escape',

        # Arrows
        'right',
        'left',
        'up',
        'down',

        # common mods
        'control',
        'shift',
    }

class IO(object):

    def __init__(self, screen_size):

        self.screen_size = screen_size      # Size of the full screen

        # The size (width, height) of the window or space that the mouse is in
        # This can change if the window is resized, etc.
        self.size   = None
        self.aspect = False     # True if the window aspect ratio is locked.

        # Mouse state
        self.mouse_x    = 0
        self.mouse_y    = 0
        self.mouse_down = False

        # Keyboard state, some subset of known_keys
        self.keys_down = set()

        # Accumulators (see clear() for available list of accumulators):
        self.clear()

        self.play    = None     # Browser sets this to relative path of video to play when selected.
        self.quitted = False    # Flag set by .quit() saying we should wrap things up.

    def clear(self):
        """Clear accumulators.
        It's also fine to directly edit this, as in removing consumed events from keys_input
        or selectively clearly consumed mouse scroll or text_input.

        BUT you should generally call clear() at least once per cycle to make sure you
            aren't letting unused accumulators grow (including any that may be added in
            the future!).
        """
        self.mouse_scroll_x = 0.0   # Cumulative scroll since last clear
        self.mouse_scroll_y = 0.0
        self.text_input     = ''    # Plain text types since last clear
        self.keystrokes     = []    # Sequence of known_keys (incl single letter) hit (down) since last clear.
        # Note keystrokes and text_input are highly redundant

    def set_size(self, size, aspect_ratio=True):
        """This must be called at least once as it established the window.

        The passed size is a request.  The actual size may be reduced to fit
            on the screen.

        If aspect_ratio is True, the aspect ratio will be preserved when
            the window is either shrunk to fit the screen or resized by the
            user.

        size may be None to get the full screen.
        """
        assert False, "Subclass needs to implement this"

    def crop_size(self, size, max_size=None, aspect_ratio=True):
        "Just a handy utility that crops size to max_size, optionally preserving aspect ratio."
        if max_size is None:
            return size
        if size is None:
            return max_size
        if aspect_ratio:
            scale = min(max_size[i]/size[i] for i in (0, 1))
            if scale < 1.0:
                return tuple(int(size[i]*scale) for i in (0, 1))
            return size
        else:
            return tuple(min(size[i], max_size[i]) for i in (0, 1))

    def crop_to_screen(self, size, aspect_ratio=True):
        return self.crop_size(size, self.screen_size, aspect_ratio)

    def swap_buffers(self):
        # Swap render buffers.  Back-end specific...
        pass

    def poll_events(self):
        # Call this at least once a cycle to handle events; most likely accumulators
        #  and such will update during this.
        pass

    def quit(self):
        "Sets the flag, but also subclass may initiate back-end specific exit."
        self.quitted = True

    def close(self):
        "Clean up, shut everything down..."
        pass

