

class Button(object):
    """Just detects False->True transitions."""
    __slots__ = ('down',)

    def __init__(self):
        self.down = False

    def click(self, down):
        """Call this with the current state.
        Returns True once when down first transitions to True.
        """
        click     = down and not self.down
        self.down = down
        return click

