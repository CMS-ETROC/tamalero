import sys
import select
import termios
import tty
from tamalero.colors import yellow

class TerminalHandler:
    def __init__(self):
        self.old_settings = None

    def start_non_blocking(self):
        """Sets terminal to read input without waiting for Enter key"""
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.cbreak(sys.stdin.fileno())
        except:
            pass

    def restore(self):
        """Restores terminal to normal mode"""
        try:
            if self.old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        except:
            pass

    def check_for_q(self):
        """Returns True if user pressed 'q'"""
        try:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                char = sys.stdin.read(1)
                if char.lower() == 'q':
                    print(yellow("\nUser pressed 'q', stopping..."))
                    return True
        except:
            pass
        return False