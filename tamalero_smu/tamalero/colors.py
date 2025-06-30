#!/usr/bin/env python3

class Colors:
    BLACK = '\033[30m'
    BLUE = '\033[94m'
    BLUE = '\033[34m'
    CYAN = '\033[96m'
    CYAN = '\033[36m'
    DARKGREY = '\033[90m'
    GREEN = '\033[92m'
    LIGHTBLUE = '\033[94m'
    LIGHTCYAN = '\033[96m'
    LIGHTGREEN = '\033[92m'
    LIGHTGREY = '\033[37m'
    LIGHTRED = '\033[91m'
    MAGENTA = '\033[95m'
    ORANGE = '\033[33m'
    PINK = '\033[95m'
    PURPLE = '\033[35m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'


def color(s, a_color):
    return(a_color + s + Colors.ENDC)


def green(s):
    return(Colors.GREEN + s + Colors.ENDC)


def red(s):
    return(Colors.RED + s + Colors.ENDC)


def blue(s):
    return(Colors.BLUE + s + Colors.ENDC)


def yellow(s):
    return(Colors.YELLOW + s + Colors.ENDC)


def magenta(s):
    return(Colors.MAGENTA + s + Colors.ENDC)


def cyan(s):
    return(Colors.CYAN + s + Colors.ENDC)

def dummy(s):
    return s

def conditional(val, threshold=1):
    if val>=threshold:
        return green(str(val))
    else:
        return red(str(val))
