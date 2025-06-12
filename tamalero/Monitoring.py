#!/usr/bin/env python3
import time

def module_mon(module, sleep=10):
    from threading import Thread
    mon = Monitoring(module.monitor)
    t = Thread(target = mon.run, args=(sleep,))
    t.start()
    return mon

def blink_rhett(rb, iterations=5):
    from threading import Thread
    mon = Monitoring(rb.bad_boy)
    t = Thread(target = mon.run_limited, args=(iterations,))
    t.start()
    return mon

class Monitoring:

    def __init__(self, fun):
        self._running = True
        self.fun = fun

    def terminate(self):
        self._running = False

    def run(self, sleep=60):
        while self._running and True:
            self.fun()
            time.sleep(sleep)

    def run_limited(self, iterations=5):
        for it in range(iterations):
            self.fun()
        self._running = False

class Lock:
    def __init__(self, to_lock):
        self.to_lock = to_lock
        self.to_lock.locked = to_lock.locked
    def __enter__(self):
        while self.to_lock.locked:
            time.sleep(0.001)  # this is not for performance computing
        self.to_lock.locked = True
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.to_lock.locked = False
        return False
