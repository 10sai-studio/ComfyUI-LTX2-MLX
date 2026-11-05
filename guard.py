"""Runs a command and takes it down when whoever started this process is gone.

macOS has no PR_SET_PDEATHSIG, and a ComfyUI killed outright runs no cleanup, so nothing in it can stop a generation
holding the whole memory of the machine. The parent keeps the write end of this process's stdin: when the parent dies
for any reason the kernel closes it, and the read below returns.
"""
import os
import signal
import subprocess
import sys
import threading


def main() -> int:
    child = subprocess.Popen(sys.argv[1:], stdin=subprocess.DEVNULL)

    def watch() -> None:
        # `os.read` and not `sys.stdin`: a daemon thread blocked on the buffered object aborts the interpreter at exit.
        while os.read(0, 4096):
            pass
        # This process leads the group (the parent starts it in a session of its own), so ffmpeg goes too.
        os.killpg(0, signal.SIGKILL)

    threading.Thread(target=watch, daemon=True).start()
    code = child.wait()
    return 128 - code if code < 0 else code


if __name__ == "__main__":
    # Past the interpreter's shutdown, for the same reason as `os.read`: the watching thread is still blocked.
    sys.stdout.flush()
    os._exit(main())
