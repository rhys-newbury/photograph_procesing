from time import sleep
from datetime import datetime
from sh import gphoto2 as gp
import signal, os, subprocess


def kill_gphoto2():

    p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
    out, err = p.communicate()

    for line in out.splitlines():
        if b'gvfs-gphoto2' in line:
            id = int(line.split(None, 1)[0])
            os.kill(id, signal.SIGKILL)
import timeit
def take_pic():
    a = timeit.default_timer()
    for i in range(3):
        print(i)
        gp(["--trigger-capture"])
    b = timeit.default_timer()
    print(b-a)



kill_gphoto2()
take_pic()