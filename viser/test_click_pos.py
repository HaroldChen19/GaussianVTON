import random
import numpy as np
import time

import viser

server = viser.ViserServer()

@server.on_scene_click
def on_click(pointer):
    print(pointer.click_pos)

while True:
    time.sleep(0.01)