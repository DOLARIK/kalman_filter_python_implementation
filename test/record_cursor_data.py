import time
import numpy as np
import pyautogui

def record_data(noise = 20, time_max = 10):

    print("RECORDING DATA")

    coords = []
    T = time.time()

    while time.time() - T < time_max:

        t = time.time()
        while True:
            if time.time() - t > .02:

                pos = np.asarray(pyautogui.position()) + 2*noise*np.asarray([np.random.rand(), np.random.rand()]) - noise*np.asarray([np.random.rand(), np.random.rand()])
                coords.append(pos)

                break

    print("RECORDING DATA COMPLETED")

    return coords
