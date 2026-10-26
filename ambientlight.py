import numpy as np

def lights(lux_readings, maxlight=10, interval=10):
    lux = np.asarray(lux_readings, dtype=float)

    return [np.sum(lux > maxlight * interval), np.max(lux), np.var(lux)]
