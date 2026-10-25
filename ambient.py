import numpy as np

def lights(lux_readings, threshold=10, interval_seconds=10):
    lux = np.asarray(lux_readings, dtype=float)

    return [int(np.sum(lux > threshold) * interval_seconds), float(np.max(lux)), float(np.var(lux))]