import numpy as np

baseline = None

def skintemp(data):
    global baseline

    npdata = np.asarray(data, dtype=float).flatten()

    meantemp = np.mean(npdata)
    stdtemp = np.std(npdata)
    if baseline is None:
        baseline = meantemp
        deviation = 0.0
    else:
        deviation = meantemp - baseline
        baseline = 0.9 * baseline + 0.1 * meantemp

    return [meantemp, stdtemp, deviation]
