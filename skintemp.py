import numpy as np

baseline = None
def skintemp(data):
    global baseline
    datas = np.asarray(data, dtype=float).flatten()
    mtemp = np.mean(datas)
    stdtemp = np.std(datas)
    
    if baseline is None:
        baseline = mtemp
        deviation = 0
    else:
        deviation = mtemp-baseline
        baseline = 0.9 * baseline + 0.1 * mtemp
    return [mtemp, stdtemp, deviation]
