import numpy as np
import pickle
with open(rf"C:\Users\yona\Downloads\S2.pkl", 'rb') as f:
    data = pickle.load(f, encoding='latin1')
tpr = data["signal"]["wrist"]["TEMP"]

idex = 0
baseline = [0, 0]


def deviate(data, idx):
    temp = []
    for i in data[120*idx:120*(idx + 1)]:
        temp.append(i)

    return np.mean(temp)


def newscore():
    baseline[1] = deviate(tpr, idex)
    deviation = baseline[1] - baseline[0]
    baseline[0] = baseline[-2] * 0.9 + baseline[-1] * 0.1
    return deviation


while True:
    idex += 1
    calc = newscore()
    if 120 * (idex + 1) > len(tpr):
        break
    print(calc)

