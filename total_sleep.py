import csv
import numpy as np
from agcounts.extract import get_counts
from datetime import datetime
import scipy.io

def thing(night, person):
    with open(rf"C:\Users\yona\Downloads\heartrateandstuff\a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0\{person}\{night}\motion.csv", mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        accel = []
        for i in reader:
            accel.append(i)
        accel.pop(0)
    with open(rf"C:\Users\yona\Downloads\heartrateandstuff\a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0\{person}\{night}\hr.csv", mode='r', newline='', encoding='utf-8') as file:
        heart = []
        reader1 = csv.reader(file)
        for i in reader1:
            heart.append(i)
    mat_data = scipy.io.loadmat(rf"C:\Users\yona\Downloads\heartrateandstuff\a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0\{person}\{night}\labels.mat")
    expert_labels = mat_data['expert_label'].flatten()
    biglist = [[heart[0]]]
    year = mat_data['recStart']
    converted = datetime.strptime(year[0], '%Y-%m-%d %H:%M:%S')
    unixe = int(converted.timestamp())
    actualfinal = []

    for i in range(1, len(heart)):
        if abs(float(heart[i-1][-2]) - float(heart[i][-2])) <= 4.8:
            biglist[-1].append(heart[i])
        else:
            biglist.append([heart[i]])

    beep = []
    for i in biglist:
        timelist = []
        hr = []
        for x in i:
            timelist.append(float(x[0]))
            hr.append(float(x[1]))
        beep.append([(max(timelist)+min(timelist))/2, np.mean(hr), max(timelist)-min(timelist)+4.8])

    time = np.array([float(i[0]) for i in accel])
    for i in range(len(beep)):
        start_idx, end_idx = np.searchsorted(time, [beep[i][0]-2.4, beep[i][0]+2.4], side='right')
        beep[i].append(end_idx-start_idx)
        for x in range(start_idx, end_idx):
            float_list = [float(item) for item in accel[x][1:4]]
            beep[i].append(float_list)

    working = []
    for i in beep:
        scale = i[2]
        samples = i[3]
        ratio = samples/(scale*51.4)
        if ratio >= 0.6:
            working.append(i)

    final = []
    for i in working:
        matrix = np.array(i[4:])
        window_counts = get_counts(matrix, freq=50, epoch=1)
        if window_counts.shape[0] == 0:
            continue
        window_counts = window_counts.sum(axis=0)
        final.append([i[0:2], window_counts])

    labeled = []
    for i in range(len(expert_labels)):
        labeled.append([expert_labels[i]])

    for i in final:
        epoch = int(i[0][0]-unixe)//30
        if epoch < 0 or epoch >= len(labeled):
            continue
        labeled[epoch].append(i)

    for i in labeled:
        if i[0] == 0:
            i[0] = False
            actualfinal.append(i)
        elif 1 <= i[0] <= 4:
            i[0] = True
            actualfinal.append(i)

    finals = []
    epoch_amount = []
    for i in actualfinal:
        heartr = []
        count = []
        for x in i[1:]:
            heartr.append(x[0][1])
            count.append(x[1])
        if len(heartr) == 0:
            continue
        total_count = np.sum(count, axis=0)
        epochs = total_count.sum()
        finals.append([i[0], np.mean(heartr), total_count])
        epoch_amount.append(epochs)

    windows = [0]*11
    sadeh = []
    for i in epoch_amount:
        sadeh_counts.append(i)
        if len(sadeh_counts) > 11:
            sadeh_counts.pop(0)

        if windows[-1] == 0:
          LG = 0
        else:
           np.log(windows[10])
        AVG = np.mean(windows)
        SD = np.std(windows[:-6])
        counts = 0
        for x in windows:
          if 50<= x < 100:
            counts += 1
        NATS = counts
        NATS = sum(1 for x in windows if 50 <= x < 100)
        raw = (7.601 - (0.065*AVG) - (1.08*NATS) - (0.056*SD) - (0.703*LG))
        sadeh.append(raw <= -4)

    result = []
    for i in range(len(finals)):
        result.append([finals[i][0], finals[i][1], finals[i][2], sadeh[i]])

    return result

completed1 = []
for i in range(48):
    bob1 = []
    for x in range(1, 8):
        try:
            bob1.append(thing(x, i))
        except FileNotFoundError as e:
            break
        except Exception as e:
            print(f"Subject {i}, night {x}: error — {type(e).__name__}: {e}")
            break
    completed1.append(bob1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

subject_X = []
subject_y = []

for subject_nights in completed1:
    rows = []
    labels = []
    for night in subject_nights:
        for row in night:
            groundtruth, hr, counts, sadeh_pred = row
            finall = [hr, counts[0], counts[1], counts[2], int(sadeh_pred)]
            rows.append(finall)
            labels.append(int(groundtruth))
    if len(rows) == 0:
        continue
    subject_X.append(np.array(rows))
    subject_y.append(np.array(labels))

n_subjects = len(subject_X)-30
print(f"Amount of subjects: {n_subjects}")

precision = []
recall = []
accuracy = []
specificity = []

for x in range(n_subjects):
    X_train = np.vstack([subject_X[i] for i in range(n_subjects) if i != x])
    y_train = np.concatenate([subject_y[i] for i in range(n_subjects) if i != x])
    X_test = subject_X[x]
    y_test = subject_y[x]

    if len(np.unique(y_train)) < 2:
        continue

    forest = RandomForestClassifier(n_estimators=100, class_weight={0:1.0,1:1.0}, random_state=42)
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)

    precision.append(precision_score(y_test, y_pred, zero_division=0))
    recall.append(recall_score(y_test, y_pred, zero_division=0))
    accuracy.append(accuracy_score(y_test, y_pred))

    tneg, fpos, fneg, tpos = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    if (tneg + fpos) > 0:
      specificityz = tneg/(tneg + fpos) 
    else:
      specificityz = 0
    specificity.append(specificityz)

print(f"Avg precision:   {np.mean(precision):.3f}")
print(f"Avg recall:      {np.mean(recall):.3f}")
print(f"Avg accuracy:    {np.mean(accuracy):.3f}")
print(f"Avg specificity: {np.mean(specificity):.3f}")
