import csv
import numpy as np
from agcounts.extract import get_counts
from datetime import datetime
import scipy.io
import joblib
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import resample_poly

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
        else:
            i[0] = "Remove"

    finals = []
    epoch_counts_seq = []
    for i in actualfinal:
        heartr = []
        count = []
        for x in i[1:]:
            heartr.append(x[0][1])
            count.append(x[1])
        if len(heartr) == 0:
            continue
        total_count = np.sum(count, axis=0)
        epoch_scalar = total_count.sum()
        finals.append([i[0], np.mean(heartr), total_count])
        epoch_counts_seq.append(epoch_scalar)

    sadeh_counts = [0]*11
    sadeh_predictions = []
    for c in epoch_counts_seq:
        sadeh_counts.append(c)
        if len(sadeh_counts) > 11:
            sadeh_counts.pop(0)

        LG = 0 if sadeh_counts[-1] == 0 else np.log(sadeh_counts[10])
        AVG = np.mean(sadeh_counts)
        SD = np.std(sadeh_counts[:-6])
        NATS = sum(1 for x in sadeh_counts if 50 <= x < 100)
        raw = (7.601 - (0.065*AVG) - (1.08*NATS) - (0.056*SD) - (0.703*LG))
        sadeh_predictions.append(raw <= -4)

    result = []
    for i in range(len(finals)):
        result.append([finals[i][0], finals[i][1], finals[i][2], sadeh_predictions[i]])

    return result

completed1 = []
for i in range(46):
    bob1 = []
    for x in range(1, 8):
        try:
            bob1.append(thing(x, i))
        except FileNotFoundError as e:
            break
        except Exception as e:
            print(f"Subject {i}, night {x}: unexpected error — {type(e).__name__}: {e}")
            break
    completed1.append(bob1)



subject_X = []
subject_y = []

for subject_nights in completed1:
    rows = []
    labels = []
    for night in subject_nights:
        for row in night:
            gt, hr, counts, sadeh_pred = row
            features = [hr, counts[0], counts[1], counts[2], int(sadeh_pred)]
            rows.append(features)
            labels.append(int(gt))
    if len(rows) == 0:
        continue
    subject_X.append(np.array(rows))
    subject_y.append(np.array(labels))

X_train = np.vstack(subject_X).astype(float)
y_train = np.concatenate(subject_y).astype(int)

if len(np.unique(y_train)) < 2:
    raise ValueError("The selected subject must contain both classes.")

clf = RandomForestClassifier(
    n_estimators=100,
    class_weight={0: 1.0, 1: 0.5},
    random_state=42
)

clf.fit(X_train, y_train)

joblib.dump(clf, "random_forest_model.pkl")

#print(f"Trained on subject index: {train_subject_idx}")
#print(f"Training samples: {len(y_train)}")
#print("Training accuracy:", accuracy_score(y_train, clf.predict(X_train)))

def sadeh(epoch_amount):
    sadeh_counts = [0] * 11
    sadeh = []
    for i in epoch_amount:
        sadeh_counts.append(i)
        if len(sadeh_counts) > 11:
            sadeh_counts.pop(0)

        LG = 0 if sadeh_counts[-1] == 0 else np.log(sadeh_counts[10])
        AVG = np.mean(sadeh_counts)
        SD = np.std(sadeh_counts[:-6])
        NATS = sum(1 for x in sadeh_counts if 50 <= x < 100)

        raw = (7.601 - (0.065 * AVG) - (1.08 * NATS) - (0.056 * SD) - (0.703 * LG))
        sadeh.append(raw <= -4)

    return sadeh

def predictions(acc, hrm, epochs):
    sadeh_preds = sadeh(epochs) # true sleep false awake
    sadeh_predi = sadeh_preds[-1]  # Latest sleep/wake prediction

    model = joblib.load("random_forest_model.pkl")
    hrm = np.asarray(hrm, dtype=float).flatten()
    acc = resample_poly(np.asarray(acc,dtype=float), up=1, down=2, axis=0)

    counting = get_counts(acc, freq=50, epoch=1)
    counts = counting.sum(axis=0)


    features1 = [[np.mean(hrm), counts[0], counts[1], counts[2], int(sadeh_predi)]]

    verdict = model.predict(features1)
    return verdict


def activity_counts(acc):
    acc = np.asarray(acc, dtype=float)
    downsample = resample_poly(acc, up=1, down=2, axis=0)
    counts = get_counts(downsample, freq=50, epoch=1)
    return counts.sum()
