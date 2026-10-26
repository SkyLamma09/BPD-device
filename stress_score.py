import pickle
import numpy as np
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from scipy.signal import resample
from pathlib import Path
if __name__ == "__main__":

    complete_data = []
    def dataload(files):
      secs = 60

      sizebvp  = 64 * secs
      sizeeda  = 4 * secs
      sizetemp = 4 * secs
      sizeacc  = 32 * secs
      sizelabel = 42000

      with open(rf"C:\Users\yona\Downloads\{str(files)}", 'rb') as f:
          data = pickle.load(f, encoding='latin1')
      nums = min(
          len(data["signal"]["wrist"]["ACC"]) // sizeacc,
          len(data["signal"]["wrist"]["BVP"]) // sizebvp,
          len(data["signal"]["wrist"]["EDA"]) // sizeeda,
          len(data["signal"]["wrist"]["TEMP"]) // sizetemp,
          len(data["label"]) // sizelabel)
      ACC = data["signal"]["wrist"]["ACC"][:sizeacc*nums]
      BVP = data["signal"]["wrist"]["BVP"][:sizebvp*nums]
      EDA = data["signal"]["wrist"]["EDA"][:sizeeda*nums]
      TEMP = data["signal"]["wrist"]["TEMP"][:sizetemp*nums]
      LABEL = data["label"][:sizelabel*nums]
      windowsaccmag = []
      windowsaccstd = []
      windowsbvp = []
      windowsbvpstd = []
      windowseda = []
      windowsedastd = []
      windowstemp = []
      windowslabel = []
      #accelerometer magnitude mean and std
      for x in range(nums):
        samplesacc = []
        for i in range(sizeacc*x,sizeacc*(x+1)):
          magnitude = np.sqrt(ACC[i][0]**2 + ACC[i][1]**2 + ACC[i][2]**2)
          samplesacc.append(magnitude)
        windowsaccmag.append((np.mean(samplesacc)))
        windowsaccstd.append(np.std(samplesacc))
      #temp mean
      for x in range(nums):
        samplestemp = []
        for i in range(sizetemp*x,sizetemp*(x+1)):
          samplestemp.append(TEMP[i])
        windowstemp.append((np.mean(samplestemp)))
      #heart rate mean std
      for x in range(nums):
        samplesbvp = []
        for i in range(sizebvp*x,sizebvp*(x+1)):
          samplesbvp.append(BVP[i])
        windowsbvp.append((np.mean(samplesbvp)))
        windowsbvpstd.append(np.std(samplesbvp))
      #eda mean std
      for x in range(nums):
        sampleseda = []
        for i in range(sizeeda*x,sizeeda*(x+1)):
          sampleseda.append(EDA[i])
        windowseda.append((np.mean(sampleseda)))
        windowsedastd.append(np.std(sampleseda))
        #groundtruth maker
      for x in range(nums):
        sampleslab = []
        for i in range(sizelabel*x,sizelabel*(x+1)):
          sampleslab.append(LABEL[i])
        windowslabel.append([statistics.mode(sampleslab), x])
    #compiling all markers and groundtruth
      finallist = []
      for i in range(nums):
        if windowslabel[i][0] not in (1,2):
          continue
        adding = [windowsedastd[i],windowseda[i],windowsbvp[i],windowsbvpstd[i],windowstemp[i],windowsaccmag[i],windowsaccstd[i],windowslabel[i][0]]
        finallist.append(adding)
      return finallist
    for i in range(2,18):
      if i == 12:
        continue
      complete_data.append(dataload(f"S{i}.pkl"))
    inputlist = []
    labellist = []
    testlist = []
    labeltest = []
    for i in range(len(complete_data)): #brings down to subjects
        if i == 14:
          for j in complete_data[i]:
            testlist.append(j[:7])
            labeltest.append(j[7])
          continue
        for x in complete_data[i]:  # brings down to sublists
          inputlist.append(x[:7])
          labellist.append(x[7])

    #splits results into training set and training groundtruths and test set and test groundtruth
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    model.fit(inputlist, labellist)

    predictions = model.predict(testlist)

    accuracy = accuracy_score(labeltest, predictions)
    print(f"Accuracy: {accuracy}")
    print(classification_report(labeltest, predictions, target_names=['baseline', 'stress']))

    model_path = Path(__file__).resolve().parent /"stress_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
model_path = Path(__file__).resolve().parent /"stress_model.pkl"
model = joblib.load(model_path)


def stress_score(ACC, BVP, EDA, TEMP):

    print("Model classes:", model.classes_)  # TEMPORARY DEBUG LINE

    ACC = resample(ACC, 1920, axis=0)
    BVP = resample(BVP, 3840)
    EDA = resample(EDA, 240)
    TEMP = resample(TEMP, 240)
    magnitude = np.sqrt(np.sum(ACC**2, axis=1))
    features = [[np.std(EDA), np.mean(EDA), np.mean(BVP), np.std(BVP), np.mean(TEMP), np.mean(magnitude), np.std(magnitude)]]

    classes = list(model.classes_)

    # Class 1 is assumed to represent stress; verify this against your labels.
    if 1 not in classes:
        raise ValueError(f"Stress class 1 is missing from model classes: {classes}")

    stress_index = classes.index(1)

    return model.predict_proba(features)[0][stress_index]
