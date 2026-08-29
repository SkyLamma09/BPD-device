import numpy as np
import joblib

def interstab():
  datapw = [[]*7] #seven days long so seven sublists
  n = 24 * len(datapw) #total number of datapoints
  hourd = [[] for i in range(24)]
  for x in range(24):
    for i in datapw:
      hourd[x].append(i[x])
  print(hourd)
  hmeans = [np.mean(x) for x in hourd] # Mean value for each specific hour
  print(hmeans)
  imean = [x for i in datapw for x in i]
  gmean = np.mean([x for i in datapw for x in i])
  print(gmean) # mean of all datapoints
  return ((24*sum((hmeans[i]-gmean)**2 for i in range(24)))/(n*sum((imean[x]-gmean)**2 for x in range(n))))
# DO ROLLING WINDOW SLIDE WINDOW EVERY DAY AFTER 7

def intravar():
  indiv = []
  smean = np.mean(indiv)
  n = 24
  return ((n*(sum(indiv[i]-indiv[i-1])**2 for i in range(1,n)))/(23*sum(indiv[x]-smean)**2 for x in range(n)))
def relatamp():
  hourd = [[] for i in range(24)]
  comparem = []
  comparel = []
  hmeans = [np.mean(x) for x in hourd]*2 # mean value for each specific hour
  for i in range(24):
    comparem.append(np.mean(hmeans[i:(i+10)]))
  for x in range(24):
    comparel.append(np.mean(hmeans[i:(i+5)]))
  mten = max(comparem)
  lfive = min(comparel)
  return (mten-lfive)/(mten+lfive)

model = joblib.load("random_forest_model.pkl")
prediction = []
prediction.append(model.predict(inputs)[0]) #inputs is the input of the model

print(prediction)

