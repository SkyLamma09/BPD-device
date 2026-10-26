import numpy as np

def interstab(datapw):
  n = 48 * len(datapw) #total number of datapoints
  hourd = [[] for i in range(48)]
  for x in range(48):
    for i in datapw:
      hourd[x].append(i[x])
  hmeans = [np.mean(x) for x in hourd]
  imean = [x for i in datapw for x in i]
  gmean = np.mean([x for i in datapw for x in i]) #mean of all datapoints

  if (48 * sum((imean[x] - gmean) ** 2 for x in range(n))) == 0:
    return 0
  return (n * sum((hmeans[i]-gmean) ** 2 for i in range(48))) / (48*sum((imean[x]-gmean)**2 for x in range(n)))


def intravar(indiv):
  smean = np.mean(indiv)
  n = len(indiv)
  denominator = (n - 1) * sum((indiv[x] - smean) ** 2 for x in range(n))
  if denominator == 0:
    return 0
  return (n*(sum((indiv[i]-indiv[i-1])**2 for i in range(1,n)))/((n-1)*sum((indiv[x]-smean)**2 for x in range(n))))

def relatamp(datapw):
  hourd = [[] for i in range(48)]
  for x in range(48):
    for i in datapw:
      hourd[x].append(i[x])
  comparem = []
  comparel = []
  hmeans = [np.mean(x) for x in hourd]*2 # mean value for each specific hour
  for i in range(48):
    comparem.append(np.mean(hmeans[i:(i+10*2)]))
  for x in range(48):
    comparel.append(np.mean(hmeans[x:(x+5*2)]))
  mten = max(comparem)
  lfive = min(comparel)
  denominator = mten+lfive
  if denominator == 0:
    return 0
  return (mten-lfive)/(mten+lfive)
