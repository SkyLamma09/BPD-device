import numpy as np

#datapw = [[] for i in range(7)] #seven days long so seven sublists
def interstab(datapw):
  n = 24 * len(datapw) #total number of datapoints
  hourd = [[] for i in range(24)]
  for x in range(24):
    for i in datapw:
      hourd[x].append(i[x])
  print(hourd)
  hmeans = [np.mean(x) for x in hourd]
  print(hmeans)
  imean = [x for i in datapw for x in i]
  gmean = np.mean([x for i in datapw for x in i])
  print(gmean) # mean of all datapoints
  return (n * sum((hmeans[i]-gmean) ** 2 for i in range(24))) / (24*sum((imean[x]-gmean)**2 for x in range(n)))


#indiv = []
def intravar(indiv): #indiv is flattened list of all sleep values over week
  smean = np.mean(indiv)
  n = len(indiv)
  denominator = (n - 1) * sum((indiv[x] - smean) ** 2 for x in range(n))
  if denominator == 0:
    return np.nan
  return (n*(sum((indiv[i]-indiv[i-1])**2 for i in range(1,n)))/((n-1)*sum((indiv[x]-smean)**2 for x in range(n))))
def relatamp(datapw):
  hourd = [[] for i in range(24)]
  for x in range(24):
    for i in datapw:
      hourd[x].append(i[x])
  comparem = []
  comparel = []
  hmeans = [np.mean(x) for x in hourd]*2 # mean value for each specific hour
  for i in range(24):
    comparem.append(np.mean(hmeans[i:(i+10)]))
  for x in range(24):
    comparel.append(np.mean(hmeans[x:(x+5)]))
  mten = max(comparem)
  lfive = min(comparel)
  return (mten-lfive)/(mten+lfive)
