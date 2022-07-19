from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

ytrainm = []
ytestm = []
ytestv = []

# df = pd.read_excel('simResults.xlsx', usecols="D")
xepoch = np.array(pd.read_excel('simResults.xlsx', usecols="D").fillna('').values.tolist()[6:66])
xepoch = np.squeeze(xepoch)
# print(xepoch)
#
ytrain3m = np.array(pd.read_excel('simResults.xlsx', usecols="AP").fillna('').values.tolist()[6:66])
ytrain3m = np.squeeze(ytrain3m)
ytest3m = np.array(pd.read_excel('simResults.xlsx', usecols="AQ").fillna('').values.tolist()[6:66])
ytest3m = np.squeeze(ytest3m)
ytest3v = np.sqrt(np.array(pd.read_excel('simResults.xlsx', usecols="AS").fillna('').values.tolist()[6:66]))
ytest3v = np.squeeze(ytest3v)
ytrainm.append(ytrain3m)
ytestm.append(ytest3m)
ytestv.append(ytest3v)

ytrain5m = np.array(pd.read_excel('simResults.xlsx', usecols="T").fillna('').values.tolist()[6:66])
ytest5m = np.array(pd.read_excel('simResults.xlsx', usecols="U").fillna('').values.tolist()[6:66])
ytest5m = np.squeeze(ytest5m)
ytest5v = np.sqrt(np.array(pd.read_excel('simResults.xlsx', usecols="W").fillna('').values.tolist()[6:66]))
ytest5v = np.squeeze(ytest5v)
ytrainm.append(ytrain5m)
ytestm.append(ytest5m)
ytestv.append(ytest5v)

ytrain9m = np.array(pd.read_excel('simResults.xlsx', usecols="BK").fillna('').values.tolist()[6:66])
ytest9m = np.array(pd.read_excel('simResults.xlsx', usecols="BL").fillna('').values.tolist()[6:66])
ytest9m = np.squeeze(ytest9m)
ytest9v = np.sqrt(np.array(pd.read_excel('simResults.xlsx', usecols="BN").fillna('').values.tolist()[6:66]))
ytest9v = np.squeeze(ytest9v)
ytrainm.append(ytrain9m)
ytestm.append(ytest9m)
ytestv.append(ytest9v)

plt.plot(xepoch, ytestm[0], 'red', label='2 steps')
plt.fill_between(xepoch, ytestm[0]-ytestv[0], ytestm[0]+ytestv[0], color='red', alpha=0.1)

plt.plot(xepoch, ytestm[1], 'blue', label='4 steps')
plt.fill_between(xepoch, ytestm[1]-ytestv[1], ytestm[1]+ytestv[1], color='blue', alpha=0.1)

plt.plot(xepoch, ytestm[2], 'green', label='8 steps')
plt.fill_between(xepoch, ytestm[2]-ytestv[2], ytestm[2]+ytestv[2], color='green', alpha=0.1)

plt.ylim([0, 100])
plt.xlim([0, 110])
plt.grid(True)
plt.xlabel('Communication round')
plt.ylabel('Test accuracy (%)')
plt.legend()
plt.show()