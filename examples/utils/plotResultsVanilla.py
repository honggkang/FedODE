from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

ytrainm = []
ytestm = []
ytestv = []

# df = pd.read_excel('simResults.xlsx', usecols="D")
start = 8
end = 168
xepoch = np.array(pd.read_excel('simResults.xlsx', sheet_name="Sheet2", usecols="E").fillna('').values.tolist()[start:end])
xepoch = np.squeeze(xepoch)
# print(xepoch)
# #
# ytrain3m = np.array(pd.read_excel('simResults.xlsx', sheet_name="Sheet2", usecols="L").fillna('').values.tolist()[6:66])
# ytrain3m = np.squeeze(ytrain3m)
ytest3m = np.array(pd.read_excel('simResults.xlsx', sheet_name="Sheet2", usecols="L").fillna('').values.tolist()[start:end])*100
ytest3m = np.squeeze(ytest3m)
ytest3v = np.sqrt(np.array(pd.read_excel('simResults.xlsx', sheet_name="Sheet2", usecols="M").fillna('').values.tolist()[start:end]))*100
ytest3v = np.squeeze(ytest3v)
# ytrainm.append(ytrain3m)
ytestm.append(ytest3m)
ytestv.append(ytest3v)

# ytrain5m = np.array(pd.read_excel('simResults.xlsx', sheet_name="Sheet2", usecols="T").fillna('').values.tolist()[6:66])
ytest5m = np.array(pd.read_excel('simResults.xlsx', sheet_name="Sheet2", usecols="U").fillna('').values.tolist()[start:end])*100
ytest5m = np.squeeze(ytest5m)
ytest5v = np.sqrt(np.array(pd.read_excel('simResults.xlsx', sheet_name="Sheet2", usecols="V").fillna('').values.tolist()[start:end]))*100
ytest5v = np.squeeze(ytest5v)
# ytrainm.append(ytrain5m)
ytestm.append(ytest5m)
ytestv.append(ytest5v)

# ytrain9m = np.array(pd.read_excel('simResults.xlsx', sheet_name="Sheet2", usecols="BK").fillna('').values.tolist()[6:66])
ytest9m = np.array(pd.read_excel('simResults.xlsx', sheet_name="Sheet2", usecols="AD").fillna('').values.tolist()[start:end])*100
ytest9m = np.squeeze(ytest9m)
ytest9v = np.sqrt(np.array(pd.read_excel('simResults.xlsx', sheet_name="Sheet2", usecols="AE").fillna('').values.tolist()[start:end]))*100
ytest9v = np.squeeze(ytest9v)
# ytrainm.append(ytrain9m)
ytestm.append(ytest9m)
ytestv.append(ytest9v)

plt.plot(xepoch, ytestm[0], 'red', label='2 steps')
plt.fill_between(xepoch, ytestm[0]-ytestv[0], ytestm[0]+ytestv[0], color='red', alpha=0.1)

plt.plot(xepoch, ytestm[1], 'blue', label='4 steps')
plt.fill_between(xepoch, ytestm[1]-ytestv[1], ytestm[1]+ytestv[1], color='blue', alpha=0.1)

plt.plot(xepoch, ytestm[2], 'green', label='8 steps')
plt.fill_between(xepoch, ytestm[2]-ytestv[2], ytestm[2]+ytestv[2], color='green', alpha=0.1)

# plt.ylim([0, 100])
plt.xlim([0, 5])
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Test accuracy (%)')
plt.legend()
plt.show()