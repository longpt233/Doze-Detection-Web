import numpy as np

a = np.load('BlinksTest_30_Fold1.npy')
# print(a[1])

# TestL=[[1]]

# print(TestL)

# b = np.load('LabelsTest_30_Fold1.npy')
# # print(a[1])

print(type(a[3]))
print(a[3])
print(a[2].shape)



# list30=np.asndarray(list([1,2,3,4]))

npa = np.asarray(list([[1,2,3,4],[1,5,6,7]]))


print(type(npa))
print(npa)