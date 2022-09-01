'''
Exercise 1.3
'''
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

#Load the Yale B image data base and compute the
#economy SVD using a standard svd command
mat_contents = scipy.io.loadmat(os.path.join('allFaces.mat'))
faces = mat_contents['faces']
m = int(mat_contents['m'])
n = int(mat_contents['n'])
nfaces = np.ndarray.flatten(mat_contents['nfaces'])

print('nfaces shape  -', nfaces.shape)
print('faces shape  -',faces.shape)

X = faces[:,:np.sum(nfaces[:])]
U_hat, S_hat, VT = np.linalg.svd(X,full_matrices=0)

#checking for zero singular values
#n_zeros = np.count_nonzero(S_hat==0)
#print(n_zeros)



#Compute the SVD with the method of snapshots
#compute xTx two vectors at a time

n = X.shape[0]
m = X.shape[1]
'''
XtX = np.zeros((m,m))
for i in range(m):
    for j in range(m):
        XtX[i,j] = np.dot(x[i,:] ,x[j,:]) #  < x[i,:] , x[j,:] >
'''

XtX = X.T @ X
V2, S, V2t = = np.linalg.svd(XtX,full_matrices=0)
S2 = np.sqrt(S)

U2 = X @ V2 @ np.linalg.inv(S2)

#checking for zero singular values
n_zeros = np.count_nonzero(S2==0)
print(n_zeros)
#############################################################################
#Compare the singular value spectra on a log plot.

plt.figure(1)
plt.semilogy(S)
plt.title('Singular values for X using economy SVD')
plt.show()

plt.figure(2)
plt.semilogy(S2)
plt.title('Square root of XtX singular values using snapshot')
plt.show()

#Compare the first 10 left singular vectors using each method
#(remember to reshape them into shape of a face).
eig_econ = np.copy(U_hat[:,0:10])
eig_snap = np.copy(U2[:,0:10])

#Now compare a few singular vectors farther down the spectrum.
#Explain your findings
