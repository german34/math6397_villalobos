#SET UP ENVIRONMENT
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os


##################################################################
fname = 'dog.jpg'
r = 10
n = 1000#None
##################################################################
#Loading data

#img_data = imread(os.path.join('..','data','dog.jpg')) #if moved to 'hw' directory
img_data = imread(fname)
X = np.mean(img_data, -1); # Convert RGB to grayscale

#compute full svd
U, S, Vt = np.linalg.svd(X, full_matrices = True)

#choose rank r< m to confirm U*U is 'r x r' matrix
X_r = U[:,:r].T @ U[:,:r]
rank_X = np.linalg.matrix_rank(X_r, tol=None)
print('Rank of truncated matrix at r = ', r, ' is actually ', rank_X)

#compute the norm of error between UU* and the 'n x n' identity matrix
#as the rank r varies from 1 to n and plot the error
print('Now computing differences/norm.......')
norms = []
for i in range(n):
    #compute UU*
    mat = U[:,:r] @ U[:, :r].T
    m = len(mat)
    Id = np.eye(m)

    #take difference
    diff = mat - Id
    err = np.linalg.norm(diff,ord=2)
    err = err/np.linalg.norm(mat,ord=2)
    #print(i)
    #store
    norms.append(err)

norms = np.array(norms)

#plot figures
#plt.rcParams['figure.figsize'] = [16, 8]

plt.figure()
plt.plot(norms)
plt.title('Error plot (absolute 2-nom)')
plt.show()
