'''
Exercise 1.4
'''
import numpy as np
import matplotlib.pyplot as plt


#Generate a random 100 by 100 matrix, whose entries are samples from a normal distribution
dim1 = 10
mu = 1.0
std = 0.4


nsamples = dim1 * dim1
X = np.random.normal(mu, std, nsamples)
X = np.reshape(X, (dim1,dim1))

#compute the SVD of matrix
print(X)
print('X shape -- ', X.shape)
U, S, Vt = np.linalg.svd(X, full_matrices = 0)
S2 = np.diag(S)
print('S shape -- ',S.shape)
print('\n')
print(S2)

#Plot the distribution of singular values in a box-and-whisker plot.
plt.rcParams['figure.figsize'] = [16, 8]
fig = plt.figure()
plt.boxplot(S)
plt.title('Singular values distribution')
plt.show()

meds = []
means = []
#Plot the mean and median singular values as a function of r.
for r in range(0,nsamples):
    S_r = np.copy(S[0:r])
    meds.append(np.median(S_r))
    means.append(np.mean(S_r))

meds = np.array(meds)
means = np.array(means)
indx = np.array(range(nsamples)) + 1


plt.plot(indx, means, color='k', linewidth=2, label='Mean') # True relationship
plt.plot(indx, meds, 'x', color='r', markersize = 10, label='Median') # Noisy measurements
plt.xlabel('Rank')
plt.ylabel('Summary stats')
plt.grid(linestyle='--')
plt.legend()
plt.show()
################################################################################
#Now repeat this for different matrix sizes
#Ex: 50by50, 200by200, 500by500, 1000by1000


for dim in [50, 200, 500, 1000]:
    nsamples = dim1 * dim1
    X = np.random.normal(mu, std, nsamples)
    X = np.reshape(X, (dim1,dim1))

    #compute the SVD of matrix
    print(X)
    print('X shape -- ', X.shape)
    U, S, Vt = np.linalg.svd(X, full_matrices = 0)
    #S2 = np.diag(S)  # < ---- not needed for this particular exercise
    #print('S shape -- ',S.shape)
    #print('\n')

    #Plot the distribution of singular values in a box-and-whisker plot.
    plt.rcParams['figure.figsize'] = [16, 8]
    fig = plt.figure()
    plt.boxplot(S)
    plt.title('Singular values distribution')
    plt.show()

    meds = []
    means = []
    #Plot the mean and median singular values as a function of r.
    for r in range(0,nsamples):
        S_r = np.copy(S[0:r])
        meds.append(np.median(S_r))
        means.append(np.mean(S_r))

    meds = np.array(meds)
    means = np.array(means)
    indx = np.array(range(nsamples)) + 1


    plt.plot(indx, means, color='k', linewidth=2, label='Mean') # True relationship
    plt.plot(indx, meds, 'x', color='r', markersize = 10, label='Median') # Noisy measurements
    plt.xlabel('Rank')
    plt.ylabel('Summary stats')
    plt.grid(linestyle='--')
    plt.legend()
    plt.show()
