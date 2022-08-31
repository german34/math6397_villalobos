'''
Exercise 1.2
'''

#SET UP ENVIRONMENT
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os


##################################################################
fname = 'dog.jpg'
n = 1300
##################################################################
#Loading data

#img_data = imread(os.path.join('..','data','dog.jpg')) #if moved to 'hw' directory
img_data = imread(fname)
X = np.mean(img_data, -1); # Convert RGB to grayscale


#compute economy SVD
U, S, Vt = np.linalg.svd(X, full_matrices = False)
S = np.diag(S)

err_abs = []
err_rel = []

#compute the reconstruction error of the truncated SVD in the Frobenious norm
#as of function of the rank
print('Computing reconstruction errors.....')
for r in range(1,n):
    # Construct approximate image
    Xapprox = U[:,:r] @ S[0:r,:r] @ Vt[:r,:]

    diff = abs(X - Xapprox)
    err_a = np.linalg.norm(diff,'fro')
    err_r = err_a / np.linalg.norm(X,'fro')
    err_abs.append(err_a)
    err_rel.append(err_r)

err_abs = np.array(err_abs)
err_rel = np.array(err_rel)

print('Squaring this error.....')#
# Can be done in one line
# square this error to compute the fraction of missing variance as a function of r)
sq_abs = []
sq_rel = []
for j in range(0,len(err_abs)):
    err1 = err_abs[j]
    err2 = err_rel[j]
    err1  = err1**2
    err2 = err2**2
    sq_abs.append(err1)
    sq_rel.append(err2)

sq_abs = np.array(sq_abs)
sq_rel = np.array(sq_rel)

#May also plot 1 minus the error or missing variance to visualize the amount of norms
#or variance captured at a given rank r
print('Plotting (1 minus the relative error)')
plt.figure(1)
minus = 1.0 - err_rel
plt.semilogy(minus)
plt.title('1 minus error')
plt.show()


#plot these quantities along with the cumulative sum of singular values
#as a function of r
plt.figure(2)
plt.semilogy(err_rel)
plt.title('Frobenious Norm Error')
plt.show()

plt.figure(3)
plt.semilogy(sq_rel)
plt.title('Fraction of Missing Variance')
plt.show()


print('now displaying cumulative sum of singular values')
plt.figure(4)
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.title('Singular Values: Cumulative Sum')
plt.show()


cumulative_sum = np.cumsum(np.diag(S))/np.sum(np.diag(S))
var_sum = np.cumsum(sq_rel)/np.sum(sq_rel)
fro_sum = np.cumsum(err_rel)/np.sum(err_rel)

print('Now working with totals')

#find the rank r where the reconstruction captures 99% of the total variance
for j in range(len(var_sum)):
    if var_sum[j] > 0.99:
        print(j)
        break

#compare this with the rank r where the reconstruction captures 99% in the Frobenius norm
for j in range(len(fro_sum)):
    if fro_sum[j] > 0.99:
        print(j)
        break

for j in range(len(cumulative_sum)):
    if cumulative_sum[j] > 0.99:
        print(j)
        break
