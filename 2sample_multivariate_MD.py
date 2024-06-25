"""
A python script in implementing a 2-sample, multivariate Mahalanobis Distance signifiance test. 

Written by: Haiyang S. Wang
Date: 25-04-2024

References: 
Mahalanobis, P. C. 1930, J. Proceedings, Asiat. Soc. Bengal, 26, 541

Mardia, K. V., Kent, J. T., & Bibby, J. M. 1979, in Probability and Mathematical
Statistics, ed. Z. W. Birnbaum & E. Lukacs (London: Academic Press), 124

https://online.stat.psu.edu/stat505/lesson/4/4.3
""

import numpy as np
import scipy.stats as stats


def mahalanobis_2sample_multivariate(arr1, arr2, arr1_err=None, arr2_err=None, p=2, alpha=0.05, pooled=True, silence=False):
    """
    A 2-sample, multivariate Mahalanobis Distance signficance test, with a choice of considering or not data errors
    arr1, arr2 -- the input 2 samples of multivariate data (here by default, two dimensional each, i.e. p=2; and input as Pandas DataFrame)
    arr1_err, arr2_err -- the optional errors of the input 2 samples
    alpha -- the statistical level (0.05, by default) of the critical statistic and the p-value of the test statistic
    pooled -- pool the covariance of the two samples or not (recommended to be 'pooled', by default)
    silence -- auto printing test results or not ('False', by default)
    """
    
    ##retireve the sizes of the input samples
    N_arr1 = len(arr1.x)  ##arr1 and arr2 are not necessarily equal sized.
    N_arr2 = len(arr2.x)

    ##calculate the weights and joint weights for calculating the weighted averages and variance-covariance matricies.
    if arr1_err is not None and arr2_err is not None:
        arr1_weights = 1/np.array(arr1_err)**2 #, 1/np.array(arr1_err.y)**2
        arr1_aweights = np.sqrt(arr1_weights[:,0]**2 + arr1_weights[:,1]**2)

        arr2_weights = 1/np.array(arr2_err)**2 #, 1/np.array(arr1_err.y)**2
        arr2_aweights = np.sqrt(arr2_weights[:,0]**2 + arr2_weights[:,1]**2)
    else:
        arr1_weights, arr2_weights = None, None
        arr1_aweights, arr2_aweights = None, None

    ##calculate the weighted averages and their differences
    arr1_xy0 = np.average(arr1, weights=arr1_weights, axis=0)
    arr2_xy0 = np.average(arr2, weights=arr2_weights, axis=0)
    diff = arr1_xy0 - arr2_xy0

    ##calculate the variance-covariance matrices of the indiviual samples and their common variance-covariances ('pooled' is recommended)
    S1 = np.cov(arr1.x, arr1.y, aweights=arr1_aweights) 
    S2 = np.cov(arr2.x, arr2.y, aweights=arr2_aweights)
    S = ((N_arr1-1)*S1 + (N_arr2-1)*S2) / (N_arr1 + N_arr2 -2) if pooled==True else S1/N_arr1 + S2/N_arr2
    inv_S = np.linalg.inv(S)
    
    ##calculate the test statistic, i.e., the squared Mahalanobis Distance
    tmp = np.dot(diff.T, inv_S)
    mahal_stat = np.dot(tmp, diff)  #mahalanobis distance, which is a squared distance and also the resultant statistic
    
    ##calculate the p-value of the test statistic, which follows a chi-square distribution
    dof = p   ##(the degree of freedom is set to be p, i.e., the number of variables or data dimensions) https://online.stat.psu.edu/stat505/lesson/4/4.3
    mahal_pvalue = 1 - stats.chi2.cdf(mahal_stat, dof) ##return the p-value by following the chi-square distribution for the statistic and p degrees of freedom, where p is the number of variables

    ##calculate the critical value at the designated 'alpha' level.     
    chi2_crit = stats.chi2.ppf(1-alpha, dof) 
    
    ##print the test results: the test statistic, the critical value, and p-value
    match silence:
        case False: 
             print(f"\n---mahalanobis_2sample result----")
             print(f"mahal_stat: {mahal_stat}\n chi2_stat: {chi2_crit}\n p-value: {mahal_pvalue}")
        case True: pass
        case other: pass

    ##return the results
    return mahal_stat, chi2_crit, mahal_pvalue
