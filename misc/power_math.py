import numpy as np
from scipy.stats import norm

def power_pdf(x,n,f=norm()):
    return(n*f.cdf(x)**(n-1)*f.pdf(x))
def power_cdf(x,n,f=norm()):
    return(f.cdf(x)**(n))