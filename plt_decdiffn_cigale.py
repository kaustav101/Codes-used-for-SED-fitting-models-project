
#################################################

from astropy.table import Table,Column
import matplotlib
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.lines as mlines
import matplotlib.cm as cm
import scipy
from scipy.stats import binned_statistic_2d as bs2d
import numpy.ma as ma
from scipy.optimize import curve_fit


#---------------------------------------------------------------------------------------------

t1=Table.read("eagle_parameters_z=0.1,0.18.dat",format="ascii")
t2=Table.read("results_cigale_EDGEON0.1,0.18.dat",format="ascii")

#--------------------------------------------------------------------------------------------------

plt.figure()

cut=np.where(t2['best.reduced_chi_square']<5)


#----------------------------------------------------------------------------------------------------

#x0=t1['SFR']
#x0=t1['MASS']
x0=t1['MDUST']
x1=x0[cut]
y0=t2['best.dust.mass']/2e30
#y0=t2['bayes.stellar.m_star']
#y0=t2['bayes.sfh.sfr']
y1=y0[cut]
#x1=t1['MASS_ap']
#-----------------------------------------------------------------------------------------------------

x11=np.log10(x1)
y11=np.log10(y1)-np.log10(x1)

#plt.hist2d(x11,y11,bins=100,cmap='hot',cmin=0.00000000001)
#-----------------------------------------------------------------------------------------------------
H, xedges, yedges = np.histogram2d(x11, y11, bins=100) 
plt.pcolor(xedges, yedges, H.transpose(), cmap='ocean_r')
#cmap = matplotlib.cm.jet
#cmap.set_bad('white',1.)

#------------------------------------------------------------------------------------------------------
"""
xmax=-9
xmin=-12
ymax=0.5
ymin=-1.2
plt.ylabel("log($sSFR_{CIGALE}$)-log($sSFR_{EAGLE}$)")
plt.xlabel("log($sSFR_{EAGLE}$)")
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])

xmax=1.5
xmin=-1.5
ymax=0.5
ymin=-1.2
plt.ylabel("log($SFR_{CIGALE-faceon}$)-log($SFR_{EAGLE}$)")
plt.xlabel("log($SFR_{EAGLE}$)")
"""
xmax=8.5
xmin=5.5
ymax=1.25
ymin=-0.4
plt.ylabel("log($MDUST_{CIGALE-edgeon}$)-log($MDUST_{EAGLE}$)")
plt.xlabel("log($MDUST_{EAGLE}$)")
"""

xmax=12
xmin=8
ymax=0.5
ymin=-0.7
plt.ylabel("log($M*_{CIGALE-faceon}$)-log($M*_{EAGLE}$)")
plt.xlabel("log($M*_{EAGLE}$)")
"""
plt.ylim(ymax=ymax)
plt.ylim(ymin=ymin)
plt.xlim(xmin=xmin)
plt.xlim(xmax=xmax)

#pol1=np.polyfit(x11,y11,2)
#f1 = np.poly1d(pol1)

#plt.text(xmin+0.23*(xmax-xmin), ymin+(0.07*(ymax-ymin)), "f(xmin)=%f \n f(max) = %f"%(f1(xmin),f1(xmax)),bbox=dict(facecolor='white', alpha=0.4), horizontalalignment='center',verticalalignment='center')
######################################################
chi_sqr=t2['best.reduced_chi_square']#get the chi_sqr
chi_sqr=chi_sqr[cut]
def f(x,a,b,c):
  return a*x*x+b*x+c

popt, pcov = curve_fit(f,x11,y11,sigma=chi_sqr,absolute_sigma=True)
par=popt
plt.text(xmin+0.23*(xmax-xmin), ymin+(0.07*(ymax-ymin)), "f(xmin)=%f \n f(max) = %f"%(f(xmin,*popt),f(xmax,*popt)),bbox=dict(facecolor='white', alpha=0.4), horizontalalignment='center',verticalalignment='center')
###################################################

m = np.arange(xmin, xmax, 0.001)
y_fit=f(m,
 *popt)
ref_l=0*m

plt.plot(m,ref_l,linewidth=1,color='b')
plt.plot(m,y_fit,linewidth=1,color='g')
plt.colorbar()


#----------------------------------------------------------------------------------------------------#----------------------------------------------------------------------------------------------------

#plt.savefig("sfr_DECDIFFN_CIGALE_2018.pdf")
plt.show()

