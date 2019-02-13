import numpy as np 
from astropy.io import fits
import itertools as itr
import collections
#returns a tuple with the names of the columns of the two fits files to compare

def colnames(c,d):
	hdulist = fits.open('%s'%c)
	hdulist_1 = fits.open('%s'%d)

	return hdulist[1].columns, hdulist_1[1].columns
'''
Returns a tuple of dictionary of the two parameters from the 
two files from the intersection set and chi_square tolerance given by the ctol argument. 
'''
def compare(a,b,file1,file2,ctol=5.0):
	hdulist = fits.open('%s'%file1)
	tbdata = hdulist[1].data

	hdulist_1 = fits.open('%s'%file2)
	tbdata_1 = hdulist_1[1].data
	
	gal_list = tbdata.field('Galaxy')
	gal_list_1 = tbdata_1.field('Galaxy')

	chi_square = tbdata.field('chi_square')
	chi_square_1 = tbdata_1.field('chi_square')
	
	gal_dic = dict(itr.izip(gal_list,chi_square)) #galaxy dict with chi_squares
	gal_dic_1 = dict(itr.izip(gal_list_1,chi_square_1))	
		
	gal_param_dic = dict(itr.izip(gal_list,tbdata.field(a))) #galay dicts with params
	gal_param_dic_1 = dict(itr.izip(gal_list_1,tbdata_1.field(b)))
 
	intersection = set(gal_list).intersection(gal_list_1)

	galaxy = collections.deque()		
	param , param_1 = collections.deque(), collections.deque() #defining empty deques
	for gal in intersection:
		if all( x < ctol for x in [gal_dic[gal],gal_dic_1[gal]]):
			galaxy.append(gal)
			param.append(gal_param_dic[gal])
			param_1.append(gal_param_dic_1[gal])
	
	return dict(itr.izip(galaxy,param)), dict(itr.izip(galaxy,param_1))
'''
Returns a dictionary of the param with the galaxy name with a chi_square filter 
give by ctol
'''

def param(a,fits_file,ctol=5.0):
    hdulist = fits.open('%s'%fits_file)
    tbdata = hdulist[1].data
    
    gal_list = tbdata.field('Galaxy')
    chi_square = tbdata.field('chi_square')

    gal_dic = dict(itr.izip(gal_list,chi_square)) #galaxy dict with chi_squares

    gal_param_dic = dict(itr.izip(gal_list,tbdata.field(a))) #galay dicts with params

    galaxy = collections.deque()

    param = collections.deque()

    for gal in gal_list:
        if gal_dic[gal] < ctol:
            galaxy.append(gal)
            param.append(gal_param_dic[gal])

    return dict(itr.izip(galaxy,param))




