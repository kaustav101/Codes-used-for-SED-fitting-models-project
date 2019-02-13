import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import os
from subprocess import Popen,PIPE,call
from astropy.table import Table, Row
from multiprocessing import Pool
import time


start = time.clock()

data = Table.read('./user_files/observations.dat',format='ascii')

cores = int(raw_input('No. of cores:'))
#z_grid = Popen('./make_zgrid',stdin=PIPE)
#z_grid.communicate(os.linsep.join(['N','']))

	
def sed(i):
	obs = Row(data,i)

	i_gal = obs[0] #Galaxy identity
	print i_gal

	z = obs[1] #redshift
	print z

	optical = Popen('./get_optic_colors', stdin=PIPE)
	optical.communicate(os.linesep.join([str(z), '70.,0.3,0.7']))
	
	infrared = Popen('./get_infrared_colors',stdin=PIPE)
	infrared.communicate(os.linesep.join([str(z),'70.,0.3,0.7']))

	fitting = Popen('./fit_sed',stdin=PIPE)
	fitting.communicate(os.linesep.join([str(i + 1)]))
	
	
if __name__ == '__main__':	
	pool = Pool(processes=cores)
	print len(data)
	galaxies = np.arange(len(data))
	print galaxies
	pool.map(sed,galaxies) 


end = time.clock()

print(end - start)
