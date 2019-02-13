import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM
import astropy.units as units

from matplotlib import gridspec
from scipy.interpolate import interp1d

filters = Table.read('./user_files/filters.dat',format='ascii')
data = Table.read('./user_files/observations.dat',format='ascii')

# Creating a dictionary for each filter

FUV = {'lambda_eff': filters['lambda_eff'][0], 'fit?': filters['fit?'][0], 'color': 'blue', 'flux':np.nan,'e_flux':np.nan,'lum': np.nan,'e_lum': np.nan,'p_flux':np.nan,'residue':np.nan,'e_residue':np.nan}
NUV = {'lambda_eff': filters['lambda_eff'][1], 'fit?': filters['fit?'][1], 'color': 'blue', 'flux':np.nan,'e_flux':np.nan,'lum': np.nan,'e_lum': np.nan,'p_flux':np.nan,'residue':np.nan,'e_residue':np.nan}
		
u = {'lambda_eff': filters['lambda_eff'][2], 'fit?': filters['fit?'][2], 'color': 'green', 'flux':np.nan,'e_flux':np.nan, 'lum': np.nan,'e_lum': np.nan,'p_flux':np.nan,'residue':np.nan,'e_residue':np.nan}
g = {'lambda_eff': filters['lambda_eff'][3], 'fit?': filters['fit?'][3], 'color': 'green', 'flux':np.nan,'e_flux':np.nan, 'lum': np.nan,'e_lum': np.nan,'p_flux':np.nan,'residue':np.nan,'e_residue':np.nan}
r = {'lambda_eff': filters['lambda_eff'][4], 'fit?': filters['fit?'][4], 'color': 'green', 'flux':np.nan,'e_flux':np.nan, 'lum': np.nan,'e_lum': np.nan,'p_flux':np.nan,'residue':np.nan,'e_residue':np.nan}
i = {'lambda_eff': filters['lambda_eff'][5], 'fit?': filters['fit?'][5], 'color': 'green', 'flux':np.nan,'e_flux':np.nan, 'lum': np.nan,'e_lum': np.nan,'p_flux':np.nan,'residue':np.nan,'e_residue':np.nan}
z = {'lambda_eff': filters['lambda_eff'][6], 'fit?': filters['fit?'][6], 'color': 'green', 'flux':np.nan,'e_flux':np.nan, 'lum': np.nan,'e_lum': np.nan,'p_flux':np.nan,'residue':np.nan,'e_residue':np.nan}

J = {'lambda_eff': filters['lambda_eff'][7], 'fit?': filters['fit?'][7], 'color': 'orange', 'flux':np.nan,'e_flux':np.nan,'lum': np.nan,'e_lum': np.nan,'p_flux':np.nan,'residue':np.nan,'e_residue':np.nan}
H = {'lambda_eff': filters['lambda_eff'][8], 'fit?': filters['fit?'][8], 'color': 'orange', 'flux':np.nan,'e_flux':np.nan, 'lum': np.nan,'e_lum': np.nan,'p_flux':np.nan,'residue':np.nan,'e_residue':np.nan}
Ks = {'lambda_eff': filters['lambda_eff'][9], 'fit?': filters['fit?'][9], 'color': 'orange', 'flux':np.nan,'e_flux':np.nan, 'lum': np.nan,'e_lum': np.nan,'p_flux':np.nan,'residue':np.nan,'e_residue':np.nan}

W1 = {'lambda_eff': filters['lambda_eff'][10], 'fit?': filters['fit?'][10], 'color': 'red', 'flux':np.nan,'e_flux':np.nan, 'lum': np.nan,'e_lum': np.nan,'p_flux':np.nan,'residue':np.nan,'e_residue':np.nan}
W2 = {'lambda_eff': filters['lambda_eff'][11], 'fit?': filters['fit?'][11], 'color': 'red', 'flux':np.nan,'e_flux':np.nan, 'lum': np.nan,'e_lum': np.nan,'p_flux':np.nan,'residue':np.nan,'e_residue':np.nan}
W3 = {'lambda_eff': filters['lambda_eff'][12], 'fit?': filters['fit?'][12], 'color': 'red', 'flux':np.nan,'e_flux':np.nan,'lum': np.nan,'e_lum': np.nan,'p_flux':np.nan,'residue':np.nan,'e_residue':np.nan}
W4 = {'lambda_eff': filters['lambda_eff'][13], 'fit?': filters['fit?'][13], 'color': 'red', 'flux': np.nan,'e_flux':np.nan, 'lum': np.nan,'e_lum': np.nan,'p_flux':np.nan,'residue':np.nan,'e_residue':np.nan}


#f_60 = {'lambda_eff': filters['lambda_eff'][14], 'fit?': filters['fit?'][14], 'color': 'brown', 'flux': np.nan,'e_flux':np.nan, 'lum': np.nan,'e_lum': np.nan,'p_flux':np.nan,'residue':np.nan,'e_residue':np.nan}
#f_100 = {'lambda_eff': filters['lambda_eff'][15], 'fit?': filters['fit?'][15], 'color': 'brown', 'flux': np.nan,'e_flux':np.nan, 'lum': np.nan,'e_lum': np.nan,'p_flux':np.nan,'residue':np.nan,'e_residue':np.nan}


#F250 = {'lambda_eff': filters['lambda_eff'][14], 'fit?': filters['fit?'][14], 'color': 'brown', 'flux': np.nan,'e_flux':np.nan, 'lum': np.nan,'e_lum': np.nan,'p_flux':np.nan,'residue':np.nan,'e_residue':np.nan}
#F350 = {'lambda_eff': filters['lambda_eff'][15], 'fit?': filters['fit?'][15], 'color': 'brown', 'flux': np.nan,'e_flux':np.nan, 'lum': np.nan,'e_lum': np.nan,'p_flux':np.nan,'residue':np.nan,'e_residue':np.nan}
#F500 = {'lambda_eff': filters['lambda_eff'][16], 'fit?': filters['fit?'][16], 'color': 'brown', 'flux': np.nan,'e_flux':np.nan, 'lum': np.nan,'e_lum': np.nan,'p_flux':np.nan,'residue':np.nan,'e_residue':np.nan}
filters_list = np.array([FUV,NUV,u,g,r,i,z,J,H,Ks,W1,W2,W3,W4]) #add extra filters here as needed. 


def plot(galaxies):
	for i_gal in galaxies:
		
		gal_index = np.where(data['i_gal']==i_gal)[0][0]
		
		k = 2
		#Updating fluxes and errors in fluxes
		for m in filters_list:
			m['flux'] = data[gal_index][k]
			m['e_flux'] = data[gal_index][k + 1]
			k += 2	

################ Flux Reading #################################
		obs_flux_all = np.array([filt['flux'] for filt in filters_list])*units.Jy #all fluxes including unused
		lambda_eff = np.array([filt['lambda_eff'] for filt in filters_list])*units.micron
		
		obs_flux_all[obs_flux_all==0] = np.nan #replacing zeros with nan to avoid it from plotting
		e_flux_all =  np.array([filt['e_flux'] for filt in filters_list])*units.Jy  #errors in the fluxes
		e_flux_all[e_flux_all == 0] = np.nan
	
		fit_read = Table.read('%s.fit'%i_gal,format='ascii',data_start=2,data_end=3) #chi square and redshift

		chi_square = fit_read[0][2]
		z = fit_read[0][3]		
		
		fit_read_1 = Table.read('%s.fit'%i_gal,format='ascii',data_start=4,data_end=5)
		
		p_flux = np.array([fit_read_1[0][m] for m in range(len(fit_read_1[0]))]) #flux predicted by the model
		
		
		sed = Table.read('%s.sed'%i_gal,format='ascii',data_start=2)

		fit_read_2 = Table.read('%s.fit'%i_gal,format='ascii',data_start=0,data_end=1)
		obs_fit_flux = np.array([fit_read_2[0][m] for m in range(len(fit_read_2[0]))])
				
		fit_read_3 = Table.read('%s.fit'%i_gal,format='ascii',data_start=1,data_end=2)
		e_obs_fit_flux = np.array([fit_read_3[0][m] for m in range(len(fit_read_3[0]))])

################ Calculations ################################
		x = 10**(sed['col1'])*units.AA  #wavelength in A
		lum_at = 10**(sed['col2'])*(units.AA)**(-1) #unattenuated lum(L_lambda/L_sunA^(-1))
		lum_un = 10**(sed['col3'])*(units.AA)**(-1) #attenuated lum in same units

		y_at = x*lum_at #total attenuated SED
	
		y_un = x*lum_un #total unattenuated SED

		xx = x.to(units.micron)/(1 + z) #wavelenght in microns in the rest frame


		# Converting the fluxes in different photometric bands into luminosities lambda*L

		#Observed Fluxes
		cosmo = FlatLambdaCDM(H0=70, Om0=0.3) #using FlatLambdaCDM cosmology
		
		D_L = cosmo.luminosity_distance(z) #luminoisty distance
		D_L = D_L.to(units.micron)  #converting Mpc to micron
		
		#adding the luminosity distance and dividing by L_sun, also converting from Jy to W/m^2/Hz hence the 10^(-26) term 
		
		c = (4*np.pi*D_L**2/const.L_sun) #finding the constant factor	

		L_lum = ((1+z)*obs_flux_all.decompose()*3*10**14/lambda_eff)*c*units.micron/units.second 
		L_lum = L_lum.decompose()

		e_lum = ((1+z)*e_flux_all*3*10**14/lambda_eff)*c*units.micron/units.second
		e_lum = e_lum.decompose()
	
		#p_lum = ((1+z)*p_flux*3*10**14/lambda_eff)*c*units.micron/units.second 		
		#p_lum = p_lum.decompose()
		
		l_1 = 0  #Updating luminosities to the filter dict
		for l in filters_list:
			l['lum'] = L_lum[l_1]
			l['e_lum'] = e_lum[l_1]
			l_1 += 1 	
		#residues calculations
		#residues = (obs_fit_flux - p_flux)/obs_fit_flux #finding the residues
		
		#e_residues = p_flux*e_obs_fit_flux/obs_fit_flux**2 #errors in the residues		
		
		int_lum = interp1d(xx,y_at)
		lum_new = int_lum(lambda_eff.value)
	
		residues = (L_lum - lum_new)/L_lum
		e_residues = lum_new*e_lum/L_lum**2
		a_1 = 0  
		for a in filters_list:  #updating the residues in the dict
			a['residue'] = residues[a_1]
			a['e_residue'] = e_residues[a_1]
			a_1 += 1	

##################### Physical Parameters ###############################

		
		phy_params_1 = Table.read('%s.sed'%i_gal,format='ascii',data_start=0,data_end=1)
		#add the physical paramters and then plot the residues using a proper subplot
		fmu_opt = phy_params_1['col1'][0]
		fmu_ir = phy_params_1['col2'][0]
		tform_yr = phy_params_1['col3'][0]
		gamma = phy_params_1['col4'][0]
		metallicity = phy_params_1['col5'][0]
		tauV = phy_params_1['col6'][0]
		mu = phy_params_1['col7'][0]
		M_star =  phy_params_1['col8'][0]/10**10
		SFR =  phy_params_1['col9'][0]*10**14
		Ld =  phy_params_1['col10'][0]/10**9

		phy_params_2 =  Table.read('%s.sed'%i_gal,format='ascii',data_start=1,data_end=2)
		xi_C_ism = phy_params_2['col1'][0]
		M_dust = phy_params_2['col7'][0]/10**7
		T_W = phy_params_2['col2'][0]
		T_C = phy_params_2['col3'][0]



##################### Plotting #########################################

		fig = plt.figure()
		gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
		ax1 = fig.add_subplot(gs[0])
		ax2 = fig.add_subplot(gs[1], sharex=ax1)
		fig.suptitle('%s'%i_gal, size=15)
		plt.setp(ax1.get_xticklabels(), visible=False)
		gs.update(hspace=0.1)
		
		#SED plot
		ax1.set_ylim([10**4,10**12])
		ax1.set_xlim([10**(-4),10**4])
                ax1.loglog(xx,y_un,color='blue',alpha=0.5)
                ax1.loglog(xx,y_at,color='green',alpha=0.6)
		#ax1.errorbar(lambda_eff.value,L_lum,yerr=e_lum.value,fmt='o',capsize=1.5)
		for d in filters_list:
			if d['fit?'] == 1:
				ax1.errorbar(d['lambda_eff'],d['lum'],yerr=d['e_lum'],color=d['color'],mec=d['color'],marker='.',capsize=1.5)
			else:	
				ax1.errorbar(d['lambda_eff'],d['lum'],yerr=d['e_lum'],color=d['color'],mec=d['color'],marker='*',capsize=1.5)
					

		textstr = '$\chi^2 = %s$\n$M_{star}/M_{\odot} = %.2f\\times 10^{10}$\n$M_{dust}/M_{\odot} = %.2f\\times 10^{7}$\n$L_{dus    t}/L_{\odot} = %.2f\\times 10^{9}$\n$sSFR = %.2f\\times 10^{-14}$\n$\gamma = %.2f$\n$Z/Z_\odot = %.2f$\n$\\tau_V = %.2f$\n$T_W^{BC} = %.2f K$\n$T_C^{ISM} = %.2f K$'%(chi_square,M_star, M_dust, Ld, SFR, gamma, metallicity, tauV,T_W,T_C)

		props = dict(facecolor='white', alpha=0.0)
		ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props)
		ax1.set_ylabel(r'$\lambda L_\lambda/L_\odot$',size=20)





		
		#residue plot
		ax2.axhline(0,ls='dashed')
		ax2.set_xscale('log',nonposx='clip')
		for s in filters_list:
			if s['fit?'] == 1:
				ax2.errorbar(s['lambda_eff'],s['residue'],yerr=s['e_residue'],color=s['color'],mec=s['color'],fmt='.',capsize=1.5)
			else:
				continue  #ax2.errorbar(s['lambda_eff'],s['residue'],yerr=s['e_residue'],color=s['color'],mec=s['color'],fmt='*',capsize=1.5)
		ax2.set_ylim([-1,1])
		ax2.set_xlabel(r'$\lambda/\mu m$',size=20)
		plt.savefig('%s.png'%i_gal,dpi=200)
		plt.close()
		print i_gal + ' plot saved...'
		#plt.show()	
