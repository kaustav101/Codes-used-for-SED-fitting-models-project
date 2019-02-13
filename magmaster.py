import numpy as np
from astropy.io import fits
import glob
from astropy.table import Table,Row

galfit = glob.glob('*.fit')
galsed = [g.strip('.fit') + '.sed' for g in galfit]


name = np.array([])
redshift = np.array([])
chi_square = np.array([])
fmu_opt = np.array([])
fmu_ir = np.array([])
tform_yr = np.array([])
gamma = np.array([])
Z = np.array([])
tauV = np.array([])
mu = np.array([])

sSFR_16 = np.array([])
sSFR_50 = np.array([])
sSFR_84 = np.array([])
sSFR = np.array([])

SFR = np.array([])

M_star_16 = np.array([])
M_star_50 = np.array([])
M_star_84 = np.array([])
M_star = np.array([])

L_dust = np.array([])

M_dust_16 = np.array([])
M_dust_50 = np.array([])
M_dust_84 = np.array([])
M_dust = np.array([])

T_W_BC = np.array([])
T_C_ISM = np.array([])
xi_C_tot = np.array([])
xi_C_ISM = np.array([])
xi_W_tot = np.array([])
xi_W_BC = np.array([])
xi_PAH_tot = np.array([])
xi_PAH_BC = np.array([])
xi_MIR_tot = np.array([])
xi_MIR_BC = np.array([])
tvism = np.array([])
for i in range(len(galfit)):
	fit = Table.read(galsed[i],format='ascii',data_start=2)
	phy_params_0 =Table.read(galfit[i],format='ascii',data_start=2,data_end=3)
	phy_params_1 = Table.read(galsed[i],format='ascii',data_start=0,data_end=1)
	phy_params_2 =  Table.read(galsed[i],format='ascii',data_start=1,data_end=2)
	phy_params_3 = Table.read(galfit[i],format='ascii',data_start=3,data_end=4)

        with open(galfit[i],'r') as f:
            read_data = f.readlines()
        
        sSFR_percentile = np.array([np.float64(x) for x in read_data[207].split()])

        sSFR_16 = np.append(sSFR_16,sSFR_percentile[1])
        sSFR_50 = np.append(sSFR_50,sSFR_percentile[2])
        sSFR_84 = np.append(sSFR_84,sSFR_percentile[3])


        M_star_percentile = np.array([np.float64(x) for x in read_data[270].split()])

        M_star_16 = np.append(M_star_16,M_star_percentile[1])
        M_star_50 = np.append(M_star_50,M_star_percentile[2])
        M_star_84 = np.append(M_star_84,M_star_percentile[3])

        M_dust_percentile = np.array([np.float64(x) for x in read_data[617].split()])
        
        M_dust_16 = np.append(M_dust_16,M_dust_percentile[1])
        M_dust_50 = np.append(M_dust_50,M_dust_percentile[2])
        M_dust_84 = np.append(M_dust_84,M_dust_percentile[3])
	
        name = np.append(name,galfit[i].strip('.fit'))
	redshift = np.append(redshift,Row(phy_params_0,0)[3])
	chi_square = np.append(chi_square,Row(phy_params_0,0)[2])	
	fmu_opt = np.append(fmu_opt,phy_params_1['col1'][0])
	fmu_ir = np.append(fmu_ir,phy_params_1['col2'][0])
	tform_yr = np.append(tform_yr,phy_params_1['col3'][0])
	gamma = np.append(gamma,phy_params_1['col4'][0])
	Z = np.append(Z,phy_params_1['col5'][0])
	tauV = np.append(tauV,phy_params_1['col6'][0])
	mu = np.append(mu,phy_params_1['col7'][0])
	sSFR = np.append(sSFR,phy_params_3['col5'][0])
	M_star = np.append(M_star,phy_params_3['col6'][0])
	SFR = np.append(SFR,phy_params_3['col16'][0])
	L_dust = np.append(L_dust,phy_params_1['col10'][0])
	M_dust = np.append(M_dust,phy_params_2['col7'][0])
	T_W_BC = np.append(T_W_BC,phy_params_2['col2'][0])
	T_C_ISM = np.append(T_C_ISM,phy_params_2['col3'][0])	
	xi_C_tot = np.append(xi_C_tot,phy_params_3['col10'][0])
	xi_C_ISM = np.append(xi_C_ISM,phy_params_2['col1'][0])
	xi_W_tot = np.append(xi_W_tot,phy_params_3['col13'][0])
	xi_W_BC = np.append(xi_W_BC,phy_params_2['col6'][0])
	xi_PAH_tot = np.append(xi_PAH_tot,phy_params_3['col11'][0])
	xi_PAH_BC = np.append(xi_PAH_BC,phy_params_2['col4'][0])
	xi_MIR_tot = np.append(xi_MIR_tot,phy_params_3['col12'][0])
	xi_MIR_BC = np.append(xi_MIR_BC,phy_params_2['col5'][0])
	tvism = np.append(tvism,phy_params_3['col14'][0])


col1 = fits.Column(name='Galaxy', format='20A', array=name)
col2 = fits.Column(name='redshift', format='E', array=redshift)
col3  = fits.Column(name='chi_square', format='E', array=chi_square)
col4  = fits.Column(name='fmu_opt', format='E', array=fmu_opt)
col5  = fits.Column(name='fmu_ir', format='E', array=fmu_ir)
col6  = fits.Column(name='tform_yr', format='E', array=tform_yr)
col7  = fits.Column(name='gamma', format='E', array=gamma)
col8  = fits.Column(name='Z', format='E', array=Z)
col9  = fits.Column(name='tauV', format='E', array=tauV)
col10  = fits.Column(name='mu', format='E', array=mu)
col11  = fits.Column(name='sSFR', format='E', array=sSFR)
col12  = fits.Column(name='M_star', format='E', array=M_star)
col13  = fits.Column(name='SFR', format='E', array=SFR)
col14  = fits.Column(name='L_dust', format='E', array=L_dust)
col15  = fits.Column(name='M_dust', format='E', array=M_dust)
col16  = fits.Column(name='T_W_BC', format='E', array=T_W_BC)
col17  = fits.Column(name='T_C_ISM', format='E', array=T_C_ISM)
col18  = fits.Column(name='xi_C_tot', format='E', array=xi_C_tot)
col19  = fits.Column(name='xi_C_ISM', format='E', array=xi_C_ISM)
col20  = fits.Column(name='xi_W_tot', format='E', array=xi_W_tot)
col21  = fits.Column(name='xi_W_BC', format='E', array=xi_W_BC)
col22  = fits.Column(name='xi_PAH_tot', format='E', array=xi_PAH_tot)
col23  = fits.Column(name='xi_PAH_BC', format='E', array=xi_PAH_BC)
col24  = fits.Column(name='xi_MIR_tot', format='E', array=xi_MIR_tot)
col25  = fits.Column(name='xi_MIR_BC', format='E', array=xi_MIR_BC)
col26 = fits.Column(name='tvism',format='E',array=tvism)
col27 = fits.Column(name='M_star_16',format='E',array=10**(M_star_16))
col28 = fits.Column(name='M_star_50',format='E',array=10**(M_star_50))
col29 = fits.Column(name='M_star_84',format='E',array=10**(M_star_84))
col30 = fits.Column(name='sSFR_16',format='E',array=10**(sSFR_16))
col31 = fits.Column(name='sSFR_50',format='E',array=10**(sSFR_50))
col32 = fits.Column(name='sSFR_84',format='E',array=10**(sSFR_84))
col33 = fits.Column(name='M_d_16',format='E',array=10**(M_dust_16))
col34 = fits.Column(name='M_d_50',format='E',array=10**(M_dust_50))
col35 = fits.Column(name='M_d_84',format='E',array=10**(M_dust_84))


cols = fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col33,col34,col35])

tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto('master.fits')






