# This code generates an ensemble of Monte Carlo (MC) sets of acceleration data (g_N, g) for a sample of wide binaries. It also calculates a control ensemble of acceleration data (g_N, g) for a virtual Newtonian sample with Gaia proper motions replaced by the Newtonian predicted values. It uses a python class 'accel_ortho_wb_v3' from the file 'accel_ortho_wb_v3.py'
# As in 'accel_plane_ortho_wb_v5.py', acceleration data are calculated from deprojected quantities in 3D space. While 'accel_plane_ortho_wb_v5.py' generates just one MC set, this code generates as many MC sets as desired. However, only the medians of the residuals are recorded.
# The user can define the sample by choosing the parameter ranges and input options. The default choice is the `new sample' of arXiv:2402.05720.
# The input file 'gaia_dr3_MSMS_d200pc.csv' was extracted from the El-Badry, Rix, & Heintz (2021) catalog.
# The input file 'Newton_dr3_MSMS_d200pc_5.csv' is a virtual Newtonian sample generated with 'make_Newton_sample_gaia_5.py' for f_multi = 0.43. It is the same as 'gaia_dr3_MSMS_d200pc.csv' except that proper motions were replaced by Newtonian predicted values.
# Version 3 corrected a bug as described in Appendix A of Chae, K.-H. 2024 (Astrophysical Journal, 960, 114) and including more input options.
# This is version 5 including a minor change in drwaing individual eccentricities as described Section 2.5 of arXiv:2402.05720 (Chae, K.-H. 2024, submitted to the Astrophysical Journal) 
# Questions can be directed to kyuhyunchae@gmail.com or chae@sejong.ac.kr.
# Last modified February 13, 2024.

from accel_ortho_wb_v5 import *
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

nIF=1.
def fEFE(p,x):  # Simple, AQUAL
    a0=p[0]
    p[1]=np.abs(p[1])
    esign=np.sign(p[1])
    e=np.abs(p[1])
    bet, gam = 1.1, 1.2
    eN=bet*e**2/(1+e)
    yi=10**x/a0
    y=np.sqrt(yi**2+eN**2)
    nu0=((1+np.sqrt(1+4/yi**nIF))/2)**(1./nIF)
    nu=((1+np.sqrt(1+4/y**nIF))/2)**(1./nIF)
    x_y=y*(0.5+0.5*np.sqrt(1+4/y**nIF))**(1/nIF)
    muhat=1-(x_y/nIF)/(1+x_y**nIF)
    nuhat=-muhat/(1+muhat)
    res=np.maximum(1.e-30,nu0*((nu/nu0)*(1+np.tanh(eN/yi)**gam*nuhat/3))**esign)
    return x+np.log10(res)

print('number of MC sampling:')
n = int(input('n = '))
print('multiplicity fraction:')
multi = input('multi_frac = ')
f_multi = float(multi)
######################################## input options #############################
d_min, d_max = 0, 200   # pc
s_min, s_max = 0.2, 30   # kau
MagG_min, MagG_max = 4, 14
ML_choice = 'v'   # mass-magnitude relation choice
R_chance_max = np.infty   # maximum allowed chance probability
pmerr_max = 0.005   # maximum allowed fractional error of proper motions
derr_max =  0.01   # maximum allowed fractional error of distances (parallaxes)
RVerr_max = 0.5       # maximum allowed fractional error of radial velocities
vterr_max = np.infty #maximum allowed error of v^tilde = vp/vc (vp = sky-projected vel, vc = Newtonian circular vel at s)
vperr_max = np.infty   #  maximum allowed error of vp divided by vp 
ruwe_max = np.infty # 1.2       # maximum allowed Gaia ruwe values
ext_corr = True   # dust extinction correction
pm_scatter = False   # allow PM scatter in MC sampling. When data quality is very high as in the `new sample' of arXiv:2402.05720, True/False makes little difference.  
sigd = 3   # tolerance for the difference in distances between two components in combined errors
include_orbit = True  # include the maximum orbit size for the distance difference
sigRV =  3   # tolerance for the difference in radial velocities between two components in combined errors
x0bin = np.array([-11.5,-9.8,-8.5,-7.5])  # 3 bins in x0
#x0bin = np.array([-11.5,-9.8,-8.5,-8.0,-7.5])  # 4 bins in x0
#x0bin = np.array([-12.,-10.5,-9.8,-8.5,-7.5,-6.5])  # 5 bins in x0
#x0bin = np.array([-12.,-10.6,-9.9,-9.2,-8.6,-8.0,-7.5,-7.])  # 7 bins in x0
echoice = 'i'  # eccentricity choice: 'i' - individual ranges from Hwang et al. (2022), 's' - separation-dependent power-law distribution from Hwang et al. (2022), a number 'a' (e.g. 2: 'thermal' distrubution) - p(e;a) = a*e**(a-1),  0 .LEQ. e .LEQ. 1, a > 0.
Gboost = 1.  # boost factor for Newton's constant: It allows testing modified gravity with a boosted G.
name_spec = '3bins'  # user-desired additional name tag
####################################### ################################################
filename='gaia_dr3_MSMS_d200pc.csv'
#filename='Newton_dr3_MSMS_d200pc_5.csv'  # no pmscatter

junk_A, junk_B, R_chance, rp, d_A, d_A_err, d_B, d_B_err, MagG_A, MagG_B, M_A, M_B, mux_A, mux_A_err, muy_A, muy_A_err, mux_B, mux_B_err, muy_B, muy_B_err, RV_A, RV_A_err, RV_B, RV_B_err, gal_b, ruwe_A, ruwe_B, bp_rp_A, bp_rp_B, ra_A, dec_A, ra_B, dec_B, e, e0, e1, A_G_A, A_G_B = np.loadtxt(filename,skiprows=1,delimiter=',',unpack=True,dtype=float)

mask = (R_chance < R_chance_max) & (rp > s_min) & (rp < s_max) #& (d_A > d_min) & (d_A < d_max) # initial selection

xmc, ymc = [], []
x1mc, y1mc = [], []

if ext_corr == True:
    dustoption = '_dust'
elif ext_corr == False:
    dustoption = '_nodust'
if pm_scatter == True:
    pmoption = '_pmscatter'
elif pm_scatter == False:
    pmoption = '_nopmscatter'

for i in range(n):
    data = [R_chance[mask], rp[mask], d_A[mask], d_A_err[mask], d_B[mask], d_B_err[mask], MagG_A[mask], MagG_B[mask], M_A[mask], M_B[mask], mux_A[mask], mux_A_err[mask], muy_A[mask], muy_A_err[mask], mux_B[mask], mux_B_err[mask], muy_B[mask], muy_B_err[mask], RV_A[mask], RV_A_err[mask], RV_B[mask], RV_B_err[mask], gal_b[mask], ruwe_A[mask], ruwe_B[mask], bp_rp_A[mask], bp_rp_B[mask], ra_A[mask], dec_A[mask], ra_B[mask], dec_B[mask], e[mask], e0[mask], e1[mask], A_G_A[mask], A_G_B[mask]]    
    a = accel_ortho_wb_v5(data=data,d_min=d_min,d_max=d_max,rp_min=s_min,rp_max=s_max,MagG_min=MagG_min,MagG_max=MagG_max,R_chance_max=R_chance_max,pmerr_max=pmerr_max,derr_max=derr_max,RVerr_max=RVerr_max,vterr_max=vterr_max,vperr_max=vperr_max,ruwe_max=ruwe_max,sigd=sigd,include_orbit=include_orbit,multi_frac=f_multi,gam_M=-0.7,ML_choice=ML_choice,ext_corr=ext_corr,pm_scatter=pm_scatter,sigRV=sigRV,x0bin=x0bin,nrand=1)
    x,y,yerr,npts_bin = a.run_mc(mod='g',echoice=echoice,Gboost=Gboost)
    xmc.append(x)
    ymc.append(y)
    npts_bin = list(map(int,npts_bin))
    print(i+1,npts_bin)
    print(' ',y)
    x1,y1,yerr1,npts_bin = a.run_mc(mod='e',echoice=echoice,Gboost=Gboost)
    print(' ',y1)
    x1mc.append(x1)
    y1mc.append(y1)
    np.savetxt('mc_x0_gaia_1_'+name_spec+'_v5.dat',np.array(xmc))
    np.savetxt('mc_dorth_gaia_1_'+name_spec+'_v5.dat',np.array(ymc))
    np.savetxt('mc_x0_Newton_1_'+name_spec+'_v5.dat',np.array(x1mc))
    np.savetxt('mc_dorth_Newton_1_'+name_spec+'_v5.dat',np.array(y1mc))

    fig = plt.figure(figsize=(20,30))
    plt.rc('xtick',labelsize=50)
    plt.rc('ytick',labelsize=50)
    plt.rc('axes',labelsize=60)
    plt.rc('font', family='STIXGeneral')
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.linewidth'] = 1.5
    plt.subplots_adjust(left=0.15,right=0.97,bottom=0.01,top=0.94) 
    ax = plt.subplot2grid((100,100), (1, 1), rowspan=55, colspan=98)   # panel 1
    ax.tick_params(direction='inout',length=7,width=2,bottom=True,top=True,left=True,right=True)

    for l in range(i+1):
        x, y = xmc[l], ymc[l]
        x1, y1 = x1mc[l], y1mc[l]
        ax.scatter(x,y,s=10,c='r',alpha=0.5)
        ax.scatter(x1,y1,s=10,c='b',alpha=0.5)
    xxmc, yymc = np.array(xmc), np.array(ymc)
    xx1mc, yy1mc = np.array(x1mc), np.array(y1mc)
    del_obs_newt = yymc - yy1mc
    x0mc = (xxmc+xx1mc)/2
    xmed, ymed = [], []
    x1med, y1med = [], []
    x0med, delmed, delsig = [], [], []
    for l in range(len(xmc[0])):
        xmed.append(np.median(xxmc[:,l]))
        ymed.append(np.median(yymc[:,l]))
        x1med.append(np.median(xx1mc[:,l]))
        y1med.append(np.median(yy1mc[:,l]))
        x0med.append(np.median(x0mc[:,l])) 
        delmed.append(np.median(del_obs_newt[:,l])) 
        delsig.append(np.std(del_obs_newt[:,l])) 
    ax.plot(xmed,ymed,lw=4,ls='solid',c='r')
    ax.errorbar(xmed,ymed,yerr=yerr,elinewidth=3,c='r',capsize=5,alpha=0.5)
    ax.plot(x1med,y1med,lw=4,ls='solid',c='b')
    ax.errorbar(x1med,y1med,yerr=yerr1,elinewidth=3,c='b',capsize=5,alpha=0.5)
    ax.grid(True, 'major', 'both', ls='--', lw=1, c='c', alpha=.5)
    ax.text(0.5,1.05,r'$f_{\rm{multi}}=%s$' %multi,horizontalalignment='center',transform=ax.transAxes,fontsize=80)
    #ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'orthogonal residual')
    ax.set_xlim(-12,-7)
    ax.set_ylim(-0.6,0.15)
    #ax.set_yticks(np.linspace(-0.5,0.15,14))
    #ax.set_yticks(np.linspace(-0.5,0.1,16))

    ax = plt.subplot2grid((100,100), (60, 1), rowspan=33, colspan=98)   # panel 2
    ax.tick_params(direction='inout',length=7,width=2,bottom=True,top=True,left=True,right=True)
    xpts = np.linspace(-13.5,-6.5,100)
    yaq = fEFE([1.2e-10,1.9],xpts)
    dy = yaq-xpts
    x0 = xpts + dy/2
    d_orth = dy/np.sqrt(2)
    ax.plot([-13,-6],[0,0],lw=3,ls='dashed',c='k')
    ax.plot(x0,d_orth,lw=3,ls='solid',c='m')
    for l in range(i+1):
        x, y = x0mc[l], del_obs_newt[l]
        ax.scatter(x,y,s=10,c='r',alpha=0.5)
    ax.scatter(x0med,delmed,s=100,c='k')
    ax.plot(x0med,delmed,lw=5,ls='solid',c='k')
    ax.errorbar(x0med,delmed,yerr=delsig,elinewidth=5,c='k',capsize=10,capthick=4,alpha=0.7)
    ax.grid(True, 'major', 'both', ls='--', lw=1, c='c', alpha=.5)
    ax.set_xlim(-12,-7)
    ax.set_ylim(-0.1,0.20)
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$\delta_{\rm{obs-newt}}$')

    fig.tight_layout()
    #plt.show()
    plt.savefig('fig_ortho_res.png')
    plt.close(fig)
