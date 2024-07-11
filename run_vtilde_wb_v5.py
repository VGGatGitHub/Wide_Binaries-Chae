# This code generates an ensemble of Monte Carlo (MC) sets of normalized velocity profile data (vtilde[:=v_p/v_c], s/r_M) for a sample of wide binaries. It also calculates a control ensemble for a virtual Newtonian sample with Gaia proper motions replaced by the Newtonian predicted values. It uses a python class 'vtilde_wb_v5' from the file 'vtilde_wb_v5.py'.
# Here v_p is the sky-projected relative velocity between the pair, v_c is the theoretical Newtonian circular velocity for the sky-projected separaion s, and r_M is the MOND radius for the binary.
# While 'plot_velprofile_oneMC.py' generates just one MC set for the input sample of binaries, this code generates as many MC sets as desired for the input sample and the control (virtual) sample. However, only the medians of (vtilde, s/r_M) are recorded in the user-defined bins of log(s/r_M).
# The user can define the input sample by choosing the parameter ranges and input options. The default choice is the `new sample' of the main reference below.
# The input file 'gaia_dr3_MSMS_d200pc.csv' was extracted from the El-Badry, Rix, & Heintz (2021) catalog.
# The input file 'Newton_dr3_MSMS_d200pc_5.csv' is a virtual Newtonian sample generated with 'make_Newton_sample_gaia_v5.py' for f_multi = 0.43. It is the same as 'gaia_dr3_MSMS_d200pc.csv' except that proper motions were replaced by Newtonian predicted values. The user can also generate his/her own virtual Newtonian sample with 'make_Newton_sample_gaia_v5.py'. 
# Main reference: arXiv:2402.05720 (Chae, K.-H. 2024, submitted to the Astrophysical Journal).
# Other references: (1) Astrophysical Journal, 952, 128 (Chae, K.-H. 2023)
#                   (2) Astrophysical Journal, 960, 114 (Chae, K.-H. 2024)
# Questions can be directed to kyuhyunchae@gmail.com or chae@sejong.ac.kr.
# Last modified February 13, 2024.

from vtilde_wb_v5 import *
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
R_chance_max = np.infty  # 0.01 # maximum allowed chance probability
pmerr_max = 0.005  # maximum allowed fractional error of proper motions
derr_max =  0.01  #np.infty  #0.01  #   # maximum allowed fractional error of distances (parallaxes)
RVerr_max =  0.5  #np.infty  #0.5  # maximum allowed fractional error of radial velocities
vterr_max =  np.infty  #0.1   #  maximum allowed error of v^tilde = vp/vc (vp = sky-projected vel, vc = Newtonian circular vel at s)
vperr_max =  np.infty   #  maximum allowed error of vp divided by vp 
ruwe_max =  np.infty      # maximum allowed Gaia ruwe values
ext_corr = True   # dust extinction correction
pm_scatter = False  # allow PM scatter in MC sampling. When data quality is very high as in the `new sample' of the main reference, True/False makes little difference. 
sigd = 3   # tolerance for the difference in distances between two components in combined errors
include_orbit = True  # include the maximum orbit size for the distance difference
sigRV = 3    # np.infty  #3    # tolerance for the difference in radial velocities between two components in combined errors
nbin = 6
x0bin = np.logspace(-1.4,0.4,nbin+1)  #np.logspace(-1.5,0.3,7)  
echoice = 'i'  # eccentricity choice: 'i' - individual ranges from Hwang et al. (2022), 's' - separation-dependent power-law distribution from Hwang et al. (2022), a number 'a' (e.g. 2: 'thermal' distrubution) - p(e;a) = a*e**(a-1),  0 .LEQ. e .LEQ. 1, a > 0.
Gboost = 1.  # boost factor for Newton's constant: It allows testing modified gravity with a boosted G.
name_spec = '6bins'  # user-desired additional name tag
####################################### ################################################
filename='gaia_dr3_MSMS_d200pc.csv'
#filename='Newton_dr3_MSMS_d200pc_5.csv'  # no pmscatter

junk_A, junk_B, R_chance, rp, d_A, d_A_err, d_B, d_B_err, MagG_A, MagG_B, M_A, M_B, mux_A, mux_A_err, muy_A, muy_A_err, mux_B, mux_B_err, muy_B, muy_B_err, RV_A, RV_A_err, RV_B, RV_B_err, gal_b, ruwe_A, ruwe_B, bp_rp_A, bp_rp_B, ra_A, dec_A, ra_B, dec_B, e, e0, e1, A_G_A, A_G_B = np.loadtxt(filename,skiprows=1,delimiter=',',unpack=True,dtype=float)

G = 6.6743e-11     # Newton's constant
Msun = 1.989e30    # Msolar in kg
au = 1.4959787e11  # AU in m

vc = 1.e-3*np.sqrt(G*Msun*(M_A+M_B)/(rp*au*1000)) # km/s
del_mu = np.sqrt((mux_A-mux_B)**2+(muy_A-muy_B)**2)
d_M = (d_A/d_A_err**2+d_B/d_B_err**2)/(1/d_A_err**2+1/d_B_err**2)
vp = 4.7404e-3*del_mu*d_M
mu = np.maximum(np.sqrt(mux_A**2+muy_A**2),np.sqrt(mux_B**2+muy_B**2))
TV = 4.7404e-3*mu*d_A
RV = np.maximum(np.abs(RV_A),np.abs(RV_B))
v = np.sqrt(TV**2+RV**2)

mask = (R_chance < R_chance_max) & (rp > s_min) & (rp < s_max) #& (v*(rp*1.e3/206265)/d_M < 0.08*vc) # initial selection

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
    a = vtilde_wb_v5(data=data,d_min=d_min,d_max=d_max,rp_min=s_min,rp_max=s_max,MagG_min=MagG_min,MagG_max=MagG_max,R_chance_max=R_chance_max,pmerr_max=pmerr_max,derr_max=derr_max,RVerr_max=RVerr_max,vterr_max=vterr_max,vperr_max=vperr_max,ruwe_max=ruwe_max,sigd=sigd,include_orbit=include_orbit,multi_frac=f_multi,gam_M=-0.7,ML_choice=ML_choice,ext_corr=ext_corr,pm_scatter=pm_scatter,sigRV=sigRV,x0bin=x0bin,nrand=1)
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
    np.savetxt('mc_rnorm_gaia_1_'+name_spec+'_v5.dat',np.array(xmc))
    np.savetxt('mc_vtilde_gaia_1_'+name_spec+'_v5.dat',np.array(ymc))
    np.savetxt('mc_rnorm_Newton_1_'+name_spec+'_v5.dat',np.array(x1mc))
    np.savetxt('mc_vtilde_Newton_1_'+name_spec+'_v5.dat',np.array(y1mc))

    fig = plt.figure(figsize=(40,50))
    plt.rc('xtick',labelsize=70)
    plt.rc('ytick',labelsize=70)
    plt.rc('axes',labelsize=90)
    plt.rc('font', family='STIXGeneral')
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.linewidth'] = 1.5
    plt.subplots_adjust(left=0.12,right=0.97,bottom=0.07,top=0.94) 
    ax = plt.subplot2grid((200,100), (1, 1), rowspan=95, colspan=98)   # panel 1
    ax.tick_params(direction='inout',length=7,width=2,bottom=True,top=True,left=True,right=True)
    xpts = np.linspace(-13.5,-6.5,100)
    rnorm = (1/1.2)*10**(0.5*(np.log10(1.2e-10)-xpts))
    yaq = fEFE([1.2e-10,1.9],xpts)
    yaq1 = (fEFE([1.2e-10,1.9-0.19],xpts)-yaq)/2
    yaq2 = (fEFE([1.2e-10,1.9+0.19],xpts)-yaq)/2
    dy = yaq-xpts
    logy = 0.5*dy

    for l in range(i+1):
        x, y = xmc[l], ymc[l]
        x1, y1 = x1mc[l], y1mc[l]
        if l == 0:
            ax.scatter(x,y,s=30,c='r',alpha=0.5,label=r'input (observed) data ')
            ax.scatter(x1,y1,s=30,c='b',alpha=0.5,label=r'control (Newtonian) data')
        else:
            ax.scatter(x,y,s=30,c='r',alpha=0.5)
            ax.scatter(x1,y1,s=30,c='b',alpha=0.5)
    xxmc, yymc = np.array(xmc), np.array(ymc)
    xx1mc, yy1mc = np.array(x1mc), np.array(y1mc)
    xmed, ymed = [], []
    x1med, y1med = [], []
    for l in range(len(xmc[0])):
        xmed.append(np.median(xxmc[:,l]))
        ymed.append(np.median(yymc[:,l]))
        x1med.append(np.median(xx1mc[:,l]))
        y1med.append(np.median(yy1mc[:,l]))
    ax.plot(xmed,ymed,lw=4,ls='solid',c='r')
    ax.plot(x1med,y1med,lw=4,ls='solid',c='b')
    ax.grid(True, 'major', 'both', ls='--', lw=1, c='c', alpha=.5)
    ax.text(0.5,1.05,r'$f_{\rm{multi}}=%s$' %multi,horizontalalignment='center',transform=ax.transAxes,fontsize=100)
    #ax.set_xlabel(r'$s/r_{\rm{M}}$')
    ax.set_ylabel(r'$\tilde{v}$')
    ax.legend(loc=2,fontsize=55)
    ax.set_xlim(0,2)
    
    ax = plt.subplot2grid((200,100), (104, 1), rowspan=95, colspan=98)   # panel 2
    ax.tick_params(direction='inout',length=7,width=2,bottom=True,top=True,left=True,right=True)
    xbin1, xbin2, xbin3, xbin4, xbin5, xbin6, ydel1, ydel2, ydel3, ydel4, ydel5, ydel6 = [], [], [], [], [], [], [], [], [], [], [], []
    for l in range(len(xmc)):
        x, y = xmc[l], ymc[l]
        x1, y1 = x1mc[l], y1mc[l]
        ax.scatter((x+x1)/2,np.log10(y/y1),s=30,c='r',alpha=0.6)
        xbin1.append((x[0]+x1[0])/2)
        xbin2.append((x[1]+x1[1])/2)
        xbin3.append((x[2]+x1[2])/2)
        xbin4.append((x[3]+x1[3])/2)
        xbin5.append((x[4]+x1[4])/2)
        xbin6.append((x[5]+x1[5])/2)
        ydel1.append(np.log10(y[0]/y1[0]))
        ydel2.append(np.log10(y[1]/y1[1]))
        ydel3.append(np.log10(y[2]/y1[2]))
        ydel4.append(np.log10(y[3]/y1[3]))
        ydel5.append(np.log10(y[4]/y1[4]))
        ydel6.append(np.log10(y[5]/y1[5]))
    xbin1, xbin2, xbin3, xbin4, xbin5, xbin6, ydel1, ydel2, ydel3, ydel4, ydel5, ydel6 = np.array(xbin1), np.array(xbin2), np.array(xbin3), np.array(xbin4), np.array(xbin5), np.array(xbin6), np.array(ydel1), np.array(ydel2), np.array(ydel3), np.array(ydel4), np.array(ydel5), np.array(ydel6)
    yerr = [ydel1.std(), ydel2.std(), ydel3.std(), ydel4.std(), ydel5.std(), ydel6.std()]

    ax.scatter([np.median(xbin1),np.median(xbin2),np.median(xbin3),np.median(xbin4),np.median(xbin5),np.median(xbin6)],[np.median(ydel1),np.median(ydel2),np.median(ydel3),np.median(ydel4),np.median(ydel5),np.median(ydel6)],s=300,c='k',alpha=1)
    ax.errorbar([np.median(xbin1),np.median(xbin2),np.median(xbin3),np.median(xbin4),np.median(xbin5),np.median(xbin6)],[np.median(ydel1),np.median(ydel2),np.median(ydel3),np.median(ydel4),np.median(ydel5),np.median(ydel6)],yerr=yerr,elinewidth=5,c='k',lw=5,capsize=15,capthick=5,alpha=1)

    ax.text(np.median(xbin1)-0.02,-0.05,r'$%.3f\pm%.3f$' %(np.median(ydel1),yerr[0]),fontsize=65, horizontalalignment='left', verticalalignment='center')
    ax.text(np.median(xbin2)-0.02,-0.07,r'$%.3f\pm%.3f$' %(np.median(ydel2),yerr[1]),fontsize=65, horizontalalignment='left', verticalalignment='center')
    ax.text(np.median(xbin3)-0.02,-0.09,r'$%.3f\pm%.3f$' %(np.median(ydel3),yerr[2]),fontsize=65, horizontalalignment='left', verticalalignment='center')
    ax.text(np.median(xbin4)-0.02,-0.11,r'$%.3f\pm%.3f$' %(np.median(ydel4),yerr[3]),fontsize=65, horizontalalignment='left', verticalalignment='center')
    ax.text(np.median(xbin5)-0.02,-0.13,r'$%.3f\pm%.3f$' %(np.median(ydel5),yerr[4]),fontsize=65, horizontalalignment='left', verticalalignment='center')
    ax.text(np.median(xbin6)-0.02,-0.15,r'$%.3f\pm%.3f$' %(np.median(ydel6),yerr[5]),fontsize=65, horizontalalignment='left', verticalalignment='center')
    
    running_median_x = np.array([np.median(xbin1),np.median(xbin2),np.median(xbin3),np.median(xbin4),np.median(xbin5),np.median(xbin6)])
    xpts = np.log10(1.2e-10) - 2*np.log10(running_median_x) -2*np.log10(1.2)
    yaq = fEFE([1.2e-10,1.9],xpts)
    dyaq = (fEFE([1.2e-10,1.9-0.19],xpts)-yaq)/2
    dy = yaq-xpts
    running_median_y = 0.5*dy
    yerr_aq, yerr_newt = np.zeros(6), np.zeros(6)
    for i in range(6):
        yerr_aq[i]=np.sqrt(yerr[i]**2+dyaq[i]**2)
        yerr_newt[i]=yerr[i]
    chi_newt = (1*(np.median(ydel1)/yerr_newt[0])**2+(np.median(ydel2)/yerr_newt[1])**2+(np.median(ydel3)/yerr_newt[2])**2+(np.median(ydel4)/yerr_newt[3])**2+(np.median(ydel5)/yerr_newt[4])**2+(np.median(ydel6)/yerr_newt[5])**2)/5
    chi_aq = (1*((np.median(ydel1)-running_median_y[0])/yerr_aq[0])**2+((np.median(ydel2)-running_median_y[1])/yerr_aq[1])**2+((np.median(ydel3)-running_median_y[2])/yerr_aq[2])**2+((np.median(ydel4)-running_median_y[3])/yerr_aq[3])**2+((np.median(ydel5)-running_median_y[4])/yerr_aq[4])**2+((np.median(ydel6)-running_median_y[5])/yerr_aq[5])**2)/5
    print('reduced chi^2=',chi_newt,chi_aq)
    ax.fill_between(rnorm,logy-yaq1,logy+yaq1,color='m',alpha=0.3)
    ax.plot(rnorm,logy,c='m',lw=4,label=r'AQUAL: $\chi^2_\nu=%.1f$' %chi_aq)
    ax.plot([0,2],[0,0],c='k',lw=4,ls='dashed',label=r'Newton: $\chi^2_\nu=%.1f$' %chi_newt)
    ax.grid(True, 'major', 'both', ls='--', lw=1, c='c', alpha=.5)
    ax.legend(loc=2,fontsize=55)
    ax.set_xlabel(r'$s/r_{\rm{M}}$')
    ax.set_ylabel(r'$\log_{10}(\tilde{v}_{\rm{obs}}/\tilde{v}_{\rm{newt}})$')
    ax.set_xlim(0.,2)
    ax.set_ylim(-0.20,0.20)
    ax.set_xticks(np.arange(0.,2.,0.25))
    ax.set_yticks(np.arange(-0.2,0.25,0.05))

    plt.savefig('fig_vtilde_rnorm.png')
    fig.tight_layout()
    #plt.show()
    plt.close(fig)
