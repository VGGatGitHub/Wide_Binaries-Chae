# This code generates a distribution of normalized velocity vtilde := v_p/v_c with respect to normalized separation s/r_M for a wide binary sample. Here v_p is the sky-projected relative velocity between the pair, v_c is the theoretical Newtonian circular velocity for the sky-projected separaion s, and r_M is the MOND radius for the binary.
# Three auxiliary figures are also generated.
# The total mass of the binary is assigned through a Monte Carlo method including hidden close star(s).
# The user can define the sample to be plotted by setting the parameter ranges in lines 55 -- 58 and 170 -- 181. The default choice is the `new sample' of the main reference below.
# The input file 'gaia_dr3_MSMS_d200pc.csv' was extracted from the El-Badry, Rix, & Heintz (2021) catalog.
# A virtual Newtonian file 'Newton_dr3_MSMS_d200pc_1.csv' is also provided.
# A mass-magnitude relation is needed. Three options are possible.
# Main reference: arXiv:2402.05720 (Chae, K.-H. 2024, submitted to the Astrophysical Journal).
# Other references: (1) Astrophysical Journal, 952, 128 (Chae, K.-H. 2023)
#                   (2) Astrophysical Journal, 960, 114 (Chae, K.-H. 2024)
# Questions can be directed to kyuhyunchae@gmail.com or chae@sejong.ac.kr.
# Last modified Feb 13, 2024.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time

def dsq(p,*args):
    phi = p
    t, eps = args
    f = lambda x: 1/(1+eps*np.cos(x))**2
    val, err = integrate.quad(f, 0, phi)
    res = (t-val)**2
    return res

#opt = input('Mass-mag choice (v (V-band), j (J-band), or f (Flame masses)) = ')
opt = 'v'
model = 'gaia'
    
if opt == 'v':
    ML = 'M_MagG_polyfit_V_Ic.dat'
    LM = 'MagG_M_polyfit_V_Ic.dat'
elif opt == 'j':
    ML = 'M_MagG_polyfit_J_Bp_Rp.dat'
    LM = 'MagG_M_polyfit_J_Bp_Rp.dat'
elif opt == 'f':
    ML = 'M_MagG_polyfit_F.dat'
    LM = 'MagG_M_polyfit_F.dat'
else:
    ML = 'M_MagG_polyfit_V_Ic.dat'  # defaul choice
    LM = 'MagG_M_polyfit_V_Ic.dat'

fac = 1.e-3*(1/(648000/np.pi))*3.085677e13/3.15582e7  # pc*mas/yr  to  km/s
G=6.6743e-11     # Newton's constant
Msun=1.989e30   # Msolar in kg
au=1.4959787e11      # AU in m

filename='gaia_dr3_MSMS_d200pc.csv'
#filename='Newton_dr3_MSMS_d200pc_5.csv'  # no pmscatter  f_multi = 0.43
junk1, junk2, R_chance, rp, d_A, d_A_err, d_B, d_B_err, MagG_A, MagG_B, M_A, M_B, mux_A, mux_A_err, muy_A, muy_A_err, mux_B, mux_B_err, muy_B, muy_B_err, RV_A, RV_A_err, RV_B, RV_B_err, gal_b, ruwe_A, ruwe_B, bp_rp_A, bp_rp_B, ra_A, dec_A, ra_B, dec_B, e, e0, e1, A_G_A, A_G_B = np.loadtxt(filename,skiprows=1,delimiter=',',unpack=True,dtype=float)

### set the preliminary parameters #################################################################
extinction_correction = True   # dust extinction correction choice
R_chance_max =  np.infty # 0.01   #### maximum allowed chance probability from El-Badry et al.
###################################### Set it to be np.infty if the use does not use this parameter. 
multi_frac = 0.18 #  #0.18  #0.43   ###   f_multi   ###
####################################################################################################

if extinction_correction == True:
    dec_min = -28
    MagG_A -= A_G_A
    MagG_B -= A_G_B
else:
    dec_min = -100
# initial selection of the sample
ruwe_bin = np.maximum(ruwe_A, ruwe_B)
a, b = 4-3.6*17/11.5+0.1, 3.6/11.5
mask0 = (R_chance < R_chance_max) & (dec_A > dec_min) & (bp_rp_A > a+b*MagG_A) & (bp_rp_B > a+b*MagG_B)
rp, d_A, d_A_err, d_B, d_B_err, MagG_A, MagG_B, M_A, M_B, mux_A, mux_A_err, muy_A, muy_A_err, mux_B, mux_B_err, muy_B, muy_B_err, RV_A, RV_A_err, RV_B, RV_B_err, gal_b, ruwe_A, ruwe_B, bp_rp_A, bp_rp_B, ra_A, dec_A, ra_B, dec_B, e, e0, e1 = rp[mask0], d_A[mask0], d_A_err[mask0], d_B[mask0], d_B_err[mask0], MagG_A[mask0], MagG_B[mask0], M_A[mask0], M_B[mask0], mux_A[mask0], mux_A_err[mask0], muy_A[mask0], muy_A_err[mask0], mux_B[mask0], mux_B_err[mask0], muy_B[mask0], muy_B_err[mask0], RV_A[mask0], RV_A_err[mask0], RV_B[mask0], RV_B_err[mask0], gal_b[mask0], ruwe_A[mask0], ruwe_B[mask0], bp_rp_A[mask0], bp_rp_B[mask0], ra_A[mask0], dec_A[mask0], ra_B[mask0], dec_B[mask0], e[mask0], e0[mask0], e1[mask0]
ruwe_bin = ruwe_bin[mask0]

aML = np.loadtxt(ML,skiprows=1,unpack=True,usecols=(1),dtype=float)
np.random.seed(int(str(int(time.time()*1.e7))[-7:]))
for i in range(len(MagG_A)):
    MG1 = MagG_A[i]
    MG2 = MagG_B[i]
    mn = np.array([MG1**j for j in range(11)]) 
    M_A[i] = 10**np.sum(aML*mn)
    mn = np.array([MG2**j for j in range(11)]) 
    M_B[i] = 10**np.sum(aML*mn)

M_A_1, M_A_2, M_B_1, M_B_2 = np.zeros(len(rp)), np.zeros(len(rp)), np.zeros(len(rp)), np.zeros(len(rp))

M_A_0, M_B_0 = np.zeros(len(rp)), np.zeros(len(rp))
aML = np.loadtxt(ML,skiprows=1,unpack=True,usecols=(1),dtype=float)
gam_M = -0.7
a = gam_M+1
for i in range(len(MagG_A)):
    np.random.seed(int(str(int(time.time()*1.e7))[-7:]))
    MG1 = MagG_A[i]
    MG2 = MagG_B[i]
    mn = np.array([MG1**j for j in range(11)]) 
    M_A[i] = 10**np.sum(aML*mn)
    M_A_0[i] = 10**np.sum(aML*mn)
    mn = np.array([MG2**j for j in range(11)]) 
    M_B[i] = 10**np.sum(aML*mn)
    M_B_0[i] = 10**np.sum(aML*mn)
    p_multi = np.random.rand()
    if p_multi > multi_frac: 
        M_A_1[i] = M_A[i]
        M_B_1[i] = M_B[i]
    else:
        p_multi_option = np.random.rand()
        if p_multi_option > 0.6:  # A has an additional component
            delm = 12*np.random.power(a,1)[0]
            fL1 = 1/(1+10**(-0.4*delm))
            MG1 = -2.5*np.log10(fL1) + MagG_A[i]
            MG2 = -2.5*np.log10(np.maximum(1-fL1,1.e-20)) + MagG_A[i]
            mn = np.array([MG1**j for j in range(11)]) 
            M_A_1[i] = 10**np.sum(aML*mn)
            mn = np.array([MG2**j for j in range(11)]) 
            M_A_2[i] = 10**np.sum(aML*mn)
            M_A[i] = M_A_1[i]+M_A_2[i]
            M_B_1[i] = M_B[i]
        elif p_multi_option > 0.3:  # B has an additional component
            delm = 12*np.random.power(a,1)[0]
            fL1 = 1/(1+10**(-0.4*delm))
            MG1 = -2.5*np.log10(fL1) + MagG_B[i]
            MG2 = -2.5*np.log10(np.maximum(1-fL1,1.e-20)) + MagG_B[i]
            mn = np.array([MG1**j for j in range(11)]) 
            M_B_1[i] = 10**np.sum(aML*mn)
            mn = np.array([MG2**j for j in range(11)]) 
            M_B_2[i] = 10**np.sum(aML*mn)
            M_B[i] = M_B_1[i]+M_B_2[i]
            M_A_1[i] = M_A[i]            
        else: # Both A and B have additional components respectively
            delm = 12*np.random.power(a,1)[0]
            fL1 = 1/(1+10**(-0.4*delm))
            MG1 = -2.5*np.log10(fL1) + MagG_A[i]
            MG2 = -2.5*np.log10(np.maximum(1-fL1,1.e-20)) + MagG_A[i]
            mn = np.array([MG1**j for j in range(11)]) 
            M_A_1[i] = 10**np.sum(aML*mn)
            mn = np.array([MG2**j for j in range(11)]) 
            M_A_2[i] = 10**np.sum(aML*mn)
            M_A[i] = M_A_1[i]+M_A_2[i]
            delm = 12*np.random.power(a,1)[0]
            fL1 = 1/(1+10**(-0.4*delm))
            MG1 = -2.5*np.log10(fL1) + MagG_B[i]
            MG2 = -2.5*np.log10(np.maximum(1-fL1,1.e-20)) + MagG_B[i]
            mn = np.array([MG1**j for j in range(11)]) 
            M_B_1[i] = 10**np.sum(aML*mn)
            mn = np.array([MG2**j for j in range(11)]) 
            M_B_2[i] = 10**np.sum(aML*mn)
            M_B[i] = M_B_1[i]+M_B_2[i]
    
Mratio = M_B/M_A
d_M = (d_A/d_A_err**2+d_B/d_B_err**2)/(1/d_A_err**2+1/d_B_err**2)
d_M_err = 1/np.sqrt(1/d_A_err**2 + 1/d_B_err**2)
dd = np.abs(d_A-d_B)
deld_orb = 6*rp*1.e3/206265  # 99% limit in parsec to the difference between the two distances from the orbit
dRV = np.abs(RV_A-RV_B)
delV_orb = 0.9419*1.3*np.sqrt((M_A_0+M_B_0)/rp)*1.2  # the 3D velocity theoretical limit (the last factor 1.2 is to allow for a boost by MOND-type modified gravity)
pmerr = np.maximum(np.maximum(np.abs(mux_A_err/mux_A),np.abs(muy_A_err/muy_A)),np.maximum(np.abs(mux_B_err/mux_B),np.abs(muy_B_err/muy_B)))
derr = np.maximum(d_A_err/d_A,d_B_err/d_B)
RVerr = np.maximum(np.abs(RV_A_err/RV_A),np.abs(RV_B_err/RV_B))

del_mu = np.sqrt((mux_A-mux_B)**2+(muy_A-muy_B)**2)
sig_mu = np.sqrt(((mux_A-mux_B)/del_mu)**2*(mux_A_err**2+mux_B_err**2)+((muy_A-muy_B)/del_mu)**2*(muy_A_err**2+muy_B_err**2))

Mtot = M_A+M_B
vx = fac*(mux_A*d_M-mux_B*d_M)
vy = fac*(muy_A*d_M-muy_B*d_M)
vp = fac*del_mu*d_M
vr = dRV
v = np.sqrt(vp**2+vr**2)

### select the allowed ranges of parameters ####################################
ruwe_max = np.infty  # Gaia DR3 RUWE value
rp_min, rp_max = 0.2, 30   # kau
Mtot_min, Mtot_max = 0., np.infty  # total mass range
d_min, d_max = 0, 200    # distance range
Mag_min, Mag_max = 4, 14  # G-band abs magnitude range
Mratio_min, Mratio_max =  0., np.infty  # mass ratio range
pmerr_max = 0.005  # PM relative error cut
derr_max = 0.01 #np.infty  #0.01  # distance relative error cut
RVerr_max = 0.5 #np.infty  #0.5
sigd = 3  # distance difference between the two components in normalized units
include_orbit = True
sigRV = 3  #np.infty  #3   # radial velocity difference between the two components in normalized units: set it to be np.infty if radial velocities are not to be used.
################################################################################
dRV_max = np.sqrt((sigRV*RV_A_err)**2+(sigRV*RV_B_err)**2+delV_orb**2)
if  include_orbit == True:
    dd_max =  np.sqrt((sigd*d_A_err)**2+(sigd*d_B_err)**2+deld_orb**2)
else:
    dd_max =  np.sqrt((sigd*d_A_err)**2+(sigd*d_B_err)**2)

mask1 = (rp>rp_min) & (rp<rp_max) & (d_M>d_min) & (d_M<d_max) & (MagG_A>Mag_min) & (MagG_A<Mag_max) & (MagG_B>Mag_min) & (MagG_B<Mag_max)  & (ruwe_bin < ruwe_max)  & (derr < derr_max) & (pmerr < pmerr_max) & (Mtot > Mtot_min) & (Mtot < Mtot_max)  & (Mratio > Mratio_min) & (Mratio < Mratio_max) & (dRV < dRV_max) & (dd < dd_max) & (RVerr < RVerr_max)

Mtot=Mtot[mask1]
M_A=M_A[mask1]
M_B=M_B[mask1]
mux_A=mux_A[mask1]
mux_A_err=mux_A_err[mask1]
mux_B=mux_B[mask1]
mux_B_err=mux_B_err[mask1]
muy_A=muy_A[mask1]
muy_A_err=muy_A_err[mask1]
muy_B=muy_B[mask1]
muy_B_err=muy_B_err[mask1]
v=v[mask1]
vp=vp[mask1]
vr=vr[mask1]
rp=rp[mask1]
d_A=d_A[mask1]
d_A_err=d_A_err[mask1]
d_B=d_B[mask1]
d_B_err=d_B_err[mask1]
d_M=d_M[mask1]
ruwe_A = ruwe_A[mask1]
ruwe_B = ruwe_B[mask1]
MagG_A = MagG_A[mask1]
MagG_B = MagG_B[mask1]
pmerr = pmerr[mask1]
e = e[mask1]
e0 = e0[mask1]
e1 = e1[mask1]
ruwe_all = np.concatenate((ruwe_A,ruwe_B))

Mt = M_A+M_B
r_M = np.sqrt(G*(M_A+M_B)*Msun/1.2e-10)/(1000*au)
rnorm = rp/r_M
vc = 1.e-3*np.sqrt(G*Msun*(M_A+M_B)/(rp*au*1000)) # km/s
delmu = np.sqrt((mux_A-mux_B)**2+(muy_A-muy_B)**2)
delmu_x, delmu_y = np.abs(mux_A-mux_B), np.abs(muy_A-muy_B)
sigvp = fac*d_M*np.sqrt((mux_A_err**2+mux_B_err**2)*delmu_x**2+(muy_A_err**2+muy_B_err**2)*delmu_y**2)/delmu
vp = fac*d_M*delmu
vt = vp/vc
normvterr = np.log10(vt*np.sqrt((sigvp/vp)**2+0.05**2))
normvterr_max = np.log10(0.1*np.maximum(1,vt/2))
normvperr = np.log10(sigvp/vp)

############################################################

fig = plt.figure(figsize=(20,15))
plt.rc('xtick',labelsize=60)
plt.rc('ytick',labelsize=60)
plt.rc('axes',labelsize=70)
plt.rc('font', family='STIXGeneral')
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 1.5

plt.subplots_adjust(left=0.13,right=0.98,bottom=0.08,top=0.97) 
ax = plt.subplot2grid((100,100), (1,3), rowspan=95, colspan=95)
ax.tick_params(direction='inout',length=7,width=2,bottom=True,top=True,left=True,right=True)
ax.hist(d_M,bins=20,range=(0.,200.),alpha=0.5)
ax.text(0.05,0.8,r"$N_{\rm{binary}}=%i$"%len(normvterr),horizontalalignment='left',fontsize=55,transform=ax.transAxes)
ax.set_xlabel(r'distance')
ax.set_ylabel(r'number')
ax.set_xlim(0,200.)
#ax.set_ylim(0,2000)

fig.tight_layout()
plt.savefig("fig_hist_dist.png")
plt.close(fig)

############################################################

fig = plt.figure(figsize=(20,35))
plt.rc('xtick',labelsize=60)
plt.rc('ytick',labelsize=60)
plt.rc('axes',labelsize=70)
plt.rc('font', family='STIXGeneral')
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 1.5

plt.subplots_adjust(left=0.13,right=0.98,bottom=0.08,top=0.97) 
ax = plt.subplot2grid((200,100), (1,3), rowspan=90, colspan=95)
ax.tick_params(direction='inout',length=7,width=2,bottom=True,top=True,left=True,right=True)
ax.hist(normvterr,bins=30,range=(-3.,0.),alpha=0.5)
ax.text(0.05,0.8,r"$N_{\rm{binary}}=%i$"%len(normvterr),horizontalalignment='left',fontsize=55,transform=ax.transAxes)
ax.set_xlabel(r'$\log_{10}\sigma_{\tilde{v}}$')
ax.set_ylabel(r'number')
ax.set_xlim(-3,0.)
#ax.set_ylim(0,2000)
ax = plt.subplot2grid((200,100), (109,3), rowspan=90, colspan=95)
ax.tick_params(direction='inout',length=7,width=2,bottom=True,top=True,left=True,right=True)
ax.hist(normvperr,bins=30,range=(-3.,0.),alpha=0.5)
ax.set_xlabel(r'$\log_{10}(\sigma_{v_p}/v_p)$')
ax.set_ylabel(r'number')
ax.set_xlim(-3.,0.)
#ax.set_ylim(0,1000)

fig.tight_layout()
plt.savefig("fig_hist_verr.png")
plt.close(fig)

############################################################

fig = plt.figure(figsize=(20,15))
plt.rc('xtick',labelsize=50)
plt.rc('ytick',labelsize=50)
plt.rc('axes',labelsize=60)
plt.rc('font', family='STIXGeneral')
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 1.5

plt.subplots_adjust(left=0.12,right=0.98,bottom=0.11,top=0.95) 
ax = plt.subplot2grid((100,100), (3,3), rowspan=95, colspan=95)
ax.tick_params(direction='inout',length=7,width=2,bottom=True,top=True,left=True,right=True)
xx = np.log10(rp*1000)
ax.scatter(xx,normvterr,s=10)
ax.text(0.02,0.85,r"$N_{\rm{binary}}=%i$"%len(d_M),horizontalalignment='left',fontsize=45,transform=ax.transAxes)
ax.set_xlabel(r'$\log_{10}$ ($s$/au)')
ax.set_ylabel(r'$\log_{10}(\sigma_{\tilde{v}}/\tilde{v})$')
ax.set_xlim(2,4.5)

fig.tight_layout()
plt.savefig("fig_vterr_s.png")
plt.close(fig)

############################################################
fig = plt.figure(figsize=(60,40))
plt.rc('xtick',labelsize=100)
plt.rc('ytick',labelsize=100)
plt.rc('axes',labelsize=150)
plt.rc('font', family='STIXGeneral')
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 3.

plt.subplots_adjust(left=0.08,right=0.98,bottom=0.09,top=0.93)
ax = plt.subplot2grid((200,200), (3,3), rowspan=195, colspan=195)
ax.tick_params(direction='inout',length=15,width=5,bottom=True,top=True,left=True,right=True)
logs = np.log10(rnorm)
ax.scatter(logs,np.log10(vt),s=20,alpha=0.8,c='gray')

xlow, xtick1, nxtick, xmin, nbin = -2.0, -1.5, 6, -1.4, 6
xbin = np.linspace(xmin,0.4,nbin+1) 

running_median_x=np.zeros(nbin)
running_median_y=np.zeros(nbin)
x0 = logs
vtilde = np.log10(vt)
for i in range(nbin):
    xx=x0[(xbin[i] <= x0) & (x0 < xbin[i+1])]
    running_median_x[i]=np.median(xx)
    yy=vtilde[(xbin[i] <= x0) & (x0 < xbin[i+1])]
    mu=np.median(yy)
    running_median_y[i]=mu
xx0, yy0 = running_median_x, running_median_y
yy_med = np.median(vtilde[(xbin[0] <= x0) & (x0 < xbin[-1])])
xx_med = np.median(x0[(xbin[0] <= x0) & (x0 < xbin[-1])])
ax.scatter(xx0,yy0,s=800,c='k')
ax.plot(xx0,yy0,lw=12,c='k')
ax.plot([xmin,0.4],[yy_med,yy_med],lw=7,ls='dashed',c='k')
ax.scatter(xx_med,yy_med,s=1200,c='r',label=r"$%.2f$" %10**yy_med)

ax.plot([xmin,xmin],[-2.5,1.5],lw=5,ls='dotted',c='k')
ax.plot([0.4,0.4],[-2.5,1.5],lw=5,ls='dotted',c='k')
for i in range(1,nbin):
    ax.plot([xbin[i],xbin[i]],[-1.5,0.5],lw=3,ls='dotted',c='k')

ax.text(-0.5,1.08,r"$N_{\rm{binary}}=%i$"%len(vt),horizontalalignment='center',fontweight='heavy',fontsize=125)

ax.legend(loc=1,fontsize=110,framealpha=1)
ax.set_yticks(np.linspace(-2.5,2.,10))
ax.set_xticks(np.linspace(-1.5,1.,6))
ax.set_xlabel(r'$\log_{10}$($s/r_{\rm{M}}$)')
ax.set_ylabel(r'$\log_{10}$($\tilde{v}$)')
ax.set_xlim(-2,1)
ax.set_ylim(-2.5,1.)
fig.tight_layout()
plt.savefig("fig_vtilde.png")
plt.close(fig)
