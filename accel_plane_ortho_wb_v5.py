# This code generates two figures for a wide binary sample: 
# projected velocity (v_p) vs projected separation (s), 
# and kinematic acceleration (g) vs Newtonian acceleration (g_N).
# Accelerations are calculated from Monte Carlo deprojected 3D velocities and separations. 
# Thus, the g_N-g relation varies from one result to another due to randomness of the deprojection.
# The user can define the sample to be plotted by setting the parameter ranges in lines 82--83, 372 - 386. 
# Extinction correction can be turned on/off in line 83. 
# When it is 'True', data only with declination > -28deg are used. 
# The default choice is the `new sample' of arXiv:2402.05720.
# The input file 'gaia_dr3_MSMS_d200pc.csv' was extracted from the El-Badry, Rix, & Heintz (2021) catalog.
# The user can also use a virtual Newtonian sample rather than a Gaia sample: 
# a sample from either Newtonian cicular or Newtonian elliptical orbits is allowed. 
# A mass-magnitude relation is needed. Three options are possible.
# The user needs to choose a value for the higher-order multiplicity fraction f_multi. 
# E.g., f_multi = 0.5 means that 50% of binaries are randomly assigned close hidden companions for either component or for both components.
# For further information see: Chae, K.-H. 2023 (Astrophysical Journal, 952, 128).
# Version 3 corrected a bug as described in Appendix A of Chae, K.-H. 2024 (Astrophysical Journal, 960, 114) and including more input options.
# This is version 5 including a minor change in drwaing individual eccentricities 
# as described Section 2.5 of arXiv:2402.05720 (Chae, K.-H. 2024, submitted to the Astrophysical Journal) 
# Questions can be directed to kyuhyunchae@gmail.com or chae@sejong.ac.kr.
# Last modified February 13, 2024.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import scipy.integrate as integrate
import multiprocessing as mp
import time

def dsq(p,*args):
    phi = p
    t, eps = args
    f = lambda x: 1/(1+eps*np.cos(x))**2
    val, err = integrate.quad(f, 0, phi)
    res = (t-val)**2
    return res

mod = input('model input (c (circular), e (elliptical), or g (gaia, default) = ')
opt = input('Mass-mag choice (v (V-band), j (J-band), or f (Flame masses)) = ')
f_multi = float(input('fraction of high-order multiples = '))
echoice = 'i'  # eccentricity choice: 'i' - individual ranges from Hwang et al. (2022), 's' - separation-dependent power-law distribution from Hwang et al. (2022), a number 'a' (e.g. 2: 'thermal' distrubution) - p(e;a) = a*e**(a-1),  0 .LEQ. e .LEQ. 1, a > 0.
Gboost = 1.  # boost factor for Newton's constant: It allows testing modified gravity with a boosted G.

if mod == 'c':
    model = 'circular'
elif mod == 'e':
    model = 'elliptical'
else: # default choice
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

if model == 'circular':
    fig_title = 'Newtonian simulation: circular'
    residual_name = 'residual_simulation_circular_x0.dat'
    fig_name = 'fig_RAR_WB_simulation_circular_x0.png'
    orthodata = 'ortho_residual_circular.dat'
elif model == 'elliptical':
    fig_title = 'Newtonian simulation: elliptical'
    residual_name = 'residual_simulation_elliptical_x0.dat'
    fig_name = 'fig_RAR_WB_simulation_elliptical_x0.png'
    orthodata = 'ortho_residual_elliptical.dat'
else:
    fig_title = 'Gaia data'
    residual_name = 'residual_gaia_x0.dat'
    fig_name = 'fig_RAR_WB_gaia_x0.png'
    orthodata = 'ortho_residual_gaia.dat'

G=6.6743e-11     # Newton's constant
Msun=1.9884e30   # Msolar in kg
au=1.49598e11      # AU in m
fac = 1.e-3*(1/(648000/np.pi))*3.085677e13/3.15582e7  # pc*mas/yr  to  km/s

junk1, junk2, R_chance, rp, d_A, d_A_err, d_B, d_B_err, MagG_A, MagG_B, M_A, M_B, mux_A, mux_A_err, muy_A, muy_A_err, mux_B, mux_B_err, muy_B, muy_B_err, RV_A, RV_A_err, RV_B, RV_B_err, gal_b, ruwe_A, ruwe_B, bp_rp_A, bp_rp_B, ra_A, dec_A, ra_B, dec_B, e, e0, e1, A_G_A, A_G_B = np.loadtxt('gaia_dr3_MSMS_d200pc.csv',skiprows=1,delimiter=',',unpack=True,dtype=float)
M_A_1, M_A_2, M_B_1, M_B_2 = np.zeros(len(MagG_A)), np.zeros(len(MagG_A)), np.zeros(len(MagG_A)), np.zeros(len(MagG_A))

#### initial selection of the sample  #########
R_chance_max = np.infty #0.01
extinction_correction = True   # True or False
###############################################
if extinction_correction == True:
    dec_min = -28     # if dust correction is True, declination > -28deg only is used because of the dust map used. 
    MagG_A -= A_G_A
    MagG_B -= A_G_B
else:
    dec_min = -100
ruwe_bin = np.maximum(ruwe_A, ruwe_B)
a, b = 4-3.6*17/11.5+0.1, 3.6/11.5
mask0 = (R_chance < R_chance_max) & (dec_A > dec_min) & (bp_rp_A > a+b*MagG_A) & (bp_rp_B > a+b*MagG_B)
rp, d_A, d_A_err, d_B, d_B_err, MagG_A, MagG_B, mux_A, mux_A_err, muy_A, muy_A_err, mux_B, mux_B_err, muy_B, muy_B_err, RV_A, RV_A_err, RV_B, RV_B_err, gal_b, ruwe_A, ruwe_B, bp_rp_A, bp_rp_B, ra_A, dec_A, ra_B, dec_B, e, e0, e1 = rp[mask0], d_A[mask0], d_A_err[mask0], d_B[mask0], d_B_err[mask0], MagG_A[mask0], MagG_B[mask0], mux_A[mask0], mux_A_err[mask0], muy_A[mask0], muy_A_err[mask0], mux_B[mask0], mux_B_err[mask0], muy_B[mask0], muy_B_err[mask0], RV_A[mask0], RV_A_err[mask0], RV_B[mask0], RV_B_err[mask0], gal_b[mask0], ruwe_A[mask0], ruwe_B[mask0], bp_rp_A[mask0], bp_rp_B[mask0], ra_A[mask0], dec_A[mask0], ra_B[mask0], dec_B[mask0], e[mask0], e0[mask0], e1[mask0]
ruwe_bin = ruwe_bin[mask0]
M_A, M_A_1, M_A_2, M_B, M_B_1, M_B_2 = M_A[mask0], M_A_1[mask0], M_A_2[mask0], M_B[mask0], M_B_1[mask0], M_B_2[mask0]

gam = -0.7
gam += 1
aML = np.loadtxt(ML,skiprows=1,unpack=True,usecols=(1),dtype=float)
np.random.seed(int(str(int(time.time()*1.e7))[-7:]))
for i in range(len(MagG_A)):
    MG1 = MagG_A[i]
    MG2 = MagG_B[i]
    mn = np.array([MG1**j for j in range(11)]) 
    M_A[i] = 10**np.sum(aML*mn)
    mn = np.array([MG2**j for j in range(11)]) 
    M_B[i] = 10**np.sum(aML*mn)
    p_multi = np.random.rand()
    if p_multi > f_multi: 
        M_A_1[i] = M_A[i]
        M_B_1[i] = M_B[i]
    else:
        p_multi_option = np.random.rand()
        if p_multi_option > 0.6:  # A has an additional component
            delm = 12*np.random.power(gam,1)[0]
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
            delm = 12*np.random.power(gam,1)[0]
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
            delm = 12*np.random.power(gam,1)[0]
            fL1 = 1/(1+10**(-0.4*delm))
            MG1 = -2.5*np.log10(fL1) + MagG_A[i]
            MG2 = -2.5*np.log10(np.maximum(1-fL1,1.e-20)) + MagG_A[i]
            mn = np.array([MG1**j for j in range(11)]) 
            M_A_1[i] = 10**np.sum(aML*mn)
            mn = np.array([MG2**j for j in range(11)]) 
            M_A_2[i] = 10**np.sum(aML*mn)
            M_A[i] = M_A_1[i]+M_A_2[i]
            delm = 12*np.random.power(0.3,1)[0]
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
d_M0 = (d_A/d_A_err**2+d_B/d_B_err**2)/(1/d_A_err**2+1/d_B_err**2)
dd = np.abs(d_A-d_B) #np.abs((d_A-d_B)/np.sqrt(d_A_err**2+d_B_err**2))
deld_orb = 6*rp*1.e3/206265  # 99% limit in parsec to the difference between the two distances from the orbit
mux_M, muy_M = (mux_A+mux_B)/2, (muy_A+muy_B)/2
dRV = np.abs(RV_A-RV_B)
delV_orb = 0.9419*1.3*np.sqrt((M_A+M_B)/rp)*1.2 # the 3D velocity theoretical limit (the last factor 1.2 is to allow for a boost by MOND-type modified gravity)
pmerr = np.maximum(np.maximum(np.abs(mux_A_err/mux_A),np.abs(muy_A_err/muy_A)),np.maximum(np.abs(mux_B_err/mux_B),np.abs(muy_B_err/muy_B)))
derr = np.maximum(d_A_err/d_A,d_B_err/d_B)
RVerr = np.maximum(np.abs(RV_A_err/RV_A),np.abs(RV_B_err/RV_B))

aLM = np.loadtxt(LM,skiprows=1,unpack=True,usecols=(1),dtype=float)
def Newton_simulation(xx):
    np.random.seed(int(str(int(time.time()*1.e7))[-7:]))
    rp_sim,M_A_1_sim,M_A_2_sim,M_B_1_sim,M_B_2_sim,d_A_sim,d_A_err_sim,d_B_sim,d_B_err_sim,mux_M_sim,muy_M_sim,mux_A_err_sim,muy_A_err_sim,mux_B_err_sim,muy_B_err_sim,e_sim,e0_sim,e1_sim,echoicex,Gboostx=xx
    G = 6.6743e-11*Gboostx     # Newton's constant
    echoice = echoicex
    ### main outer orbit: motion of two barycenters ###
    s = rp_sim  # in kau
    M_A_sim, M_B_sim = M_A_1_sim+M_A_2_sim, M_B_1_sim+M_B_2_sim
    inc = np.arccos(np.random.rand())  # random number drawn from prob(inc) = sin(inc)
    if model == 'circular':
        eps = 0
    else: # eps = eccentricity
        if echoice == 'i':
            xran=np.zeros(100)
            for i in range(100):
                sige = np.random.randn()
                if sige < 0:
                    xran[i] = e_sim + sige*(e_sim-e0_sim)
                else:
                    xran[i] = e_sim + sige*(e1_sim-e_sim)
            yran = xran[(xran>0.001)&(xran<0.999)]
            if len(yran) == 0:
                eps = e_sim
            else:
                eps = yran[0]
        elif echoice == 's':
            x = np.log10(s*1000)
            gam = np.maximum(np.minimum(1 -5.12340377 + 4.61235048*x -1.09835007*x**2 + 0.08748435*x**3,2.5),1.e-3)
            eps = np.random.power(gam, 1)[0]
        else:
            gam = float(echoice)
            eps = np.random.power(gam, 1)[0]
    f = lambda x: 1/(1+eps*np.cos(x))**2
    tmax, err = integrate.quad(f, 0, 2*np.pi)  # time for one complete orbit
    t = tmax*np.random.rand()
    sol=minimize(dsq,x0=[3.],args=(t,eps),bounds=[(0,2*np.pi)])  # solve the evolved azimuthal angle for t 
    phi0 = np.random.rand()*np.pi*2   # zero-point orientation of the elliptical orbit 
    phi = phi0+ sol.x[0]  # azimuthal angle
    r = 1.e3*s/np.sqrt(np.cos(phi)**2+np.cos(inc)**2*np.sin(phi)**2)  # AU
    a = r*(1+eps*np.cos(phi-phi0))/(1-eps**2)  # semi-major axis
    M = Msun*(M_A_sim+M_B_sim)
    vc = 1.e-3*np.sqrt((G*M/au)*(2/r-1/a)) # km/s
    dr_los = r*np.sin(inc)*np.sin(phi)/206265 # pc
    d_M_sim = (d_A_sim/d_A_err_sim**2+d_B_sim/d_B_err_sim**2)/(1/d_A_err_sim**2+1/d_B_err_sim**2) # initial value
    d_A_sim = d_M_sim + (M_B_sim/(M_A_sim+M_B_sim))*dr_los 
    d_B_sim = d_M_sim - (M_A_sim/(M_A_sim+M_B_sim))*dr_los 
    d_M_sim = (d_A_sim/d_A_err_sim**2+d_B_sim/d_B_err_sim**2)/(1/d_A_err_sim**2+1/d_B_err_sim**2) # simulated value
    d_M_err_sim = 1/np.sqrt(1/d_A_err_sim**2 + 1/d_B_err_sim**2)
    psi = np.arctan(-(np.cos(phi)+eps*np.cos(phi0))/(np.sin(phi)+eps*np.sin(phi0)))
    dv_x = vc*np.cos(psi)
    dv_y = vc*np.cos(inc)*np.sin(psi)
    mux_A_sim = mux_M_sim + (M_B_sim/(M_A_sim+M_B_sim))*dv_x/(fac*d_A_sim) 
    mux_B_sim = mux_M_sim - (M_A_sim/(M_A_sim+M_B_sim))*dv_x/(fac*d_B_sim) 
    muy_A_sim = muy_M_sim + (M_B_sim/(M_A_sim+M_B_sim))*dv_y/(fac*d_A_sim) 
    muy_B_sim = muy_M_sim - (M_A_sim/(M_A_sim+M_B_sim))*dv_y/(fac*d_B_sim) 
    ### inner orbit(s) when close hidden component(s) is(are) present: ###
    alp = 3.5 # L ~ M^gam
    log_a_in_min = np.log10(0.01)
    log_a_in_max = np.log10(np.minimum(d_M_sim,a*1000*0.3))
    if (M_A_2_sim != 0.) & (M_B_2_sim == 0.): # Case that A has an additional component
        M1 = M_A_1_sim
        M2 = M_A_2_sim
        Msim = M1+M2            
        a_in = 10**((log_a_in_max-log_a_in_min)*np.random.rand()+log_a_in_min)  # semi-major axis in AU
        gam = np.minimum(np.maximum(1 -1.26411333 + 0.85366141*np.log10(a_in),0.001),2.3)
        eps = np.random.power(gam, 1)[0]
        f = lambda x: 1/(1+eps*np.cos(x))**2
        tmax, err = integrate.quad(f, 0, 2*np.pi)
        t = tmax*np.random.rand()
        sol=minimize(dsq,x0=[3.],args=(t,eps),bounds=[(0,2*np.pi)])
        phi0 = np.random.rand()*np.pi*2
        phi = phi0+ sol.x[0]
        r_in = a_in*(1-eps**2)/(1+eps*np.cos(phi-phi0))  # in AU
        inc_CL = np.arccos(np.random.rand())
        s_in = r_in*np.sqrt(np.cos(phi)**2+np.cos(inc_CL)**2*np.sin(phi)**2)
        logM = np.log10(M2)
        mn = np.array([logM**j for j in range(11)]) 
        mag2 = np.sum(aLM*mn)+5*np.log10(d_A_sim/10)
        P_in = np.sqrt(a_in**3/(M1+M2))
        if P_in < 3:
            r_CL = 0
        else:
            r_CL = M1*M2*(M1**(alp-1)-M2**(alp-1))/((M1+M2)*(M1**alp+M2**alp)) 
        v_CL = r_CL*1.e-3*np.sqrt((G*Msun*Msim/au)*(2/r_in-1/a_in)) # km/s
        psi = np.arctan(-(np.cos(phi)+eps*np.cos(phi0))/(np.sin(phi)+eps*np.sin(phi0)))
        phi_CL = psi
        v_CL_x_A = v_CL*np.cos(phi_CL)
        v_CL_y_A = v_CL*np.cos(inc_CL)*np.sin(phi_CL)
        v_CL_x_B = 0
        v_CL_y_B = 0
    elif (M_A_2_sim == 0.) & (M_B_2_sim != 0.): # Case that B has an additional component
        M1 = M_B_1_sim
        M2 = M_B_2_sim
        Msim = M1+M2            
        a_in = 10**((log_a_in_max-log_a_in_min)*np.random.rand()+log_a_in_min)  # semi-major axis in AU
        gam = np.minimum(np.maximum(1 -1.26411333 + 0.85366141*np.log10(a_in),0.001),2.3)
        eps = np.random.power(gam, 1)[0]
        f = lambda x: 1/(1+eps*np.cos(x))**2
        tmax, err = integrate.quad(f, 0, 2*np.pi)
        t = tmax*np.random.rand()
        sol=minimize(dsq,x0=[3.],args=(t,eps),bounds=[(0,2*np.pi)])
        phi0 = np.random.rand()*np.pi*2
        phi = phi0+ sol.x[0] 
        r_in = a_in*(1-eps**2)/(1+eps*np.cos(phi-phi0))  # in AU
        inc_CL = np.arccos(np.random.rand())
        s_in = r_in*np.sqrt(np.cos(phi)**2+np.cos(inc_CL)**2*np.sin(phi)**2)
        logM = np.log10(M2)
        mn = np.array([logM**j for j in range(11)]) 
        mag2 = np.sum(aLM*mn)+5*np.log10(d_A_sim/10)
        P_in = np.sqrt(a_in**3/(M1+M2))
        if P_in < 3:
            r_CL = 0
        else: 
            r_CL = M1*M2*(M1**(alp-1)-M2**(alp-1))/((M1+M2)*(M1**alp+M2**alp)) #0
        v_CL = r_CL*1.e-3*np.sqrt((G*Msun*Msim/au)*(2/r_in-1/a_in)) # km/s
        psi = np.arctan(-(np.cos(phi)+eps*np.cos(phi0))/(np.sin(phi)+eps*np.sin(phi0)))
        phi_CL = psi
        v_CL_x_B = v_CL*np.cos(phi_CL)
        v_CL_y_B = v_CL*np.cos(inc_CL)*np.sin(phi_CL)
        v_CL_x_A = 0
        v_CL_y_A = 0
    elif (M_A_2_sim != 0.) & (M_B_2_sim != 0.): # Case that both A and B have additional components
        M1 = M_A_1_sim
        M2 = M_A_2_sim
        Msim = M1+M2            
        a_in = 10**((log_a_in_max-log_a_in_min)*np.random.rand()+log_a_in_min)   # semi-major axis in AU
        gam = np.minimum(np.maximum(1 -1.26411333 + 0.85366141*np.log10(a_in),0.001),2.3)
        eps = np.random.power(gam, 1)[0]
        f = lambda x: 1/(1+eps*np.cos(x))**2
        tmax, err = integrate.quad(f, 0, 2*np.pi)
        t = tmax*np.random.rand()
        sol=minimize(dsq,x0=[3.],args=(t,eps),bounds=[(0,2*np.pi)])
        phi0 = np.random.rand()*np.pi*2
        phi = phi0+ sol.x[0] 
        r_in = a_in*(1-eps**2)/(1+eps*np.cos(phi-phi0))  # in AU
        inc_CL = np.arccos(np.random.rand())
        s_in = r_in*np.sqrt(np.cos(phi)**2+np.cos(inc_CL)**2*np.sin(phi)**2)
        logM = np.log10(M2)
        mn = np.array([logM**j for j in range(11)]) 
        mag2 = np.sum(aLM*mn)+5*np.log10(d_A_sim/10)
        P_in = np.sqrt(a_in**3/(M1+M2))
        if P_in < 3:
            r_CL = 0
        else: 
            r_CL = M1*M2*(M1**(alp-1)-M2**(alp-1))/((M1+M2)*(M1**alp+M2**alp)) #0
        v_CL = r_CL*1.e-3*np.sqrt((G*Msun*Msim/au)*(2/r_in-1/a_in)) # km/s
        psi = np.arctan(-(np.cos(phi)+eps*np.cos(phi0))/(np.sin(phi)+eps*np.sin(phi0)))
        phi_CL = psi
        v_CL_x_A = v_CL*np.cos(phi_CL)
        v_CL_y_A = v_CL*np.cos(inc_CL)*np.sin(phi_CL)
        M1 = M_B_1_sim
        M2 = M_B_2_sim
        Msim = M1+M2            
        a_in = 10**((log_a_in_max-log_a_in_min)*np.random.rand()+log_a_in_min)  # semi-major axis in AU
        gam = np.minimum(np.maximum(1 -1.26411333 + 0.85366141*np.log10(a_in),0.001),2.3)
        eps = np.random.power(gam, 1)[0]
        f = lambda x: 1/(1+eps*np.cos(x))**2
        tmax, err = integrate.quad(f, 0, 2*np.pi)
        t = tmax*np.random.rand()
        sol=minimize(dsq,x0=[3.],args=(t,eps),bounds=[(0,2*np.pi)])
        phi0 = np.random.rand()*np.pi*2
        phi = phi0+ sol.x[0] 
        r_in = a_in*(1-eps**2)/(1+eps*np.cos(phi-phi0))  # in AU
        inc_CL = np.arccos(np.random.rand())
        s_in = r_in*np.sqrt(np.cos(phi)**2+np.cos(inc_CL)**2*np.sin(phi)**2)
        logM = np.log10(M2)
        mn = np.array([logM**j for j in range(11)]) 
        mag2 = np.sum(aLM*mn)+5*np.log10(d_A_sim/10)
        P_in = np.sqrt(a_in**3/(M1+M2))
        if P_in < 3:
            r_CL = 0
        else: 
            r_CL = M1*M2*(M1**(alp-1)-M2**(alp-1))/((M1+M2)*(M1**alp+M2**alp)) #0
        v_CL = r_CL*1.e-3*np.sqrt((G*Msun*Msim/au)*(2/r_in-1/a_in)) # km/s
        psi = np.arctan(-(np.cos(phi)+eps*np.cos(phi0))/(np.sin(phi)+eps*np.sin(phi0)))
        phi_CL = psi
        v_CL_x_B = v_CL*np.cos(phi_CL)
        v_CL_y_B = v_CL*np.cos(inc_CL)*np.sin(phi_CL)
    else:
        v_CL_x_A = 0
        v_CL_y_A = 0
        v_CL_x_B = 0
        v_CL_y_B = 0
    mux_A_sim = mux_A_sim + v_CL_x_A/(fac*d_A_sim)
    muy_A_sim = muy_A_sim + v_CL_y_A/(fac*d_A_sim)
    mux_B_sim = mux_B_sim + v_CL_x_B/(fac*d_B_sim)
    muy_B_sim = muy_B_sim + v_CL_y_B/(fac*d_B_sim)
    
    values = d_M_sim,d_M_err_sim,mux_A_sim,muy_A_sim,mux_B_sim,muy_B_sim
    return values
    
if model != 'gaia':
    xx = []
    for i in range(len(rp)):
        xx.append([rp[i],M_A_1[i],M_A_2[i],M_B_1[i],M_B_2[i],d_A[i],d_A_err[i],d_B[i],d_B_err[i],mux_M[i],muy_M[i],mux_A_err[i],muy_A_err[i],mux_B_err[i],muy_B_err[i],e[i],e0[i],e1[i],echoice,Gboost])
    pool = mp.Pool(mp.cpu_count()-1)
    res = pool.map(Newton_simulation, xx)
    for i in range(len(rp)):
        d_M[i],d_M_err[i],mux_A[i],muy_A[i],mux_B[i],muy_B[i] = res[i]

### select the allowed ranges of parameters ####################################
ruwe_max = np.infty  # 1.2  # Gaia DR3 RUWE value
rp_min, rp_max = 0.2, 30   # kau
d_min, d_max = 0, 200    # distance range
Mag_min, Mag_max = 4, 14  # G-band abs magnitude range
Mtot_min, Mtot_max =  0., np.infty  # total mass range
Mratio_min, Mratio_max = 0., np.infty  # mass ratio range
pmerr_max = 0.005  # PM relative error cut
derr_max = 0.01  # 0.005  # distance relative error cut
RVerr_max =  0.5  # maximum allowed fractional error of radial velocities
normverr_max = np.infty  #  maximum allowed error of projected velocity normailzed by Newtonian circular velocity
sigd = 3  # distance difference between the two components in normalized units
include_orbit = True  # include the maximum orbit size for the distance difference
sigRV = 3  # radial velocity difference between the two components in normalized units: set it to be np.infty if radial velocities are not to be used.
nbin = 6     # number of bins in log(s) where s = sky-projected separation in astronomical unit (au).
gamma_v = 1.2  # velocity boost factor to be compared
################################################################################

vx = fac*(mux_A*d_M-mux_B*d_M)
vy = fac*(muy_A*d_M-muy_B*d_M)
v = fac*d_M*np.sqrt((mux_A-mux_B)**2+(muy_A-muy_B)**2)
vc = 1.e-3*np.sqrt(G*Gboost*(M_A+M_B)*Msun/(au*1.e3*rp))
v_tilde = v/vc
delmu = np.sqrt((mux_A-mux_B)**2+(muy_A-muy_B)**2)
delmu_x, delmu_y = np.abs(mux_A-mux_B), np.abs(muy_A-muy_B)
sigv = fac*d_M*np.sqrt((mux_A_err**2+mux_B_err**2)*delmu_x**2+(muy_A_err**2+muy_B_err**2)*delmu_y**2)/delmu
print('v_tilde range:',v_tilde.min(),v_tilde.max())
vx_tilde, vy_tilde = vx/vc, vy/vc
Mtot = M_A + M_B
dRV_max = np.sqrt((sigRV*RV_A_err)**2+(sigRV*RV_B_err)**2+delV_orb**2)
if include_orbit == True:
    dd_max =  np.sqrt((sigd*d_A_err)**2+(sigd*d_B_err)**2+deld_orb**2)
else:
    dd_max =  np.sqrt((sigd*d_A_err)**2+(sigd*d_B_err)**2)
    
mask1 = (rp>rp_min) & (rp<rp_max) & (d_M0>d_min) & (d_M0<d_max) & (MagG_A>Mag_min) & (MagG_A<Mag_max) & (MagG_B>Mag_min) & (MagG_B<Mag_max) & (dd < dd_max)  & (pmerr < pmerr_max) & (dRV < dRV_max) & (Mtot > Mtot_min) & (Mtot < Mtot_max)  & (Mratio > Mratio_min) & (Mratio < Mratio_max)  & (ruwe_bin < ruwe_max) & (derr < derr_max) & (RVerr < RVerr_max) & (sigv/vc < normverr_max)

M_A=M_A[mask1]
M_A_1=M_A_1[mask1]
M_A_2=M_A_2[mask1]
M_B=M_B[mask1]
M_B_1=M_B_1[mask1]
M_B_2=M_B_2[mask1]
v_tilde=v_tilde[mask1]
vx_tilde=vx_tilde[mask1]
vy_tilde=vy_tilde[mask1]
v=v[mask1]
vx=vx[mask1]
vy=vy[mask1]
rp=rp[mask1]
d_A=d_A[mask1]
ruwe_A = ruwe_A[mask1]
ruwe_B = ruwe_B[mask1]
MagG_A = MagG_A[mask1]
MagG_B = MagG_B[mask1]
pmerr = pmerr[mask1]
e = e[mask1]
e0 = e0[mask1]
e1 = e1[mask1]
ruwe_all = np.concatenate((ruwe_A,ruwe_B))

print('ruwe:',len(ruwe_bin),np.median(ruwe_all),np.min(ruwe_all),np.max(ruwe_all))
print('distance:',len(d_A),np.median(d_A),np.min(d_A),np.max(d_A))
print('s[kau]:',len(rp),np.median(rp),np.min(rp),np.max(rp))
print('s/d:',len(rp/(206.265*d_A)),np.median(rp/(206.265*d_A)),np.min(rp/(206.265*d_A)),np.max(rp/(206.265*d_A)))

########################################################################################
fig = plt.figure(figsize=(20,15))
plt.rc('xtick',labelsize=50)
plt.rc('ytick',labelsize=50)
plt.rc('axes',labelsize=60)
plt.rc('font', family='STIXGeneral')
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 1.5

plt.subplots_adjust(left=0.1,right=0.96,bottom=0.1,top=0.93) 
ax = plt.subplot2grid((150,150), (5,9), rowspan=140, colspan=140)
ax.tick_params(direction='inout',length=7,width=2,bottom=True,top=True,left=True,right=True)
x = np.log10(rp*1.e3)
y = np.log10(v)
ax.scatter(x,y,s=3,c='gray')
xbins = np.linspace(np.log10(199),np.log10(30001),nbin+1)
running_x = (xbins[:-1]+xbins[1:])/2
inds = np.digitize(x,xbins)
running_y = np.zeros(len(running_x))
for i in range(len(running_x)):
    running_y[i] = np.median(y[inds==i+1])
ax.scatter(running_x,running_y,s=80,c='k')
ax.plot(running_x,running_y,c='k',lw=3,label=r'$\langle v_p\rangle$')

ax.text(3.5,0.4,r"$N=%i$" %len(x),horizontalalignment='center',fontsize=50)
xline = np.linspace(0,6,100)
a = np.median(running_y[running_x<3.2]+0.5*running_x[running_x<3.2])
yline = a-0.5*xline
ax.plot(xline,yline,c='r',lw=3,ls='dashed',label=r'$\langle v_p\rangle \propto s^{-1/2}$')
yline1 = yline+np.log10(gamma_v)
ax.plot(xline[xline>3.35],yline1[xline>3.35],c='r',lw=4,ls='dotted',label=r'$\langle v_p\rangle \rightarrow %.2f\langle v_p\rangle$' %gamma_v)

ax.legend(loc=1,fontsize=40)
ax.set_xlabel(r'$\log_{10}$ $s$ (au)')
ax.set_ylabel(r'$\log_{10}$ $v_p$ (km s$^{-1}$)')
ax.set_xlim(2.1,4.8)
ax.set_ylim(-1.5,0.75)
fig.tight_layout()
plt.savefig("fig_vp_s.png")
plt.close(fig)
########################################################################################

print('N_selected=',len(rp))
print('rp range:',np.min(rp),np.max(rp),np.mean(rp),np.median(rp))
print('mass range:',np.min(M_A+M_B),np.max(M_A+M_B),np.mean(M_A+M_B),np.median(M_A+M_B),np.std(M_A+M_B))

def Newton_2d_to_3d(xx):
    np.random.seed(int(str(int(time.time()*1.e7))[-7:]))
    rp, M_A, M_B, vp, e_sim, e0_sim, e1_sim, echoicex, Gboostx = xx
    G = 6.6743e-11*Gboostx     # Newton's constant
    echoice = echoicex
    s = rp
    inc = np.arccos(np.random.rand())  # random number drawn from prob(inc) = sin(inc)
    if model == 'circular':
        eps = 0
    else:
        if echoice == 'i':
            xran=np.zeros(100)
            for i in range(100):
                sige = np.random.randn()
                if sige < 0:
                    xran[i] = e_sim + sige*(e_sim-e0_sim)
                else:
                    xran[i] = e_sim + sige*(e1_sim-e_sim)
            yran = xran[(xran>0.001)&(xran<0.999)]
            if len(yran) == 0:
                eps = e_sim
            else:
                eps = yran[0]
        elif echoice == 's':
            x = np.log10(s*1000)
            gam = np.maximum(np.minimum(1 -5.12340377 + 4.61235048*x -1.09835007*x**2 + 0.08748435*x**3,2.5),1.e-3)
            eps = np.random.power(gam, 1)[0]
        else:
            gam = float(echoice)
            eps = np.random.power(gam, 1)[0]
    f = lambda x: 1/(1+eps*np.cos(x))**2
    tmax, err = integrate.quad(f, 0, 2*np.pi)  # time for one complete orbit
    t = tmax*np.random.rand()
    sol=minimize(dsq,x0=[3.],args=(t,eps),bounds=[(0,2*np.pi)])  # solve the evolved azimuthal angle for t 
    phi0 = np.random.rand()*np.pi*2   # zero-point orientation of the elliptical orbit 
    phi = phi0+ sol.x[0]  # azimuthal angle
    r = s/np.sqrt(np.cos(phi)**2+np.cos(inc)**2*np.sin(phi)**2)  # kAU
    R = r*1000*au
    M = Msun*(M_A+M_B)
    gN = np.log10(G*M/R**2)
    psi = np.arctan(-(np.cos(phi)+eps*np.cos(phi0))/(np.sin(phi)+eps*np.sin(phi0)))
    v3d = vp/np.sqrt(np.cos(psi)**2+np.cos(inc)**2*np.sin(psi)**2)
    g = np.log10((v3d*1.e3)**2/R)
    logr = np.log10(r*1000)
    values = gN, g, logr, eps
    return values

nprob = 1  # number of sampling from each binary system. Set nprob = 1 unless you want samping many times at once.
gN, g, logr, logm, ellip = np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
for j in range(nprob):
    gN_sim, g_sim, logr_sim, logm_sim, ellip_sim = np.zeros(len(rp)), np.zeros(len(rp)), np.zeros(len(rp)), np.zeros(len(rp)), np.zeros(len(rp))
    xx = []
    for i in range(len(rp)):
        xx.append([rp[i],M_A[i],M_B[i],v[i],e[i],e0[i],e1[i],echoice,Gboost])
    pool = mp.Pool(mp.cpu_count()-1)
    res = pool.map(Newton_2d_to_3d, xx)
    for i in range(len(rp)):
        gN_sim[i],g_sim[i],logr_sim[i],ellip_sim[i] = res[i]
        logm_sim[i] = np.log10(M_A[i]+M_B[i])
    gN = np.concatenate((gN,gN_sim))
    g = np.concatenate((g,g_sim))
    logr = np.concatenate((logr,logr_sim))
    logm = np.concatenate((logm,logm_sim))
    ellip = np.concatenate((ellip,ellip_sim))

dy = g - gN
x0 = gN + dy/2
d_orth = dy/np.sqrt(2)
xbin=np.array([-11.5,-9.8,-8.5,-7.5])
nbin = len(xbin)-1

fw = open(orthodata,'w')
fw.write('x0      ortho_residual \n')
for i in range(len(x0)):
    fw.write('%.8f   %.8f \n' %(x0[i],d_orth[i]))
fw.close()
print('orthogonal residual:',np.median(d_orth),d_orth.std())

nsam=200
Ntot=len(x0)
running_median_x=np.zeros(nbin)
running_median_y=np.zeros(nbin)
running_median_y_err=np.zeros(nbin)
running_sample_n=np.zeros(nbin)
running_median_rp=np.zeros(nbin)
running_median_M=np.zeros(nbin)
print('x0 range:',np.min(x0),np.max(x0))
for i in range(nbin):
    xx=x0[(xbin[i] <= x0) & (x0 < xbin[i+1])]
    running_median_x[i]=np.median(xx)
    yy=d_orth[(xbin[i] <= x0) & (x0 < xbin[i+1])]
    n=len(yy)
    xmeansam=np.zeros(nsam)
    for k in range(nsam):
        xsam=np.zeros(n)
        for j in range(n):
            irand=np.random.randint(n)
            xsam[j]=yy[irand]
        xmeansam[k]=np.median(xsam)
    mu=np.median(yy)
    mu_err=np.std(xmeansam)
    running_median_y[i]=mu
    running_median_y_err[i]=mu_err
    running_sample_n[i]=n
    print(n,running_median_x[i],running_median_y[i],running_median_y_err[i])

running_median_x0_tot = running_median_x
running_median_dorth_tot = running_median_y
running_median_dorth_err_tot = running_median_y_err
running_sample_n_tot = running_sample_n
running_median_x_tot = running_median_x0_tot - running_median_dorth_tot/np.sqrt(2)
running_median_y_tot = running_median_x0_tot + running_median_dorth_tot/np.sqrt(2)
running_median_y_err_tot = running_median_dorth_err_tot*np.sqrt(2)

fig = plt.figure(figsize=(20,30))
plt.rc('xtick',labelsize=40)
plt.rc('ytick',labelsize=40)
plt.rc('axes',labelsize=70)
plt.rc('font', family='STIXGeneral')
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 2.5

plt.subplots_adjust(left=0.12,right=0.97,bottom=0.07,top=0.99) 
ax = plt.subplot2grid((150,100), (1, 3), rowspan=95, colspan=95)   ############ panel 1
ax.tick_params(direction='inout',length=7,width=2,bottom=True,top=True,left=True,right=True)
ax.set_aspect('equal')
if model == 'circular':
    cx = 'orange'
elif model == 'eccentric':
    cx = 'b'
else:
    cx = 'r'
x0 = gN + (g-gN)/2
ax.scatter(gN[x0<xbin[0]],g[x0<xbin[0]],s=10,c='gray',alpha=0.5)
ax.scatter(gN[x0>xbin[3]],g[x0>xbin[3]],s=10,c='gray',alpha=0.5)
ax.scatter(gN[(x0>xbin[0])&(x0<xbin[1])],g[(x0>xbin[0])&(x0<xbin[1])],s=10,c='deepskyblue',alpha=0.5)
ax.scatter(gN[(x0>xbin[1])&(x0<xbin[2])],g[(x0>xbin[1])&(x0<xbin[2])],s=10,c='b',alpha=0.5)
ax.scatter(gN[(x0>xbin[2])&(x0<xbin[3])],g[(x0>xbin[2])&(x0<xbin[3])],s=10,c='k',alpha=0.5)
x = running_median_x_tot
y = running_median_y_tot 
yerr = running_median_y_err_tot
ax.scatter(x,y,s=200,c=cx)
ax.plot([-13.,-6],[-13.,-6],c='k',lw=6,ls='dashed')
D = 2
for i in range(len(xbin)):
    xline, yline = [xbin[i]-D/np.sqrt(2), xbin[i]+D/np.sqrt(2)], [xbin[i]+D/np.sqrt(2), xbin[i]-D/np.sqrt(2)]
    ax.plot(xline,yline,lw=3,ls='dotted',c='m')
#ax.text(0.5,1.05,fig_title, fontsize=70, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
ax.set_xlabel(r'$\log_{10}$ $g_{\rm{N}}$ (m s$^{-2}$)')
ax.set_ylabel(r'$\log_{10}$ $g$ (m s$^{-2}$)')
ax.set_xlim(-13,-6)
ax.set_ylim(-13,-6)
ax.legend(fontsize=40,loc=2)
############################################################## inset 1
axins = inset_axes(ax, width="22%", height="22%", loc=4, borderpad=5.)
axins.set_aspect('equal')
axins.set_xlim(0,1)
axins.set_ylim(0,1)
axins.text(0.78,0.37,r'$\Delta_\bot$',fontsize=50,horizontalalignment='center',transform=axins.transAxes)
axins.plot([0,1],[0,1],c='k',lw=6,ls='dashed')
axins.scatter(0.75,0.25,s=100,c='b',alpha=0.5)
axins.plot([0.5,0.75],[0.5,0.25],lw=3,c='k')
axins.plot([0.5,0.5],[0.,0.5],lw=2,ls='dotted',c='k')
axins.set_xticks([ ])
axins.set_yticks([ ])
ax.text(-7.02,-12.92,r'$x_0$',horizontalalignment='center',fontsize='50')

ax = plt.subplot2grid((150,100), (109, 3), rowspan=40, colspan=95)   ######################## panel 2
ax.tick_params(direction='inout',length=7,width=2,bottom=True,top=True,left=True,right=True)
ax.plot([-13,-6],[0,0],c='k',lw=6,ls='dashed')
x = running_median_x0_tot
y = running_median_dorth_tot 
yerr = running_median_dorth_err_tot
ax.scatter(x,y,s=200,c=cx)
for i in range(len(xbin)):
    xline, yline = [xbin[i], xbin[i]], [-1, 1]
    ax.plot(xline,yline,lw=3,ls='dotted',c='m')
fw = open(residual_name,'w')
for i in range(len(x)):
    data = '%.3f  %.3f  %.3f \n' %(x[i], y[i], yerr[i])
    fw.write(data)
fw.close()
ax.grid(True, 'major', 'both', ls='--', lw=1, c='c', alpha=.5)
ax.set_xlabel(r'$x_0$')
ax.set_ylabel(r'$\langle\Delta_\bot\rangle$')
ax.set_xlim(-13,-6)
ax.set_ylim(-0.45,0.05)
ax.set_yticks([-0.4,-0.3,-0.2,-0.1,0])

fig.tight_layout()
#plt.show()
plt.savefig(fig_name)
plt.close(fig)

