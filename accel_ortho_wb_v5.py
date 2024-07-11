# This python class generates one Monte Carlo set of acceleration data (g_N, g) for an input sample of wide binaries and returns the median residuals in the user-defined bins.
# It also calculates another Monte Carlo set for a virtual Newtonian data in the same bins. This set provides the Newtonian benchmark with which the result of interest can be compared.
# The user is strongly encouraged to read carefully Chae, K.-H. 2023, published in the Astrophysical Journal (vol: 952, id: 128) before using this class. 
# Version 3 corrected a bug as described in Appendix A of Chae, K.-H. 2024 (Astrophysical Journal, 960, 114) and included more input options.
# This is version 5 including a minor change in drwaing individual eccentricities as described Section 2.5 of arXiv:2402.05720 (Chae, K.-H. 2024, submitted to the Astrophysical Journal) 
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

class accel_ortho_wb_v5:
    def __init__(self,data=0.,d_min=0,d_max=80,rp_min=0.2,rp_max=30,MagG_min=4,MagG_max=14,R_chance_max=0.01,pmerr_max=0.01,derr_max=0.01,RVerr_max=0.2,vterr_max=100.,vperr_max=100.,ruwe_max=100.,sigd=3,include_orbit=True,multi_frac=0.5,gam_M=-0.7,ML_choice='v',ext_corr=False,pm_scatter=True,sigRV=3,x0bin=np.array([-11.5,-9.8,-8.5,-7.5]),nrand=1):
        self.opt = ML_choice
        self.d_min, self.d_max = d_min, d_max   # pc
        self.rp_min, self.rp_max = rp_min, rp_max   # kau
        self.pmerr_max = pmerr_max
        self.derr_max = derr_max
        self.RVerr_max = RVerr_max
        self.vterr_max, self.vperr_max = vterr_max, vperr_max
        self.sigd = sigd
        self.nprob = nrand
        self.x0bin = x0bin

        if ML_choice == 'v':
            ML = 'M_MagG_polyfit_V_Ic.dat'
            LM = 'MagG_M_polyfit_V_Ic.dat'
        elif ML_choice == 'j':
            ML = 'M_MagG_polyfit_J_Bp_Rp.dat'
            LM = 'MagG_M_polyfit_J_Bp_Rp.dat'
        elif ML_choice == 'f':
            ML = 'M_MagG_polyfit_F.dat'
            LM = 'MagG_M_polyfit_F.dat'
        else:  # default choice
            ML = 'M_MagG_polyfit_V_Ic.dat'
            LM = 'MagG_M_polyfit_V_Ic.dat'
 
        self.fac = 1.e-3*(1/(648000/np.pi))*3.085677e13/3.15582e7  # pc*mas/yr  to  km/s
        self.G=6.6743e-11     # Newton's constant
        self.Msun=1.989e30    # Msolar in kg
        self.au=1.4959787e11  # AU in m

        R_chance, rp, d_A, d_A_err, d_B, d_B_err, MagG_A, MagG_B, M_A, M_B, mux_A, mux_A_err, muy_A, muy_A_err, mux_B, mux_B_err, muy_B, muy_B_err, RV_A, RV_A_err, RV_B, RV_B_err, gal_b, ruwe_A, ruwe_B, bp_rp_A, bp_rp_B, ra_A, dec_A, ra_B, dec_B, e, e0, e1, A_G_A, A_G_B = data

        if pm_scatter == True:
            mux_A += mux_A_err*np.random.randn(len(mux_A))
            muy_A += muy_A_err*np.random.randn(len(mux_A))
            mux_B += mux_B_err*np.random.randn(len(mux_A))
            muy_B += muy_B_err*np.random.randn(len(mux_A))

        ruwe_bin = np.maximum(ruwe_A, ruwe_B)
        if ext_corr == True:
            dec_min = -28
            MagG_A -= A_G_A
            MagG_B -= A_G_B
        else:
            dec_min = -100
        a, b = 4-3.6*17/11.5+0.1, 3.6/11.5
        mask0 = (dec_A > dec_min)  & (MagG_A > MagG_min) & (MagG_A < MagG_max) & (MagG_B > MagG_min) & (MagG_B < MagG_max) & (ruwe_bin < ruwe_max) & (bp_rp_A > a+b*MagG_A) & (bp_rp_B > a+b*MagG_B)
        rp, d_A, d_A_err, d_B, d_B_err, MagG_A, MagG_B, M_A, M_B, mux_A, mux_A_err, muy_A, muy_A_err, mux_B, mux_B_err, muy_B, muy_B_err, RV_A, RV_A_err, RV_B, RV_B_err, gal_b, ruwe_A, ruwe_B, bp_rp_A, bp_rp_B, ra_A, dec_A, ra_B, dec_B, e, e0, e1 = rp[mask0], d_A[mask0], d_A_err[mask0], d_B[mask0], d_B_err[mask0], MagG_A[mask0], MagG_B[mask0], M_A[mask0], M_B[mask0], mux_A[mask0], mux_A_err[mask0], muy_A[mask0], muy_A_err[mask0], mux_B[mask0], mux_B_err[mask0], muy_B[mask0], muy_B_err[mask0], RV_A[mask0], RV_A_err[mask0], RV_B[mask0], RV_B_err[mask0], gal_b[mask0], ruwe_A[mask0], ruwe_B[mask0], bp_rp_A[mask0], bp_rp_B[mask0], ra_A[mask0], dec_A[mask0], ra_B[mask0], dec_B[mask0], e[mask0], e0[mask0], e1[mask0]
        M_A_1, M_A_2, M_B_1, M_B_2 = np.zeros(len(rp)), np.zeros(len(rp)), np.zeros(len(rp)), np.zeros(len(rp))
                
        aML = np.loadtxt(ML,skiprows=1,unpack=True,usecols=(1),dtype=float)
        self.aLM = np.loadtxt(LM,skiprows=1,unpack=True,usecols=(1),dtype=float)
        a = gam_M+1
        for i in range(len(MagG_A)):
            np.random.seed(int(str(int(time.time()*1.e7))[-7:]))
            MG1 = MagG_A[i]
            MG2 = MagG_B[i]
            mn = np.array([MG1**j for j in range(11)]) 
            M_A[i] = 10**np.sum(aML*mn)
            mn = np.array([MG2**j for j in range(11)]) 
            M_B[i] = 10**np.sum(aML*mn)
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

        d_M = (d_A/d_A_err**2+d_B/d_B_err**2)/(1/d_A_err**2+1/d_B_err**2)
        d_M_err = 1/np.sqrt(1/d_A_err**2 + 1/d_B_err**2)
        d_M0 = (d_A/d_A_err**2+d_B/d_B_err**2)/(1/d_A_err**2+1/d_B_err**2)
        dd = np.abs(d_A-d_B)  
        deld_orb = 6*rp*1.e3/206265  # 99% limit in parsec to the difference between the two distances from the orbit
        mux_M, muy_M = (mux_A+mux_B)/2, (muy_A+muy_B)/2
        delV_orb = 0.9419*1.3*np.sqrt((M_A+M_B)/rp)*1.2 # the 3D velocity theoretical limit (the last factor 1.2 is to allow for a boost by MOND-type modified gravity)
        self.dRV_max = np.sqrt((sigRV*RV_A_err)**2+(sigRV*RV_B_err)**2+delV_orb**2)
        if include_orbit == True:
            self.dd_max = np.sqrt((sigd*d_A_err)**2+(sigd*d_B_err)**2+deld_orb**2)
        else:
            self.dd_max = np.sqrt((sigd*d_A_err)**2+(sigd*d_B_err)**2)
        pmerr = np.maximum(np.maximum(np.abs(mux_A_err/mux_A),np.abs(muy_A_err/muy_A)),np.maximum(np.abs(mux_B_err/mux_B),np.abs(muy_B_err/muy_B)))
        derr = np.maximum(d_A_err/d_A,d_B_err/d_B)
        RVerr = np.maximum(np.abs(RV_A_err/RV_A),np.abs(RV_B_err/RV_B))
        dRV = np.abs((RV_A-RV_B)) 
    
        self.M_A, self.M_A_1, self.M_A_2, self.M_B, self.M_B_1, self.M_B_2, self.rp, self.d_A, self.d_A_err, self.d_B, self.d_B_err, self.mux_A, self.mux_A_err, self.muy_A, self.muy_A_err, self.mux_B, self.mux_B_err, self.muy_B, self.muy_B_err, self.RV_A, self.RV_A_err, self.RV_B, self.RV_B_err, self.e, self.e0, self.e1 = M_A, M_A_1, M_A_2, M_B, M_B_1, M_B_2, rp, d_A, d_A_err, d_B, d_B_err, mux_A, mux_A_err, muy_A, muy_A_err, mux_B, mux_B_err, muy_B, muy_B_err, RV_A, RV_A_err, RV_B, RV_B_err, e, e0, e1

        self.d_M, self.d_M_err, self.d_M0, self.dd, self.mux_M, self.muy_M, self.dRV, self.pmerr, self.derr, self.RVerr = d_M, d_M_err, d_M0, dd, mux_M, muy_M, dRV, pmerr, derr, RVerr

    def run_mc(self,mod='g',echoice='i',Gboost=1.):        

        M_A, M_A_1, M_A_2, M_B, M_B_1, M_B_2, rp, d_A, d_A_err, d_B, d_B_err, mux_A, mux_A_err, muy_A, muy_A_err, mux_B, mux_B_err, muy_B, muy_B_err, RV_A, RV_A_err, RV_B, RV_B_err, e, e0, e1 = self.M_A, self.M_A_1, self.M_A_2, self.M_B, self.M_B_1, self.M_B_2, self.rp, self.d_A, self.d_A_err, self.d_B, self.d_B_err, self.mux_A, self.mux_A_err, self.muy_A, self.muy_A_err, self.mux_B, self.mux_B_err, self.muy_B, self.muy_B_err, self.RV_A, self.RV_A_err, self.RV_B, self.RV_B_err, self.e, self.e0, self.e1
        d_M, d_M_err, d_M0, dd, mux_M, muy_M, dRV, pmerr, derr, RVerr = self.d_M, self.d_M_err, self.d_M0, self.dd, self.mux_M, self.muy_M, self.dRV, self.pmerr, self.derr, self.RVerr

        if mod == 'c':
            model = 'circular'
        elif mod == 'e':
            model = 'eccentric'
        else:
            model = 'gaia'        

        if model != 'gaia':  # replace proper motions with simulated data
            xx = []
            for i in range(len(rp)):
                xx.append([rp[i],M_A_1[i],M_A_2[i],M_B_1[i],M_B_2[i],d_A[i],d_A_err[i],d_B[i],d_B_err[i],mux_M[i],muy_M[i],mux_A_err[i],muy_A_err[i],mux_B_err[i],muy_B_err[i],e[i],e0[i],e1[i],model,echoice,Gboost,self.aLM])
            pool = mp.Pool(mp.cpu_count()-1)
            res = pool.map(Newton_simulation, xx)
            for i in range(len(rp)):
                d_M[i],d_M_err[i],mux_A[i],muy_A[i],mux_B[i],muy_B[i] = res[i]
            
        fac, G, Msun, au = self.fac, self.G, self.Msun, self.au
        d_min, d_max, rp_min, rp_max, sigd, nprob = self.d_min, self.d_max, self.rp_min, self.rp_max, self.sigd, self.nprob

        v = fac*d_M*np.sqrt((mux_A-mux_B)**2+(muy_A-muy_B)**2)
        vc = 1.e-3*np.sqrt(G*Msun*(M_A+M_B)/(rp*au*1000)) # km/s
        vtilde = v/vc
        delmu = np.sqrt((mux_A-mux_B)**2+(muy_A-muy_B)**2)
        delmu_x, delmu_y = np.abs(mux_A-mux_B), np.abs(muy_A-muy_B)
        sigv = fac*d_M*np.sqrt((mux_A_err**2+mux_B_err**2)*delmu_x**2+(muy_A_err**2+muy_B_err**2)*delmu_y**2)/delmu
        sigvtilde = vtilde*np.sqrt((sigv/v)**2+0.05**2)

        Mtot = M_A+M_B
        mask1 = (rp>rp_min) & (rp<rp_max) & (d_M0>d_min) & (d_M0<d_max) & (dRV < self.dRV_max) & (dd < self.dd_max) & (pmerr < self.pmerr_max)  & (derr < self.derr_max)  & (RVerr < self.RVerr_max) & (sigvtilde < self.vterr_max*np.maximum(1,vtilde/2)) & (sigv/v < self.vperr_max)
        
        M_A=M_A[mask1]
        M_B=M_B[mask1]
        rp=rp[mask1]
        v=v[mask1]
        d_A=d_A[mask1]
        pmerr = pmerr[mask1]
        e = e[mask1]
        e0 = e0[mask1]
        e1 = e1[mask1]

        if mod == 'g':
            print('N_selected=',len(M_A))

        gN, g, logr = np.zeros(0), np.zeros(0), np.zeros(0)
        for j in range(nprob):
            gN_sim, g_sim, logr_sim = np.zeros(len(rp)), np.zeros(len(rp)), np.zeros(len(rp))
            xx = []
            for i in range(len(rp)):
                xx.append([rp[i],M_A[i],M_B[i],v[i],e[i],e0[i],e1[i],model,echoice,Gboost])
            pool = mp.Pool(mp.cpu_count()-1)
            res = pool.map(Newton_2d_to_3d, xx)
            for i in range(len(rp)):
                gN_sim[i],g_sim[i],logr_sim[i] = res[i]
            gN = np.concatenate((gN,gN_sim))
            g = np.concatenate((g,g_sim))
            logr = np.concatenate((logr,logr_sim))

        dy = g - gN
        x0 = gN + dy/2
        d_orth = dy/np.sqrt(2)

        nsam=200
        Ntot=len(x0)
        xbin=self.x0bin
        nbin =len(xbin)-1

        running_median_x=np.zeros(nbin)
        running_median_y=np.zeros(nbin)
        running_median_y_err=np.zeros(nbin)
        running_sample_n=np.zeros(nbin)
        running_median_rp=np.zeros(nbin)
        running_median_M=np.zeros(nbin)
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

        running_median_x0 = running_median_x
        running_median_dorth = running_median_y
        running_median_dorth_err = running_median_y_err
        running_median_x = running_median_x0 - running_median_dorth/np.sqrt(2)
        running_median_y = running_median_x0 + running_median_dorth*np.sqrt(2)
        running_median_y_err = running_median_dorth_err*np.sqrt(2)
        return running_median_x0, running_median_dorth, running_median_dorth_err,running_sample_n
 
def dsq(p,*args):
    phi = p
    t, eps = args
    f = lambda x: 1/(1+eps*np.cos(x))**2
    val, err = integrate.quad(f, 0, phi)
    res = (t-val)**2
    return res

def Newton_simulation(xx):
    np.random.seed(int(str(int(time.time()*1.e7))[-7:]))
    rp_sim,M_A_1_sim,M_A_2_sim,M_B_1_sim,M_B_2_sim,d_A_sim,d_A_err_sim,d_B_sim,d_B_err_sim,mux_M_sim,muy_M_sim,mux_A_err_sim,muy_A_err_sim,mux_B_err_sim,muy_B_err_sim,e_sim,e0_sim,e1_sim,modelx,echoicex,Gboostx,aLM=xx
    fac = 1.e-3*(1/(648000/np.pi))*3.085677e13/3.15582e7  # pc*mas/yr  to  km/s
    G = 6.6743e-11*Gboostx     # Newton's constant
    Msun = 1.9884e30   # Msolar in kg
    au = 1.496e11      # AU in m
    model = modelx
    echoice = echoicex
    # main outer orbit: motion of two barycenters
    s = rp_sim
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
    mux_A_sim += v_CL_x_A/(fac*d_A_sim)
    muy_A_sim += v_CL_y_A/(fac*d_A_sim)
    mux_B_sim += v_CL_x_B/(fac*d_B_sim)
    muy_B_sim += v_CL_y_B/(fac*d_B_sim)    

    values = d_M_sim,d_M_err_sim,mux_A_sim,muy_A_sim,mux_B_sim,muy_B_sim
    return values

def Newton_2d_to_3d(xx):
    np.random.seed(int(str(int(time.time()*1.e7))[-7:]))
    rp, M_A, M_B, vp, e_sim, e0_sim, e1_sim, modelx, echoicex, Gboostx = xx
    G=6.6743e-11*Gboostx     # Newton's constant
    Msun=1.989e30    # Msolar in kg
    au=1.4959787e11  # AU in m
    model = modelx
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
    values = gN, g, logr
    return values
