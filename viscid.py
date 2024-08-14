#!/usr/bin/env python
import numpy as np
import argparse
import json
import os
import timeit
import sys
from scipy.linalg import eig,lu_factor,lu_solve
from scipy.special import iv
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

def makeindices (argsdict):
    lps = np.arange(-argsdict['Nt'], argsdict['Nt'] + 1)[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    ls = np.arange(-argsdict['Nt'], argsdict['Nt'] + 1)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
    mps = np.arange(-argsdict['Nx'], argsdict['Nx'] + 1)[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    ms = np.arange(-argsdict['Nx'], argsdict['Nx'] + 1)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    nps = np.arange(-argsdict['Ny'], argsdict['Ny'] + 1)[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    ns = np.arange(-argsdict['Ny'], argsdict['Ny'] + 1)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]

    indices=[lps,ls,mps,ms,nps,ns]

    return indices

def viscid_mat2d (omega, argsdict):
    lps,ls,mps,ms,nps,ns=makeindices(argsdict)
    kappax = (argsdict['kx'] + argsdict['k1x']*ms + argsdict['k2x']*ns)
    kappay = (argsdict['ky'] + argsdict['k1y']*ms + argsdict['k2y']*ns)
    kappa = (kappax**2+kappay**2)**0.5
    kappapx = (argsdict['kx'] + argsdict['k1x']*mps + argsdict['k2x']*nps)
    kappapy = (argsdict['ky'] + argsdict['k1y']*mps + argsdict['k2y']*nps)
    kappap = (kappapx**2+kappapy**2)**0.5
    Omega = (1j*(omega + 2*np.pi*argsdict['freq']*ls) + argsdict['mu']/argsdict['rho']*kappa**2)
    EpOmega=np.exp((argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0'])
    EmOmega=np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0'])

    C = (np.exp(-kappa*argsdict['h0']) * iv(ms-mps, kappa*argsdict['As']*0.5) * iv(ns-nps, kappa*argsdict['As']*0.5) + np.exp(kappa*argsdict['h0']) * iv(ms-mps, -kappa*argsdict['As']*0.5) * iv(ns-nps, -kappa*argsdict['As']*0.5))
    S = (- np.exp(-kappa*argsdict['h0']) * iv(ms-mps, kappa*argsdict['As']*0.5) * iv(ns-nps, kappa*argsdict['As']*0.5) + np.exp(kappa*argsdict['h0']) * iv(ms-mps, -kappa*argsdict['As']*0.5) * iv(ns-nps, -kappa*argsdict['As']*0.5))


    Ctilde = (EmOmega * iv(ms-mps, (argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5) * iv(ns-nps, (argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5) + EpOmega * iv(ms-mps, -(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5) * iv(ns-nps, -(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5))
    Stilde = (- EmOmega * iv(ms-mps, (argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5) * iv(ns-nps, (argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5) + EpOmega * iv(ms-mps, -(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5) * iv(ns-nps, -(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5))

    #kp k
    vals00 =((1-argsdict['mu']/(argsdict['rho']*Omega)*kappax*(kappax-kappapx))*Ctilde - (2*argsdict['mu']*kappax*kappapx*C)/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))
    vals01 = -(2*argsdict['mu']*kappapx*kappay*C/(argsdict['rho']*Omega+argsdict['mu']*kappa**2)+argsdict['mu']/(argsdict['rho']*Omega)*kappay*(kappax-kappapx)*Ctilde)
    vals02 = -1j*((argsdict['mu']*(argsdict['rho']/argsdict['mu']*Omega)**0.5*kappax-argsdict['mu']*kappa**2*(kappax-kappapx)/(argsdict['rho']/argsdict['mu']*Omega)**0.5) * (Ctilde-Stilde) + (argsdict['rho']*Omega+argsdict['mu']*kappa**2)*kappapx*S/(2*kappa) + (argsdict['g']*argsdict['rho']**2+kappa**2*(argsdict['rho']*argsdict['sigma'] - 4*argsdict['mu']**2*(argsdict['rho']*Omega/argsdict['mu'])**0.5))*kappapx*C/(2*(argsdict['rho']*Omega+argsdict['mu']*kappa**2)))

    vals10 = -(2*argsdict['mu']*kappapy*kappax*C/(argsdict['rho']*Omega+argsdict['mu']*kappa**2)+argsdict['mu']/(argsdict['rho']*Omega)*kappax*(kappay-kappapy)*Ctilde)
    vals11 = ((1-argsdict['mu']/(argsdict['rho']*Omega)*kappay*(kappay-kappapy))*Ctilde - (2*argsdict['mu']*kappay*kappapy*C)/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))
    vals12 = -1j*((argsdict['mu']*(argsdict['rho']/argsdict['mu']*Omega)**0.5*kappay-argsdict['mu']*kappa**2*(kappay-kappapy)/(argsdict['rho']/argsdict['mu']*Omega)**0.5) * (Ctilde-Stilde) + (argsdict['rho']*Omega+argsdict['mu']*kappa**2)*kappapy*S/(2*kappa) + (argsdict['g']*argsdict['rho']**2+kappa**2*(argsdict['rho']*argsdict['sigma'] - 4*argsdict['mu']**2*(argsdict['rho']*Omega/argsdict['mu'])**0.5))*kappapy*C/(2*(argsdict['rho']*Omega+argsdict['mu']*kappa**2)))

    vals20 = 1j*(-2*argsdict['mu']*kappax*(kappax*kappapx+kappay*kappapy)/kappa*S/(argsdict['rho']*Omega+argsdict['mu']*kappa**2)+kappapx*Stilde/(argsdict['rho']*Omega/argsdict['mu'])**0.5)
    vals21 = 1j*(-2*argsdict['mu']*kappay*(kappax*kappapx+kappay*kappapy)/kappa*S/(argsdict['rho']*Omega+argsdict['mu']*kappa**2)+kappapy*Stilde/(argsdict['rho']*Omega/argsdict['mu'])**0.5)
    vals22 = (kappax*kappapx+kappay*kappapy)*(-argsdict['mu']*(Ctilde-Stilde)+(argsdict['rho']*Omega + argsdict['mu']*kappa**2)*C/(2*kappa**2)+(argsdict['g']*argsdict['rho']**2+kappa**2*(argsdict['rho']*argsdict['sigma']-4*argsdict['mu']**2*(argsdict['rho'] * Omega/argsdict['mu'])**0.5))*S/(2*kappa*(argsdict['rho']*Omega+argsdict['mu']*kappa**2)))

    # rfac0=EmOmega*(argsdict['rho']*Omega+argsdict['mu']*kappa**2)
    # rfac1=EmOmega*(argsdict['rho']*Omega+argsdict['mu']*kappa**2)
    # rfac2=4*(argsdict['rho']*Omega + argsdict['mu']*kappa**2)
    rfac0=EmOmega
    rfac1=EmOmega
    rfac2=1
    E=np.zeros((3,2*argsdict['Nt']+1,2*argsdict['Nx']+1,2*argsdict['Ny']+1,3,2*argsdict['Nt']+1,2*argsdict['Nx']+1,2*argsdict['Ny']+1),dtype=np.complex128)
    for lind in range(2*argsdict['Nt']+1):
        E[0,lind,:,:,0,lind,:,:]=(vals00*rfac0)[0,0,:,:,0,lind,:,:]
        E[0,lind,:,:,1,lind,:,:]=(vals01*rfac1)[0,0,:,:,0,lind,:,:]
        E[0,lind,:,:,2,lind,:,:]=(vals02*rfac2)[0,0,:,:,0,lind,:,:]
        E[1,lind,:,:,0,lind,:,:]=(vals10*rfac0)[0,0,:,:,0,lind,:,:]
        E[1,lind,:,:,1,lind,:,:]=(vals11*rfac1)[0,0,:,:,0,lind,:,:]
        E[1,lind,:,:,2,lind,:,:]=(vals12*rfac2)[0,0,:,:,0,lind,:,:]
        E[2,lind,:,:,0,lind,:,:]=(vals20*rfac0)[0,0,:,:,0,lind,:,:]
        E[2,lind,:,:,1,lind,:,:]=(vals21*rfac1)[0,0,:,:,0,lind,:,:]
        E[2,lind,:,:,2,lind,:,:]=(vals22*rfac2)[0,0,:,:,0,lind,:,:]

    vals02 = -1j*(argsdict['rho']**2*kappapx*C/(4*(argsdict['rho']*Omega+argsdict['mu']*kappa**2)))
    vals12 = -1j*(argsdict['rho']**2*kappapy*C/(4*(argsdict['rho']*Omega + argsdict['mu']*kappa**2)))
    vals22 = (argsdict['rho']**2*(kappax*kappapx+kappay*kappapy)*S/(4*kappa*(argsdict['rho']*Omega + argsdict['mu']*kappa**2)))
    F=np.zeros((3,2*argsdict['Nt']+1,2*argsdict['Nx']+1,2*argsdict['Ny']+1,3,2*argsdict['Nt']+1,2*argsdict['Nx']+1,2*argsdict['Ny']+1),dtype=np.complex128)
    for lind in range(2*argsdict['Nt']):
        F[0,lind,:,:,2,lind+1,:,:]+=argsdict['g']*(vals02*rfac2)[0,0,:,:,0,lind+1,:,:]
        F[1,lind,:,:,2,lind+1,:,:]+=argsdict['g']*(vals12*rfac2)[0,0,:,:,0,lind+1,:,:]
        F[2,lind,:,:,2,lind+1,:,:]+=argsdict['g']*(vals22*rfac2)[0,0,:,:,0,lind+1,:,:]
        F[0,lind+1,:,:,2,lind,:,:]+=argsdict['g']*(vals02*rfac2)[0,0,:,:,0,lind,:,:]
        F[1,lind+1,:,:,2,lind,:,:]+=argsdict['g']*(vals12*rfac2)[0,0,:,:,0,lind,:,:]
        F[2,lind+1,:,:,2,lind,:,:]+=argsdict['g']*(vals22*rfac2)[0,0,:,:,0,lind,:,:]
    n_flat=np.prod((3,2*argsdict['Nt']+1,2*argsdict['Nx']+1,2*argsdict['Ny']+1))
    return E.reshape((n_flat,n_flat)),F.reshape((n_flat,n_flat))

def inviscid_mat_undriven (argsdict):
    lps,ls,mps,ms,nps,ns=makeindices(argsdict)

    kappax = argsdict['kx'] + argsdict['k1x']*ms + argsdict['k2x']*ns
    kappay = argsdict['ky'] + argsdict['k1y']*ms + argsdict['k2y']*ns
    kappapx = argsdict['kx'] + argsdict['k1x']*mps + argsdict['k2x']*nps
    kappapy = argsdict['ky'] + argsdict['k1y']*mps + argsdict['k2y']*nps
    kappa = (kappax**2+kappay**2)**0.5

    C = (np.exp(-kappa*argsdict['h0']) * iv(ms-mps, kappa*argsdict['As']*0.5) * iv(ns-nps, kappa*argsdict['As']*0.5) + np.exp(kappa*argsdict['h0']) * iv(ms-mps, -kappa*argsdict['As']*0.5) * iv(ns-nps, -kappa*argsdict['As']*0.5))
    S = (- np.exp(-kappa*argsdict['h0']) * iv(ms-mps, kappa*argsdict['As']*0.5) * iv(ns-nps, kappa*argsdict['As']*0.5) + np.exp(kappa*argsdict['h0']) * iv(ms-mps, -kappa*argsdict['As']*0.5) * iv(ns-nps, -kappa*argsdict['As']*0.5))

    F=kappa*(argsdict['g']+argsdict['sigma']/argsdict['rho']*kappa**2)*(1-(kappax*(kappax-kappapx)+kappay*(kappay-kappapy))/kappa**2)*S
    G=(1-(kappax*(kappax-kappapx)+kappay*(kappay-kappapy))/kappa**2)*C
    n_flat=np.prod((1,1,2*argsdict['Nx']+1,2*argsdict['Ny']+1))

    return F.reshape((n_flat,n_flat)),G.reshape((n_flat,n_flat))

def inviscid_mat (argsdict):
    lps,ls,mps,ms,nps,ns=makeindices(argsdict)

    kappax = argsdict['kx'] + argsdict['k1x']*ms + argsdict['k2x']*ns
    kappay = argsdict['ky'] + argsdict['k1y']*ms + argsdict['k2y']*ns
    kappapx = argsdict['kx'] + argsdict['k1x']*mps + argsdict['k2x']*nps
    kappapy = argsdict['ky'] + argsdict['k1y']*mps + argsdict['k2y']*nps
    kappa = (kappax**2+kappay**2)**0.5

    C = (np.exp(-kappa*argsdict['h0']) * iv(ms-mps, kappa*argsdict['As']*0.5) * iv(ns-nps, kappa*argsdict['As']*0.5) + np.exp(kappa*argsdict['h0']) * iv(ms-mps, -kappa*argsdict['As']*0.5) * iv(ns-nps, -kappa*argsdict['As']*0.5))
    S = (- np.exp(-kappa*argsdict['h0']) * iv(ms-mps, kappa*argsdict['As']*0.5) * iv(ns-nps, kappa*argsdict['As']*0.5) + np.exp(kappa*argsdict['h0']) * iv(ms-mps, -kappa*argsdict['As']*0.5) * iv(ns-nps, -kappa*argsdict['As']*0.5))

    F=kappa*(argsdict['g']+argsdict['sigma']/argsdict['rho']*kappa**2)*(1-(kappax*(kappax-kappapx)+kappay*(kappay-kappapy))/kappa**2)*S
    G=(1-(kappax*(kappax-kappapx)+kappay*(kappay-kappapy))/kappa**2)*C
    F2=kappa*(argsdict['g'])*(1-(kappax*(kappax-kappapx)+kappay*(kappay-kappapy))/kappa**2)*S

    n_flat=np.prod((1,2*argsdict['Nt']+1,2*argsdict['Nx']+1,2*argsdict['Ny']+1))

    E1=np.zeros((1,2*argsdict['Nt']+1,2*argsdict['Nx']+1,2*argsdict['Ny']+1,1,2*argsdict['Nt']+1,2*argsdict['Nx']+1,2*argsdict['Ny']+1),dtype=np.complex128)
    E2=np.zeros((1,2*argsdict['Nt']+1,2*argsdict['Nx']+1,2*argsdict['Ny']+1,1,2*argsdict['Nt']+1,2*argsdict['Nx']+1,2*argsdict['Ny']+1),dtype=np.complex128)
    E3=np.zeros((1,2*argsdict['Nt']+1,2*argsdict['Nx']+1,2*argsdict['Ny']+1,1,2*argsdict['Nt']+1,2*argsdict['Nx']+1,2*argsdict['Ny']+1),dtype=np.complex128)

    for lind in range(2*argsdict['Nt']+1):
        E1[0,lind,:,:,0,lind,:,:]=F[0,0,:,:,0,0,:,:]
        E2[0,lind,:,:,0,lind,:,:]=G[0,0,:,:,0,0,:,:]
    for lind in range(2*argsdict['Nt']):
        E3[0,lind,:,:,0,lind+1,:,:]=F2[0,0,:,:,0,0,:,:]
        E3[0,lind+1,:,:,0,lind,:,:]=F2[0,0,:,:,0,0,:,:]

    return E1.reshape((n_flat,n_flat)),E2.reshape((n_flat,n_flat)),E3.reshape((n_flat,n_flat))

#Rayleigh quotient iterations
def rayleigh_mat(omega_0, v0, w0, mat, argsdict, mat2=None):
    vn=v0
    wn=w0

    omegas=[omega_0]
    vns=[v0]
    wns=[w0]
    omega=omega_0
    for n in range(argsdict['itmax']):
        if mat2 is None:
            E_n=mat(omega, argsdict)
            dE = (mat(omega+argsdict['domega_fd'],argsdict)-mat(omega-argsdict['domega_fd'],argsdict))/(2*argsdict['domega_fd'])
        else:
            E_n=mat(omega, argsdict)
            dE=mat2(omega, argsdict)
        domega=-E_n.dot(vn).dot(wn)/dE.dot(vn).dot(wn)
        lu,piv=lu_factor(E_n)
        xi=lu_solve((lu,piv),-domega*dE.dot(vn))
        zeta=lu_solve((lu,piv),-domega*dE.T.dot(wn),trans=1)

        omega=omega+domega
        vn=xi/np.linalg.norm(xi)
        wn=zeta/np.linalg.norm(zeta)

        omegas=omegas+[omega]
        vns=vns+[vn]
        wns=wns+[wn]

        E_n=mat(omega, argsdict)
        dv=np.abs(vns[n+1])-np.abs(vns[n])
        dw=np.abs(wns[n+1])-np.abs(wns[n])
        verr=np.linalg.norm(dv,ord=np.inf)/(1+np.linalg.norm(vn,ord=np.inf))
        werr=np.linalg.norm(dw,ord=np.inf)/(1+np.linalg.norm(wn,ord=np.inf))
        omegaerr=np.abs(domega)/(1+np.abs(omega))
        if argsdict['verbose']>0:
            print("n=%i dv=%e dw=%e dl=%e"%(n, verr, werr, omegaerr))
        if verr<argsdict['epsu'] and werr<argsdict['epsu'] and omegaerr<argsdict['epsl']:
            break

    return omegas,vns,wns

#pseudoarclength continuation with rayleigh quotient refinement
def pseudocont(omega, v, w, mat, argsdict, mat2=None, mat3=None, M=None, dir=None):
    # if argsdict['verbose']:
    #     print(argsdict)
    omegans=[omega]
    vns=[v.copy()]
    wns=[w.copy()]
    vni=v.copy()
    wni=w.copy()
    parns=[argsdict[argsdict['par']]]
    ds = argsdict['ds']
    dss=[ds]


    def makesys(omega,argsdict):
        E_n=mat(omega,argsdict)
        if mat2 is None:
            E_omega = (mat(omega+argsdict['domega_fd'],argsdict)-mat(omega-argsdict['domega_fd'],argsdict))/(2*argsdict['domega_fd'])
        else:
            E_omega = mat2(omega,argsdict)
        if mat3 is None:
            par0 = argsdict[argsdict['par']]
            argsdict[argsdict['par']] = par0+argsdict['dmu_fd']
            E_mup = mat(omega,argsdict)
            argsdict[argsdict['par']] = par0-argsdict['dmu_fd']
            E_mum = mat(omega,argsdict)
            E_mu = (E_mup-E_mum)/(2*argsdict['dmu_fd'])
            argsdict[argsdict['par']] = par0
        else:
            E_mu = mat3(omega,argsdict)
        return E_n,E_omega,E_mu

    E_n,E_omega,E_mu=makesys(omega,argsdict)

    s=E_n.shape[0]
    if M is None:
        M=np.zeros(2*s+3)
        M[2*s:]=argsdict['thl']
        M[:2*s]=argsdict['thu']

    def makejac(v,T,Tlambdav,Tmuv,dir):
        C=np.zeros((2*s+3,2*s+3),dtype=np.float64)
        C[:s,:s]=np.real(T)
        C[s:2*s,:s]=np.imag(T)
        C[2*s,:s]=np.real(v)
        C[2*s+1,:s]=np.imag(v)
        C[2*s+2,:s]=dir[:s]*M[:s]

        C[:s,s:2*s]=-np.imag(T)
        C[s:2*s,s:2*s]=np.real(T)
        C[2*s,s:2*s]=-np.imag(v)
        C[2*s+1,s:2*s]=np.real(v)
        C[2*s+2,s:2*s]=dir[s:2*s]*M[s:2*s]

        C[:s,2*s]=np.real(Tlambdav)
        C[s:2*s,2*s]=np.imag(Tlambdav)
        C[2*s+2,2*s]=dir[2*s]*M[2*s]

        C[:s,2*s+1]=-np.imag(Tlambdav)
        C[s:2*s,2*s+1]=np.real(Tlambdav)
        C[2*s+2,2*s+1]=dir[2*s+1]*M[2*s+1]

        C[:s,2*s+2]=np.real(Tmuv)
        C[s:2*s,2*s+2]=np.imag(Tmuv)
        C[2*s+2,2*s+2]=dir[2*s+2]*M[2*s+2]
        lu,piv=lu_factor(C)
        return C,lu,piv


    dir0=np.zeros(2*s+3,dtype=np.float64)
    dir0[-1]=1
    b=np.zeros((2*s+3),dtype=np.float64)
    delta=np.zeros(2*s+3,dtype=np.float64)

    C,lu,piv=makejac(np.conjugate(vni),E_n,E_omega.dot(vni),E_mu.dot(vni),dir0)

    if dir is None:
        b[:-1]=0
        b[-1]=1
        dir0=lu_solve((lu,piv),b)
        dir0=dir0/(dir0.dot(M*dir0))**0.5
        if argsdict['verbose']>0:
            print("newdir=(%.3f,%.3f,%.3f)"%(dir0[2*s],dir0[2*s+1],dir0[2*s+2]))

        dirs=[dir0.copy()]
    else:
        dir0=dir.copy()
        # dir0=dir0/(dir0.dot(M*dir0))**0.5
        dirs=[dir0.copy()]

    scount=0
    steps=0
    try:
        while (argsdict[argsdict['par']]-argsdict['pari'])/(argsdict['parf']-argsdict['pari'])<=1 and (argsdict[argsdict['par']]-argsdict['pari'])/(argsdict['parf']-argsdict['pari'])>=0 and steps<argsdict['Nsteps'] and np.linalg.norm(omega)<argsdict['lambdamax']:
            if argsdict['verbose']>0:
                print("step=%i ds=%.6e %s=%.6f lambda=%.6f+i%.6f dir=(%.3f, %.3f, %.3f)"%(steps, ds, argsdict['par'], parns[-1], np.real(omegans[-1]),np.imag(omegans[-1]),dir0[2*s],dir0[2*s+1],dir0[2*s+2]))
            # predictor
            domega=(dir0[2*s]+1j*dir0[2*s+1])*ds
            dmu=dir0[2*s+2]*ds
            dv=(dir0[:s]+1j*dir0[s:2*s])*ds
            vni=vns[-1]+dv
            wni=wns[-1].copy()
            omega=omegans[-1]+domega
            argsdict[argsdict['par']]=parns[-1]+dmu
            E_n,E_omega,E_mu=makesys(omega,argsdict)

            omegaerr=np.abs(domega)/(1+np.abs(omega))
            muerr=np.abs(dmu)/(1+np.abs(argsdict[argsdict['par']]))
            verr=np.linalg.norm(dv,ord=np.inf)/(1+np.linalg.norm(vni,ord=np.inf))
            stp=np.concatenate([(np.real(vni-vns[-1])/ds),(np.imag(vni-vns[-1])/ds),[(np.real(omega-omegans[-1])/ds),(np.imag(omega-omegans[-1])/ds),(argsdict[argsdict['par']]-parns[-1])/ds]])
            mstp=(stp.dot(M*stp)**0.5)-(stp.dot(M*dir0))
            if argsdict['verbose']>0:
                print("pre dv=%.3e dlambda=%.3e dmu=%.3e %s=%.6f lr=%.6f li=%.6f mstp=%.3e"%(verr, omegaerr, muerr, argsdict['par'], argsdict[argsdict['par']],np.real(omega),np.imag(omega),mstp))

            verrlast=np.inf
            muerrlast=np.inf
            omegaerrlast=np.inf
            for n in range(argsdict['itmax']):
                delta[:s]=np.real(vni-vns[-1])
                delta[s:2*s]=np.imag(vni-vns[-1])
                delta[2*s]=np.real(omega-omegans[-1])
                delta[2*s+1]=np.imag(omega-omegans[-1])
                delta[2*s+2]=argsdict[argsdict['par']]-parns[-1]

                Tv=E_n.dot(vni)
                b[:s]=-np.real(Tv)
                b[s:2*s]=-np.imag(Tv)
                b[2*s]=0
                b[2*s+1]=0
                b[2*s+2]=ds-delta.dot(M*dir0)
                C,lu,piv=makejac(np.conjugate(vni),E_n,E_omega.dot(vni),E_mu.dot(vni),dir0)
                delta=lu_solve((lu,piv),b)
                # iterative refinement for delta
                deltaerr=0
                lastcor=delta
                lasterr=np.inf
                for m in range(argsdict['itmax']):
                    res=(b.astype(np.float128)-C.astype(np.float128).dot(delta.astype(np.float128))).astype(np.float64)
                    cor=lu_solve((lu,piv),res)
                    deltaerr=np.linalg.norm(cor,ord=np.inf)/(1+np.linalg.norm(delta,ord=np.inf))
                    if argsdict['verbose']>1:
                        print("m=%i deltaerr=%.3e res=%.3e"%(m,deltaerr,np.linalg.norm(res)))
                    if deltaerr<argsdict['epsu'] or deltaerr>lasterr:
                        break
                    delta=delta+cor
                    lasterr=deltaerr

                domega=delta[2*s]+1j*delta[2*s+1]
                dmu=delta[2*s+2]
                dv=delta[:s]+1j*delta[s:2*s]
                vold=vni.copy()
                wold=wni.copy()
                omegaold=omega
                muold=argsdict[argsdict['par']]
                vni=vni+dv
                omega+=domega
                argsdict[argsdict['par']]+=dmu

                E_n,E_omega,E_mu=makesys(omega,argsdict)
                lu2,piv2=lu_factor(E_n)

                # rayleigh quotient refinement at fixed mu
                xi=lu_solve((lu2,piv2),E_omega.dot(vni))
                # iterative refinement for xi
                xierr=0
                lastcor=xi
                lasterr=np.inf
                for m in range(argsdict['itmax']):
                    res=(E_omega.dot(vni).astype(np.complex256)-E_n.astype(np.complex256).dot(xi.astype(np.complex256))).astype(np.complex128)
                    cor=lu_solve((lu2,piv2),res)
                    xierr=np.linalg.norm(cor,ord=np.inf)/(1+np.linalg.norm(xi,ord=np.inf))
                    if argsdict['verbose']>1:
                        print("m=%i xierr=%.3e res=%.3e"%(m,xierr,np.linalg.norm(res)))

                    if xierr<argsdict['epsu'] or xierr>lasterr:
                        break
                    xi=xi+cor
                    lasterr=xierr
                zeta=lu_solve((lu2,piv2),E_omega.T.dot(wni),trans=1)
                # iterative refinement for zeta
                zetaerr=0
                lastcor=zeta
                lasterr=np.inf
                for m in range(argsdict['itmax']):
                    res=(E_omega.T.dot(wni).astype(np.complex256)-E_n.T.astype(np.complex256).dot(zeta.astype(np.complex256))).astype(np.complex128)
                    cor=lu_solve((lu2,piv2),res,trans=1)
                    zetaerr=np.linalg.norm(cor,ord=np.inf)/(1+np.linalg.norm(zeta,ord=np.inf))
                    if argsdict['verbose']>1:
                        print("m=%i zetaerr=%.3e res=%.3e"%(m,zetaerr,np.linalg.norm(res)))

                    if zetaerr<argsdict['epsu'] or zetaerr>lasterr:
                        break
                    zeta=zeta+cor
                    lasterr=zetaerr

                vni=xi/np.linalg.norm(xi)*np.abs(xi.dot(np.conjugate(vni)))/(xi.dot(np.conjugate(vni)))
                wni=zeta/np.linalg.norm(zeta)*np.abs(zeta.dot(np.conjugate(wni)))/(zeta.dot(np.conjugate(wni)))

                omega+=-E_n.dot(vni).dot(wni)/E_omega.dot(vni).dot(wni)
                domega+=-E_n.dot(vni).dot(wni)/E_omega.dot(vni).dot(wni)
                E_n,E_omega,E_mu=makesys(omega,argsdict)

                omegaerr=np.abs(domega)/(1+np.abs(omega))
                muerr=np.abs(dmu)/(1+np.abs(argsdict[argsdict['par']]))
                dv=vni-vold

                verr=np.linalg.norm(dv,ord=np.inf)/(1+np.linalg.norm(vni,ord=np.inf))
                stp=np.concatenate([(np.real(vni-vns[-1])/ds),(np.imag(vni-vns[-1])/ds),[(np.real(omega-omegans[-1])/ds),(np.imag(omega-omegans[-1])/ds),(argsdict[argsdict['par']]-parns[-1])/ds]])
                mstp=(stp.dot(M*stp)**0.5)-(stp.dot(M*dir0))

                if argsdict['verbose']>0:
                    print("n=%i dv=%.3e dlambda=%.3e dmu=%.3e %s=%.6f lr=%.6f li=%.6f mstp=%.3e"%(n, verr, omegaerr, muerr, argsdict['par'], argsdict[argsdict['par']],np.real(omega),np.imag(omega),mstp))
                if verr<argsdict['epsu'] and omegaerr<argsdict['epsl'] and muerr<argsdict['epsl'] and (steps==0 or np.abs(mstp)<argsdict['epstp']):
                    break

                if (steps>0 and np.abs(mstp)>argsdict['epstp']):
                    if argsdict['verbose']>1:
                        print('Reject')
                    vni=vold
                    wni=wold
                    omega=omegaold
                    argsdict[argsdict['par']]=muold
                    E_n,E_omega,E_mu=makesys(omega,argsdict)
                    ds=ds*(1+np.random.normal(0,argsdict['epsl']))
                    continue

                # if n>2 and ((verr>5*verrlast and verr>argsdict['epsu']) or (muerr>5*muerrlast and muerr>argsdict['epsl']) or (omegaerr>5*omegaerrlast and omegaerr>argsdict['epsl'])):
                if n>2 and ((muerr>5*muerrlast and muerr>argsdict['epsl']) or (omegaerr>5*omegaerrlast and omegaerr>argsdict['epsl'])):
                    break

                verrlast=verr
                muerrlast=muerr
                omegaerrlast=omegaerr

            if verr<argsdict['epsu'] and omegaerr<argsdict['epsl'] and muerr<argsdict['epsl'] and (steps==0 or np.abs(mstp)<argsdict['epstp']):

                E_n,E_omega,E_mu=makesys(omega,argsdict)
                C,lu,piv=makejac(np.conjugate(vni),E_n,E_omega.dot(vni),E_mu.dot(vni),dir0)
                b[:-1]=0
                b[-1]=1
                dir0=lu_solve((lu,piv),b)

                # iterative refinement for direction vector
                direrr=0
                lastcor=dir0
                lasterr=np.inf

                for m in range(argsdict['itmax']):
                    res=(b.astype(np.float128)-C.astype(np.float128).dot(dir0.astype(np.float128))).astype(np.float64)
                    cor=lu_solve((lu,piv),res)
                    direrr=np.linalg.norm(M*cor,ord=np.inf)/(1+np.linalg.norm(M*dir0,ord=np.inf))
                    if argsdict['verbose']>1:
                        print("m=%i direrr=%.3e res=%.3e"%(m,direrr,np.linalg.norm(res)))

                    if direrr<argsdict['epdir'] or direrr>lasterr:
                        break
                    dir0=dir0+cor
                    lasterr=direrr

                dir0=dir0/(dir0.dot(M*dir0))**0.5*np.sign(dirs[-1].dot(M*dir0))
                dir1=stp/(stp.dot(M*stp))**0.5*np.sign(dirs[-1].dot(M*stp))
                ddir=dir1-dir0
                # deltadir=np.abs(dir0.dot(ddir)/np.linalg.norm(dir0)**2)
                deltadir=dir0.dot(M*ddir)
                if argsdict['verbose']:
                    print('deltadir=%.3e stp=(%.3f,%.3f,%.3f) newdir=(%.3f, %.3f, %.3f)'%(deltadir,dir1[-3],dir1[-2],dir1[-1],dir0[2*s],dir0[2*s+1],dir0[2*s+2]))

                if direrr>argsdict['epdir'] or np.abs(deltadir)>argsdict['epdir']:
                    # if poor direction vector convergence, use last stp
                    # dir0=(dirs[-1]+dir1)/2 #Bias to last step...bad for curvature
                    # dir0=2*dir1-dirs[-1] #Try to extrapolate...may be unstable
                    dir0=argsdict['stpweight']*dir1+(1-argsdict['stpweight'])*dirs[-1]
                    dir0=dir0/(dir0.dot(M*dir0))**0.5*np.sign(dirs[-1].dot(M*dir0))

                steps+=1
                omegans=omegans+[omega]
                vns=vns+[vni.copy()]
                wns=wns+[wni.copy()]
                dirs=dirs+[dir0.copy()]
                dss=dss+[ds]
                parns=parns+[argsdict[argsdict['par']]]
                scount=scount+1
                if scount>1 and n<5:
                    if np.abs(ds)*2<=argsdict['dsmax']:
                        ds=ds*1.5
                        scount=0
                    else:
                        ds=np.sign(ds)*argsdict['dsmax']
                        scount=0

                if argsdict['verbose']>0:
                    print("ds=%.3e %s=%.6f lambda=%.6f+%.6fi stp=(%.3f,%.3f,%.3f) mstp=%.3e"%(ds, argsdict['par'], argsdict[argsdict['par']], np.real(omegans[-1]),np.imag(omegans[-1]), stp[-3],stp[-2],stp[-1],mstp))

            else:
                scount=0
                argsdict[argsdict['par']] = parns[-1]
                omega = omegans[-1]
                vni = vns[-1].copy()
                wni = wns[-1].copy()
                E_n,E_omega,E_mu=makesys(omega,argsdict)

                ds=ds/1.5
                if argsdict['verbose']>0:
                    print('Convergence failure!')
                    print("ds=%.3e dv=%.3e dlambda=%.3e dmu=%.3e stp=(%.3f,%.3f,%.3f) mstp=%.3e"%(ds, verr, omegaerr, muerr, stp[-3],stp[-2],stp[-1],mstp))

                if np.abs(ds)<argsdict['dsmin']:
                    if argsdict['verbose']>0:
                        print('Minimum step!')
                    return omegans,vns,wns,parns,dirs,dss

    except KeyboardInterrupt:
        print("Keyboard interrupt!")

    return omegans,vns,wns,parns,dirs,dss

#Command line arguments
parser = argparse.ArgumentParser(description='Find the Floquet exponents for a given drive and wavenumber.')
parser.add_argument("--frequency", type=float, required=False, default=1.3, dest='freq', help='Driving frequency (Hz)')
parser.add_argument("--ad", type=float, required=False, default=0.0, dest='ad', help='Driving acceleration (in gravitational units)')
parser.add_argument("--viscosity", type=float, required=False, default=0.0089, dest='mu', help='Viscosity (cgs units)')
parser.add_argument("--density", type=float, required=False, default=1.0, dest='rho', help='Fluid density (cgs units)')
parser.add_argument("--gravity", type=float, required=False, default=980, dest='g', help='Gravitational acceleration (cgs units)')
parser.add_argument("--tension", type=float, required=False, default=72, dest='sigma', help='Surface tension (cgs units)')
parser.add_argument("--kx", type=float, required=False, default=0.05*np.pi, dest='kx', help='Wave vector x component')
parser.add_argument("--ky", type=float, required=False, default=0, dest='ky', help='Wave vector y component')
parser.add_argument("--height", type=float, required=False, default=1.0, dest='h0', help='Fluid depth')
parser.add_argument("--As", type=float, required=False, default=0.8, dest='As', help='Substrate height')
parser.add_argument("--k1x", type=float, required=False, default=0.1*np.pi, dest='k1x', help='Second reciprocal lattice vector x component')
parser.add_argument("--k1y", type=float, required=False, default=0, dest='k1y', help='Second reciprocal lattice vector y component')
parser.add_argument("--k2x", type=float, required=False, default=-0.05*np.pi, dest='k2x', help='First reciprocal lattice vector x component')
parser.add_argument("--k2y", type=float, required=False, default=3**0.5*0.05*np.pi, dest='k2y', help='First reciprocal lattice vector y component')
parser.add_argument("--Nt", type=int, required=False, default=3, dest='Nt', help='Number of modes to include the Floquet expansion for time')
parser.add_argument("--Nx", type=int, required=False, default=3, dest='Nx', help='Number of modes to include the Floquet expansion for spatial x')
parser.add_argument("--Ny", type=int, required=False, default=3, dest='Ny', help='Number of modes to include the Floquet expansion for spatial y')
parser.add_argument("--itmax", type=int, required=False, default=10, dest='itmax', help='Number of iterators in acceleration continuation.')
parser.add_argument("--ds", type=float, required=False, default=5e-3, dest='ds', help='Initial parameter change')
parser.add_argument("--dsmax", type=float, required=False, default=5e-3, dest='dsmax', help='Max parameter change')
parser.add_argument("--dsmin", type=float, required=False, default=1e-8, dest='dsmin', help='Min parameter change')
parser.add_argument("--Nsteps", type=int, required=False, default=1000, dest='Nsteps', help='Initial parameter change')
parser.add_argument("--domega_fd", type=float, required=False, default=1e-3, dest='domega_fd', help='Finite difference step')
parser.add_argument("--dmu_fd", type=float, required=False, default=1e-3, dest='dmu_fd', help='Finite difference step')
parser.add_argument("--epsu", type=float, required=False, default=1e-3, dest='epsu', help='Newton tolerance for state variables')
parser.add_argument("--epsl", type=float, required=False, default=1e-4, dest='epsl', help='Newton tolerance for parameters')
parser.add_argument("--lambdamax", type=float, required=False, default=np.inf, dest='lambdamax', help='Maximum eigenvalue norm')
parser.add_argument("--epstp", type=float, required=False, default=1E0, dest='epstp', help='Step direction tolerance')
parser.add_argument("--epdir", type=float, required=False, default=1E-3, dest='epdir', help='Direction vector tolerance')
parser.add_argument("--thl", type=float, required=False, default=1, dest='thl', help='Pseudoarclength weight for parameters and eigenvalues.')
parser.add_argument("--thu", type=float, required=False, default=1E-3, dest='thu', help='Pseudoarclength weight for eigenvectors.')
parser.add_argument("--stpweight", type=float, required=False, default=1, dest='stpweight', help='Weight for previous step.')
parser.add_argument("--verbose", type=int, required=False, default=True,
dest='verbose', help='Print interation details')
parser.add_argument("--par", type=str, required=False, default='ad', dest='par', help='Continuation parameter')
parser.add_argument("--pari", type=float, required=False, default=0.0, dest='pari', help='Minimum for continuation parameter')
parser.add_argument("--parf", type=float, required=False, default=0.15, dest='parf', help='Maximum for continuation parameter')
parser.add_argument("--continue", type=int, nargs=2, required=False, default=None, dest='continue', help='Branch and step index for continuation.')
parser.add_argument("--branch", type=int, required=False, default=None, dest='branch', help='Branch id for continuation.')
parser.add_argument("--mode", type=int, required=False, default=0, dest='mode', help='Mode id for continuation.')
if __name__=="__main__":
    start=timeit.default_timer()
    parser.add_argument("--filebase", type=str, required=True, dest='filebase', help='Base string for file output.')
    args = parser.parse_args()
    argsdict=args.__dict__.copy()
    lps,ls,mps,ms,nps,ns=makeindices(argsdict)
    def mat(omega,argsdict):
        E,F=viscid_mat2d(omega,argsdict)
        return (E-argsdict['ad']*F).T

    s=3*(2*argsdict['Nt']+1)*(2*argsdict['Nx']+1)*(2*argsdict['Ny']+1)
    v=np.zeros(s,np.complex128)
    w=np.zeros(s,np.complex128)
    dir0=None
    br=argsdict['branch']

    nextbr=1
    while os.path.exists('%sargsdict_%i.json'%(argsdict['filebase'],nextbr)):
        nextbr=nextbr+1

    if not os.path.exists('%sargsdict.json'%(argsdict['filebase'])):
        out=open('%sargsdict.json'%(argsdict['filebase']),'w')
        args.__dict__['argv']=sys.argv
        json.dump(args.__dict__, out)
        out.close()

    if argsdict['continue'] is not None:
        oldbr=int(argsdict['continue'][0])
        file=open('%sargsdict_%i.json'%(argsdict['filebase'],oldbr),'r')
        oldargsdict=json.load(file)
        file.close()
        ind=np.abs(int(argsdict['continue'][1]))
        omegas=np.load('%somegas_%i.npy'%(argsdict['filebase'],oldbr))
        vs=np.load('%svs_%i.npy'%(argsdict['filebase'],oldbr))
        ws=np.load('%sws_%i.npy'%(argsdict['filebase'],oldbr))
        pars=np.load('%spars_%i.npy'%(argsdict['filebase'],oldbr))
        dirs=np.load('%sdirs_%i.npy'%(argsdict['filebase'],oldbr))

        if argsdict['continue'][1]<0:
            fomega=interp1d(np.arange(len(omegas)),omegas,kind='cubic')
            fv=interp1d(np.arange(len(omegas)),np.array(vs).T,kind='cubic')
            fw=interp1d(np.arange(len(omegas)),np.array(ws).T,kind='cubic')
            fpar=interp1d(np.arange(len(omegas)),pars,kind='cubic')
            fdir=interp1d(np.arange(len(omegas)),np.array(dirs).T,kind='cubic')

            if pars[ind-2]<pars[ind] and pars[ind+2]<pars[ind]:
                find=minimize_scalar(lambda x: -fpar(x),bracket=(ind-1,ind+1)).x
            else:
                find=minimize_scalar(lambda x: fpar(x),bracket=(ind-1,ind+1)).x

            print('Trying to switch at LP near %s=%f'%(oldargsdict['par'],float(fpar(find))))
            omega0=complex(fomega(find))
            v0=fv(find)
            w0=fw(find)
            dir0=fdir(find)
            argsdict[oldargsdict['par']]=float(fpar(find))

            newdir=np.zeros(dir0.shape)
            newdir[:s]=dir0[s:2*s]
            newdir[s:2*s]=-dir0[:s]
            newdir[2*s]=dir0[2*s+1]
            newdir[2*s+1]=-dir0[2*s]
            newdir[2*s+2]=2*dir0[2*s+2]
            dir0=newdir

        else:
            if ind==0:
                ind=-1
            omega0=omegas[ind]
            v0=vs[ind]
            w0=ws[ind]
            dir0=dirs[ind]
            argsdict[oldargsdict['par']]=pars[ind]


    else:
        if br is None:
            br=nextbr
        F,G=inviscid_mat_undriven(argsdict)
        evals,revecs,levecs=eig(F.astype(complex),G.astype(complex),right=True,left=True)
        order=np.argsort(evals)
        omega=evals[order[argsdict['mode']]]**0.5+1j*0

        v=np.zeros((3,(2*argsdict['Nt']+1),(2*argsdict['Nx']+1),(2*argsdict['Ny']+1)),np.complex128)
        w=np.zeros((3,(2*argsdict['Nt']+1),(2*argsdict['Nx']+1),(2*argsdict['Ny']+1)),np.complex128)
        v[2,argsdict['Nt']]=revecs[:,order[argsdict['mode']]].reshape((2*argsdict['Nx']+1,2*argsdict['Ny']+1))
        w[2,argsdict['Nt']]=levecs[:,order[argsdict['mode']]].reshape((2*argsdict['Nx']+1,2*argsdict['Ny']+1))
        v=v.ravel()
        w=w.ravel()

        ad0=0.0
        freqf=argsdict['freq']
        adf=argsdict['ad']

        #starting frequency and driving
        argsdict['ad']=ad0
        print(argsdict)
        omegas,vs,ws=rayleigh_mat(omega,v,np.conjugate(w),mat,argsdict)
        omega0=omegas[-1]
        v0=vs[-1]
        w0=ws[-1]

    print('Branch id: %i'%(br))
    print(argsdict)
    argsdict['pari']=np.min([argsdict['pari'],argsdict[argsdict['par']]])
    argsdict['parf']=np.max([argsdict['parf'],argsdict[argsdict['par']]])
    omegas,vs,ws,pars,dirs=pseudocont(omega0,v0,w0,mat,argsdict,dir=dir0)
    np.save('%somegas_%i'%(argsdict['filebase'],br),omegas)
    np.save('%svs_%i'%(argsdict['filebase'],br),vs)
    np.save('%sws_%i'%(argsdict['filebase'],br),ws)
    np.save('%spars_%i'%(argsdict['filebase'],br),pars)
    np.save('%sdirs_%i'%(argsdict['filebase'],br),dirs)

    out=open('%sargsdict_%i.json'%(argsdict['filebase'],br),'w')
    argsdict['argv']=sys.argv
    json.dump(argsdict, out)
    out.close()

    stop=timeit.default_timer()
    print('Runtime: %f'%(stop-start))

else:
    args=parser.parse_args(args=[])
    argsdict=args.__dict__
