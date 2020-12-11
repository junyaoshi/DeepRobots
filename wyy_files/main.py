# global mvortex
# global mvortex0
# global U,V,Omega
# global TVSE,gammac,Pic,npv,workspos,t_cnt,Dzcy,Dzcydot,zci,x_count,i_cam_x,alpha_t_plot,t_plot,integral,i_integral,i_derivative,i_derivative_plot
# global maxNumOfVortices,derivative,integral_plot,derivative_plot,i_integral_plot
# global computeEnergy
# global U0,V0,Omega0,npp,T0,KE_novortex0,KE_vortex0,Pvortex0,KE_0_4plot,KE_trace

import time
import numpy as np
import scipy.integrate as integrate
import timeit
tfinal = 1
deltat = 0.001 #deltat: time step for the fixed step solver ode5
tspan = np.arange(0.0,tfinal,deltat)
t_start = time.time()
computeEnergy = False
maxNumOfVortices = 2000
n_vortex = maxNumOfVortices
mvortex = np.zeros((n_vortex, 3))
mvortex0 = np.zeros((n_vortex, 6))

#initialize all variables
npp = 0
U0 = 0
V0 = 0
Omega0 = 0
gammac = 0
T0 = 0
KE_novortex0 = 0
KE_vortex0 = 0
npv = 0
Pic = np.zeros((3,))
U = 0
V = 0
Omega = 0
Pvortex0 = np.zeros((3,1))
TVSE = 0
Dzcy = np.zeros((6*int(tfinal/deltat) + 1,))
Dzcydot = np.zeros((6*int(tfinal/deltat) + 1,))
zci = 1
alpha_t_plot = 0
t_cnt = 1
x_count = 0
t_plot = 0
integral = 0
derivative = 0
integral_plot = 0
derivative_plot = 0
i_integral = 0
i_integral_plot = 0
i_derivative = 0
i_derivative_plot = 0

KE_0_4plot = np.zeros((int(tfinal / deltat) + 1, 2))

N = 3 + 2*n_vortex
ic = np.zeros((N, ))
nt = (tfinal / deltat) + 1

def joukowski_equation_ic_new_2_t(t=None,x=None):
    global mvortex
    global mvortex0
    global U,V,Omega
    global TVSE,gammac,Pic,npv,zci,Dzcy,Dzcydot,t_cnt,x_count,alpha_t,alpha_t_plot,t_plot,integral, \
    derivative,i_integral,i_derivative
    global maxNumOfVortices,integral_plot,derivative_plot,i_integral_plot,i_derivative_plot,chk
    global computeEnergy,kp,ki,kd,setpt
    global U0,V0,Omega0,npp,T0,KE_novortex0,KE_vortex0,Pvortex0

    N = 3+2 * maxNumOfVortices;
    dx=np.zeros((N,1))
    n_vortex=maxNumOfVortices
    rc=1
    setpt=-1.2
    shpcglmt=-0.65
    ua=0.2
    uf=2
    zcy = ua * np.sin(2 * np.pi * uf * t);
    zcydot = ua * (2 * np.pi * uf) * np.cos(2 * np.pi * uf * t)
    kp=- 5
    ki=- 0.6
    kd=- 0.95

    # if setpt > 0:
    #     alpha_t=setpt - x(3)
    # if setpt < 0:
    #     alpha_t=x(3) - setpt
    # alpha_t_plot[t_cnt]=alpha_t
    # t_plot[t_cnt]=t

    # if t_cnt == 1:
    #     integral=integral + dot(alpha_t,(t - 0))
    #     i_integral=i_integral + dot(integral,(t - 0))
    #     if t != 0:
    #         derivative=(alpha_t - 0) / (t - 0)
    #         i_derivative=i_derivative + dot(derivative,(t - 0))
    #     integral_plot[t_cnt]=integral
    #     derivative_plot[t_cnt]=derivative
    #     i_integral_plot[t_cnt]=i_integral
    #     i_derivative_plot[t_cnt]=i_derivative
    #
    # if t_cnt > 1:
    #     integral=integral + dot(alpha_t,(t - t_plot(t_cnt - 1)))
    #     derivative=(alpha_t - alpha_t_plot(t_cnt - 1)) / (t - t_plot(t_cnt - 1))
    #     i_integral=i_integral + dot(integral,(t - t_plot(t_cnt - 1)))
    #     integral_plot[t_cnt]=integral
    #     i_integral_plot[t_cnt]=i_integral
    #     if t > 0 and t != t_plot(t_cnt - 1):
    #         derivative=(alpha_t - alpha_t_plot(t_cnt - 1)) / (t - t_plot(t_cnt - 1))
    #         i_derivative=i_derivative + dot(derivative,(t - t_plot(t_cnt - 1)))
    #     derivative_plot[t_cnt]=derivative
    #     i_derivative_plot[t_cnt]=i_derivative
    #
    #
    # t_cnt=t_cnt + 1
    # if alpha_t == 0 and t == 0:
    #     zcy=0
    #     zcydot=0
    #     Dzcy[zci]=zcy
    #     Dzcydot[zci]=zcydot
    #     zci=zci + 1
    # if setpt < 0:
    #     zcytest=dot(abs(kp),integral) + dot(abs(ki),i_integral) + dot(abs(kd),i_derivative)
    #     if zcytest <= 0:
    #         if zcytest >= shpcglmt:
    #             zcy=copy(zcytest)
    #             zcydot=dot(abs(kp),alpha_t) + dot(abs(ki),integral) + dot(abs(kd),derivative)
    #
    #         else:
    #             zcy=Dzcy(zci - 1)
    #             zcydot=0
    #
    #     if zcytest >= 0:
    #         if zcytest <= abs(shpcglmt):
    #             zcy=copy(zcytest)
    #             zcydot=dot(abs(kp),alpha_t) + dot(abs(ki),integral) + dot(abs(kd),derivative)
    #
    #         else:
    #             zcy=Dzcy(zci - 1)
    #             zcydot=0
    #
    #     Dzcy[zci]=zcy
    #     Dzcydot[zci]=zcydot
    #     zci=zci + 1
    #
    #
    # if setpt > 0:
    #     zcytest=dot((kp),integral) + dot((ki),i_integral) + dot((kd),i_derivative)
    #
    #     if zcytest <= 0:
    #         if zcytest >= shpcglmt:
    #             zcy=copy(zcytest)
    #             zcydot=dot((kp),alpha_t) + dot((ki),integral) + dot((kd),derivative)
    #
    #         else:
    #             zcy=Dzcy(zci - 1)
    #             zcydot=0
    #
    #     if zcytest >= 0:
    #         if zcytest <= abs(shpcglmt):
    #             zcy=copy(zcytest)
    #             zcydot=dot((kp),alpha_t) + dot((ki),integral) + dot((kd),derivative)
    #         else:
    #             zcy=Dzcy(zci - 1)
    #             zcydot=0
    #
    #     Dzcy[zci]=zcy
    #     Dzcydot[zci]=zcydot
    #     zci=zci + 1


    zcy = ua*np.sin(2*uf*np.pi*t)

    if t > 0.2:
        zcy=ua
        Dzcy[zci]=zcy
        # zcydot = ua * 2 * uf * np.pi * np.cos(2 * uf * np.pi * t)
        zcydot=0
        Dzcydot[zci]=zcydot
        zci=zci + 1

    else:
        zcy=0
        Dzcy[zci]=zcy
        # zcydot = ua * 2 * uf * np.pi * np.cos(2 * uf * np.pi * t)
        zcydot=0
        Dzcydot[zci]=zcydot
        zci=zci + 1



    ak=0.85 ** 2 / (rc ** 2 - 0.15 ** 2)
    zcx=np.dot((1 - ak) / (1 + ak),np.sqrt(rc ** 2 - zcy ** 2))
    a = -2*ak/(1+ak)*np.sqrt(rc**2 - zcy**2)
    zcxdot = (1-ak)*(-zcy*zcydot)/(1+ak)/(zcx-a)
    adot = -2 * ak / (1 + ak) * (-zcy * zcydot) / (zcx - a)
    xx=x[1]
    yy=x[2]
    ththeta=x[3]


    for j in range(1,n_vortex):
        mvortex[j,1]=mvortex0[j,1] + x[2 + 2*j]
        mvortex[j,2]=mvortex0[j,2] + x[3 + 2*j]

    veloc_zeta, vortex_flag=findUVOmegaAndVelocitiesOfVortices(t,zcx,zcy,rc,a,zcxdot,zcydot,adot,xx,yy,ththeta,n_vortex,ak)
    dx[1]=np.dot(U,np.cos(x[3])) - np.dot(V,np.sin(x[3]))
    dx[2]=np.dot(U,np.sin(x[3])) + np.dot(V,np.cos(x[3]))
    dx[3]=Omega


    for j in range(1,vortex_flag):
        dx[2 + 2*j]=np.real(veloc_zeta[j])
        dx[3 + 2*j]=np.imag(veloc_zeta[j])
    dx[N]=np.sqrt(U ** 2 + V ** 2)
    return dx

def findUVOmegaAndVelocitiesOfVortices(t=None, zcx=None,zcy=None,rc=None,a=None,zcxdot=None,zcydot=None,adot=None,xx=None,yy=None,ththeta=None,n_vortex=None,ak=None):


    global mvortex
    global mvortex0
    global U,V,Omega
    global TVSE, gammac, Pic
    global maxNumOfVortices
    global computeEnergy
    global U0,V0,Omega0,npp,T0,KE_novortex0,KE_vortex0,Pvortex0, npv
    vortex_flag=0

    for j in range(1,maxNumOfVortices):
        if mvortex[j,2] != 0:
            vortex_flag=j


    zc=zcx + np.dot(j,zcy)
    conjzc=zcx - np.dot(j,zcy)
    zcdot=zcxdot + np.dot(j,zcydot)
    NewI11=np.dot(np.dot(2,np.pi),(rc ** 2 - a ** 2))
    NewI22=np.dot(np.dot(2,np.pi),(rc ** 2 + a ** 2))
    NewI13=np.real(np.dot(np.dot(np.dot(2,np.pi),j),(np.dot(rc ** 2,zc) + np.dot(a ** 2,conjzc) - np.dot(np.dot(a ** 2,ak),zc))))
    NewI23=np.imag(np.dot(np.dot(np.dot(2,np.pi),j),(np.dot(rc ** 2,zc) + np.dot(a ** 2,conjzc) - np.dot(np.dot(a ** 2,ak),zc))))
    NewI33=np.dot(np.dot(np.pi,rc ** 4),(1 - ak ** 4)) / 2 + np.dot(np.dot(np.dot(np.pi,a ** 2),(1 + ak ** 2)),(zc ** 2 + conjzc ** 2)) + np.dot(np.dot(np.dot(np.dot(2,np.pi),(rc ** 2 - np.dot(ak,a ** 2))),zc),conjzc) + np.dot(np.dot(2,np.pi),a ** 4)
    NewI33=abs(NewI33)
    Imatrix=[[NewI11,0,NewI13],[0,NewI22,NewI23],[NewI13,NewI23,NewI33]]


    us1=- rc ** 2
    us2=a ** 2
    vs1=np.dot(-j,rc ** 2)
    vs2=np.dot(-j,a ** 2)
    omegas1=np.dot(np.dot(- j,rc ** 2),(zc + a ** 2 / zc))
    omegas2=np.dot(np.dot(np.dot(- j,a ** 2),(- 1)),(np.dot(ak,zc) + a ** 2 / ak / zc))

    NewA1=np.dot(- rc ** 2,(zcdot + np.dot(np.dot(2,a),adot) / zc - np.dot(a ** 2,zcdot) / zc ** 2))
    NewA2=(np.dot(np.dot(rc ** 2,zcdot),(a ** 2 / zc ** 2 - ak ** 2)) + np.dot(np.conj(zcdot),(a ** 2 - np.dot(ak ** 2,zc ** 2))) - np.dot(np.dot(np.dot(np.dot(2,a),adot),ak),zc))
    NewA3=np.dot(rc ** 2,(np.dot(np.dot(2,a),adot) / zc ** 2 - np.dot(np.dot(2,a ** 2),zcdot) / zc ** 3))




    Rmatrix=[[np.cos(ththeta),np.sin(ththeta)],[- np.sin(ththeta),np.cos(ththeta)]]
    Lb=np.dot(Rmatrix,np.asarray([xx,yy]).T)
    xxb=Lb[0]
    yyb=Lb[1]
    VSE=0


    deltaTForVortex=0.005

    if ((abs(round(t / deltaTForVortex) - t / deltaTForVortex) < 1e-10) & t != TVSE):

        z0=(a - zc)
        zeta0=(a - zc)
        te=zeta0 + zc + a ** 2 / (zeta0 + zc)
        beta = np.arctan(zcy / (zcx - a))
        Zstart=np.dot(1.2,zeta0) + zc + a ** 2 / (np.dot(1.2,zeta0) + zc)
        if vortex_flag == npv:
            oldpv=Zstart
        else:
            oldpvzeta=mvortex(vortex_flag,1) + np.dot(j,mvortex(vortex_flag,2))
            oldpv=oldpvzeta + zc + (a ** 2) / (oldpvzeta + zc)
        dto=oldpv - te
        mason=np.dot(dto ** 2 / abs(dto ** 2),np.exp(np.dot(np.dot(- j,4),beta)))
        if (zcy < 0&np.unwrap(np.angle(dto)) < np.pi / 2 + np.dot(2,beta)&np.unwrap(np.angle(dto)) > 0):
            newpv=np.dot(2,a) + (oldpv - np.dot(2,a)) / (1 + np.dot(mason ** (1 / 3),np.exp(np.dot(np.dot(-j,2),np.pi) / 3)) + np.dot(mason ** (2 / 3),np.exp(np.dot(np.dot(-j,4),np.pi) / 3)))
        else:
            if (zcy < 0&np.unwrap(np.angle(dto)) > - (np.pi / 2 - np.dot(2,beta))&np.unwrap(np.angle(dto))< 0):
                newpv=np.dot(2,a) + (oldpv - np.dot(2,a)) / (1 + np.dot(mason ** (1 / 3),np.exp(np.dot(np.dot(j,2),np.pi) / 3)) + np.dot(mason ** (2 / 3),np.exp(np.dot(np.dot(j,4),np.pi) / 3)))

            else:
                if (zcy > 0&np.unwrap(np.angle(dto)) < np.pi / 2 + np.dot(2,beta)&np.unwrap(np.angle(dto)) > 0):
                    newpv=np.dot(2,a) + (oldpv - np.dot(2,a)) / (1 + np.dot(mason ** (1 / 3),np.exp(np.dot(np.dot(-j,2),np.pi) / 3)) + np.dot(mason ** (2 / 3),np.exp(np.dot(np.dot(-j,4),np.pi) / 3)))

                else:
                    if (zcy > 0&np.unwrap(np.angle(dto)) >-(np.pi / 2 - np.dot(2,beta))&np.unwrap(np.angle(dto)) < 0):
                        newpv=np.dot(2,a) + (oldpv - np.dot(2,a)) / (1 + np.dot(mason ** (1 / 3),np.exp(np.dot(np.dot(j,2),np.pi) / 3)) + np.dot(mason ** (2 / 3),np.exp(np.dot(np.dot(j,4),np.pi) / 3)))

                    else:
                        newpv=np.dot(2,a) + (oldpv - np.dot(2,a)) / (1 + mason ** (1 / 3) + mason ** (2 / 3))

        rootsp=[1,- newpv(a ** 2)]
        pzeta=np.roots(rootsp) - zc
        for pk in range(1,len(pzeta)):
            if (abs(pzeta[pk]) > rc):
                newvortex=pzeta[pk]
        z1=newvortex


        dw1dz=- us1 / z0 ** 2 - us2 / (z0 + zc) ** 2
        dw2dz=- vs1 / z0 ** 2 - vs2 / (z0 + zc) ** 2
        dw3dz=- omegas1 / z0 ** 2 - omegas2 / (z0 + zc) ** 2
        dwsdz=- NewA1 / z0 ** 2 - NewA2 / (z0 + zc) ** 2 + np.dot(NewA3,(1 / (z0 + zc) - 1 / z0))
        dwdz=np.dot(U,dw1dz) + np.dot(V,dw2dz) + np.dot(Omega,dw3dz) + dwsdz
        dwdz=dwdz + np.dot(j,gammac) / z0

        if vortex_flag != 0:
            for pn in range(1,vortex_flag):
                strengthj=mvortex[pn,3]
                zj=mvortex[pn,1] + np.dot(j,mvortex[pn,2])
                dwdz=dwdz + np.dot(np.dot(strengthj,j),(1 / (z0 - zj) - 1 / (z0 - rc ** 2 / np.conj(zj))))

        C=np.zeros((4,4))
        C[1,1]=dw1dz
        C[1,2]=dw2dz
        C[1,3]=dw3dz
        C[1,4]=np.dot(j,(1 / (z0 - z1) - 1 / (z0 - rc ** 2 / np.conj(z1))))
        C[range(2,4),range(1,3)]=Imatrix
        zk=z1
        Zk=zk + zc + (a ** 2) / (zk + zc)
        Zkx=np.real(Zk)
        Zky=np.imag(Zk)
        im,imc=getImpulseAndImpulseCoupleForVortex(zk,rc,zc,a)
        Impulsevortex=im
        Imcouplevortex=imc
        B=[np.real(Impulsevortex),np.imag(Impulsevortex),Imcouplevortex].T
        Bsh=[- Zky,Zkx(Zkx ** 2 + Zky ** 2) / 2].T
        C[range(2,4),4]=(B + np.dot(np.dot(Bsh,2),np.pi))
        xxx=- [np.real(dwdz),0,0,0].T - np.dot(j,[np.imag(dwdz),0,0,0].T)
        ss=np.linalg.solve(C,xxx)  # in matlab it is C\xxx
        deltaU=np.real(ss[1])
        deltaV=np.real(ss[2])
        deltaOmega=np.real(ss[3])
        new_strength=np.real(ss[4])
        U=U + deltaU
        V=V + deltaV
        Omega=Omega + deltaOmega
        VSE=1
        TVSE=t
        vortex_flag=vortex_flag + 1
        mvortex[vortex_flag,1]=np.real(newvortex)
        mvortex[vortex_flag,2]=np.imag(newvortex)
        mvortex[vortex_flag,3]=new_strength
        mvortex0[vortex_flag,1]=np.real(newvortex)
        mvortex0[vortex_flag,2]=np.imag(newvortex)
        mvortex0[vortex_flag,3]=xx
        mvortex0[vortex_flag,4]=yy
        mvortex0[vortex_flag,5]=ththeta
        mvortex0[vortex_flag,6]=t
    veloc_zeta=np.zeros((n_vortex,1))
    if vortex_flag != 0:
        for j in range(1,vortex_flag):
            z0=mvortex[j,1] + np.dot(j,mvortex[j,2])
            strength=mvortex[j,3]
            if (abs(z0) - rc < 1e-05):
                mvortex[j,3]=0
                veloc_zeta[j]=0
            else:
                dw1dz=- us1 / z0 ** 2 - us2 / (z0 + zc) ** 2
                dw2dz=- vs1 / z0 ** 2 - vs2 / (z0 + zc) ** 2
                dw3dz=- omegas1 / z0 ** 2 - omegas2 / (z0 + zc) ** 2
                dwsdz=- NewA1 / z0 ** 2 - NewA2 / (z0 + zc) ** 2 + np.dot(NewA3,(1 / (z0 + zc) - 1 / z0))
                dwdz=np.dot(U,dw1dz) + np.dot(V,dw2dz) + np.dot(Omega,dw3dz) + dwsdz
                dwdz=dwdz + np.dot(j,gammac) / z0 + np.dot(np.dot(strength,j),(- 1 / (z0 - rc ** 2 / np.conj(z0))))

                for pn in range(1,vortex_flag):
                    if pn != j:
                        strengthj=mvortex[pn,3]
                        zj=mvortex[pn,1] + np.dot(j,mvortex[pn,2])
                        dwdz=dwdz + np.dot(np.dot(strengthj,j),(1 / (z0 - zj) - 1 / (z0 - rc ** 2 / np.conj(zj))))

                dFdz=1 - (a ** 2) / (z0 + zc) ** 2
                dF2dz2=np.dot(2,(a ** 2)) / (z0 + zc) ** 3
                Veloc0=dwdz / dFdz - np.dot(np.dot(j,strength),dF2dz2) / 2 / dFdz ** 2
                Z0=(z0 + zc + (a ** 2) / (z0 + zc))
                veloc0=np.conj(Veloc0) - (U + np.dot(j,V) + np.dot(np.dot(j,Omega),Z0))
                dFdzc= dFdz
                dFda=np.dot(2,a) / (z0 + zc)
                veloc_zeta[j]=(veloc0 - np.dot(dFdzc,zcdot) - np.dot(dFda,adot)) / dFdz



    NewLf1=np.dot(- np.pi,(NewA1 - np.dot(a ** 2 / rc ** 2,np.conj(NewA1)) + NewA2 - np.dot(np.dot(rc ** 2,ak ** 2) / a ** 2,np.conj(NewA2))))
    NewLf2=np.dot(- np.pi,(np.dot(NewA3,zc) - np.dot(np.dot(np.conj(NewA3),ak),np.conj(zc))))
    Lf=NewLf1 + NewLf2
    Lfx=np.real(Lf)
    Lfy=np.imag(Lf)
    NewAf1=np.real(np.dot(np.dot(np.dot(np.dot(np.dot(2,np.pi),j),(1 + ak ** 2)),conjzc),NewA1) + np.dot(np.dot(np.dot(np.dot(2,np.pi),j),(np.dot(np.dot(ak ** 3,rc ** 2),conjzc) / a ** 2 + conjzc - np.dot(ak,zc))),NewA2) + np.dot(np.dot(np.dot(np.dot(2,np.pi),j) / rc ** 2,(np.dot(np.dot(ak ** 2,zc),(np.dot(2,rc ** 2) - np.dot(zc,conjzc))) - np.dot(a ** 2,conjzc))),np.conj(NewA1)) + np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(4,np.pi),j),rc ** 2),zc),ak ** 3) / a ** 2,np.conj(NewA2))) / 2
    NewAf2=np.real(np.dot(np.dot(np.dot(np.dot(np.dot(2,np.pi),j),zc),conjzc),NewA3) + np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(2,np.pi),j),zc),conjzc),ak ** 2),np.conj(NewA3))) / 2
    Af=NewAf1 + NewAf2


    Newm2=np.dot(np.dot(np.pi,rc ** 2),(1 - np.dot(2,ak ** 2) + np.dot(ak ** 4,conjzc ** 2) / a ** 2))
    Newm3=np.dot(np.pi,(np.dot(np.dot(2,a),conjzc) - np.dot(np.dot(np.dot(2,a),ak),zc) + np.dot(np.dot(np.dot(2,rc ** 2),ak ** 3),conjzc) / a))
    NewJ2=np.dot(np.dot(np.dot(- np.pi,rc ** 2),j),(conjzc + np.dot(np.dot(2,ak ** 3),zc) - np.dot(np.dot(ak ** 5,conjzc),(rc ** 2 + np.dot(zc,conjzc))) / a ** 2))
    NewJ3=np.dot(np.dot(- np.pi,j),(np.dot(2,a ** 3) + np.dot(a,conjzc ** 2) + np.dot(np.dot(a,ak ** 2),zc ** 2) - np.dot(np.dot(rc ** 2,ak ** 4),(rc ** 2 + np.dot(np.dot(2,zc),conjzc))) / a))
    Lb=np.dot(Newm2,zcdot) + np.dot(Newm3,adot)
    Ab=np.real(np.dot(zcdot,NewJ2) + np.dot(adot,NewJ3))
    Lbx=np.real(Lb)
    Lby=np.imag(Lb)
    P1=np.asarray([(Lbx + Lfx),(Lby + Lfy),(Ab + Af)]).T

    Pvortex=0

    if vortex_flag != 0:
        for pn in range(1,vortex_flag):
            strength=mvortex(pn,3)
            zk=mvortex(pn,1) + np.dot(j,mvortex(pn,2))
            Zk=(zk + zc + (a ** 2) / (zk + zc))
            Zkx=np.real(Zk)
            Zky=np.imag(Zk)
            im,imc=getImpulseAndImpulseCoupleForVortex(zk,rc,zc,a)
            Impulsevortex=im
            Imcouplevortex=imc
            B=[np.real(Impulsevortex),np.imag(Impulsevortex),Imcouplevortex].T
            Bsh=[- Zky,Zkx,(Zkx ** 2 + Zky ** 2) / 2].T
            Pvortex=Pvortex + np.dot(B,strength) + np.dot(np.dot(np.dot(Bsh,strength),2),np.pi)

    if gammac != 0:
        im0=np.dot(np.dot(np.dot(np.dot(gammac,2),np.pi),j),zc) + np.dot(np.dot(np.dot(np.dot(gammac,2),np.pi),j),(xxb + np.dot(j,yyb)))
        imc0=np.dot(np.dot(np.dot(gammac,2),np.pi),(- np.imag(np.dot((- j),(rc ** 2 + np.dot(zc,conjzc) + np.dot((a ** 2),(a ** 2)) / (rc ** 2 - np.dot(zc,conjzc)))) / 2))) + np.dot(np.dot(1 / 2,(np.dot(np.dot(- 2,np.pi),gammac))),(xxb ** 2 + yyb ** 2))
        Pvortex=[Pvortex] + np.asarray([np.real(im0),np.imag(im0),imc0]).T


    P = np.asarray([0,0,0])
    for i in range(len(P)):
        P[i] = P1[i] + Pvortex
    LMb=np.dot(Rmatrix,[[Pic[0]],[Pic[1]]])
    AMb=Pic[2] - (np.dot(xx,Pic[1]) - np.dot(yy,Pic[0]))
    Mb=np.asarray([[LMb[0]],[LMb[1]],[AMb]])
    Vbody=np.dot(np.linalg.inv(Imatrix),(Mb - P))
    U=Vbody[0]
    V=Vbody[1]
    Omega=Vbody[2]

    if computeEnergy == True:
        T0[npp]=t
        U0[npp]=U
        V0[npp]=V
        Omega0[npp]=Omega
        Pvortex0[range(),npp]=Pvortex
        delta=abs(zc)
        kea1=np.dot(U,(- rc ** 2)) + np.dot(np.dot(V,(- i)),rc ** 2) + np.dot(np.dot(np.dot(Omega,(- i)),(zc + a ** 2 / zc)),rc ** 2) + NewA1
        kea2=np.dot(U,a ** 2) + np.dot(np.dot(V,(- i)),a ** 2) + np.dot(np.dot(np.dot(np.dot(Omega,(- i)),(- 1)),(np.dot(ak,zc) + a ** 2 / (np.dot(ak,zc)))),a ** 2) + NewA2
        kek2=NewA3
        KE_novortex=np.dot(np.pi / 2,((np.dot(kea1,np.conj(kea1)) + np.dot(kea1,np.conj(kea2)) + np.dot(np.conj(kea1),kea2)) / rc ** 2 + np.dot(np.dot(kea2,np.conj(kea2)),rc ** 2) / (rc ** 2 - delta ** 2) ** 2 + np.dot(np.dot(kek2,np.conj(kek2)),np.ln(rc ** 2 / (rc ** 2 - delta ** 2))) + (np.dot(np.dot(np.conj(kea1),kek2),zc) + np.dot(np.dot(kea1,np.conj(kek2)),np.conj(zc))) / rc ** 2 + (np.dot(np.dot(np.conj(kea2),kek2),zc) + np.dot(np.dot(kea2,np.conj(kek2)),np.conj(zc))) / (rc ** 2 - delta ** 2)))
        KE_novortex0[npp]=KE_novortex
        KE1=0
        KE2=0
        KE3=0
        sumgamma=0
    KE1 = 0
    KE2 = 0
    KE3 = 0
    sumgamma = 0

    if vortex_flag != 0:
        for pk in range(1,vortex_flag):
            strengthpk=mvortex[pk,3]
            zetapk=mvortex[pk,1] + np.dot(j,mvortex[pk,2])
            for pj in range(1,vortex_flag):
                strengthpj=mvortex[pj,3]
                zetapj=mvortex[pj,1] + np.dot(j,mvortex[pj,2])

                if pj != pk:
                    KE1=KE1 - np.dot(np.dot(np.dot(np.pi,strengthpk),strengthpj), np.ln(abs(zetapk - zetapj)))
                KE2=KE2 + np.dot(np.dot(np.dot(np.pi,strengthpk),strengthpj), np.ln(abs(zetapk - rc ** 2 / np.conj(zetapj))))

            sumgamma=sumgamma + strengthpk
            KE3=KE3 + np.dot(np.dot(np.pi,strengthpk),np.ln(abs(zetapk) / rc))
    # KE_vortex0[npp]=KE1 + KE2 + np.dot(sumgamma,KE3)
    return veloc_zeta, vortex_flag

def getImpulseAndImpulseCoupleForVortex(zk=None,rc=None,zc=None,a=None):


    conjzc=np.conj(zc)
    conjzk=np.conj(zk)
    Zk=(zk + zc) + (a ** 2) / (zk + zc)
    im = (-2*np.pi*1j)*(Zk - zk  + rc**2/np.conj(zk) )
    w3zk = -1j/2 * (rc ** 2 + zc * conjzc + a**4 / (rc**2 - zc * conjzc) + 2 * rc ** 2 * (zc + a ** 2 / zc) / zk - 2 * a ** 2 * (rc ** 2 - zc * conjzc) / zc / (zk + zc) - 2 * zc * a ** 4 / (rc ** 2 - zc * conjzc) / (zk + zc))
    imc = 2*np.pi*np.imag(w3zk)

X = integrate.RK45(joukowski_equation_ic_new_2_t,0, ic, 1, max_step=0.01, atol=1, rtol=1)

T = tspan
x = X[:,1]
y = X[:,2]
theta = X[:, 3]
# there was a plot of (T,x,y theta) in matlab, not implemented here,
sumnv = 0
sumpv = 0

for jj in range(1, n_vortex):
    if abs(mvortex[jj, 3]) > 0:
        if mvortex[jj, 3] > 0:
            sumpv = sumpv + mvortex[jj, 3]

        else:
            sumnv = sumnv + mvortex[jj, 3]

t_end = time.time()

total_time = t_end-t_start
print('Total Time', total_time)

