import torch
#from pykeops.torch import Kernel, kernel_product, Genred
from pykeops.torch import Genred
from torch.autograd import grad
import numpy as np
from scipy.optimize import minimize, OptimizeResult
import time


##Deformation Kerenl:

def Kernel_f(kernel_type):
    def dfKernel(d,dim_grass):
        if kernel_type.lower() == 'gaussian':
            expr_defoker = 'Exp(-p*SqNorm2(x-y))'
            
        if dim_grass == 0:
        
            exprV = expr_defoker + '*p1'
            V_list = ['p=Pm(1)','x=Vi('+str(d)+')','y=Vj('+str(d)+')','p1=Vj('+str(d)+')']
        
        elif dim_grass == 1:
        
            exprV = expr_defoker + '*p0+Grad(Grad(' + expr_defoker +'*p1,y,dum),dum,u)'             
            V_list = ['p=Pm(1)','x=Vi(' +str(d)+ ')','y=Vj(' +str(d)+ ')','p0=Vj(' +str(d)+ ')',
               'p1=Vj(' +str(d)+ ')','u=Vj(' +str(d)+ ')','dum=Vj(' +str(d)+ ')']
                       
        elif dim_grass == 2:
        
            exprV = expr_defoker + '*p0+Grad(Grad('+expr_defoker\
                +'*p1,y,dum),dum,u1)+Grad(Grad('+expr_defoker+'*p2,y,dum),dum,u2)'
        
            V_list = ['p=Pm(1)','x=Vi(' +str(d)+ ')','y=Vj(' +str(d)+ ')','p0=Vj(' +str(d)+ ')',
                'p1=Vj(' +str(d)+ ')','p2=Vj('+str(d)+ ')','u1=Vj(' +str(d)+ ')',
               'u2=Vj(' +str(d)+ ')','dum=Vj(' +str(d)+ ')']
                
        expr_ad = 'Grad('+exprV+',x,h1)'
        exprdV = 'Grad('+expr_ad+',h1,h2)'
        expr_2ad = 'Grad('+expr_ad+',x,h2)'
    
        dV_ad_list = V_list.copy()+['h1=Vi(' + str(d)+ ')']
        dV_list = dV_ad_list.copy()+['h2=Vi(' + str(d)+ ')']
        dV_2ad_list = dV_ad_list.copy()+['h2=Vi(' + str(d)+ ')']
    
        V = Genred(exprV,V_list,reduction_op='Sum',axis=1)
        dV_ad = Genred(expr_ad,dV_ad_list,reduction_op='Sum',axis=1)
        dV = Genred(exprdV,dV_list,reduction_op='Sum',axis=1)
        dV_2ad = Genred(expr_2ad,dV_2ad_list,reduction_op='Sum',axis=1)
    
        if dim_grass == 0:
            return V, dV_ad
        else:
            return V, dV_ad, dV, dV_2ad
    return dfKernel


## Dynamics and Hamiltonian

def Hamiltonian(kernel_size, kernel_type = 'gaussian'):
    dfKernel = Kernel_f(kernel_type)
    
    def Ham(p0,q0):
        #oos2 = 1/kernel_size**2
        p = list(map(lambda x: x.clone().float(), p0))
        q = list(map(lambda x: x.clone().float(), q0))
        oos2 = torch.tensor([1/kernel_size**2], dtype=torch.float32, device=p0[0].device)
        dim_grass = len(q)-1
        d = q[0].shape[1]
        #dummies
        dum_i = q[0].clone().detach()
        dum_j = p[0].clone().detach()
        
        if dim_grass == 0:
            V, dV_ad = dfKernel(d,0)
            Vx = V(oos2,q[0],q[0],p[0]).type_as(p0[0])
            
            return .5*(p0[0]*Vx).sum()
            
        elif dim_grass == 1:
            V, dV_ad, dV, dV_2ad = dfKernel(d,1)
            Vx = V(oos2,q[0],q[0],p[0],p[1],q[1],dum_j).type_as(p0[0])
            dVx_u = dV(oos2,q[0],q[0],p[0],p[1],q[1],dum_i,dum_j,q[1]).type_as(p0[0])
            
            return .5*(p0[0]*Vx + p0[1]*dVx_u).sum()
            
        elif dim_grass == 2:
            V, dV_ad, dV, dV_2ad= dfKernel(d,2)
            Vx = V(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j).type_as(p0[0])
            dVx_ad_p1 = dV_ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_i,p[1]).type_as(p0[0])
            dVx_ad_p2 = dV_ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_i,p[2]).type_as(p0[0])
            
            return .5*(p0[0]*Vx + q0[1]*dVx_ad_p1 + q0[2]*dVx_ad_p2).sum()
    return Ham

def HamiltonianSystem(kernel_size,kernel_type = 'gaussian'):
    #compute -Hq and Hp
    dfKernel = Kernel_f(kernel_type)
    
    def HS(p0,q0):
        #oos2 = 1/kernel_size**2
        p = list(map(lambda x: x.clone().float(), p0))
        q = list(map(lambda x: x.clone().float(), q0))
        oos2 = torch.tensor([1/kernel_size**2], dtype=torch.float32, device=p0[0].device)
        dim_grass = len(q)-1
        d = q[0].shape[1]
        #dummies
        dum_i = q[0].clone().detach()
        dum_j = p[0].clone().detach()
        
        if dim_grass == 0:
            V, dV_ad = dfKernel(d,0)
            Hp = [V(oos2,q[0],q[0],p[0])]
            nHq = [-dV_ad(oos2,q[0],q[0],p[0],p[0])]
            
        elif dim_grass == 1:
            V, dV_ad, dV, dV_2ad = dfKernel(d,1)
            
            Hp = [V(oos2,q[0],q[0],p[0],p[1],q[1],dum_j)]\
                +[dV(oos2,q[0],q[0],p[0],p[1],q[1],dum_j,dum_i,q[1])]
            nHq = [-dV_ad(oos2,q[0],q[0],p[0],p[1],q[1],dum_j,p[0])\
                   -dV_2ad(oos2,q[0],q[0],p[0],p[1],q[1],dum_j,p[1],q[1])]\
                + [-dV_ad(oos2,q[0],q[0],p[0],p[1],q[1],dum_j,p[1])]
            
        elif dim_grass == 2:
            V, dV_ad, dV, dV_2ad= dfKernel(d,2)
            
            Hp = [V(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j)]
            Hp.append(dV(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,dum_i,q[1]))
            Hp.append(dV(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,dum_i,q[2]))
            
            nHq = [-dV_ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,p[0])\
                   -dV_2ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,p[1],q[1])\
                   -dV_2ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,p[2],q[2])]
            nHq.append(-dV_ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,p[1]))
            nHq.append(-dV_ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,p[2]))
            
        #return nHq+Hp
        return list(map(lambda x: x.clone().type_as(p0[0]), nHq+Hp))
    return HS

def Hamiltonian_fr(kernel_size, meta_weight=1, kernel_type = 'gaussian'):
    dfKernel = Kernel_f(kernel_type)
    
    def Ham(p0,q0):
        #oos2 = 1/kernel_size**2
        p = list(map(lambda x: x.clone().float(), p0))
        q = list(map(lambda x: x.clone().float(), q0))
        oos2 = torch.tensor([1/kernel_size**2], dtype=torch.float32, device=p0[0].device)
        dim_grass = len(q)-1
        d = q[0].shape[1]
        #dummies
        dum_i = q[0].clone().detach()
        dum_j = p[0].clone().detach()
        
        if dim_grass == 0:
            V, dV_ad = dfKernel(d,0)
            Vx = V(oos2,q[0],q[0],p[0]).type_as(p0[0])
            
            return .5*(p0[0]*Vx).sum()
            
        elif dim_grass == 1:
            V, dV_ad, dV, dV_2ad = dfKernel(d,1)

            Gr_norm = pvec_Norm([q[1]]).unsqueeze(1)
            #Gr1_normalized = q[1]/Gr_norm
            Inn_div_W = ((p[1]*q[1]).sum(1).unsqueeze(1)**2)/Gr_norm
            meta_reg = .5*(Inn_div_W).sum()/meta_weight

            Vx = V(oos2,q[0],q[0],p[0],p[1],q[1],dum_j).type_as(p0[0])
            dVx_u = dV(oos2,q[0],q[0],p[0],p[1],q[1],dum_i,dum_j,q[1]).type_as(p0[0])
            
            
            return .5*(p0[0]*Vx + p0[1]*dVx_u).sum() + meta_reg, meta_reg
            
        elif dim_grass == 2:
            V, dV_ad, dV, dV_2ad= dfKernel(d,2)

            Gr_norm = pvec_Norm(q[1:]).unsqueeze(1)
            #eta = (p[1]*q[1] + p[2]*q[2]).sum(1).unsqueeze(1)/(2*meta_weight*Gr_norm)
            #meta_reg = .5*meta_weight*((eta**2)*Gr_norm).sum()
            Inn_div_W = ((p[1]*q[1] + p[2]*q[2]).sum(1).unsqueeze(1)**2)/Gr_norm
            meta_reg = Inn_div_W.sum()/(8*meta_weight)

            Vx = V(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j).type_as(p0[0])
            dVx_ad_p1 = dV_ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_i,p[1]).type_as(p0[0])
            dVx_ad_p2 = dV_ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_i,p[2]).type_as(p0[0])
            
            
            return .5*(p0[0]*Vx + q0[1]*dVx_ad_p1 + q0[2]*dVx_ad_p2).sum() + meta_reg, meta_reg
            
    return Ham

def HamiltonianSystem_fr(kernel_size, meta_weight=1, kernel_type = 'gaussian'):
    #compute -Hq and Hp
    dfKernel = Kernel_f(kernel_type)
    
    def HS(p0,q0):
        p = list(map(lambda x: x.clone().float(), p0))
        q = list(map(lambda x: x.clone().float(), q0))
        oos2 = torch.tensor([1/kernel_size**2], dtype=torch.float32, device=p0[0].device)
        dim_grass = len(q)-1
        d = q[0].shape[1]
        #dummies
        dum_i = q[0].clone().detach()
        dum_j = p[0].clone().detach()
        
        if dim_grass == 0:
            V, dV_ad = dfKernel(d,0)
            Hp = [V(oos2,q[0],q[0],p[0])]
            nHq = [-dV_ad(oos2,q[0],q[0],p[0],p[0])]
            
        elif dim_grass == 1:
            V, dV_ad, dV, dV_2ad = dfKernel(d,1)
            
            Gr_norm = pvec_Norm([q[1]]).reshape(q[1].shape[0],1)
            Gr1_normalized = q[1]/Gr_norm
            Inn_pu_  = (p[1]*Gr1_normalized).sum(1).unsqueeze(1)
            eta = Inn_pu_/meta_weight
            
            Hp = [V(oos2,q[0],q[0],p[0],p[1],q[1],dum_j)]\
                +[dV(oos2,q[0],q[0],p[0],p[1],q[1],dum_j,dum_i,q[1]) + eta*q[1]]
            nHq = [-dV_ad(oos2,q[0],q[0],p[0],p[1],q[1],dum_j,p[0])\
                   -dV_2ad(oos2,q[0],q[0],p[0],p[1],q[1],dum_j,p[1],q[1])]\
                + [-dV_ad(oos2,q[0],q[0],p[0],p[1],q[1],dum_j,p[1])  - eta*p[1] + (Inn_pu_**2)*Gr1_normalized/(2*meta_weight)]
            
        elif dim_grass == 2:
            V, dV_ad, dV, dV_2ad= dfKernel(d,2)
            
            Gr_norm = pvec_Norm(q[1:]).unsqueeze(1)
            B = (p[1]*q[1] + p[2]*q[2]).sum(1).unsqueeze(1)/(2*Gr_norm)
            temp = ((B**2)/(2*Gr_norm*meta_weight))
            eta = B/meta_weight
            
            Hp = [V(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j)]
            Hp.append(dV(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,dum_i,q[1]) + eta*q[1]/2)
            Hp.append(dV(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,dum_i,q[2]) + eta*q[2]/2)



            Inn_11 = (q[1]**2).sum(1).unsqueeze(1)
            Inn_22 = (q[2]**2).sum(1).unsqueeze(1)
            Inn_12 = (q[1]*q[2]).sum(1).unsqueeze(1)

            nHq = [-dV_ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,p[0])\
                   -dV_2ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,p[1],q[1])\
                   -dV_2ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,p[2],q[2])]
            nHq.append(-dV_ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,p[1])\
                       -eta*p[1]/2 + temp*(Inn_22*q[1]-Inn_12*q[2]))
            nHq.append(-dV_ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,p[2])\
                       -eta*p[2]/2 + temp*(Inn_11*q[2]-Inn_12*q[1]))
            
        return list(map(lambda x: x.clone().type_as(p0[0]), nHq+Hp))
    return HS


def Hamiltonian_pf(kernel_size, meta_weight=1, kernel_type = 'gaussian'):
    dfKernel = Kernel_f(kernel_type)
    
    def Ham(p0,q0,p_alpha0,alpha0):
        
        p = list(map(lambda x: x.clone().float(), p0))
        q = list(map(lambda x: x.clone().float(), q0))
        alpha = alpha0.clone().float()
        p_alpha = p_alpha0.clone().float()
        oos2 = torch.tensor([1/kernel_size**2], dtype=torch.float32, device=p0[0].device)
        dim_grass = len(q)-1
        d = q[0].shape[1]
        #dummies
        dum_i = q[0].clone().detach()
        dum_j = p[0].clone().detach()
        
        if dim_grass == 0:
            V, dV_ad = dfKernel(d,0)
            Vx = V(oos2,q[0],q[0],p[0]).type_as(p0[0])
            
            return .5*(p0[0]*Vx).sum()
            
        elif dim_grass == 1:
            V, dV_ad, dV, dV_2ad = dfKernel(d,1)

            Gr_norm = pvec_Norm([q[1]])
            
            meta_reg = ((p_alpha**2)/Gr_norm).sum()/(4*meta_weight)

            Vx = V(oos2,q[0],q[0],p[0],p[1],q[1],dum_j).type_as(p0[0])
            dVx_u = dV(oos2,q[0],q[0],p[0],p[1],q[1],dum_i,dum_j,q[1]).type_as(p0[0])
            
            
            return .5*((p0[0]*Vx + p0[1]*dVx_u).sum() + meta_reg), meta_reg
            
        elif dim_grass == 2:
            V, dV_ad, dV, dV_2ad= dfKernel(d,2)

            Gr_norm = pvec_Norm(q[1:])

            meta_reg = ((p_alpha**2)/Gr_norm).sum()/(4*meta_weight)

            Vx = V(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j).type_as(p0[0])
            dVx_ad_p1 = dV_ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_i,p[1]).type_as(p0[0])
            dVx_ad_p2 = dV_ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_i,p[2]).type_as(p0[0])
            
            
            return .5*((p0[0]*Vx + q0[1]*dVx_ad_p1 + q0[2]*dVx_ad_p2).sum() + meta_reg), meta_reg
            
    return Ham


def HamiltonianSystem_pf(kernel_size, meta_weight=1, kernel_type = 'gaussian'):
    #compute -Hq and Hp
    dfKernel = Kernel_f(kernel_type)
    
    def HS(p0,q0,p_alpha0,alpha0):
        
        p = list(map(lambda x: x.clone().float(), p0))
        q = list(map(lambda x: x.clone().float(), q0))
        alpha = alpha0.clone().float()
        p_alpha = p_alpha0.clone().float()
        oos2 = torch.tensor([1/kernel_size**2], dtype=torch.float32, device=p0[0].device)
        dim_grass = len(q)-1
        d = q[0].shape[1]
        #dummies
        dum_i = q[0].clone().detach()
        dum_j = p[0].clone().detach()
        
        if dim_grass == 0:
            V, dV_ad = dfKernel(d,0)
            Hp = [V(oos2,q[0],q[0],p[0])]
            nHq = [-dV_ad(oos2,q[0],q[0],p[0],p[0])]
            
        elif dim_grass == 1:
            V, dV_ad, dV, dV_2ad = dfKernel(d,1)

            Gr_norm = pvec_Norm([q[1]]).unsqueeze(1)

            eta = p_alpha.unsqueeze(1)/(2*meta_weight*Gr_norm) ###
            C = (p_alpha**2).unsqueeze(1)/(8*meta_weight*(Gr_norm**3))###

            Hp = [V(oos2,q[0],q[0],p[0],p[1],q[1],dum_j)]\
                +[dV(oos2,q[0],q[0],p[0],p[1],q[1],dum_j,dum_i,q[1])]
            nHq = [-dV_ad(oos2,q[0],q[0],p[0],p[1],q[1],dum_j,p[0])\
                   -dV_2ad(oos2,q[0],q[0],p[0],p[1],q[1],dum_j,p[1],q[1])]\
                + [-dV_ad(oos2,q[0],q[0],p[0],p[1],q[1],dum_j,p[1]) + C*q[1]]
            Hp_alpha = [eta.flatten()/2]###
            nH_alpha = [0*p_alpha]

            
        elif dim_grass == 2:
            V, dV_ad, dV, dV_2ad= dfKernel(d,2)
            
            Gr_norm = pvec_Norm(q[1:]).unsqueeze(1)

            eta = p_alpha.unsqueeze(1)/(2*meta_weight*Gr_norm)###
            C = (p_alpha**2).unsqueeze(1)/(8*meta_weight*(Gr_norm**3))###

            Hp = [V(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j)]
            Hp.append(dV(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,dum_i,q[1]))
            Hp.append(dV(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,dum_i,q[2]))



            Inn_11 = (q[1]**2).sum(1).unsqueeze(1)
            Inn_22 = (q[2]**2).sum(1).unsqueeze(1)
            Inn_12 = (q[1]*q[2]).sum(1).unsqueeze(1)

            nHq = [-dV_ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,p[0])\
                   -dV_2ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,p[1],q[1])\
                   -dV_2ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,p[2],q[2])]
            nHq.append(-dV_ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,p[1])\
                    + C*(Inn_22*q[1]-Inn_12*q[2]))
            nHq.append(-dV_ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,p[2])\
                    + C*(Inn_11*q[2]-Inn_12*q[1]))

            Hp_alpha = [eta.flatten()/2]###
            nH_alpha = [0*p_alpha]

            
        
        return list(map(lambda x: x.clone().type_as(p0[0]), nHq+Hp+nH_alpha+Hp_alpha))
    return HS

def Integrator(method):
    
    if method.lower() == 'rk4':
        def f(ODESystem, x0, nt, deltat=1.0):
            x = tuple(map(lambda x: x.clone(), x0))
            dt = deltat / nt
            x_evol = [x]
            w = [1,2,2,1]
        
            for i in range(nt):
                y_new = x_evol[i]
                for j in range(4):
                    if j == 0:
                        temp = x_evol[i]
                    else:
                        temp = add_list(x_evol[i],k,a=1,b=1/w[j])
    
                    k = tuple(map(lambda T: T*dt,ODESystem(temp)))
                    y_new = add_list(y_new,k,a=1,b=w[j]/6)
                x_evol.append(y_new)
        
            return x_evol
    
    if method.lower() == 'middle_point':
        def f(ODESystem, x0, nt, deltat=1.0):
            x = tuple(map(lambda x: x.clone(), x0))
            dt = deltat / nt
            x_evol = [x]
               
            for i in range(nt):
                temp = add_list(x_evol[i],ODESystem(x_evol[i]),a=1,b=dt/2)
                y_new = add_list(x_evol[i],ODESystem(temp),a=1,b=dt)
                x_evol.append(y_new)
        
            return x_evol
    
    if method.lower() == 'ralston':
        def f(ODESystem, x0, nt, deltat=1.0):
            x = tuple(map(lambda x: x.clone(), x0))
            dt = deltat / nt
            l = [x]
            for i in range(nt):
                xdot = ODESystem(x)
                xi = tuple(map(lambda x, xdot: x + (2 * dt / 3) * xdot, x, xdot))
                xdoti = ODESystem(xi)
                x = tuple(map(lambda x, xdot, xdoti: x + (.25 * dt) * (xdot + 3 * xdoti), x, xdot, xdoti))
                l.append(x)
            return l
    return f

def flow(x0,p0,q0,**options):
    if 'model' in options:
        model = options['model'].lower()
    else:
        model = 'pure_varifold'
        
    if 'defo_kernel_type' not in options:
        options['defo_kernel_type'] = 'gaussian'
    
    if 'odemethod' not in options:  
        options['odemethod'] = 'rk4'
        
    if 'nb_euler_steps' not in options:
        options['nb_euler_steps'] = 15
    
    
    dfKernel = Kernel_f(options['defo_kernel_type'])
    
    if model == 'fisher_rao':
        HS = HamiltonianSystem_fr(options['defo_kernel_size'], meta_weight=options['weight_meta'],\
                              kernel_type = options['defo_kernel_type'])
    else:
        HS = HamiltonianSystem(options['defo_kernel_size'],options['defo_kernel_type'])

    def FlowEq(x1, p1, q1):

        x = x1.clone().float()
        p = list(map(lambda x: x.clone().float(), p1))
        q = list(map(lambda x: x.clone().float(), q1))
        oos2 = torch.tensor([1/options['defo_kernel_size']**2], dtype=torch.float32, device=p0[0].device)
        dim_grass = len(q)-1
        d = q[0].shape[1]
        #dummies
        dum_i = q[0].clone().detach()
        dum_j = p[0].clone().detach()

        if dim_grass == 0:
            V, dV_ad = dfKernel(d,0)
            Hp = V(oos2,x,q[0],p[0])

        elif dim_grass == 1:
            V, dV_ad, dV, dV_2ad = dfKernel(d,1)

            Hp = V(oos2,x,q[0],p[0],p[1],q[1],dum_j)

        elif dim_grass == 2:
            V, dV_ad, dV, dV_2ad= dfKernel(d,2)

            Hp = V(oos2,x,q[0],p[0],p[1],p[2],q[1],q[2],dum_j)

        return list(map(lambda x: x.clone().type_as(p1[0]), [Hp] + HS(p1, q1)))
        
    def ODESystem(Q):
        n = int((len(Q)-1)/2)
        return FlowEq(Q[0],Q[1:1+n],Q[1+n:])
    
    odesolver = Integrator(options['odemethod'])
    W_dyn = odesolver(ODESystem,[x0]+p0+q0,options['nb_euler_steps'])
    o_evol = [W_dyn[t][0] for t in range(len(W_dyn))]
    pq_evol = [W_dyn[t][1:] for t in range(len(W_dyn))]
    return o_evol, pq_evol
    
def flow_fr(x0,p0,q0,**options):
    #flow for fisher-rao model
        
    if 'defo_kernel_type' not in options:
        options['defo_kernel_type'] = 'gaussian'
    
    if 'odemethod' not in options:  
        options['odemethod'] = 'rk4'
        
    if 'nb_euler_steps' not in options:
        options['nb_euler_steps'] = 15
    
    
    dfKernel = Kernel_f(options['defo_kernel_type'])
    
    kernel_size = options['defo_kernel_size']
    meta_weight = options['weight_meta']
    
    #HS = HamiltonianSystem_fr(options['defo_kernel_size'], meta_weight=options['weight_meta'],\
    #                  kernel_type = options['defo_kernel_type'])

    def FlowEq(pt1, p1, q1):

        #x = list(map(lambda x: x.clone().float(), x1))
        pt = pt1.clone().float()
        p = list(map(lambda x: x.clone().float(), p1))
        q = list(map(lambda x: x.clone().float(), q1))

        oos2 = torch.tensor([1/kernel_size**2], dtype=torch.float32, device=p0[0].device)
        dim_grass = len(q)-1
        d = q[0].shape[1]
        #dummies
        dum_i = q[0].clone().detach()
        dum_j = p[0].clone().detach()

        if dim_grass == 0:
            V, dV_ad = dfKernel(d,0)
            Hp = [V(oos2,q[0],q[0],p[0])]
            nHq = [-dV_ad(oos2,q[0],q[0],p[0],p[0])]

        elif dim_grass == 1:
            V, dV_ad, dV, dV_2ad = dfKernel(d,1)

            Gr_norm = pvec_Norm([q[1]]).reshape(q[1].shape[0],1)
            Gr1_normalized = q[1]/Gr_norm
            Inn_pu_  = (p[1]*Gr1_normalized).sum(1).unsqueeze(1)
            eta = Inn_pu_/meta_weight


            Hp = [V(oos2,q[0],q[0],p[0],p[1],q[1],dum_j)]\
                +[dV(oos2,q[0],q[0],p[0],p[1],q[1],dum_j,dum_i,q[1]) + eta*q[1]]
            nHq = [-dV_ad(oos2,q[0],q[0],p[0],p[1],q[1],dum_j,p[0])\
                   -dV_2ad(oos2,q[0],q[0],p[0],p[1],q[1],dum_j,p[1],q[1])]\
                + [-dV_ad(oos2,q[0],q[0],p[0],p[1],q[1],dum_j,p[1])  - eta*p[1] + (Inn_pu_**2)*Gr1_normalized/(2*meta_weight)]

            dpt = [V(oos2,pt,q[0],p[0],p[1],q[1],dum_j)]
            dlogalpha = [eta.flatten()]

        elif dim_grass == 2:
            V, dV_ad, dV, dV_2ad= dfKernel(d,2)

            Gr_norm = pvec_Norm(q[1:]).unsqueeze(1)
            B = (p[1]*q[1] + p[2]*q[2]).sum(1).unsqueeze(1)/(2*Gr_norm)
            temp = ((B**2)/(2*Gr_norm*meta_weight))
            eta = B/meta_weight

            Hp = [V(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j)]
            Hp.append(dV(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,dum_i,q[1]) + eta*q[1]/2)
            Hp.append(dV(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,dum_i,q[2]) + eta*q[2]/2)



            Inn_11 = (q[1]**2).sum(1).unsqueeze(1)
            Inn_22 = (q[2]**2).sum(1).unsqueeze(1)
            Inn_12 = (q[1]*q[2]).sum(1).unsqueeze(1)

            nHq = [-dV_ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,p[0])\
                   -dV_2ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,p[1],q[1])\
                   -dV_2ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,p[2],q[2])]
            nHq.append(-dV_ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,p[1])\
                       -eta*p[1]/2 + temp*(Inn_22*q[1]-Inn_12*q[2]))
            nHq.append(-dV_ad(oos2,q[0],q[0],p[0],p[1],p[2],q[1],q[2],dum_j,p[2])\
                       -eta*p[2]/2 + temp*(Inn_11*q[2]-Inn_12*q[1]))

            dpt = [V(oos2,pt,q[0],p[0],p[1],p[2],q[1],q[2],dum_j)]
            dlogalpha = [eta.flatten()]

        return list(map(lambda x: x.clone().type_as(p1[0]), dpt+nHq+Hp+dlogalpha))


    def ODESystem(Q):
        n = int((len(Q)-2)/2)
        return FlowEq(Q[0], Q[1:1+n], Q[1+n:1+2*n])

    odesolver = Integrator(options['odemethod'])

    W_ini = [x0] + p0 + q0 + [torch.zeros(p0[0].shape[0]).to(dtype=p0[0].dtype, device=p0[0].device)]
    W_dyn = odesolver(ODESystem,W_ini,options['nb_euler_steps'])
    o_evol = [W_dyn[t][0] for t in range(len(W_dyn))]
    pq_evol = [W_dyn[t][1:-1] for t in range(len(W_dyn))]
    alpha_evol = [torch.exp(W_dyn[t][-1]) for t in range(len(W_dyn))]
    return o_evol, pq_evol, alpha_evol
    
def Flow_set(**options):
    '''Create a function computing flow map'''
    
    
    #Default settings for model, kernel, ODE method and ODE steps----------------
    if 'model' in options:
        model = options['model'].lower()
    else:
        model = 'pure_varifold'
        
    if 'defo_kernel_type' not in options:
        options['defo_kernel_type'] = 'gaussian'
    
    if 'odemethod' not in options:  
        options['odemethod'] = 'rk4'
        
    if 'nb_euler_steps' not in options:
        options['nb_euler_steps'] = 15
        
    dfKernel = Kernel_f(options['defo_kernel_type'])
    
    odesolver = Integrator(options['odemethod'])
    
    #Hamiltonian system apply to the grids-----------------------------------
    def Hp_grid(x1, p1, q1):
        x = x1.clone().float()
        p = list(map(lambda x: x.clone().float(), p1))
        q = list(map(lambda x: x.clone().float(), q1))
        oos2 = torch.tensor([1/options['defo_kernel_size']**2], dtype=torch.float32, device=p1[0].device)
        dim_grass = len(q)-1
        d = q[0].shape[1]
        #dummies
        dum_i = q[0].clone().detach()
        dum_j = p[0].clone().detach()

        if dim_grass == 0:
            V, dV_ad = dfKernel(d,0)
            Hp = V(oos2,x,q[0],p[0])

        elif dim_grass == 1:
            V, dV_ad, dV, dV_2ad = dfKernel(d,1)
            Hp = V(oos2,x,q[0],p[0],p[1],q[1],dum_j)

        elif dim_grass == 2:
            V, dV_ad, dV, dV_2ad= dfKernel(d,2)
            Hp = V(oos2,x,q[0],p[0],p[1],p[2],q[1],q[2],dum_j)
            
        return Hp.type_as(p1[0])
    
    #Set up Hamiltonian systems for the Source shape------------------------------------ 
    if model == 'fisher_rao':
        HS = HamiltonianSystem_fr(options['defo_kernel_size'], meta_weight=options['weight_meta'],\
                              kernel_type = options['defo_kernel_type'])
    
    elif model == 'metapf':
        HS = HamiltonianSystem_pf(options['defo_kernel_size'], meta_weight=options['weight_meta'],\
                              kernel_type = options['defo_kernel_type'])

    else:
        HS = HamiltonianSystem(options['defo_kernel_size'],options['defo_kernel_type'])
        
    #Flow map--------------------------------------------------------------------------------
    
    if model == 'metapf':
        def ODESystem(Q):
            n = int((len(Q)-3)/2)
            return [Hp_grid(Q[0], Q[1:1+n], Q[1+n:1+2*n])] + HS(Q[1:1+n], Q[1+n:1+2*n], Q[-2], Q[-1])
        
        def FlowMap(x0,p0,q0,p_alpha0):
            alpha0 = torch.ones(p0[0].shape[0]).to(dtype=p0[0].dtype,device=p0[0].device)
            init = [x0]+p0+q0+[p_alpha0]+[alpha0]
            W_dyn = odesolver(ODESystem,init,options['nb_euler_steps'])
            o_evol = [W_dyn[t][0] for t in range(len(W_dyn))]
            pq_evol = [W_dyn[t][1:] for t in range(len(W_dyn))]
            return o_evol, pq_evol
    else:
        def ODESystem(Q):
            n = int((len(Q)-1)/2)
            return [Hp_grid(Q[0], Q[1:1+n], Q[1+n:])] + HS(Q[1:1+n], Q[1+n:])
        
        def FlowMap(x0,p0,q0):
            W_dyn = odesolver(ODESystem,[x0]+p0+q0,options['nb_euler_steps'])
            o_evol = [W_dyn[t][0] for t in range(len(W_dyn))]
            pq_evol = [W_dyn[t][1:] for t in range(len(W_dyn))]
            return o_evol, pq_evol
        
    #-----------------------------------------------------------------------------------------  
    
    return FlowMap
    
def add_list(X,Y,a=1,b=1):
    return tuple(map(lambda x, y: a*x+b*y ,X,Y))





##Cost functional----------------------------------------
def enr_match(Source,Target,**options):
    '''Energy function for pure varifold matching'''
    
    if 'defo_kernel_type' not in options:
        options['defo_kernel_type'] = 'gaussian'
    
    if 'odemethod' not in options:  
        options['odemethod'] = 'rk4'
        
    if 'nb_euler_steps' not in options:
        options['nb_euler_steps'] = 15
    
    #options['defo_kernel_size'] = torch.tensor([options['defo_kernel_size']], dtype=Source[0].dtype, device=Target[0].device)
    #options['sig_geom'] = torch.tensor([options['sig_geom']], dtype=Source[0].dtype, device=Target[0].device)
    #options['sig_grass'] = torch.tensor([options['sig_grass']], dtype=Source[0].dtype, device=Target[0].device)

    
    options['dim_varifold'] = len(Source) - 1
    Ham = Hamiltonian(options['defo_kernel_size'],options['defo_kernel_type'])
    
    odesolver = Integrator(options['odemethod'])
    #vfloss = vfnorm(options['kernel_geom'],options['kernel_grass'],options['sig_geom'],options['sig_grass'],len(Source) - 1)
    vfloss = varifold_dist(**options)
    
    HS = HamiltonianSystem(options['defo_kernel_size'],options['defo_kernel_type'])
    def ODESystem(Q):
        n = int(len(Q)/2)
        return HS(Q[:n],Q[n:])
    
    def enr_lddmm(P):
        n = len(P)
        X_end = odesolver(ODESystem,P+Source,options['nb_euler_steps'])[-1][n:]
        ham = Ham(P,Source)
        dat = options['weight_varifold_dist']*vfloss(X_end,Target)
        #print(ham,dat)
        return ham+dat,ham,dat
        
    return enr_lddmm

def enr_match_fr(Source,Target,**options):
    '''Energy function for Fisher-Rao metric'''
    
    if 'defo_kernel_type' not in options:
        options['defo_kernel_type'] = 'gaussian'
    
    if 'odemethod' not in options:  
        options['odemethod'] = 'rk4'
        
    if 'nb_euler_steps' not in options:
        options['nb_euler_steps'] = 15

    
    options['dim_varifold'] = len(Source) - 1
    
    Ham = Hamiltonian_fr(options['defo_kernel_size'], meta_weight=options['weight_meta'],\
                         kernel_type = options['defo_kernel_type'])
    
    odesolver = Integrator(options['odemethod'])
    vfloss = varifold_dist(**options)
    
    HS = HamiltonianSystem_fr(options['defo_kernel_size'], meta_weight=options['weight_meta'],\
                              kernel_type = options['defo_kernel_type'])
    def ODESystem(Q):
        n = int(len(Q)/2)
        return HS(Q[:n],Q[n:])
    
    def enr_lddmm(P):
        n = len(P)
        X_end = odesolver(ODESystem,P+Source,options['nb_euler_steps'])[-1][n:]
        ham, meta_reg = Ham(P,Source)
        dat = options['weight_varifold_dist']*vfloss(X_end,Target)
        
        return ham+dat, ham, dat, meta_reg
        
    return enr_lddmm

def enr_metagw_match(Source,Target,**options):
    if 'defo_kernel_type' not in options:
        options['defo_kernel_type'] = 'gaussian'
    
    if 'odemethod' not in options:  
        options['odemethod'] = 'rk4'
        
    if 'nb_euler_steps' not in options:
        options['nb_euler_steps'] = 15

    options['dim_varifold'] = len(Source) - 1
    Ham = Hamiltonian(options['defo_kernel_size'],options['defo_kernel_type'])
    
    odesolver = Integrator(options['odemethod'])
    vfinn = varifold_Inn(**options)
    
    HS = HamiltonianSystem(options['defo_kernel_size'],options['defo_kernel_type'])
    def ODESystem(Q):
        n = int(len(Q)/2)
        return HS(Q[:n],Q[n:])
    
    def enr_lddmm(P):
        n = len(P)
        X_end = odesolver(ODESystem,P+Source,options['nb_euler_steps'])[-1][n:]
        ham, Inn = Ham(P,Source), vfinn(X_end,Target)
        delh, lam = options['weight_meta']/2, options['weight_varifold_dist']
        
        #print(Inn)
        alpha_1 =(delh+lam*Inn[2])/(delh+lam*Inn[0])
        #print(alpha_1)
        meta_reg = delh*((alpha_1 - 1)**2)
        dat = lam*((alpha_1**2)*Inn[0] - 2*alpha_1*Inn[2] + Inn[1])
        
        return ham+meta_reg+dat, ham, dat, meta_reg, alpha_1
        
    return enr_lddmm


def enr_metapwm_match(Source,Target,**options):
    #Pushforward of weight measure from Eulerian coordinate
    if 'defo_kernel_type' not in options:
        options['defo_kernel_type'] = 'gaussian'
    
    if 'odemethod' not in options:  
        options['odemethod'] = 'rk4'
        
    if 'nb_euler_steps' not in options:
        options['nb_euler_steps'] = 15

    options['dim_varifold'] = len(Source) - 1
    Ham = Hamiltonian(options['defo_kernel_size'],options['defo_kernel_type'])
    
    odesolver = Integrator(options['odemethod'])
    vfloss = varifold_dist(**options)
    
    HS = HamiltonianSystem(options['defo_kernel_size'],options['defo_kernel_type'])
    def ODESystem(Q):
        n = int(len(Q)/2)
        return HS(Q[:n],Q[n:])
    
    def enr_lddmm(P,alpha_1):
        n = len(P)
        delh, lam = options['weight_meta']/2, options['weight_varifold_dist']
        
        #Compute Hamiltonian
        X_end = list(odesolver(ODESystem,P+Source,options['nb_euler_steps'])[-1][n:])
        #print(X_end)
        ham = Ham(P,Source)
        #Compute regularization for alpha
        r_0 = pvec_Norm(Source[1:])        
        meta_reg = delh*(((alpha_1 - 1)*r_0)**2).sum()
        #data attachment term
        X_end[1] = X_end[1]*alpha_1[:,None]
        dat = options['weight_varifold_dist']*vfloss(X_end,Target)
        
        
        return ham+meta_reg+dat, ham, dat, meta_reg, alpha_1
        
    return enr_lddmm   
    

def enr_match_pf(Source,Target,**options):
    '''Energy function for pushforward metamorphosis model'''
    
    if 'defo_kernel_type' not in options:
        options['defo_kernel_type'] = 'gaussian'
    
    if 'odemethod' not in options:  
        options['odemethod'] = 'rk4'

    if 'nb_euler_steps' not in options:
        options['nb_euler_steps'] = 15


    options['dim_varifold'] = len(Source) - 1

    Ham = Hamiltonian_pf(options['defo_kernel_size'], meta_weight=options['weight_meta'],\
                         kernel_type = options['defo_kernel_type'])

    odesolver = Integrator(options['odemethod'])
    vfloss = varifold_dist(**options)

    HS = HamiltonianSystem_pf(options['defo_kernel_size'], meta_weight=options['weight_meta'],\
                              kernel_type = options['defo_kernel_type'])
    def ODESystem(Q):
        n = len(Source)
        return HS(Q[:n],Q[n:2*n],Q[-2],Q[-1])
    
    def enr_lddmm(P,P_alpha):
        n = len(P)
        Alpha = torch.ones(P[0].shape[0]).to(dtype=P[0].dtype,device=P[0].device)
        Dyn_end = odesolver(ODESystem,P+Source+[P_alpha]+[Alpha],options['nb_euler_steps'])[-1]
        X_end, alpha_end = list(Dyn_end[n:2*n]), (Dyn_end[-1]**2) ##
        X_end[1] = X_end[1]*alpha_end.unsqueeze(1)

        ham, meta_reg = Ham(P,Source,P_alpha,Alpha)
        dat = options['weight_varifold_dist']*vfloss(X_end,Target)
        
        return ham+dat, ham, dat, meta_reg, alpha_end
        
    return enr_lddmm

def enr_scipy(Source,Target,model='pure_varifold',**options):
    n,d = Source[0].shape
    if 'model' not in options:
        options['model'] = model
    model = options['model']
    
    options['dim_varifold'] = len(Source)-1
    
    if model.lower() == 'pure_varifold':
        energy = enr_match(Source,Target,**options)
    elif model.lower() == 'fisher_rao':
        energy = enr_match_fr(Source,Target,**options)
    elif model.lower() == 'metagw':
        energy = enr_metagw_match(Source,Target,**options)
    elif model.lower() == 'metapwm':
        energy = enr_metapwm_match(Source,Target,**options)
    elif model.lower() == 'metapf':
        energy = enr_match_pf(Source,Target,**options)
    
    if model.lower() in {'pure_varifold','metagw','fisher_rao'}:  
        def enr(x):
            x = torch.from_numpy(x).clone().detach().to(dtype=Source[0].dtype, device=Source[0].device).requires_grad_(True)
            X = vec2list(x,n,d)

            enr_list = energy(X)
            
            Gx = grad(enr_list[0], [x])[0].detach().cpu().numpy().astype('float64')
            

            E = float(enr_list[0].detach().cpu().numpy())
            Enr_dic = {}
            Enr_dic['energy'] = E
            Enr_dic['hamiltonian'] = float(enr_list[1].detach().cpu().numpy())
            Enr_dic['data_attachment'] = float(enr_list[2].detach().cpu().numpy())

            if model.lower() == 'metagw':
                Enr_dic['weight_variation_reg'] = float(enr_list[3].detach().cpu().numpy())
                Enr_dic['alpha'] = float(enr_list[4].detach().cpu().numpy())
            elif model.lower() == 'fisher_rao':
                Enr_dic['weight_variation_reg'] = float(enr_list[3].detach().cpu().numpy())
            else:
                Enr_dic['alpha'] = 1.0

            return E,Gx, Enr_dic
    
    
    else:
        def enr(x):
            x = torch.from_numpy(x).clone().detach().to(dtype=Source[0].dtype, device=Source[0].device).requires_grad_(True)
            p, alpha_1 = x[:(options['dim_varifold']+1)*n*d], x[(options['dim_varifold']+1)*n*d:]
            P = vec2list(p,n,d)

            #start = time.time()
            enr_list = energy(P,alpha_1)
            #print( 'energy time: ', round(time.time() - start, 2), ' seconds')

            #start = time.time()
            Gx = grad(enr_list[0], [x])[0].detach().cpu().numpy().astype('float64')
            #print( 'gradient time: ', round(time.time() - start, 2), ' seconds')

            E = float(enr_list[0].detach().cpu().numpy())
            Enr_dic = {}
            Enr_dic['energy'] = E
            Enr_dic['hamiltonian'] = float(enr_list[1].detach().cpu().numpy())
            Enr_dic['data_attachment'] = float(enr_list[2].detach().cpu().numpy())            
            Enr_dic['weight_variation_reg'] = float(enr_list[3].detach().cpu().numpy())
            Enr_dic['alpha'] = enr_list[4].detach().cpu().numpy()

            return E,Gx, Enr_dic
    
    
    return enr

'''
def list2vec(P):
    #n,d = P[0].shape
    dim_var = len(P)-1
    p = P[0].flatten()
    if dim_var >0:
        for i in range(1,len(P)):
            p = torch.cat([p,P[i].flatten()])
    return p
'''

def list2vec(P):
    if isinstance(P[0], np.ndarray):
        p = np.concatenate([P[i].flatten() for i in range(len(P))])
    elif isinstance(P[0], torch.Tensor):
        p = torch.cat([P[i].flatten() for i in range(len(P))])
                
    return p
    
def vec2list(p,n,d):
    m = int(len(p)/(n*d))
    P = []
    for i in range(m):
        P.append(p[i*n*d:(i+1)*n*d].reshape(n,d))
    return P


## Varifold distance----------------------------------------------------------

def  pvec_Norm(XiX):
    '''Compute the norm for p-vectors
    Input:-- a list XiX of length p. XiX[i] is a N by d Tensor 
          -- A p-vector is represented by XiX[0][i,:]^...^XiX[p-1][i,:]
    Output: A float number represents |XiX[0][i,:]^...^XiX[p-1][i,:]|
    
    '''
    F_X = XiX[0].shape[0]
    m = len(XiX)
    A = torch.empty(m,m,F_X, dtype=XiX[0].dtype, device=XiX[0].device)
    for i in range(m):
        for j in range(m):
            A[i,j,:] = (XiX[i]*XiX[j]).sum(dim=1)
    #print(A)    
    if m == 1:
        #G = torch.squeeze(A)
        G = A.reshape(F_X)
    elif m == 2:
        G = A[0,0,:]*A[1,1,:]-A[0,1,:]*A[1,0,:]
        
    G = G.sqrt()
    return G

def VKerenl(kernel_geom,kernel_grass,sig_geom,sig_grass,dim_varifold):
    #kerenel on spatial domain
    if kernel_geom.lower() == "gaussian":
        expr_geom = 'Exp(-SqDist(x,y)*a)'
    elif kernel_geom.lower() == "cauchy":
        expr_geom = 'IntCst(1)/(IntCst(1)+SqDist(x,y)*a)'
        
    #kernel on Grassmanian  
    #dimension of Grassmanian = 1
    if dim_varifold == 1:
        if kernel_grass.lower() == 'constant':
            expr_grass = 'IntCst(1)'
                
        elif kernel_grass.lower() == 'linear':
            expr_grass = '(u|v)'
            
        elif kernel_grass.lower() == 'gaussian_oriented':
            expr_grass = 'Exp(IntCst(2)*b*((u|v)-IntCst(1)))'
            
        elif kernel_grass.lower() == 'binet':
            expr_grass = 'Square((u|v))'
            
        elif kernel_grass.lower() == 'gaussian_unoriented':
            expr_grass='Exp(IntCst(2)*b*(Square((u|v))-IntCst(1)))'
    
        def K(x, y, u, v,weight_y):
            d = x.shape[1]
            pK = Genred(expr_geom + '*' + expr_grass + '*r',
                ['a=Pm(1)','b=Pm(1)','x=Vi('+str(d)+')','y=Vj('+str(d)+')','u=Vi('+str(d)+')',
                'v=Vj('+str(d)+')','r=Vj(1)'],
                reduction_op='Sum',
                axis=1)
            geom_inv_sq = torch.tensor([1/sig_geom**2], dtype=x.dtype, device=x.device)
            grass_inv_sq = torch.tensor([1/sig_grass**2], dtype=x.dtype, device=x.device)
            #return pK(1/sig_geom**2,1/sig_grass**2,x,y,u,v,weight_y)
            return pK(geom_inv_sq,grass_inv_sq,x,y,u,v,weight_y)
    #dimension of Grassmanian = 2
    elif dim_varifold == 2:
        if kernel_grass.lower() == 'constant':
            expr_grass='IntCst(1)'
                    
        elif kernel_grass.lower() == 'linear':
            expr_grass='((u1|v1)*(u2|v2)-(u1|v2)*(u2|v1))'
            
        elif kernel_grass.lower() == 'gaussian_oriented':
            expr_grass='Exp(IntCst(2)*b*((u1|v1)*(u2|v2)-(u1|v2)*(u2|v1)-IntCst(1)))'
            
        elif kernel_grass.lower() == 'binet':
            expr_grass='Square((u1|v1)*(u2|v2)-(u1|v2)*(u2|v1))'
            
        elif kernel_grass.lower() == 'gaussian_unoriented':
            expr_grass='Exp(IntCst(2)*b*(Square((u1|v1)*(u2|v2)-(u1|v2)*(u2|v1))-IntCst(1)))'
            
        def K(x, y, u1, u2, v1, v2, weight_y):
            d = x.shape[1]
            
            pK = Genred(expr_geom + '*' + expr_grass + '*r',
                ['a=Pm(1)','b=Pm(1)','x=Vi('+str(d)+')','y=Vj('+str(d)+')','u1=Vi('+str(d)+')','u2=Vi('+str(d)+')',
                'v1=Vj('+str(d)+')','v2=Vj('+str(d)+')','r=Vj(1)'],
                reduction_op='Sum',
                axis=1)
            
            geom_inv_sq = torch.tensor([1/sig_geom**2], dtype=x.dtype, device=x.device)
            grass_inv_sq = torch.tensor([1/sig_grass**2], dtype=x.dtype, device=x.device)
            
            return pK(geom_inv_sq,grass_inv_sq,x,y,u1,u2,v1,v2,weight_y)
      
    return K

def varifold_dist(**options):
    
    
    K =VKerenl(options['kernel_geom'],options['kernel_grass'],options['sig_geom'],options['sig_grass'],options['dim_varifold'])
    
    def vfdist(U,V):
        X = list(map(lambda x: x.clone(), U))
        Y = list(map(lambda x: x.clone(), V))
        weight_x = pvec_Norm(X[1:])[:, None]
        weight_y = pvec_Norm(Y[1:])[:, None]
        X[1] = X[1].div(weight_x)
        Y[1] = Y[1].div(weight_y)
        
        X = list(map(lambda x: x.float(), X))
        Y = list(map(lambda x: x.float(), Y))
        
        if options['dim_varifold'] == 1:
            KXX = K(X[0], X[0], X[1], X[1] ,weight_x.float()).type_as(U[0])
            KXY = K(X[0], Y[0], X[1], Y[1] ,weight_y.float()).type_as(U[0])
            KYY = K(Y[0], Y[0], Y[1], Y[1] ,weight_y.float()).type_as(U[0])
            
            
        elif options['dim_varifold'] == 2:
            KXX = K(X[0], X[0], X[1], X[2], X[1], X[2], weight_x.float()).type_as(U[0])
            KXY = K(X[0], Y[0], X[1], X[2], Y[1], Y[2], weight_y.float()).type_as(U[0])
            KYY = K(Y[0], Y[0], Y[1], Y[2], Y[1], Y[2], weight_y.float()).type_as(U[0])
            
        PXX = (weight_x*KXX).sum()
        PXY = (weight_x*KXY).sum()
        PYY = (weight_y*KYY).sum()
            
        return PXX+PYY-2*PXY
    return vfdist
    
def varifold_Inn(**options):
    K =VKerenl(options['kernel_geom'],options['kernel_grass'],options['sig_geom'],options['sig_grass'],options['dim_varifold'])
    
    def vfinn(U,V):
        X = list(map(lambda x: x.clone(), U))
        Y = list(map(lambda x: x.clone(), V))
        weight_x = pvec_Norm(X[1:])[:, None]
        weight_y = pvec_Norm(Y[1:])[:, None]
        X[1] = X[1].div(weight_x)
        Y[1] = Y[1].div(weight_y)
        
        X = list(map(lambda x: x.float(), X))
        Y = list(map(lambda x: x.float(), Y))
        
        if options['dim_varifold'] == 1:
            KXX = K(X[0], X[0], X[1], X[1] ,weight_x.float()).type_as(U[0])
            KXY = K(X[0], Y[0], X[1], Y[1] ,weight_y.float()).type_as(U[0])
            KYY = K(Y[0], Y[0], Y[1], Y[1] ,weight_y.float()).type_as(U[0])
            
            
        elif options['dim_varifold'] == 2:
            KXX = K(X[0], X[0], X[1], X[2], X[1], X[2], weight_x.float()).type_as(U[0])
            KXY = K(X[0], Y[0], X[1], X[2], Y[1], Y[2], weight_y.float()).type_as(U[0])
            KYY = K(Y[0], Y[0], Y[1], Y[2], Y[1], Y[2], weight_y.float()).type_as(U[0])
            
        PXX = (weight_x*KXX).sum()
        PXY = (weight_x*KXY).sum()
        PYY = (weight_y*KYY).sum()
            
        return [PXX,PYY,PXY]
    return vfinn
    
def vfnorm(kernel_geom,kernel_grass,sig_geom,sig_grass,dim_varifold):
    #Input paramters of kernels, returns a function which computes distance between two varifolds
    K = VKerenl(kernel_geom,kernel_grass,sig_geom,sig_grass,dim_varifold)
    
    def vfdist(U,V):
        X = {'center':U['center'].clone(),'vector':[U['vector'][0].clone(),U['vector'][1].clone()]}
        Y = {'center':V['center'].clone(),'vector':[V['vector'][0].clone(),V['vector'][1].clone()]}
        
        weight_x = pvec_Norm(X['vector'])[:, None]
        weight_y = pvec_Norm(Y['vector'])[:, None]
        X["vector"][0] = X["vector"][0].div(weight_x)
        Y["vector"][0] = Y["vector"][0].div(weight_y)
        
        
        
        if dim_varifold == 1:
            PXX = (weight_x*K(X['center'], X['center'], X['vector'][0], X['vector'][0] ,weight_x)).sum()
            PXY = (weight_x*K(X['center'], Y['center'], X['vector'][0], Y['vector'][0] ,weight_y)).sum()
            PYY = (weight_y*K(Y['center'], Y['center'], Y['vector'][0], Y['vector'][0] ,weight_y)).sum()
        elif dim_varifold == 2:
            PXX = (weight_x*K(X['center'], X['center'], X['vector'][0], X['vector'][1], 
                X['vector'][0], X['vector'][1], weight_x)).sum()
            PXY = (weight_x*K(X['center'], Y['center'], X['vector'][0], X['vector'][1], 
                Y['vector'][0], Y['vector'][1], weight_y)).sum()
            PYY = (weight_y*K(Y['center'], Y['center'], Y['vector'][0], Y['vector'][1], 
                Y['vector'][0], Y['vector'][1], weight_y)).sum()
            
        return PXX+PYY-2*PXY
    return vfdist

def mesh2var(pts,tri):
    #compute the dirac representation of meshes (1D or 2D)
    pts = pts.clone()
    T,M = tri.shape
    d = pts.shape[1]
    
    var = [pts[tri.flatten(),:].reshape(T,M,d).sum(1)/M]
    for i in range(M-1):
        var.append(pts[tri[:,i+1],:]-pts[tri[:,0],:])
    return var         


#Optimization------------------
def GD_adapted(fun,x_ini,bounds=None,**options):
    '''Gradient descent with adapted steps'''
    
    if 'step_size' not in options:
        options['step_size'] = 0.001
    
    if 'maxiter' not in options:
        options['maxiter'] = 300
    
    if 'gtol' not in options:
        options['gtol'] = 1e-6
        
    if 'ftol' not in options:
        options['ftol'] = 2.220446049250313e-09
    
    x_old, x_new = x_ini, x_ini
    Enr_gd = fun(x_new)
    enr_hist, gd_hist = [Enr_gd[0]], [Enr_gd[1]]
    
    for iter_ in range(options['maxiter']):
        for i in range(21):
            x_new = x_old - options['step_size']*gd_hist[iter_]
            
            #Project x_new to contrained domain
            #if 'bounds' in options:
            #    ind_lower = np.where(x_new<options['bounds'][:,0])[0]
            #    ind_upper = np.where(x_new>options['bounds'][:,1])[0]
            #    x_new[ind_lower] = options['bounds'][:,0][ind_lower]
            #    x_new[ind_upper] = options['bounds'][:,1][ind_upper]
            if bounds is not None:
                ind_lower = np.where(x_new<bounds[:,0])[0]
                ind_upper = np.where(x_new>bounds[:,1])[0]
                x_new[ind_lower] = bounds[:,0][ind_lower]
                x_new[ind_upper] = bounds[:,1][ind_upper]
            
            Enr_gd = fun(x_new)
            if Enr_gd[0] - enr_hist[iter_]<0:
                options['step_size'] = options['step_size']*1.1
                break
            else:
                options['step_size'] = options['step_size']/2
                
        x_old = x_new
        enr_hist.append(Enr_gd[0])
        gd_hist.append(Enr_gd[1])
        msg = 'iter:{} | fun:{:.5e} | grad norm:{:.5e} | step size:{:.5e} | # step size search {}'.format(iter_+1, 
                Enr_gd[0], np.linalg.norm(Enr_gd[1]), options['step_size'],i)
        print(msg)
        
        if i==20:
            print('Unable to decease the cost!')
            break
            
        if np.abs(enr_hist[iter_+1] - enr_hist[iter_])/np.max([enr_hist[iter_+1],enr_hist[iter_],1]) < options['ftol']:
            print('Reached tolerance of termination')
            break
            
        if np.linalg.norm(Enr_gd[1]) < options['gtol']:
            print('Reached tolerance of termination for gradient')
            break
    
    print('Optimization terminates at iteration {}, function value {:.5f}'.format(iter_+1,Enr_gd[0]))
    #return x_new, enr_hist, gd_hist
    return OptimizeResult(fun=Enr_gd[0], jac=Enr_gd[1], x=x_new, nit=iter_+1, fun_hist=enr_hist, jac_hist=gd_hist)

#Sparse--------------------------------------
def proj2_M_dirac(Ini_vf,Target,options):
    '''
    Find closest linear combination of M Diracs to the Target discrete varifold
    Input:
    Target: Original varifold
    Ini_vf: initial linear combination of M diracs
    Output:
    vf_app: closest linear combination of M diracs to Target
    '''
    
    n, d = Ini_vf[0].shape
    options['dim_varifold'] = len(Ini_vf)  - 1
    vfloss = varifold_dist(**options)
    
    def funopt(x):
        x = torch.from_numpy(x).clone().detach().to(dtype=Target[0].dtype, device=Target[0].device).requires_grad_(True)
        Z = vec2list(x,n,d)
        f = vfloss(Z,Target)
        g = grad(f, x)[0].detach().cpu().numpy().astype('float64')
        f = float(f.detach().cpu().numpy())
        return f,g
    
    x0 = list2vec(Ini_vf).cpu().numpy()
    res = minimize(funopt, x0, method='L-BFGS-B', jac=True, options=options)
    vf_app = vec2list(torch.from_numpy(res.x),n,d)
    return vf_app
    

'''
Fiber bundle processing----------------------------------
'''

def curves_2_graph(curves):
    '''
    Input:List of curve indices(list of 1d numpy array) , eg, [[0,1,2],[3,4]]
    Output:Connectivity matrix (2d numpy array), eg [[0,1],[1,2],[3,4]]
    '''
    first = np.array([]).astype(int)
    sec = np.array([]).astype(int)
    for cur in curves:
        first = np.concatenate((first,cur[:-1]))
        sec = np.concatenate((sec,cur[1:]))
    first = first.reshape(first.shape[0],1)
    sec = sec.reshape(sec.shape[0],1)
    A = np.concatenate((first,sec),axis=1)
    return A

def sample_curves(fiber,num_sample):
    '''Sample num_sample curves in fiber bundle fiber:
    Input:fiber: a dictionary contains points fiber['points'] (2d numpy array) and
          list of curve indices fiber['curves'] (list of 1d numpy array)
    
    Output:new_fib: a dictionary contains sampled points and curve indices
    '''
    num_curves = len(fiber['curves'])
    ind_sample_cur = np.sort(np.random.choice(num_curves,num_sample,replace=False))
    
    ind = fiber['curves'][ind_sample_cur[0]]
    pts = fiber['points'][ind,:]
    curves = [np.arange(ind.shape[0])]
    for i in range(1,num_sample):
        ind = fiber['curves'][ind_sample_cur[i]]
        #print(ind)
        
        curves.append(np.arange(ind.shape[0])+pts.shape[0])
        pts = np.concatenate((pts,fiber['points'][ind,:]),axis=0)
        
    new_fib = {'points':pts,'curves':curves}
    return new_fib