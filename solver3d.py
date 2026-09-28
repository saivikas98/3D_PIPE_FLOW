import numpy as np
from dataclasses import dataclass

@dataclass
class Config:
    nx:int=24; ny:int=16; nz:int=16
    lx:float=1.; ly:float=.08; lz:float=.08
    rho:float=998.; mu0:float=.001002; gradp:float=25.
    rheology:str="newtonian"; closure:str="laminar"
    K:float=.01; n:float=.8; tau0:float=0.; mu_inf:float=.0001; lam:float=1.; a:float=2.
    energy:bool=True; species:bool=True; Tin:float=298.15; Tw:float=308.15
    cp:float=4180.; kcond:float=.6; Dm:float=1e-8
    steps:int=200; dtmax:float=2e-4; cfl:float=.3; piter:int=30; tol:float=1e-4

def d(a,h,ax): return (np.roll(a,-1,ax)-np.roll(a,1,ax))/(2*h)
def lap(a,dx,dy,dz): return (np.roll(a,-1,0)-2*a+np.roll(a,1,0))/dx**2+(np.roll(a,-1,1)-2*a+np.roll(a,1,1))/dy**2+(np.roll(a,-1,2)-2*a+np.roll(a,1,2))/dz**2
def wall(a,val=0):
    a[:,0,:]=val;a[:,-1,:]=val;a[:,:,0]=val;a[:,:,-1]=val;return a

def shear(u,v,w,dx,dy,dz):
    ux,uy,uz=d(u,dx,0),d(u,dy,1),d(u,dz,2);vx,vy,vz=d(v,dx,0),d(v,dy,1),d(v,dz,2);wx,wy,wz=d(w,dx,0),d(w,dy,1),d(w,dz,2)
    return np.sqrt(np.maximum(2*(ux*ux+vy*vy+wz*wz)+(uy+vx)**2+(uz+wx)**2+(vz+wy)**2,1e-16))
def viscosity(g,c):
    g=np.maximum(g,1e-6)
    if c.rheology=='power_law': return np.clip(c.K*g**(c.n-1),1e-6,1e3)
    if c.rheology=='carreau': return c.mu_inf+(c.mu0-c.mu_inf)*(1+(c.lam*g)**c.a)**((c.n-1)/c.a)
    if c.rheology=='herschel_bulkley': return np.clip(c.tau0/g+c.K*g**(c.n-1),1e-6,1e3)
    return np.full_like(g,c.mu0)
def turbulence(u,v,w,g,c,dy,dz):
    if c.closure=='laminar': return np.zeros_like(u),np.zeros_like(u),np.zeros_like(u)
    yy=np.minimum(np.arange(c.ny)*dy,(c.ny-1-np.arange(c.ny))*dy)[None,:,None];zz=np.minimum(np.arange(c.nz)*dz,(c.nz-1-np.arange(c.nz))*dz)[None,None,:];dist=np.maximum(np.minimum(yy,zz),min(dy,dz)/2)
    if c.closure=='mixing_length':
        lm=.41*dist*(1-np.exp(-dist*np.maximum(abs(u),1e-9)/(26*c.mu0/c.rho)));return c.rho*lm*lm*g,np.zeros_like(u),np.zeros_like(u)
    k=np.maximum(1e-10,.015*(u*u+v*v+w*w));om=np.maximum(np.sqrt(k)/(.09**.25*dist),1e-8);F2=np.tanh(np.maximum(2*np.sqrt(k)/(.09*om*dist),500*(c.mu0/c.rho)/(dist*dist*om))**2);mut=c.rho*.31*k/np.maximum(.31*om,g*F2)
    return np.clip(mut,0,200*c.mu0),k,om

def solve(c,progress=None):
    sh=(c.nx,c.ny,c.nz);dx=c.lx/(c.nx-1);dy=c.ly/(c.ny-1);dz=c.lz/(c.nz-1)
    u=np.zeros(sh);v=np.zeros(sh);w=np.zeros(sh);p=np.zeros(sh);T=np.full(sh,c.Tin);C=np.zeros(sh);C[0]=1;hist=[]
    for it in range(c.steps):
        g=shear(u,v,w,dx,dy,dz);mu=viscosity(g,c);mut,k,om=turbulence(u,v,w,g,c,dy,dz);nu=(mu+mut)/c.rho
        speed=np.max(np.sqrt(u*u+v*v+w*w));dt=min(c.dtmax,c.cfl*min(dx,dy,dz)/max(speed,1e-4),.15*min(dx,dy,dz)**2/max(nu.max(),1e-12));uo=u.copy();vo=v.copy();wo=w.copy()
        us=u+dt*(-u*d(u,dx,0)-v*d(u,dy,1)-w*d(u,dz,2)+nu*lap(u,dx,dy,dz)+c.gradp/c.rho);vs=v+dt*(-u*d(v,dx,0)-v*d(v,dy,1)-w*d(v,dz,2)+nu*lap(v,dx,dy,dz));ws=w+dt*(-u*d(w,dx,0)-v*d(w,dy,1)-w*d(w,dz,2)+nu*lap(w,dx,dy,dz));wall(us);wall(vs);wall(ws)
        rhs=c.rho*(d(us,dx,0)+d(vs,dy,1)+d(ws,dz,2))/dt;pn=p.copy()
        for _ in range(c.piter):
            pn=((np.roll(pn,-1,0)+np.roll(pn,1,0))*dy**2*dz**2+(np.roll(pn,-1,1)+np.roll(pn,1,1))*dx**2*dz**2+(np.roll(pn,-1,2)+np.roll(pn,1,2))*dx**2*dy**2-rhs*dx**2*dy**2*dz**2)/(2*(dy**2*dz**2+dx**2*dz**2+dx**2*dy**2));pn[-1]=0;pn[0]=pn[1];pn[:,0]=pn[:,1];pn[:,-1]=pn[:,-2];pn[:,:,0]=pn[:,:,1];pn[:,:,-1]=pn[:,:,-2]
        p=pn;u=us-dt/c.rho*d(p,dx,0);v=vs-dt/c.rho*d(p,dy,1);w=ws-dt/c.rho*d(p,dz,2);wall(u);wall(v);wall(w)
        if c.energy:
            T+=dt*(-u*d(T,dx,0)-v*d(T,dy,1)-w*d(T,dz,2)+c.kcond/(c.rho*c.cp)*lap(T,dx,dy,dz));T[:,0]=c.Tw;T[:,-1]=c.Tw;T[:,:,0]=c.Tw;T[:,:,-1]=c.Tw;T[0]=c.Tin
        if c.species:C=np.clip(C+dt*(-u*d(C,dx,0)-v*d(C,dy,1)-w*d(C,dz,2)+c.Dm*lap(C,dx,dy,dz)),0,1);C[0]=1
        div=d(u,dx,0)+d(v,dy,1)+d(w,dz,2);mom=max(abs(u-uo).max(),abs(v-vo).max(),abs(w-wo).max())/max(abs(u).max(),1e-9);cont=abs(div).max()*c.lx/max(abs(u).max(),1e-9);pres=abs(lap(p,dx,dy,dz)-rhs).max()/max(abs(rhs).max(),1e-9)
        if it%5==0:hist.append((it+1,mom,cont,pres,dt))
        if progress and it%10==0:progress((it+1)/c.steps)
        if it>20 and max(mom,cont,pres)<c.tol:break
    xyz=(np.linspace(0,c.lx,c.nx),np.linspace(0,c.ly,c.ny),np.linspace(0,c.lz,c.nz))
    return dict(u=u,v=v,w=w,p=p,T=T,C=C,mu=mu,k=k,omega=om,gdot=g,hist=np.array(hist),xyz=xyz,dt=dt)

def trace(r,c,n=40,steps=120):
    rng=np.random.default_rng(4);x=np.zeros(n);y=rng.uniform(.1*c.ly,.9*c.ly,n);z=rng.uniform(.1*c.lz,.9*c.lz,n);exp=np.zeros(n);paths=[];dx=c.lx/(c.nx-1);dy=c.ly/(c.ny-1);dz=c.lz/(c.nz-1);dt=.35*dx/max(r['u'].max(),1e-8)
    for _ in range(steps):
        ix=np.clip((x/dx).astype(int),0,c.nx-1);iy=np.clip((y/dy).astype(int),0,c.ny-1);iz=np.clip((z/dz).astype(int),0,c.nz-1);x+=r['u'][ix,iy,iz]*dt;y+=r['v'][ix,iy,iz]*dt;z+=r['w'][ix,iy,iz]*dt;exp+=r['gdot'][ix,iy,iz]*dt;x=np.clip(x,0,c.lx);y=np.clip(y,0,c.ly);z=np.clip(z,0,c.lz);paths.append(np.c_[x.copy(),y.copy(),z.copy()])
    return np.stack(paths),exp
