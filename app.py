import streamlit as st
import numpy as np, pandas as pd, plotly.graph_objects as go
from solver3d import Config,solve,trace
st.set_page_config(page_title="3D Process CFD Lab",layout="wide");st.title("3D Process CFD Lab");st.caption("Python 3D finite-difference backend with flow, heat, species, rheology and particle shear exposure")
with st.sidebar:
 st.header("Domain and mesh");lx=st.number_input("Length (m)",.05,10.,1.,.1);ly=st.number_input("Width (m)",.005,1.,.08,.01);lz=st.number_input("Height (m)",.005,1.,.08,.01);nx=st.slider("Nx",12,60,24);ny=st.slider("Ny",8,40,16);nz=st.slider("Nz",8,40,16)
 st.header("Physics");rho=st.number_input("Density kg/m3",1.,3000.,998.);mu=st.number_input("Reference viscosity Pa.s",1e-5,100.,.001002,format="%.6f");gp=st.number_input("Pressure gradient Pa/m",.01,1e7,25.);rheo=st.selectbox("Rheology",["newtonian","power_law","carreau","herschel_bulkley"]);closure=st.selectbox("Closure",["laminar","mixing_length","sst_surrogate"]);K=st.number_input("Consistency K",1e-5,1000.,.01);n=st.number_input("Flow index n",.05,2.,.8);tau0=st.number_input("Yield stress Pa",0.,1e5,0.);energy=st.checkbox("Energy",True);species=st.checkbox("Species",True);steps=st.slider("Steps",50,800,200,50);run=st.button("Run 3D solver",type="primary",use_container_width=True)
c=Config(nx=nx,ny=ny,nz=nz,lx=lx,ly=ly,lz=lz,rho=rho,mu0=mu,gradp=gp,rheology=rheo,closure=closure,K=K,n=n,tau0=tau0,energy=energy,species=species,steps=steps)
if run or 'r' not in st.session_state:
 b=st.progress(0);st.session_state.r=solve(c,lambda q:b.progress(min(q,1.)));st.session_state.c=c;b.empty()
r=st.session_state.r;c=st.session_state.c;u,v,w=r['u'],r['v'],r['w'];spd=np.sqrt(u*u+v*v+w*w);x,y,z=r['xyz'];X,Y,Z=np.meshgrid(x,y,z,indexing='ij');umean=float(u[-1,1:-1,1:-1].mean());Dh=2*c.ly*c.lz/(c.ly+c.lz);Re=c.rho*abs(umean)*Dh/c.mu0
cols=st.columns(5);cols[0].metric("Outlet velocity",f"{umean:.5f} m/s");cols[1].metric("Re",f"{Re:.0f}");cols[2].metric("Pressure drop",f"{c.gradp*c.lx:.3f} Pa");cols[3].metric("Max shear",f"{r['gdot'].max():.2f} 1/s");cols[4].metric("dt",f"{r['dt']:.2e} s")
def volume(a,title,unit):
 lo,hi=np.nanpercentile(a,[5,95]);fig=go.Figure(go.Volume(x=X.ravel(),y=Y.ravel(),z=Z.ravel(),value=a.ravel(),isomin=float(lo),isomax=float(hi),opacity=.12,surface_count=13,colorscale='Turbo',colorbar_title=unit));fig.update_layout(title=title,height=520,scene_aspectmode='data',margin=dict(l=0,r=0,t=40,b=0));return fig
t1,t2,t3,t4,t5,t6=st.tabs(["3D fields","Slices","Residuals","Particles","Mesh study","Scope"])
with t1:
 choice=st.selectbox("Field",["Velocity","Pressure","Temperature","Concentration","Viscosity","k","omega"]);data={"Velocity":(spd,"m/s"),"Pressure":(r['p'],"Pa"),"Temperature":(r['T'],"K"),"Concentration":(r['C'],"-"),"Viscosity":(r['mu'],"Pa.s"),"k":(r['k'],"m2/s2"),"omega":(r['omega'],"1/s")};a,unit=data[choice];st.plotly_chart(volume(a,choice,unit),width='stretch')
 skip=(slice(None,None,max(1,c.nx//9)),slice(None,None,max(1,c.ny//6)),slice(None,None,max(1,c.nz//6)));fig=go.Figure(go.Cone(x=X[skip].ravel(),y=Y[skip].ravel(),z=Z[skip].ravel(),u=u[skip].ravel(),v=v[skip].ravel(),w=w[skip].ravel(),colorscale='Viridis',sizemode='absolute',sizeref=max(spd.max(),1e-8)/8));fig.update_layout(height=500,scene_aspectmode='data',title='Velocity vectors');st.plotly_chart(fig,width='stretch')
with t2:
 ix=st.slider("x index",0,c.nx-1,c.nx//2);fig=go.Figure(go.Heatmap(x=z,y=y,z=spd[ix],colorscale='Turbo',colorbar_title='m/s'));fig.update_layout(title=f"Velocity slice x={x[ix]:.3f} m",xaxis_title='z',yaxis_title='y');st.plotly_chart(fig,width='stretch')
with t3:
 df=pd.DataFrame(r['hist'],columns=['Iteration','Momentum','Continuity','Pressure','dt']).set_index('Iteration');st.line_chart(df[['Momentum','Continuity','Pressure']]);st.dataframe(df.tail(10),width='stretch')
with t4:
 paths,exp=trace(r,c);fig=go.Figure();
 for j in range(paths.shape[1]):fig.add_trace(go.Scatter3d(x=paths[:,j,0],y=paths[:,j,1],z=paths[:,j,2],mode='lines',line=dict(width=2,color=float(exp[j]),colorscale='Plasma'),showlegend=False))
 fig.update_layout(height=520,scene_aspectmode='data',title='Particle tracks and shear exposure');st.plotly_chart(fig,width='stretch');st.metric("Max cumulative shear exposure",f"{exp.max():.3f}")
with t5:
 if st.button("Run coarse/base/fine"):
  rows=[]
  for f in [.65,1,1.35]:
   cc=Config(**{**c.__dict__,'nx':max(10,int(c.nx*f)),'ny':max(8,int(c.ny*f)),'nz':max(8,int(c.nz*f)),'steps':min(c.steps,120)});rr=solve(cc);q=rr['u'][-1,1:-1,1:-1].mean()*cc.ly*cc.lz;rows.append({'Grid':f'{cc.nx}x{cc.ny}x{cc.nz}','Cells':cc.nx*cc.ny*cc.nz,'Outlet flow m3/s':q})
  ref=rows[-1]['Outlet flow m3/s'];[a.update({'Deviation from fine %':100*abs(a['Outlet flow m3/s']-ref)/max(abs(ref),1e-12)}) for a in rows];st.dataframe(pd.DataFrame(rows),width='stretch')
with t6:
 st.warning("Educational structured-grid prototype. The SST option is a compact algebraic surrogate, not a validated full k-omega SST transport implementation. Use OpenFOAM or ANSYS Fluent for arbitrary CAD, VOF, rotating machinery, sliding mesh and safety-critical design.")
 st.markdown("**Included:** Newtonian, power-law, Carreau, Herschel-Bulkley, adaptive time step, pressure projection, energy, species, 3D fields, particles, shear exposure and mesh study.")
