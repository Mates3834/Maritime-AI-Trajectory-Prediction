"""Synthetic AIS-like trajectories in local east/north metres (not real AIS)."""
import numpy as np
import pandas as pd

def simulate(seed=42, n=360, dt=5.0, sigma=28.0):
    rng=np.random.default_rng(seed)
    t=np.arange(n)*dt
    # Nonlinear, smooth motion with distinct manoeuvring periods.
    heading=np.deg2rad(35+4*np.sin(t/280)+np.where(t>650,22*np.clip((t-650)/180,0,1),0)-np.where(t>1260,18*np.clip((t-1260)/180,0,1),0))
    speed=6.2+0.65*np.sin(t/210)
    velocity=np.column_stack([speed*np.sin(heading),speed*np.cos(heading)])
    pos=np.zeros((n,2))
    pos[1:]=np.cumsum(velocity[:-1]*dt,axis=0)
    measured=pos+rng.normal(0,sigma,size=pos.shape)
    # Deliberately anomalous GPS-like observations, with explicit labels.
    anomaly=np.zeros(n,dtype=bool)
    anomaly[[85,185,270]]=True
    measured[anomaly]+=np.array([[220,-170],[-230,180],[200,200]])
    return pd.DataFrame(dict(time_s=t,east_true_m=pos[:,0],north_true_m=pos[:,1],east_measured_m=measured[:,0],north_measured_m=measured[:,1],speed_true_mps=speed,heading_true_deg=np.rad2deg(heading),injected_outlier=anomaly))
