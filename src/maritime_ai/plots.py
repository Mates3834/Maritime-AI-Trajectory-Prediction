"""Publication-friendly visualizations, generated from actual computations."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def make_plots(df,est,nis,rejected,metrics,outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    truth=df[['east_true_m','north_true_m']].to_numpy()
    measured=df[['east_measured_m','north_measured_m']].to_numpy()
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    def save(name):
        plt.tight_layout();plt.savefig(outdir/name,bbox_inches='tight',dpi=170);plt.close()
    fig,ax=plt.subplots(figsize=(11,5.8))
    ax.plot(truth[:,0],truth[:,1],label='Simulated vessel track',lw=2.4)
    ax.scatter(measured[::3,0],measured[::3,1],label='Noisy AIS-like observations',s=10,alpha=.35)
    ax.plot(est[:,0],est[:,1],label='Kalman estimate',lw=1.8)
    ax.scatter(measured[rejected,0],measured[rejected,1],marker='x',s=50,label='Gated observations')
    ax.set(xlabel='East (m)',ylabel='North (m)',title='Vessel tracking | simulated maritime trajectory');ax.legend();ax.axis('equal');save('01_trajectory_tracking.png')
    raw=np.linalg.norm(measured-truth,axis=1);filtered=np.linalg.norm(est[:,:2]-truth,axis=1)
    fig,ax=plt.subplots(figsize=(11,4.2));ax.plot(df.time_s,raw,alpha=.65,label='Measurement error');ax.plot(df.time_s,filtered,label='Kalman estimation error');ax.set(xlabel='Time (s)',ylabel='Position error (m)',title='Position error over time');ax.legend();save('02_position_error.png')
    fig,ax=plt.subplots(figsize=(7.5,4.6));vals=[metrics['measurement_rmse_m'],metrics['kalman_rmse_m']];bars=ax.bar(['Noisy measurements','Kalman filter'],vals,width=.52);ax.bar_label(bars,fmt='%.1f m',padding=4);ax.set(ylabel='2D position RMSE (m)',title='Position estimation performance');ax.set_ylim(0,max(vals)*1.23);save('03_rmse_comparison.png')
    fig,ax=plt.subplots(figsize=(11,4.2));ax.plot(df.time_s,nis,label='Normalized innovation squared (NIS)',lw=1);ax.axhline(13.82,ls='--',label='Gating threshold (2D, 99.9%)');ax.scatter(df.time_s[rejected],nis[rejected],marker='x',label='Rejected updates');ax.set(xlabel='Time (s)',ylabel='NIS',title='Innovation-based observation gating');ax.legend();save('04_innovation_gating.png')
    fig,ax=plt.subplots(figsize=(10,4.5));ax.plot(df.time_s,df.speed_true_mps,label='Simulated speed');ax.plot(df.time_s,np.linalg.norm(est[:,2:4],axis=1),label='Estimated speed');ax.set(xlabel='Time (s)',ylabel='Speed (m/s)',title='Speed state estimation');ax.legend();save('05_speed_estimation.png')
