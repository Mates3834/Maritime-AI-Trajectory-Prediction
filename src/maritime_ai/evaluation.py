"""Tracking and short-horizon forecast metrics."""
import numpy as np

def position_rmse(actual, predicted, mask=None):
    a=np.asarray(actual);p=np.asarray(predicted)
    if mask is not None:
        a=a[mask];p=p[mask]
    return float(np.sqrt(np.mean(np.sum((a-p)**2,axis=1))))

def forecast_constant_velocity(estimates, truth, dt=5.0, horizon_steps=12, start=20):
    # Forecast only from estimated state at t; no future observations used.
    errors=[];endpoint=[]
    for i in range(start,len(estimates)-horizon_steps):
        h=np.arange(1,horizon_steps+1)[:,None]
        pred=estimates[i,:2]+h*dt*estimates[i,2:4]
        actual=truth[i+1:i+1+horizon_steps]
        distances=np.linalg.norm(pred-actual,axis=1)
        errors.extend(distances.tolist());endpoint.append(distances[-1])
    return dict(ade_m=float(np.mean(errors)),fde_m=float(np.mean(endpoint)))
