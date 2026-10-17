"""Constant-velocity Kalman filtering with innovation gating."""
import numpy as np

def track(df, dt=5.0, measurement_sigma=28.0, accel_sigma=0.14, gate_threshold=13.82):
    F=np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],float)
    H=np.array([[1,0,0,0],[0,1,0,0]],float)
    G=np.array([[dt*dt/2,0],[0,dt*dt/2],[dt,0],[0,dt]])
    Q=(accel_sigma**2)*G@G.T
    R=np.eye(2)*measurement_sigma**2
    x=np.array([df.east_measured_m.iloc[0],df.north_measured_m.iloc[0],0,0],float)
    P=np.diag([measurement_sigma**2]*2+[16.0]*2)
    estimates=[]; nis=[]; rejected=[]; innovations=[]
    I=np.eye(4)
    for i,row in enumerate(df.itertuples()):
        if i:
            x=F@x;P=F@P@F.T+Q
        z=np.array([row.east_measured_m,row.north_measured_m])
        y=z-H@x
        S=H@P@H.T+R
        d2=float(y@np.linalg.solve(S,y))
        is_outlier=bool(d2>gate_threshold and i>3)
        if not is_outlier:
            K=np.linalg.solve(S,(P@H.T).T).T
            x=x+K@y
            # Joseph form for covariance numerical stability.
            A=I-K@H
            P=A@P@A.T+K@R@K.T
        estimates.append(x.copy());nis.append(d2);rejected.append(is_outlier);innovations.append(y.copy())
    return np.asarray(estimates),np.asarray(nis),np.asarray(rejected),np.asarray(innovations)
