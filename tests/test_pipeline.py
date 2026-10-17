import numpy as np
from maritime_ai.simulation import simulate
from maritime_ai.filter import track
from maritime_ai.evaluation import position_rmse,forecast_constant_velocity

def test_simulation_is_reproducible():
    a=simulate(seed=12);b=simulate(seed=12)
    assert a.equals(b)

def test_filter_reduces_position_rmse():
    df=simulate()
    est,nis,rejected,_=track(df)
    true=df[['east_true_m','north_true_m']].to_numpy()
    noisy=df[['east_measured_m','north_measured_m']].to_numpy()
    assert position_rmse(true,est[:,:2])<position_rmse(true,noisy)
    assert np.isfinite(nis).all()
    assert rejected.dtype==bool

def test_forecast_is_finite():
    df=simulate();est,*_=track(df)
    m=forecast_constant_velocity(est,df[['east_true_m','north_true_m']].to_numpy())
    assert m['ade_m']>0 and np.isfinite(m['fde_m'])
