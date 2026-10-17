"""Run the complete reproducible simulation, estimation and reporting pipeline."""
import argparse
import json
from pathlib import Path
import numpy as np
from maritime_ai.simulation import simulate
from maritime_ai.filter import track
from maritime_ai.evaluation import position_rmse,forecast_constant_velocity
from maritime_ai.plots import make_plots

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',default='results')
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    df=simulate(seed=args.seed)
    est,nis,rejected,_=track(df)
    truth=df[['east_true_m','north_true_m']].to_numpy()
    obs=df[['east_measured_m','north_measured_m']].to_numpy()
    valid=~df.injected_outlier.to_numpy()
    forecast=forecast_constant_velocity(est,truth)
    metrics={'seed':args.seed,'samples':len(df),'sample_interval_s':5,'measurement_rmse_m':position_rmse(truth,obs),'kalman_rmse_m':position_rmse(truth,est[:,:2]),'measurement_rmse_clean_m':position_rmse(truth,obs,valid),'kalman_rmse_clean_m':position_rmse(truth,est[:,:2],valid),'gated_observations':int(rejected.sum()),'injected_outliers':int(df.injected_outlier.sum()),'forecast_horizon_s':60,**forecast}
    metrics['rmse_reduction_pct']=100*(1-metrics['kalman_rmse_m']/metrics['measurement_rmse_m'])
    df.assign(east_estimated_m=est[:,0],north_estimated_m=est[:,1],east_velocity_estimated_mps=est[:,2],north_velocity_estimated_mps=est[:,3],nis=nis,gated=rejected).to_csv(out/'synthetic_tracking_results.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    make_plots(df,est,nis,rejected,metrics,out)
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
