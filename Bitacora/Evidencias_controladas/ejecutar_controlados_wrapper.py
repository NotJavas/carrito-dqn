"""Ejecuta una corrida real con las fuentes aisladas y evalúa su modelo final."""
import argparse
import csv
import json
import os
import platform
import runpy
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
from zoneinfo import ZoneInfo

parser=argparse.ArgumentParser()
parser.add_argument('--source',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--seed',required=True,type=int)
parser.add_argument('--coefficient',required=True,type=float)
args=parser.parse_args()
source=Path(args.source).resolve()
out=Path(args.output).resolve()
out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(source))
os.environ['MPLCONFIGDIR']=str(out.parent/'cache_matplotlib')
os.environ['MPLBACKEND']='Agg'
import numpy as np
import torch
torch.set_num_threads(1)
from config import TrainConfig,TrackConfig,CarConfig
assert TrainConfig().success_window==100
assert TrainConfig().early_stop_rate==0.99
def now():
    t=datetime.now(timezone.utc)
    return {'utc':t.isoformat(),'america_mexico_city':t.astimezone(ZoneInfo('America/Mexico_City')).isoformat()}
metadata={'status':'running','start':now(),'real_execution':True,'source_directory':str(source),
    'source_manifest':str(source/'manifest_fuentes.json'),'output_directory':str(out),
    'condition_step_coefficient':args.coefficient,'seed':args.seed,'max_episodes':600,
    'cpu_threads':torch.get_num_threads(),'cpu_interop_threads':torch.get_num_interop_threads(),
    'device':'cpu','python':platform.python_version(),'pytorch':torch.__version__,
    'numpy':np.__version__,'snapshot_train_config_defaults':asdict(TrainConfig()),
    'track_config':asdict(TrackConfig()),'car_config':asdict(CarConfig()),
    'argv':['train.py','--episodes','600','--seed',str(args.seed),'--output',str(out)]}
meta_path=out/'metadata_corrida.json'
def write_meta():meta_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
write_meta()
start=time.perf_counter()
try:
    sys.argv=metadata['argv']
    train_start=time.perf_counter()
    runpy.run_path(str(source/'train.py'),run_name='__main__')
    metadata['training_end']=now()
    metadata['training_wall_seconds']=time.perf_counter()-train_start
    with (out/'training_log.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    metadata['episodes_executed']=len(rows)
    metadata['total_training_steps']=sum(int(r['steps']) for r in rows)
    ck=torch.load(out/'final_model.pt',map_location='cpu',weights_only=True)
    metadata['effective_train_config']=ck['config']
    (out/'metadata_traincfg.json').write_text(json.dumps(ck['config'],ensure_ascii=False,indent=2),encoding='utf-8')
    recent=rows[-100:]
    recent_rate=sum(int(r['success']) for r in recent)/len(recent)
    metadata['last_window_success_rate']=recent_rate
    metadata['stop_reason']='early_stop' if len(recent)==100 and recent_rate>=0.99 else 'episode_budget'
    from car_env import CarTrackEnv
    from dqn_agent import DQNAgent
    environment=CarTrackEnv(render_mode=None,track_config=TrackConfig(),car_config=CarConfig())
    observation,_=environment.reset(seed=10000,options={'random_start':False})
    agent=DQNAgent(len(observation),environment.action_space.n,TrainConfig(**ck['config']),torch.device('cpu'))
    agent.policy_net.load_state_dict(ck['policy_state_dict'])
    agent.target_net.load_state_dict(ck['target_state_dict'])
    agent.policy_net.eval()
    evaluations=[]
    metadata['evaluation_start']=now()
    ev_start=time.perf_counter()
    try:
        for episode in range(10):
            observation,_=environment.reset(seed=10000+episode,options={'random_start':False})
            done=False; reward_sum=0.0; steps=0; info={}
            while not done:
                action=agent.select_action(observation,training=False)
                observation,reward,terminated,truncated,info=environment.step(action)
                reward_sum+=reward; steps+=1; done=terminated or truncated
            evaluations.append({'episode':episode+1,'reset_seed':10000+episode,'random_start':False,
                'reward':reward_sum,'success':int(info.get('success',0)),'collision':int(info.get('collision',0)),
                'steps':steps,'simulated_seconds':steps*CarConfig().dt})
    finally:environment.close()
    with (out/'evaluation_episodes.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(evaluations[0]));writer.writeheader();writer.writerows(evaluations)
    successful=[r for r in evaluations if r['success']]
    metadata['evaluation']={'checkpoint':'final_model.pt','episodes':10,'render':None,'training':False,
        'random_start':False,'reset_seeds':list(range(10000,10010)),
        'unique_initial_conditions':1,'repeated_same_scenario':True,
        'success_rate':sum(r['success'] for r in evaluations)/10,
        'collision_rate':sum(r['collision'] for r in evaluations)/10,
        'mean_return':sum(r['reward'] for r in evaluations)/10,
        'successful_episodes':len(successful),
        'mean_steps_successful':sum(r['steps'] for r in successful)/len(successful) if successful else None,
        'mean_simulated_seconds_successful':sum(r['simulated_seconds'] for r in successful)/len(successful) if successful else None,
        'all_episode_outcomes_identical':len({(r['reward'],r['success'],r['collision'],r['steps']) for r in evaluations})==1,
        'wall_seconds':time.perf_counter()-ev_start}
    metadata['status']='completed'
except BaseException as e:
    metadata['status']='failed';metadata['error']=repr(e);metadata['traceback']=traceback.format_exc()
    raise
finally:
    metadata['end']=now();metadata['total_wall_seconds']=time.perf_counter()-start;write_meta()
    print('RUN_METADATA',json.dumps(metadata,ensure_ascii=False),flush=True)
