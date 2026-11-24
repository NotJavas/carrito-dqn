"""Evaluación común posterior al entrenamiento; no modifica ni reentrena modelos."""
import os
os.environ['SDL_VIDEODRIVER']='dummy'
os.environ['SDL_AUDIODRIVER']='dummy'
import sys, json, csv, time, hashlib, dataclasses, datetime, statistics, argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--project',type=Path,default=ROOT)
parser.add_argument('--output',type=Path,default=ROOT/'runs'/'comparacion_politicas')
args=parser.parse_args()
PROJECT=args.project.resolve()
sys.path.insert(0,str(PROJECT))
import torch
import numpy as np
import pygame
from config import TrainConfig, TrackConfig, CarConfig
from car_env import CarTrackEnv
from dqn_agent import DQNAgent
from heuristic_baseline import heuristic_action
torch.set_num_threads(1)
OUT=args.output.resolve()
if OUT.exists() and any(OUT.iterdir()):
    raise FileExistsError(f'Usa una salida nueva para preservar evidencia: {OUT}')
OUT.mkdir(parents=True,exist_ok=True)
started=datetime.datetime.now().astimezone().isoformat()
model=PROJECT/'results_seed76'/'final_model.pt'
cfg=TrainConfig(**torch.load(model,map_location='cpu',weights_only=False)['config'])
agent=DQNAgent(9,9,cfg,torch.device('cpu')); agent.load(str(model))
records=[]
for mode in ['fijo','aleatorio']:
    for policy in ['DQN_final_seed76','reglas','azar']:
        env=CarTrackEnv()
        for episode in range(30):
            seed=1000+episode
            obs,_=env.reset(seed=seed,options={'random_start':mode=='aleatorio'})
            env.action_space.seed(2000+episode)
            total=0.; done=False; started_run=time.perf_counter()
            while not done:
                action=agent.select_action(obs,training=False) if policy.startswith('DQN') else heuristic_action(obs) if policy=='reglas' else env.action_space.sample()
                obs,reward,terminated,truncated,info=env.step(action)
                total+=reward; done=terminated or truncated
            records.append(dict(inicio=mode,politica=policy,episodio=episode+1,semilla_entorno=seed,semilla_acciones=2000+episode if policy=='azar' else '',retorno=total,exito=int(info.get('success',0)),colision=int(info.get('collision',0)),pasos=int(info['steps']),segundos_simulados=int(info['steps'])/30,tiempo_evaluacion_s=time.perf_counter()-started_run))
        env.close()
with (OUT/'evaluacion_politicas_episodios.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=records[0]); w.writeheader(); w.writerows(records)
summary=[]
for mode in ['fijo','aleatorio']:
    for policy in ['DQN_final_seed76','reglas','azar']:
        rows=[r for r in records if r['inicio']==mode and r['politica']==policy]
        good=[r['pasos'] for r in rows if r['exito']]
        summary.append(dict(inicio=mode,politica=policy,episodios=len(rows),exitos=sum(r['exito'] for r in rows),colisiones=sum(r['colision'] for r in rows),tasa_exito=statistics.mean(r['exito'] for r in rows),tasa_colision=statistics.mean(r['colision'] for r in rows),retorno_medio=statistics.mean(r['retorno'] for r in rows),pasos_vueltas_exitosas=statistics.mean(good) if good else None,DE_pasos_exitosos=statistics.stdev(good) if len(good)>1 else None))
with (OUT/'resumen_politicas.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=summary[0]); w.writeheader(); w.writerows(summary)
frames=OUT/'fotogramas_video'; frames.mkdir(exist_ok=True)
env=CarTrackEnv(render_mode='rgb_array'); obs,_=env.reset(seed=1000,options={'random_start':False})
pygame.init(); font=pygame.font.Font(None,28)
frame_count=0; done=False;total=0
while not done:
    frame=env.render(); surface=pygame.surfarray.make_surface(np.transpose(frame,(1,0,2)))
    label=font.render('DQN final | semilla entrenamiento 76 | inicio fijo | sin exploracion',True,(255,255,255));surface.blit(label,(18,660))
    pygame.image.save(surface,str(frames/f'{frame_count:05d}.jpg'));frame_count+=1
    action=agent.select_action(obs,training=False)
    obs,reward,terminated,truncated,info=env.step(action);total+=reward;done=terminated or truncated
frame=env.render(); surface=pygame.surfarray.make_surface(np.transpose(frame,(1,0,2)))
label=font.render(f"Resultado real: exito {int(info.get('success',0))} | pasos {info['steps']} | retorno {total:.2f}",True,(255,255,255)); surface.blit(label,(18,660))
for _ in range(75):pygame.image.save(surface,str(frames/f'{frame_count:05d}.jpg')); frame_count+=1
env.close()
metadata={'inicio_real':started,'fin_real':datetime.datetime.now().astimezone().isoformat(),'modelo':str(model),'sha256_modelo':hashlib.sha256(model.read_bytes()).hexdigest(),'recompensa_actual':'progress_reward + speed_reward - 0.02 - control_cost','train_config_modelo':dataclasses.asdict(cfg),'track_config':dataclasses.asdict(TrackConfig()),'car_config':dataclasses.asdict(CarConfig()),'protocolo':'30 episodios por política y tipo de inicio; seeds 1000..1029; acciones azar 2000..2029; CPU1 hilo; DQN training=False','seleccion_modelo':'Modelo histórico favorable elegido para demostración; no es réplica independiente de la matriz controlada ni selección universal óptima.','resumen':summary,'video':{'fps':30,'fotogramas':frame_count,'exito':int(info.get('success',0)),'pasos':int(info['steps']),'retorno':total,'inicio_aleatorio':False}}
(OUT/'metadata_evaluacion.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(metadata,ensure_ascii=True,indent=2))
