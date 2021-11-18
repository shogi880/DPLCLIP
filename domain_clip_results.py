import argparse
import functools
import glob
import pickle
import itertools
import json
import os
import random
import sys

import numpy as np
import tqdm

from domainbed import datasets
from domainbed import algorithms
from domainbed.lib import misc, reporting
from domainbed import model_selection
from domainbed.lib.query import Q

import subprocess
import json

import torch

# if __name__ == "__main__":

#     # DEFAULT_ATTRIBUTES = (
#     #     'index',
#     #     'memory.free',
#     # )
#     # def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
#     #     nu_opt = '' if not no_units else ',nounits'
#     #     cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
#     #     output = subprocess.check_output(cmd, shell=True)
#     #     lines = output.decode().split('\n')
#     #     lines = [ line.strip() for line in lines if line.strip() != '' ]

#     #     return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]
#     # gpu_info = get_gpu_info()

#     # n_gpus = torch.cuda.device_count()
#     # procs_by_gpu = [None] * n_gpus

#     # gpu_list = []
#     # for info in gpu_info:
#     #     if int(info['memory.free']) > 10000:
#     #         gpu_list.append(int(info['index']))
#     # print(gpu_list)
#     ALGS = ['UDGCLIP']
#     # ALGS = ['DPICLIP']
#     # ALGS = ['APCLIP']
#     # ALGS = ['DANNCLIP']
#     # ALGS = ['CoOp', 'DPCLIP', 'ERMDPCLIP']
#     # ALGS = ['DPICLIP', 'UDGCLIP', 'DANNCLIP']
#    # ALGS = ['ERMDPCLIP']
    
    
    
#     for ALG in ALGS:
#         merged_dict = []
#         dict_seed = {}
#         file = f'./{ALG}_results_iid.jsonl'
#         results_seed0 = []
#         results_seed1 = []
#         results_seed2 = []
#         with open(file, 'r') as f:
#             for line in f:
#                 l = json.loads(line)
#                 if l['args']['algorithm'] != ALG:
#                     continue
#                 if not dict_seed.get(l['args']['seed']):
#                     if l['args']['hparams_seed'] < 20 and l['args']['trial_seed']  < 3:
#                         dict_seed[l['args']['seed']] = 1
#                         # merged_dict.append(l)
#                         if l['args']['trial_seed'] == 0:
#                             results_seed0.append(l)
#                         elif l['args']['trial_seed'] == 1:
#                             results_seed1.append(l)
#                         elif l['args']['trial_seed'] == 2:
#                             results_seed2.append(l)
#         # print(len(merged_dict))
#         # print(merged_dict[0])
#         # out_file = f'./{ALG}_merged_results.jsonl'
#         # with open(out_file, 'a') as f:
#         #     for i in merged_dict:
#         #         f.write(json.dumps(i, sort_keys=True) + "\n")

#         print(results_seed0[0]['args'])
#         f = {}
#         c = 0
#         _c = 0
#         trial = {}
#         seed = {}
#         print(len(results_seed0) + len(results_seed1) + len(results_seed2))
#         for data in ['PACS', 'VLCS', 'OfficeHome', 'TerraIncognita']:
#         # for data in ['PACS']:
#             for test in [0, 1, 2, 3]:
#             # for test in [3]:
#                 trial[f'{data} - {test}'] = {}
                
#                 # for 3 seed.
#                 i = 0
#                 for results in [results_seed0, results_seed1, results_seed2]:
#                     # print(i)
#                     i += 1
#                     acc_idd = []

#                     # for every result.
#                     acc = {}
#                     for result in results:
#                         k = f"{result['args']['trial_seed']} - {result['args']['hparams_seed']}"
#                         # if f.get(k) is None:
#                         #     f[k] = 1
#                         assert result['args']['trial_seed'] < 3
#                         assert result['args']['hparams_seed'] < 20
#                         assert result['args']['algorithm'] == 'UDGCLIP'
#                         assert result['args']['test_envs'] in [[0], [1], [2], [3]]
                        
#                         # print(result['args']['seed'])
#                         # print(result['args']['seed'])
#                         # print(seed)
#                         # print(f"{result['args']['hparams_seed']} | {result['args']['dataset']} | {result['args']['test_envs']} | {result['acc_iid_best']}")
                        
#                         # if the result is the correct data and test_env.
#                         if result['args']['dataset'] == data and result['args']['test_envs'][0] == test:                        
                            
#                             # filter results with the same hparam_seed.
#                             if seed.get(result['args']['seed']) is None:
#                                 seed[result['args']['seed']] = 1
#                                 # if test in [2]:
#                                 #     print(result['acc_iid_best'])
#                                 acc_idd.append(result['acc_iid_best'])
                    
#                     if len(acc_idd) < 20:
#                         print(len(acc_idd), data, test)
#                     if len(acc_idd) > 20:
#                         print(len(acc_idd), data, test)
#                         print(acc_idd)
#                     if len(acc_idd) >= 20:
#                         # print(len(acc_idd))
#                         trial[f'{data} - {test}'][result['args']['trial_seed']] = np.max(acc_idd)
#                     # else:
#                     #     import ipdb; ipdb.set_trace()
#         # print(_c)
#         # print(trial)
#         for key in trial.keys():
#             # print(key, trial[key])
#             if len(trial[key]) != 3:
#                 print(key, trial[key])
#             # i trial[key] is None:
#             #     print(key)
#         # print(trial)
             
#     # coment out to use.
#     # if False:
#         # print(trial)
#         total = 0
#         data_avg = {'PACS': [], 'VLCS': [], 'OfficeHome': [], 'TerraIncognita': []}
#         data_std = {'PACS': [], 'VLCS': [], 'OfficeHome': [], 'TerraIncognita': []}
#         #  dataset-test-env acc.
        
#         print(trial.keys())
#         # print(trial)
#         # assert len(trial.keys()) == 16
#         for name in trial.keys():
#             # print(trial[name])
#             try:
#                 if int(name.split(' ')[-1]) == 0:
#                     print(ALG)
#                     traial_0 = []
#                     traial_1 = []
#                     traial_2 = []
#                 # trial[name][0] :trial_seed = 0, best iid acc.
#                 # print(max(trial[name][0], trial[name][1], trial[name][2]), min(trial[name][0], trial[name][1], trial[name][2]))
#                 trial_avg = (max(trial[name][0], trial[name][1], trial[name][2]) + min(trial[name][0], trial[name][1], trial[name][2])) / 2
#                 trial_std = (max(trial[name][0], trial[name][1], trial[name][2]) - min(trial[name][0], trial[name][1], trial[name][2])) / 2
#                 print(f'{name}: {trial_avg*100:.1f} $\pm$ {trial_std*100:.1f}')
            
#                 traial_0.append(trial[name][0])
#                 traial_1.append(trial[name][1])
#                 traial_2.append(trial[name][2])
#                 if int(name.split(' ')[-1]) == 3:
#                     avg_0 = np.mean(traial_0)
#                     avg_1 = np.mean(traial_1)
#                     avg_2 = np.mean(traial_2)
#                     # print(f'{name }_max', [avg_0, avg_1, avg_2])
#                     avg = (np.max([avg_0, avg_1, avg_2]) + np.min([avg_0, avg_1, avg_2])) / 2
#                     std = (np.max([avg_0, avg_1, avg_2]) - np.min([avg_0, avg_1, avg_2])) / 2
#                     print(f"{name.split(' ')[0]}: {avg*100:.1f} $\pm$ {std*100:.1f}")
#                     total += avg
#             except:
#                 continue

#         print(f'total: {(total * 100 / 4):.1f}')
        
        
        
#     """MEMO - 11.15
#     DANNCLIP VLCS. test_0
#     DANNCLIP VLCS. test_1,  s7.
#     DANNCLIP VLCS. test_2,  w1.
#     DANNCLIP VLCS. test_3,  s1.
#     DANNCLIP VLCS. OfficeHome_0,  s4.
#     DANNCLIP VLCS. OfficeHome_1,  s4.
#     """

# CLIP and CLIP*
if __name__ == "__main__":

    dataset = ['PACS', 'VLCS', 'OfficeHome', 'TerraIncognita']
    test_env = [0, 1, 2, 3]
    algorithm = ['CLIP', 'DomainCLIP'] 

    file = f'./results/DomainCLIP_results_iid.jsonl'
    results_seed0 = []
    results_seed1 = []
    results_seed2 = []
    dict_seed = {}
    with open(file, 'r') as f:
        for line in f:
            l = json.loads(line)
            # print(l)
            # print(l['args']['seed'])
            if not dict_seed.get(l['args']['seed']):
                dict_seed[l['args']['seed']] = 1
                if l['args']['trial_seed'] == 0:
                    results_seed0.append(l)
                elif l['args']['trial_seed'] == 1:
                    results_seed1.append(l)
                elif l['args']['trial_seed'] == 2:
                    results_seed2.append(l)
    print(len(results_seed0) + len(results_seed1) + len(results_seed2))

    trial = {}
    seed = {}
    for data in ['PACS', 'VLCS', 'OfficeHome', 'TerraIncognita']:
# for data in ['PACS']:
        for test in [0, 1, 2, 3]:
        # for test in [3]:
            trial[f'{data} - {test}'] = {}
            
            # for 3 seed.
            i = 0
            for results in [results_seed0, results_seed1, results_seed2]:
                # print(i)
                i += 1
                acc_idd = []

                # for every result.
                acc = {}
                for result in results:
                    k = f"{result['args']['trial_seed']} - {result['args']['hparams_seed']}"
                    # if f.get(k) is None:
                    #     f[k] = 1
                    assert result['args']['trial_seed'] < 3
                    assert result['args']['hparams_seed'] < 20
                    assert result['args']['algorithm'] == 'DomainCLIP'
                    assert result['args']['test_envs'] in [[0], [1], [2], [3]]
                    
                    # print(result['args']['seed'])
                    # print(result['args']['seed'])
                    # print(seed)
                    # print(f"{result['args']['hparams_seed']} | {result['args']['dataset']} | {result['args']['test_envs']} | {result['acc_iid_best']}")
                    
                    # if the result is the correct data and test_env.
                    if result['args']['dataset'] == data and result['args']['test_envs'][0] == test:                        
                        # print(result)        
                        # filter results with the same hparam_seed.
                        if seed.get(result['args']['seed']) is None:
                            seed[result['args']['seed']] = 1
                            # if test in [2]:
                            #     print(result['acc_iid_best'])
                            acc_idd.append(result['acc_iid_best'])
                
                
                            trial[f'{data} - {test}'][result['args']['trial_seed']] = result['acc_iid_best']
                # else:
        # print(trial)

    for key in trial.keys():
        # print(key, trial[key])
        if len(trial[key]) != 3:
            print(key, trial[key])
        # i trial[key] is None:

    # if False:
    # print(trial)
    total = 0
    data_avg = {'PACS': [], 'VLCS': [], 'OfficeHome': [], 'TerraIncognita': []}
    data_std = {'PACS': [], 'VLCS': [], 'OfficeHome': [], 'TerraIncognita': []}
    #  dataset-test-env acc.
    
    print(trial.keys())
    # print(trial)
    # assert len(trial.keys()) == 16
    for name in trial.keys():
        # print(trial[name])
        try:
            if int(name.split(' ')[-1]) == 0:
                traial_0 = []
                traial_1 = []
                traial_2 = []
            # trial[name][0] :trial_seed = 0, best iid acc.
            # print(max(trial[name][0], trial[name][1], trial[name][2]), min(trial[name][0], trial[name][1], trial[name][2]))
            trial_avg = np.mean([trial[name][0], trial[name][1], trial[name][2]])
            trial_std = np.std([trial[name][0], trial[name][1], trial[name][2]])
            print(f'{name}: {trial_avg*100:.1f} $\pm$ {trial_std*100:.1f}')
        
            traial_0.append(trial[name][0])
            traial_1.append(trial[name][1])
            traial_2.append(trial[name][2])
            if int(name.split(' ')[-1]) == 3:
                avg_0 = np.mean(traial_0)
                avg_1 = np.mean(traial_1)
                avg_2 = np.mean(traial_2)
                # print(f'{name }_max', [avg_0, avg_1, avg_2])
                avg = np.mean([avg_0, avg_1, avg_2])
                std = np.std([avg_0, avg_1, avg_2])
                print(f"{name.split(' ')[0]}: {avg*100:.1f} $\pm$ {std*100:.1f}")
                total += avg
        except:
            continue

    print(f'total: {(total * 100 / 4):.1f}')