# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess
import time
import torch

momery = 3000

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

def dummy_launcher(commands):
    """
    Doesn't run anything; instead, prints each command.
    Useful for testing.
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    

    DEFAULT_ATTRIBUTES = (
        'index',
        'memory.free',
    )

    def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
        nu_opt = '' if not no_units else ',nounits'
        cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
        output = subprocess.check_output(cmd, shell=True)
        lines = output.decode().split('\n')
        lines = [ line.strip() for line in lines if line.strip() != '' ]

        return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]
    gpu_info = get_gpu_info()

    n_gpus = torch.cuda.device_count()
    procs_by_gpu = [None] * n_gpus

    gpu_list = []
    for info in gpu_info:
        if int(info['memory.free']) > momery:
            gpu_list.append(int(info['index']))
    print('GPU: ', gpu_list)
    
    while len(commands) > 0:
        for gpu_idx in gpu_list:
            # to address out of gpu memory error.
            proc = procs_by_gpu[gpu_idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[gpu_idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()
    # n_gpus = torch.cuda.device_count()
    # procs_by_gpu = [None]*n_gpus

    # while len(commands) > 0:
    #     for gpu_idx in range(n_gpus):
    #         proc = procs_by_gpu[gpu_idx]
    #         if (proc is None) or (proc.poll() is not None):
    #             # Nothing is running on this GPU; launch a command.
    #             cmd = commands.pop(0)
    #             new_proc = subprocess.Popen(
    #                 f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
    #             procs_by_gpu[gpu_idx] = new_proc
    #             break
    #     time.sleep(1)

    

REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'multi_gpu': multi_gpu_launcher
}

try:
    from domainbed import facebook
    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass
