#!/bin/sh

# <Argments>
# $1 : trial_seed
# $2 : datasets
# $3 : test_envs
# $4 : algorithms (ERM, CORAL, etc)
# 
# <Example>
# sh scripts/hparam_search.sh 0 PACS 0 DPCLIP

# trial_seed = 0

# python domainbed/scripts/sweep.py delete_incomplete --data_dir=/root/datasets --output_dir=/root/share/sweep_hparam/TerraIncognita --command_launcher multi_gpu --trial_seed 0 --algorithms ERMDPCLIP --datasets TerraIncognita --test_envs 1 --n_hparams_from 13 --n_hparams 14 --skip_confirmation

# python domainbed/scripts/sweep.py launch --data_dir=/root/datasets --output_dir=/root/share/sweep_hparam/TerraIncognita --command_launcher multi_gpu --trial_seed 0 --algorithms ERMDPCLIP --datasets TerraIncognita --test_envs 1 --n_hparams_from 0 --n_hparams 20 --skip_confirmation &!




# python domainbed/scripts/sweep.py delete_incomplete \
# --data_dir=/root/datasets \
# --output_dir=/root/share/sweep_hparam/$1 \
# --command_launcher multi_gpu \
# --trial_seed 2 \
# --algorithms $3 \
# --datasets $1 \
# --test_envs $2 \
# --n_hparams_from 0 \
# --n_hparams 20 \
# --skip_confirmation

python domainbed/scripts/sweep.py delete_incomplete \
--data_dir=/root/datasets \
--output_dir=/root/share/sweep_hparam/$1 \
--command_launcher multi_gpu \
--trial_seed $4 \
--algorithms $3 \
--datasets $1 \
--test_envs $2 \
--n_hparams_from $5 \
--n_hparams $6 \
--skip_confirmation

sleep 3
python domainbed/scripts/sweep.py launch \
--data_dir=/root/datasets \
--output_dir=/root/share/sweep_hparam/$1 \
--command_launcher multi_gpu \
--trial_seed $4 \
--algorithms $3 \
--datasets $1 \
--test_envs $2 \
--n_hparams_from $5 \
--n_hparams $6 \
--skip_confirmation
