python domainbed/scripts/sweep.py delete_incomplete --data_dir=/root/datasets --output_dir=/root/share/sweep_hparam/VLCS --command_launcher multi_gpu --trial_seed 2 --algorithms ERMDPCLIP --datasets VLCS --test_envs 3 --n_hparams_from 3 --n_hparams 4 --skip_confirmation

python domainbed/scripts/sweep.py launch --data_dir=/root/datasets --output_dir=/root/share/sweep_hparam/OfficeHome --command_launcher multi_gpu --trial_seed 2 --algorithms ERMDPCLIP --datasets OfficeHome --test_envs 3 --n_hparams_from 0 --n_hparams 20 --skip_confirmation &!

python domainbed/scripts/sweep.py launch --data_dir=/root/datasets --output_dir=/root/share/sweep_hparam/VLCS --command_launcher multi_gpu --trial_seed 2 --algorithms ERMDPCLIP --datasets VLCS --test_envs 3 --n_hparams_from 0 --n_hparams 20 --skip_confirmation &!

python domainbed/scripts/sweep.py launch --data_dir=/root/datasets --output_dir=/root/share/sweep_hparam/OfficeHome --command_launcher multi_gpu --trial_seed 2 --algorithms ERMDPCLIP --datasets OfficeHome --test_envs 2 --n_hparams_from 10 --n_hparams 20 --skip_confirmation &!


# ==== TerraIncognita
python domainbed/scripts/sweep.py delete_incomplete --data_dir=/root/datasets --output_dir=/root/share/sweep_hparam/TerraIncognita --command_launcher multi_gpu --trial_seed 0 --algorithms DPICLIP --datasets TerraIncognita --test_envs 3 --n_hparams_from 0 --n_hparams 20 --skip_confirmation

python domainbed/scripts/sweep.py launch --data_dir=/root/datasets --output_dir=/root/share/sweep_hparam/TerraIncognita --command_launcher multi_gpu --trial_seed 2 --algorithms DPICLIP --datasets TerraIncognita --test_envs 0 --n_hparams_from 0 --n_hparams 1 --skip_confirmation &!

# ==== PACS
python domainbed/scripts/sweep.py delete_incomplete --data_dir=/root/datasets --output_dir=/root/share/sweep_hparam/PACS --command_launcher multi_gpu --trial_seed 2 --algorithms UDGCLIP --datasets PACS --test_envs 0 --n_hparams_from 0 --n_hparams 20 --skip_confirmation

python domainbed/scripts/sweep.py launch --data_dir=/root/datasets --output_dir=/root/share/sweep_hparam/PACS --command_launcher multi_gpu --trial_seed 2 --algorithms UDGCLIP --datasets PACS --test_envs 0 --n_hparams_from 13 --n_hparams 20 --skip_confirmation &!

# ==== OfficeHome
python domainbed/scripts/sweep.py delete_incomplete --data_dir=/root/datasets --output_dir=/root/share/sweep_hparam/OfficeHome --command_launcher multi_gpu --trial_seed 2 --algorithms APCLIP --datasets OfficeHome --test_envs 3 --n_hparams_from 0 --n_hparams 20 --skip_confirmation


python domainbed/scripts/sweep.py launch --data_dir=/root/datasets --output_dir=/root/share/sweep_hparam/OfficeHome --command_launcher multi_gpu --trial_seed 2 --algorithms APCLIP --datasets OfficeHome --test_envs 3 --n_hparams_from 10 --n_hparams 12 --skip_confirmation &!


# ==== VLCS
python domainbed/scripts/sweep.py delete_incomplete --data_dir=/root/datasets --output_dir=/root/share/sweep_hparam/VLCS --command_launcher multi_gpu --trial_seed 2 --algorithms APCLIP --datasets VLCS --test_envs 3 --n_hparams_from 0 --n_hparams 20 --skip_confirmation

python domainbed/scripts/sweep.py launch --data_dir=/root/datasets --output_dir=/root/share/sweep_hparam/VLCS --command_launcher multi_gpu --trial_seed 0 --algorithms DANNCLIP --datasets VLCS --test_envs 0 --n_hparams_from 0 --n_hparams 20 --skip_confirmation &!


# single_gpu
