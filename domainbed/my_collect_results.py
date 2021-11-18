import glob
import os
import pandas as pd
import shutil

pd.set_option("display.max_colwidth", 80)
# files_dir = [f for f in glob.glob("./domainbed/results/*") if os.path.isdir(f)]
# # print("files_dir\n", files_dir)

# for files in files_dir:
#     dataset_domain = files.split("/")[-1]
#     print(dataset_domain)
algorithms = [f for f in glob.glob("/root/results/TerraIncognita-test3-trail0/*") if os.path.isdir(f)]

"""
df:
ALG, iid, hold-out, oracle
alg1, X, X, X
alg2, X, X, X
alg3, X, X, X
"""
alg_df = pd.DataFrame(columns=["IID_ACC", "Hold_out_ACC", "Oracle_ACC"])
c = 0
for algorithm in algorithms:
    # if 'DPCLIP-kl-prompt16-gamma0.2' not in algorithm:
    #     continue
    alg = algorithm.split("/")[-1]
    try:
        with open(algorithm + '/out.txt') as f:
            lines = f.readlines()
            if len(lines) > 19:
                print(len(lines))
                c += 1
                name = algorithm.split("/")[-1]
                for i, line in enumerate(lines[:19]):
                    lines[i] = [i for i in line.split(" ") if i != ""][:-1]
                df = pd.DataFrame(lines[1:19], columns=lines[0])
                assert pd.to_numeric(df.loc[17], errors="coerce").notna().sum() == 12
                # print(df.iloc[:])
                # IID.
                df = df.iloc[1:].astype(float)
                idx_oracle_acc = df.env3_out_acc == df.env3_out_acc.max()
                
                LAST_STEP = True
                if LAST_STEP:
                    #  Regard the last step acc as the ORACLE_ACC.
                    oracle_acc = float(df.iloc[[-1], :].env3_in_acc)
                elif len(df[idx_oracle_acc].env3_in_acc) == 1:
                    #  Regard the last step acc as the ORACLE_ACC.
                    oracle_acc = float(df[idx_oracle_acc].env3_in_acc)
                else:
                    oracle_acc = float(df[idx_oracle_acc].env3_in_acc.iloc[0])
                
                # import ipdb; ipdb.set_trace()
                idx_iid_acc = df.loc[:, ["env0_out_acc", "env1_out_acc", "env2_out_acc"]].mean(axis=1).idxmax()
                # print("IID: ", df.iloc[idx_iid_acc-1].env3_in_acc)
                
                iid_acc = df.iloc[idx_iid_acc - 1].env3_in_acc
                # print(iid_acc)
                hold_out_acc = 0
                
                alg_df.loc[name] = [iid_acc, hold_out_acc, oracle_acc]
    except: pass  # runing the experiment.
# print(alg_df)
print(c)
print("TOP_IID_ACC:", alg_df["IID_ACC"].max())

# print(len(alg_df))
print(alg_df[alg_df["IID_ACC"] == alg_df["IID_ACC"].max()])
print("TOP_Oracle_ACC:", alg_df["Oracle_ACC"].max())
print(alg_df[alg_df["Oracle_ACC"] == alg_df["Oracle_ACC"].max()])