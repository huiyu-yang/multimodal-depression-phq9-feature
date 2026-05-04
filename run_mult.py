import os
import gc
import time
import random
import logging
import torch
import pynvml
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool

from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import MMDataLoader
from config.config_classification import ConfigClassification

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run(args):
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    args.model_save_path = os.path.join(args.model_save_dir,\
                                        f'{args.modelName}-{args.datasetName}-{args.train_mode}.pth')
    # device
    print("gpu_ids raw:", args.gpu_ids)
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    logger.info("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')
    args.device = device
    # add tmp tensor to increase the temporary consumption of GPU
    tmp_tensor = torch.zeros((100, 100)).to(args.device)
    # load data and models
    dataloader = MMDataLoader(args)
    model = AMIO(args).to(device)

    del tmp_tensor

    def count_parameters(model):
        answer = 0
        for p in model.parameters():
            if p.requires_grad:
                answer += p.numel()
                # print(p)
        return answer
    logger.info(f'The model has {count_parameters(model)} trainable parameters')

    atio = ATIO().getTrain(args)
    atio.do_train(model, dataloader)      # do train

    assert os.path.exists(args.model_save_path)
    model.load_state_dict(torch.load(args.model_save_path))    # load pretrained model
    model.to(device)

    results = atio.do_test(model, dataloader['test'], mode="TEST")  # do test

    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)
 
    return results

def run_normal(args):
    args.res_save_dir = os.path.join(args.res_save_dir, 'normals')
    init_args = args
    model_results = []
    seeds = args.seeds

    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)

    save_path = os.path.join(args.res_save_dir, f'{args.datasetName}-{args.train_mode}.csv')

    # PF9 importance csv (NEW, separate)
    pf9_save_path = os.path.join(
        args.res_save_dir,
        f"{args.datasetName}-{args.train_mode}-{args.modelName}-PF9imp.csv"
    )

    # accumulate for mean row
    pf9_logit_all = []
    pf9_gate_all  = []

    # run results
    for i, seed in enumerate(seeds):
        args = init_args
        config = ConfigClassification(args)
        args = config.get_config()

        setup_seed(seed)
        args.seed = seed
        logger.info('Start running %s...' % (args.modelName))
        logger.info(args)

        # running
        args.cur_time = i + 1
        test_results = run(args)
        test_results["seed"] = seed

        # restore results
        model_results.append(test_results)

        # =========================
        # (A) save PF9 vectors immediately (per seed)
        # =========================
        if ("PF9_imp_logit_vec" in test_results) and ("PF9_imp_gate_vec" in test_results):
            logit_vec = np.asarray(test_results["PF9_imp_logit_vec"], dtype=float).reshape(9,)
            gate_vec  = np.asarray(test_results["PF9_imp_gate_vec"], dtype=float).reshape(9,)

            pf9_logit_all.append(logit_vec)
            pf9_gate_all.append(gate_vec)

            cols = (["Model", "Seed"] +
                    [f"logit_dim{i}" for i in range(9)] +
                    [f"gate_dim{i}" for i in range(9)])

            if os.path.exists(pf9_save_path):
                dfp = pd.read_csv(pf9_save_path)
                # remove old mean row (we'll re-add at end)
                if "Seed" in dfp.columns:
                    dfp = dfp[dfp["Seed"].astype(str) != "mean"]
            else:
                dfp = pd.DataFrame(columns=cols)

            row = {"Model": args.modelName, "Seed": int(seed)}
            for j in range(9):
                row[f"logit_dim{j}"] = float(logit_vec[j])
                row[f"gate_dim{j}"]  = float(gate_vec[j])

            dfp.loc[len(dfp)] = row
            dfp.to_csv(pf9_save_path, index=False)
            logger.info(f"Saved PF9 importance for seed={seed} to {pf9_save_path}")

    # =========================
    # (B) after all seeds: append mean row to PF9 csv
    # =========================
    if len(pf9_logit_all) > 0:
        logit_mean = np.stack(pf9_logit_all, axis=0).mean(axis=0)
        gate_mean  = np.stack(pf9_gate_all, axis=0).mean(axis=0)

        cols = (["Model", "Seed"] +
                [f"logit_dim{i}" for i in range(9)] +
                [f"gate_dim{i}" for i in range(9)])

        dfp = pd.read_csv(pf9_save_path) if os.path.exists(pf9_save_path) else pd.DataFrame(columns=cols)
        if "Seed" in dfp.columns:
            dfp = dfp[dfp["Seed"].astype(str) != "mean"]

        mean_row = {"Model": args.modelName, "Seed": "mean"}
        for j in range(9):
            mean_row[f"logit_dim{j}"] = float(logit_mean[j])
            mean_row[f"gate_dim{j}"]  = float(gate_mean[j])

        dfp.loc[len(dfp)] = mean_row
        dfp.to_csv(pf9_save_path, index=False)
        logger.info(f"Added mean PF9 importance row to {pf9_save_path}")
    else:
        logger.info("No PF9 importance vectors found; PF9imp.csv not created/updated.")

    # =========================
    # (C) original performance saving (unchanged style, but skip PF9 vector keys)
    # =========================
    criterions = list(model_results[0].keys())

    # Remove PF9 vector fields from performance CSV saving to avoid float(list)
    skip_keys = {"PF9_imp_logit_vec", "PF9_imp_gate_vec"}

    # ---- columns ----
    base_cols = ["Model", "Seed"]
    all_cols = base_cols + [c for c in criterions if c not in skip_keys]

    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        for col in all_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[all_cols]
    else:
        df = pd.DataFrame(columns=all_cols)

    # 1) save each seed (one row per seed)
    for r in model_results:
        row = {"Model": args.modelName, "Seed": int(r.get("seed", args.seed))}
        for c in all_cols:
            if c in ("Model", "Seed"):
                continue
            v = r.get(c, None)
            # only save numeric scalars
            if isinstance(v, (int, float, np.floating, np.integer)):
                row[c] = round(float(v) * 100, 2)
            else:
                row[c] = np.nan
        df.loc[len(df)] = row

    # 2) save ONE summary row: (mean, std) for numeric scalars
    summary = {"Model": args.modelName, "Seed": "mean"}
    for c in all_cols:
        if c in ("Model", "Seed"):
            continue
        values = []
        for r in model_results:
            v = r.get(c, None)
            if isinstance(v, (int, float, np.floating, np.integer)):
                values.append(float(v) * 100)
        if len(values) > 0:
            mean = round(np.mean(values), 2)
            std  = round(np.std(values), 2)
            summary[c] = (mean, std)
        else:
            summary[c] = np.nan

    df.loc[len(df)] = summary
    df.to_csv(save_path, index=None)
    logger.info(f'Per-seed rows + one summary (mean,std) row are added to {save_path}...')


def set_log(args):
    log_file_path = f'logs/{args.modelName}-{args.datasetName}.log'
    # set logging
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)

    for ph in logger.handlers:
        logger.removeHandler(ph)
    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    return logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', type=str, default="classification",
                        help='regression / classification')
    parser.add_argument('--modelName', type=str, default='mult',
                        help='support mult')
    parser.add_argument('--datasetName', type=str, default='cmdc',
                        help='support cmdc')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers of loading data')
    parser.add_argument('--model_save_dir', type=str, default='results/mult-cmdc-f1PF',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='results/mult-cmdc-f1PF',
                        help='path to save results.')
    parser.add_argument('--gpu_ids', type=list, default=[0],
                        help='indicates the gpus will be used. If none, the most-free gpu will be used!')
    # modified by huiyu
    parser.add_argument('--epochs', type=int, default=30, help='indicates the epochs!')
    parser.add_argument('--personal_hidden', type=int, default=32)
    parser.add_argument('--pf_fusion', type=str, default="feature_gate_dim",
                        choices=["decision", "feature_gate", "feature_gate_dim", "feature_concat", "none"])
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    logger = set_log(args)
    args.seeds = [1111, 1112, 1113, 1114, 1115]
    run_normal(args)
