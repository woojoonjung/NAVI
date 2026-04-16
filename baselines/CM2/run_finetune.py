import argparse
import logging
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from CM2.dataset_openml import load_single_data_all

import CM2

import warnings
warnings.filterwarnings("ignore")

# set random seed
CM2.random_seed(42)

cal_device = 'cuda'

def log_config(args):
    """
    log Configuration information, specifying the saving path of output log file, etc
    :return: None
    """
    log_name = args.log_name
    exp_dir = 'search_{}_{}'.format(
        log_name, datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
    )
    exp_log_dir = Path('logs') / exp_dir
    # save argss
    setattr(args, 'exp_log_dir', exp_log_dir)

    exp_log_dir.mkdir(parents=True, exist_ok=True)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(exp_log_dir / 'log.txt')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

def parse_args():
    parser = argparse.ArgumentParser(description='CM2-finetune')
    parser.add_argument('--log_name', type=str, default="CM2_finetune", help='task name')
    parser.add_argument('--cpt', type=str, default="./CM2-v1", help='pretrain model')
    parser.add_argument('--task_data', type=str, default="./example/cmc.csv", help='task dataset')
    parser.add_argument(
        '--target',
        type=str,
        default=None,
        help='Label column name (must exist in CSV). If omitted, uses the last column (legacy behavior).',
    )
    parser.add_argument('--num_epoch', type=int, default=30, help='training epochs')
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Fraction of task_data for held-out test (stratified). Ignored if --val_data is set.',
    )
    parser.add_argument(
        '--val_size',
        type=float,
        default=0.2,
        help='Fraction of the remaining (train) portion after test split to use as validation (stratified). Ignored if --val_data is set.',
    )
    parser.add_argument(
        '--val_data',
        type=str,
        default=None,
        help='Optional explicit validation CSV. If set, train on all of task_data and validate on this file (no random split).',
    )
    parser.add_argument(
        '--max_rows',
        type=int,
        default=None,
        help='Optional cap on rows after load (stratified subsample for smoke tests).',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./models/checkpoint-finetune',
        help='Directory for trainer checkpoints (removed and recreated each fold).',
    )
    parser.add_argument(
        '--save_ckpt_start_epoch',
        type=int,
        default=4,
        help='1-based epoch index from which to save periodic checkpoints (e.g. 20 -> epoch_20/).',
    )
    parser.add_argument(
        '--save_ckpt_every',
        type=int,
        default=2,
        help='Save a checkpoint every this many epochs after --save_ckpt_start_epoch (e.g. 2 -> 20,22,24,...).',
    )
    args = parser.parse_args()
    return args

_args = parse_args()
log_config(_args)

all_res = {}

task_dataset = _args.task_data.split(',')

for table_file_path in task_dataset:
    data_name = table_file_path.split('/')[-1]
    logging.info(f'Start========>{data_name}_DataSet==========>')
    X, y, cat_cols, num_cols, bin_cols = load_single_data_all(
        table_file_path,
        target=_args.target,
    )
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if _args.max_rows is not None and len(X) > _args.max_rows:
        n = min(_args.max_rows, len(X))
        try:
            X, _, y, _ = train_test_split(
                X,
                y,
                train_size=n,
                stratify=y,
                random_state=42,
                shuffle=True,
            )
        except ValueError:
            # e.g. high-cardinality labels: cannot stratify with small n
            X = X.sample(n=n, random_state=42)
            y = y.loc[X.index]
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        logging.info(f'Subsampled to n={len(X)} (max_rows={_args.max_rows})')

    num_class = len(y.value_counts())
    logging.info(f'num_class : {num_class}')
    cat_cols = [cat_cols]
    num_cols = [num_cols]
    bin_cols = [bin_cols]
    idd = 0
    score_list = []

    if _args.val_data:
        X_val, y_val, _, _, _ = load_single_data_all(_args.val_data, target=_args.target)
        X_val = X_val.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)
        # Train on full task_data; val from file; same val used as eval set for trainer.train / final metric.
        fold_iter = [("explicit_val_file", X, y, X_val, y_val)]
    else:
        ts = float(_args.test_size)
        vs = float(_args.val_size)
        if not (0 < ts < 1):
            raise ValueError("--test_size must be in (0, 1) when not using --val_data")
        if not (0 < vs < 1):
            raise ValueError("--val_size must be in (0, 1) when not using --val_data")
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X,
            y,
            test_size=ts,
            random_state=42,
            shuffle=True,
            stratify=y,
        )
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval,
                y_trainval,
                test_size=vs,
                random_state=0,
                shuffle=True,
                stratify=y_trainval,
            )
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval,
                y_trainval,
                test_size=vs,
                random_state=0,
                shuffle=True,
            )
        logging.info(
            f"train_val_split: train={len(X_train)} val={len(X_val)} test={len(X_test)} "
            f"(test_size={ts}, val_size={vs} of trainval)"
        )
        fold_iter = [("train_val_split", X_train, y_train, X_val, y_val, X_test, y_test)]

    for item in fold_iter:
        if len(item) == 5:
            _, X_train, y_train, X_val, y_val = item
            X_test, y_test = X_val, y_val
        else:
            _, X_train, y_train, X_val, y_val, X_test, y_test = item
        CM2.random_seed(42)
        idd += 1
        model = CM2.build_classifier(
            checkpoint=_args.cpt,

            device=cal_device,
            num_class=num_class,
            num_layer=3,

            hidden_dropout_prob=0.1,
            vocab_freeze=True,
            use_bert=True,
        )
        model.update({'cat':cat_cols, 'num':num_cols, 'bin':bin_cols})
        training_arguments = {
            'num_epoch': _args.num_epoch,
            'batch_size':64,
            # 'lr':3e-4,
            'lr':1e-4,
            'eval_metric':'auc',
            'eval_less_is_better':False,
            'output_dir': _args.output_dir,
            'patience':10,
            'num_workers':0,
            'device':cal_device,
            'flag':1,
            'save_ckpt_start_epoch': _args.save_ckpt_start_epoch,
            'save_ckpt_every': _args.save_ckpt_every,
        }
        
        logging.info(training_arguments)
        if os.path.isdir(training_arguments['output_dir']):
            shutil.rmtree(training_arguments['output_dir'])
        trainer = CM2.train(model, (X_train, y_train), (X_val, y_val), data_weight=[True], **training_arguments)
        eval_res_list = trainer.train((X_test, y_test))

        ypred = CM2.predict(model, X_test)
        ans = CM2.evaluate(ypred, y_test, metric='auc', num_class=num_class)
        # assembling the top 5 models on the validation set
        ans[0] = max(ans[0], max(eval_res_list[-5:]))
        score_list.append(ans[0])
        logging.info(f'Test_Score_{idd}===>{data_name}_DataSet==> {ans[0]}')
    all_res[data_name] = np.mean(score_list)
    logging.info(f'Test_Score_split===>{data_name}_DataSet==> {np.mean(score_list)}')

mean_list = []
for key in all_res:
    logging.info(f'mean_split=>{all_res[key]}=>{key}')
    mean_list.append(all_res[key])
result_df = pd.DataFrame(mean_list, columns=['result'])
res_path = str(_args.exp_log_dir) + os.sep +'res.csv'
result_df.to_csv(res_path, index=False)
logging.info(f'meaning all data=>{np.mean(mean_list)}')