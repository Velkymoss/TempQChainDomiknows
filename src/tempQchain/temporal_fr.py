import os
import random
from datetime import datetime
from typing import Any

import mlflow
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

from tempQchain.logger import get_logger
from tempQchain.programs.program_fr import (
    program_declaration_tb_dense_fr,
)
from tempQchain.readers.temporal_reader import TemporalReader
from tempQchain.utils import get_train_labels

logger = get_logger(__name__)


def main(args: Any) -> None:
    SEED = 382
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    logger.info("Starting FR training...")
    logger.info(f"Model: {args.model}")
    logger.info("Using Hyperparameters:")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Weight Decay: {args.weight_decay}")
    logger.info(f"Epochs: {args.epoch}")
    logger.info(f"Patience: {args.patience}")
    logger.info(f"Batch Size: {args.batch_size}")
    if args.constraints:
        logger.info("Using Constraints")
    if args.dropout:
        logger.info("Using Dropout")
    if args.pmd:
        logger.info(f"Using Primal Dual Method with Beta: {args.beta}")
    if args.sampling:
        logger.info(f"Using Sampling Loss with Size: {args.sampling_size}")

    if args.use_mlflow:
        if not args.run_name:
            run_name = f"{args.model}_{datetime.now().strftime('%Y-%d-%m_%H:%M:%S')}"
        logger.info(f"Starting run with id {args.run_name if args.run_name else run_name}")
        mlflow.set_experiment("Temporal_FR")
        mlflow.start_run(run_name=args.run_name if args.run_name else run_name)

        mlflow.log_param("model", args.model)
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("weight_decay", args.weight_decay)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs", args.epoch)
        mlflow.log_param("patience", args.patience)
        mlflow.log_param("use_constraints", args.constraints)
        mlflow.log_param("use_dropout", args.dropout)
        mlflow.log_param("use_pmd", args.pmd)
        if args.pmd:
            mlflow.log_param("pmd_beta", args.beta)
        mlflow.log_param("use_sampling", args.sampling)
        if args.sampling:
            mlflow.log_param("sampling_size", args.sampling_size)
        mlflow.log_param("use_class_weights", args.use_class_weights)
        mlflow.log_param("cuda", args.cuda)

    cuda_number = args.cuda
    if cuda_number == -1:
        cur_device = "cpu"
    else:
        cur_device = "cuda:" + str(cuda_number) if torch.cuda.is_available() else "cpu"

    logger.info("Loading training data...")
    train_file = "tb_dense_train.json"
    training_set = TemporalReader.from_file(
        file_path=os.path.join(args.data_path, train_file), question_type="FR", batch_size=args.batch_size
    )

    if args.use_class_weights:
        train_labels = get_train_labels(training_set)
        class_weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
        logger.info(f"Calculated class weigths {class_weights}")
        class_weights = torch.FloatTensor(class_weights).to(cur_device)

    logger.info("Loading development data...")
    eval_file = "tb_dense_dev.json"
    eval_set = TemporalReader.from_file(
        file_path=os.path.join(args.data_path, eval_file), question_type="FR", batch_size=args.batch_size
    )

    logger.info("Loading test data...")
    test_file = "tb_dense_test.json"
    test_set = TemporalReader.from_file(
        file_path=os.path.join(args.data_path, test_file), question_type="FR", batch_size=args.batch_size
    )

    program = program_declaration_tb_dense_fr(
        cur_device,
        pmd=args.pmd,
        beta=args.beta,
        sampling=args.sampling,
        sampleSize=args.sampling_size,
        dropout=args.dropout,
        constraints=args.constraints,
        class_weights=class_weights if args.use_class_weights else None,
    )

    results = program.train(
        training_set=training_set,
        valid_set=eval_set,
        test_set=test_set,
        train_epoch_num=args.epoch,
        Optim=lambda param: torch.optim.AdamW(param, lr=args.lr, weight_decay=args.weight_decay),
        patience=args.patience,
        device=cur_device,
        model_dir=args.best_model_dir,
        best_model_name=args.best_model_name,
        # batch_size=args.batch_size,
        c_lr=args.c_lr,
        c_warmup_iters=args.c_warmup_iters,
        c_freq_increase=args.c_freq_increase,
        c_freq_increase_freq=args.c_freq_increase_freq,
        c_lr_decay=args.c_lr_decay,
        c_lr_decay_param=args.c_lr_decay_param,
    )

    history = results["history"]
    if args.use_mlflow:
        for epoch, (train_loss, val_loss, train_f1, val_f1) in enumerate(
            zip(history["train_loss"], history["val_loss"], history["train_f1"], history["val_f1"]), start=1
        ):
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("train_f1", train_f1, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)

        mlflow.log_metric("best_epoch", results["best_epoch"])
        mlflow.log_metric("best_val_f1", results["best_val_f1"])

        best_model_path = os.path.join(args.best_model_dir, args.best_model_name)
        mlflow.log_artifact(best_model_path)

        if "test_loss" in results:
            mlflow.log_metric("test_loss", results["test_loss"])
        if "test_f1_macro" in results:
            mlflow.log_metric("test_f1_macro", results["test_f1_macro"])
        if "test_f1_per_class" in results:
            for class_label, f1_score in results["test_f1_per_class"].items():
                mlflow.log_metric(f"test_f1_{class_label}", f1_score)

    if args.use_mlflow:
        mlflow.end_run()
