import os
import random
from typing import Any

import numpy as np
import torch
from domiknows.program.lossprogram import LearningBasedProgram
from sklearn.utils.class_weight import compute_class_weight

from tempQchain.logger import get_logger
from tempQchain.programs.program_fr import (
    program_declaration_tb_dense_fr,
)
from tempQchain.readers.temporal_reader import TemporalReader
from tempQchain.utils import get_train_labels

logger = get_logger(__name__)

LABEL_STRINGS = ["after", "before", "includes", "is_included", "simultaneous", "vague"]


def train(
    program: LearningBasedProgram,
    train_set: list[dict[str, str]],
    eval_set: list[dict[str, str]],
    test_set: list[dict[str, str]] | None,
    cur_device: str | None,
    lr: float,
    program_name: str = "DomiKnow",
    args: Any = None,
) -> None:
    logger.info("Starting FR training...")
    logger.info(f"Model: {args.model}")
    logger.info("Using Hyperparameters:")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Batch Size: {args.batch_size}")
    if args.constraints:
        logger.info("Using Constraints")
    if args.dropout:
        logger.info("Using Dropout")
    if args.pmd:
        logger.info(f"Using Primal Dual Method with Beta: {args.beta}")
    if args.sampling:
        logger.info(f"Using Sampling Loss with Size: {args.sampling_size}")

    program.train(
        training_set=train_set,
        valid_set=eval_set,
        train_epoch_num=args.epoch,
        Optim=lambda param: torch.optim.AdamW(param, lr=args.lr),
        device=cur_device,
    )

    if program.model.metric:
        metrics = program.model.metric["argmax"].value()["answer_class"]
        logger.info(metrics)
        loss = program.model.loss.value()["answer_class"]
        logger.info(loss)

    if test_set:
        logger.info("Final evaluation on test set")
        program.test(test_set)
        if program.model.metric:
            test_metrics = program.model.metric["argmax"].value()["answer_class"]
            logger.info(f"Test Metrics: {test_metrics}")
        if program.model.loss:
            test_loss = program.model.loss.value()["answer_class"]
            logger.info(f"Test Loss: {test_loss}")


def main(args: Any) -> None:
    SEED = 382
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

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

    program_name = "PMD" if args.pmd else "Sampling" if args.sampling else "Base"

    train(
        program=program,
        train_set=training_set,
        eval_set=eval_set,
        cur_device=cur_device,
        lr=args.lr,
        program_name=program_name,
        test_set=test_set,
        args=args,
    )
