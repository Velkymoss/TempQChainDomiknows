import os
import random
from typing import Any

import numpy as np
import torch
from domiknows.program.model.base import Mode

from tempQchain.logger import get_logger
from tempQchain.programs.program import program_declaration_tb_dense
from tempQchain.readers.temporal_reader import TemporalReader
from tempQchain.utils import get_class_distribution

logger = get_logger(__name__)


def main(args: Any) -> None:
    SEED = args.seed
    logger.info(f"Model: {args.model}")
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Constraint analysis - loading constraints test dataset...")

    cuda_number = args.cuda
    if cuda_number == -1:
        cur_device = "cpu"
    else:
        cur_device = "cuda:" + str(cuda_number) if torch.cuda.is_available() else "cpu"

    test_constraints_file = "tb_dense_test_constraints.json"
    test_constraints_path = os.path.join(args.data_path, test_constraints_file)

    test_constraints_set = TemporalReader.from_file(
        file_path=test_constraints_path, question_type="FR", batch_size=args.batch_size
    )

    class_distribution = get_class_distribution(test_constraints_set)
    logger.info(f"Test constraints class distribution: {class_distribution}")

    program = program_declaration_tb_dense(
        cur_device,
        pmd=args.pmd,
        beta=args.beta,
        sampling=args.sampling,
        sampleSize=args.sampling_size,
        disable_dropout=args.disable_dropout,
        constraints=args.constraints,
        transitive_enabled=args.transitive_enabled,
        inverse_enabled=args.inverse_enabled,
        class_weights=None,
    )

    pretrain_model = torch.load(
        os.path.join("models", args.model),
        map_location={
            "cuda:0": cur_device,
            "cuda:1": cur_device,
            "cuda:2": cur_device,
            "cuda:3": cur_device,
            "cuda:4": cur_device,
            "cuda:5": cur_device,
        },
    )
    pretrain_dict = pretrain_model
    current_dict = program.model.state_dict()

    new_state_dict = {k: v if k not in pretrain_dict else pretrain_dict[k] for k, v in current_dict.items()}
    program.model.load_state_dict(new_state_dict)

    logger.info("Program instance created for constraint analysis.")

    program.model.to(cur_device)
    program.model.mode(Mode.TEST)
    program.model.reset()

    logger.info("Verifying results for the test constraints set...")
    program.verifyResultsLC(test_constraints_set, device=cur_device)
    logger.info("Constraint analysis completed.")
