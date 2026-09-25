from typing import Sequence

import pytest
import torch
from domiknows.sensor.pytorch.learners import TorchLearner

from tempQchain.logger import get_logger

logger = get_logger(__name__)


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # return torch.device("cpu")


def str_to_int_list(x, device=None) -> torch.LongTensor:
    return (
        torch.LongTensor([int(i) for i in x]).to(device)
        if device is not None
        else torch.LongTensor([int(i) for i in x])
    )


def check_symmetric(**kwargs) -> bool:
    args = list(kwargs.values())
    if len(args) != 2:
        return False
    arg1, arg2 = args[0], args[1]

    if arg1 == arg2:
        return False
    relation_arg2 = arg2.getAttribute("relation")
    if relation_arg2 == "":
        return False
    relation_describe = relation_arg2.split(",")
    if relation_describe[0] == "symmetric":
        qid1 = arg1.getAttribute("id").item()
        if qid1 == int(relation_describe[1]):
            return True
    return False


def check_inverse(**kwargs) -> bool:
    """Check if arg2 is the inverse of arg1."""
    args = list(kwargs.values())
    if len(args) != 2:
        return False
    arg1, arg2 = args[0], args[1]

    if arg1 == arg2:
        return False
    relation_arg2 = arg2.getAttribute("relation")
    if relation_arg2 == "":
        return False
    relation_describe = relation_arg2.split(",")
    if relation_describe[0] == "inverse":
        qid1 = arg1.getAttribute("id").item()
        if qid1 == int(relation_describe[1]):
            return True
    return False


def check_transitive(**kwargs) -> bool:
    args = list(kwargs.values())
    if len(args) != 3:
        return False
    arg11, arg22, arg33 = args[0], args[1], args[2]

    if arg11 == arg22 or arg11 == arg33 or arg22 == arg33:
        return False
    relation_arg3 = arg33.getAttribute("relation")
    if relation_arg3 == "":
        return False
    # exmple of relation_describe: ['transitive', '1', '4']
    relation_describe = relation_arg3.split(",")

    if relation_describe[0] == "transitive":
        qid1 = arg11.getAttribute("id").item()
        qid2 = arg22.getAttribute("id").item()
        if qid1 == int(relation_describe[1]) and qid2 == int(relation_describe[2]):
            return True
    return False


def assert_local_softmax(q_node, label, expected_tensor, device=None):
    """Assert local/softmax predictions match expected values"""
    result = q_node.getAttribute(label, "local/softmax")
    if device is not None:
        result = result.to(device)
        expected_tensor = expected_tensor.to(device)
    assert torch.allclose(result, expected_tensor, atol=1e-4), (
        f"Label {label}: Expected {expected_tensor}, got {result}"
    )


def assert_ilp_result(q_node, label, expected_tensor, device=None):
    """Assert ILP predictions match expected value"""
    result = q_node.getAttribute(label, "ILP")
    if device is not None:
        result = result.to(device)
        expected_tensor = expected_tensor.to(device)
    assert torch.allclose(result, expected_tensor), f"Label {label}: Expected {expected_tensor}, got {result}"


class FrSpecificDummyLearner(TorchLearner):
    """
    Learner that outputs one-hot-like predictions based on predefined label indices.

    Args:
        *pre: Positional arguments for TorchLearner
        predictions: List of label indices for each batch position.
                     Use -1 to output all zeros for that position.
        num_labels: Total number of labels
        high_score: Score for the predicted label (default: 1000)
        low_score: Score for non-predicted labels (default: -1000)
        device: Device for tensors
    """

    def __init__(
        self,
        *pre,
        predictions: list[int],
        num_labels: int,
        high_score: float = 500.0,
        low_score: float = -500.0,
        logit_weight_conclusion: float = 1.0,
        device=None,
    ):
        TorchLearner.__init__(self, *pre)
        self.predictions = predictions
        self.num_labels = num_labels
        self.high_score = high_score
        self.low_score = low_score
        self.logit_weight_conclusion = logit_weight_conclusion
        self.device = device

    def forward(self, x: Sequence) -> torch.Tensor:
        batch_size = len(x)
        device = self.device or (x.device if isinstance(x, torch.Tensor) else "cpu")

        result = torch.full((batch_size, self.num_labels), self.low_score, device=device)

        for i in range(min(batch_size, len(self.predictions))):
            pred_idx = self.predictions[i]

            if pred_idx == -1:
                result[i, :] = 0
            elif 0 <= pred_idx < self.num_labels:
                result[i, pred_idx] = self.high_score

            if self.logit_weight_conclusion != 1.0 and i == 2:
                result[i, :] *= self.logit_weight_conclusion
        logger.info(f"Dummy learner logits: \n{result}")
        return result


def make_question(
    questions: str, stories: str, relations: str, q_ids: str, labels: str, device=None
) -> tuple[torch.Tensor, list[str], list[str], list[str], torch.LongTensor, torch.LongTensor]:
    num_labels = str_to_int_list(labels.split("@@"), device=device)
    ids = str_to_int_list(q_ids.split("@@"), device=device)
    return (
        torch.ones(len(questions.split("@@")), 1, device=device)
        if device is not None
        else torch.ones(len(questions.split("@@")), 1),
        questions.split("@@"),
        stories.split("@@"),
        relations.split("@@"),
        ids,
        num_labels,
    )
