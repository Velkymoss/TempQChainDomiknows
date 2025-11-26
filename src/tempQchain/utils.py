import torch
import tqdm
from collections import Counter
from domiknows.program.lossprogram import LearningBasedProgram
from domiknows.program.model.base import Mode


def get_avg_loss(
    program: LearningBasedProgram, dataset: list[dict[str, str]], cur_device: str | None, mode: str
) -> float:
    if cur_device is not None:
        program.model.to(cur_device)
    program.model.mode(Mode.TEST)
    program.model.reset()
    train_loss = 0
    total_loss = 0
    with torch.no_grad():
        for data_item in tqdm.tqdm(dataset, f"Calculating {mode} loss" if mode else "Calculating loss"):
            loss, _, *output = program.model(data_item)
            total_loss += 1
            train_loss += loss
    return train_loss / total_loss

def get_train_labels(dataset: list[dict[str, str]]) -> list[int]:
    labels = []
    for batch in dataset:
        batch_labels = batch["labels"].split("@@")
        batch_labels = [int(label) for label in batch_labels]
        labels.extend(batch_labels)
    return labels

def get_train_labels(dataset: list[dict[str, str]]) -> list[int]:
    labels = []
    for batch in dataset:
        batch_labels = batch["labels"].split("@@")
        batch_labels = [int(label) for label in batch_labels]
        labels.extend(batch_labels)
    return labels

def get_class_distribution(dataset: list[dict[str, str]]) -> dict[int, dict[str, int | float]]:
    labels = get_train_labels(dataset)
    total_samples = len(labels)
    
    class_counts = Counter(labels)
    
    distribution = {}
    for class_label in sorted(class_counts.keys()):
        count = class_counts[class_label]
        distribution[class_label] = {
            'count': count,
            'percentage': (count / total_samples) * 100 if total_samples > 0 else 0.0
        }
    
    return distribution