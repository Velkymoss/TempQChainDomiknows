import torch
from domiknows.program import SolverPOIProgram
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import PrimalDualProgram, SampleLossProgram
from domiknows.program.metric import DatanodeCMMetric, MacroAverageTracker, PRF1Tracker
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ReaderSensor

from tempQchain.graphs.graph_yn import (
    answer_class,
    graph,
    question,
    s_quest1,
    s_quest2,
    story,
    story_contain,
    symmetric,
    t_quest1,
    t_quest2,
    t_quest3,
    transitive,
)
from tempQchain.logger import get_logger
from tempQchain.programs.models import (
    Bert,
    BERTTokenizer,
)
from tempQchain.programs.utils import check_symmetric, check_transitive, read_label, str_to_int_list

logger = get_logger(__name__)


def program_declaration_tb_dense_yn(
    device: torch.device,
    *,
    pmd=False,
    beta=0.5,
    sampling=False,
    sampleSize=1,
    dropout=False,
    constraints=False,
):
    story["questions"] = ReaderSensor(keyword="questions")
    story["stories"] = ReaderSensor(keyword="stories")
    story["relations"] = ReaderSensor(keyword="relation")
    story["question_ids"] = ReaderSensor(keyword="question_ids")
    story["labels"] = ReaderSensor(keyword="labels")

    def make_labels(label_list: str) -> torch.LongTensor:
        labels = label_list.split("@@")
        label_nums = [0 if label == "Yes" else 1 if label == "No" else 2 for label in labels]
        return str_to_int_list(label_nums)

    def make_question(
        questions: str, stories: str, relations: str, q_ids: str, labels: str
    ) -> tuple[torch.Tensor, list[str], list[str], list[str], torch.LongTensor, torch.LongTensor]:
        num_labels = make_labels(labels)
        ids = str_to_int_list(q_ids.split("@@"))
        return (
            torch.ones(len(questions.split("@@")), 1),
            questions.split("@@"),
            stories.split("@@"),
            relations.split("@@"),
            ids,
            num_labels,
        )

    question[story_contain, "question", "story", "relation", "id", "label"] = JointSensor(
        story["questions"],
        story["stories"],
        story["relations"],
        story["question_ids"],
        story["labels"],
        forward=make_question,
        device=device,
    )

    question[answer_class] = FunctionalSensor(story_contain, "label", forward=read_label, label=True, device=device)

    tokenizer = BERTTokenizer()
    question["input_ids", "attention_mask"] = JointSensor(
        story_contain, "question", "story", forward=tokenizer, device=device
    )
    classifier = Bert(
        device="cuda" if torch.cuda.is_available() else "cpu", drp=dropout, num_classes=2, tokenizer=tokenizer.tokenizer
    )
    question[answer_class] = ModuleLearner("input_ids", "attention_mask", module=classifier, device=device)

    poi_list = [question, answer_class]

    if constraints:
        symmetric[s_quest1.reversed, s_quest2.reversed] = CompositionCandidateSensor(
            relations=(s_quest1.reversed, s_quest2.reversed), forward=check_symmetric, device=device
        )

        transitive[t_quest1.reversed, t_quest2.reversed, t_quest3.reversed] = CompositionCandidateSensor(
            relations=(t_quest1.reversed, t_quest2.reversed, t_quest3.reversed),
            forward=check_transitive,
            device=device,
        )

        poi_list.extend([symmetric, transitive])

    infer_list = ["local/argmax"]
    if pmd:
        program = PrimalDualProgram(
            graph,
            SolverModel,
            poi=poi_list,
            inferTypes=infer_list,
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            beta=beta,
            metric={"ILP": PRF1Tracker(DatanodeCMMetric()), "argmax": PRF1Tracker(DatanodeCMMetric("local/argmax"))},
            device=device,
            logger=logger,
        )
    elif sampling:
        program = SampleLossProgram(
            graph,
            SolverModel,
            poi=poi_list,
            inferTypes=infer_list,
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            metric={"ILP": PRF1Tracker(DatanodeCMMetric()), "argmax": PRF1Tracker(DatanodeCMMetric("local/argmax"))},
            sample=True,
            sampleSize=sampleSize,
            sampleGlobalLoss=False,
            beta=1,
            device=device,
        )
    else:
        program = SolverPOIProgram(
            graph,
            poi=poi_list,
            inferTypes=infer_list,
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            metric={"ILP": PRF1Tracker(DatanodeCMMetric()), "argmax": PRF1Tracker(DatanodeCMMetric("local/argmax"))},
            device=device,
        )

    return program
