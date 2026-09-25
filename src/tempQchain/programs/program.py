import torch
from domiknows.program import SolverPOIProgram
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import LearningBasedProgram, PrimalDualProgram, SampleLossProgram
from domiknows.program.metric import DatanodeCMMetric, MacroAverageTracker, PRF1Tracker
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ReaderSensor

from tempQchain.graphs.graph import (
    answer_class,
    graph,
    question,
    story,
    story_contain,
    symmetric,
    symmetric_question1,
    symmetric_question2,
    tran_quest1,
    tran_quest2,
    tran_quest3,
    transitive,
)
from tempQchain.logger import get_logger
from tempQchain.programs.models import (
    Bert,
    BERTTokenizer,
)
from tempQchain.programs.utils import check_symmetric, check_transitive, read_label, str_to_int_list

logger = get_logger(__name__)


def program_declaration_tb_dense(
    device: torch.device,
    *,
    pmd: bool = False,
    beta: float = 0.5,
    sampling: bool = False,
    sampleSize: int = 10,
    dropout: bool = False,
    constraints: bool = False,
    class_weights: torch.FloatTensor = None,
    transitive_enabled: bool = True,
    inverse_enabled: bool = True,
) -> LearningBasedProgram:
    program = None

    story["questions"] = ReaderSensor(keyword="questions")
    story["stories"] = ReaderSensor(keyword="stories")
    story["relations"] = ReaderSensor(keyword="relation")
    story["question_ids"] = ReaderSensor(keyword="question_ids")
    story["labels"] = ReaderSensor(keyword="labels")

    def make_labels(label_list: str) -> torch.LongTensor:
        labels = label_list.split("@@")
        return str_to_int_list(labels)

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
        device="cuda" if torch.cuda.is_available() else "cpu", drp=dropout, num_classes=6, tokenizer=tokenizer.tokenizer
    )
    question[answer_class] = ModuleLearner("input_ids", "attention_mask", module=classifier, device=device)

    poi_list = [
        question,
        answer_class,
    ]

    if constraints:
        if transitive_enabled and inverse_enabled:
            logger.info("Using transitive and inverse constraints")

            symmetric[symmetric_question1.reversed, symmetric_question2.reversed] = CompositionCandidateSensor(
                relations=(symmetric_question1.reversed, symmetric_question2.reversed),
                forward=check_symmetric,
                device=device,
            )

            transitive[tran_quest1.reversed, tran_quest2.reversed, tran_quest3.reversed] = CompositionCandidateSensor(
                relations=(tran_quest1.reversed, tran_quest2.reversed, tran_quest3.reversed),
                forward=check_transitive,
                device=device,
            )

            poi_list.extend([symmetric, transitive])

        elif transitive_enabled and not inverse_enabled:
            logger.info("Using transitive constraints only")
            transitive[tran_quest1.reversed, tran_quest2.reversed, tran_quest3.reversed] = CompositionCandidateSensor(
                relations=(tran_quest1.reversed, tran_quest2.reversed, tran_quest3.reversed),
                forward=check_transitive,
                device=device,
            )
            poi_list.extend([transitive])

        elif not transitive_enabled and inverse_enabled:
            logger.info("Using inverse constraints only")
            symmetric[symmetric_question1.reversed, symmetric_question2.reversed] = CompositionCandidateSensor(
                relations=(symmetric_question1.reversed, symmetric_question2.reversed),
                forward=check_symmetric,
                device=device,
            )
            poi_list.extend([symmetric])

    infer_list = ["ILP", "local/argmax"]
    if pmd:
        if class_weights is not None:
            criterion = NBCrossEntropyLoss(weight=class_weights)
        else:
            criterion = NBCrossEntropyLoss()

        program = PrimalDualProgram(
            graph,
            SolverModel,
            poi=poi_list,
            inferTypes=infer_list,
            loss=MacroAverageTracker(criterion),
            beta=beta,
            metric={"ILP": PRF1Tracker(DatanodeCMMetric()), "argmax": PRF1Tracker(DatanodeCMMetric("local/argmax"))},
            device=device,
            logger=logger,
        )
    elif sampling:
        if class_weights is not None:
            criterion = NBCrossEntropyLoss(weight=class_weights)
        else:
            criterion = NBCrossEntropyLoss()
        program = SampleLossProgram(
            graph,
            SolverModel,
            poi=poi_list,
            inferTypes=infer_list,
            loss=MacroAverageTracker(criterion),
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
