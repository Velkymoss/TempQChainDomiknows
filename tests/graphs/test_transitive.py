import numpy as np
import pytest
import torch
from domiknows.program import SolverPOIProgram
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import JointSensor, ReaderSensor

from tests.graphs.conftest import (
    FrSpecificDummyLearner,
    assert_ilp_result,
    check_transitive,
    make_question,
)
from tests.graphs.graph import get_graph


@pytest.mark.parametrize(
    "predictions,log_conclusion_weight,expected_ilp,vacuously_true",
    [
        # Case 1: uniform logits distribution for 3rd question
        (
            [0, 0, -1],
            1.0,
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            False,
        ),
        # Case 2: Third question predicts label 1, logits of third question smaller
        (
            [0, 0, 1],
            0.005,
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            False,
        ),
        # Case 3: Test vacuous satisfaction of constraint
        # Third question predicts label 1, logits have equal size for all 3 questions
        (
            [0, 0, 1],
            1.0,
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            True,
        ),
    ],
)
def test_transitive(
    device,
    predictions,
    log_conclusion_weight,
    expected_ilp,
    vacuously_true,
):
    NUM_LABELS = 6
    (
        graph,
        story,
        question,
        transitive,
        inverse,
        answer_class,
        story_contain,
        tran_quest1,
        tran_quest2,
        tran_quest3,
        inv_quest1,
        inv_quest2,
    ) = get_graph(transitive_determin=True)

    synthetic_dataset = [
        {
            "questions": "A B?@@B C?@@A C?",
            "stories": "story@@story@@story",
            "relation": "@@@@transitive,0,1",
            "question_ids": "0@@1@@2",
            "labels": "0@@0@@0",
        }
    ]

    story["questions"] = ReaderSensor(keyword="questions")
    story["stories"] = ReaderSensor(keyword="stories")
    story["relations"] = ReaderSensor(keyword="relation")
    story["question_ids"] = ReaderSensor(keyword="question_ids")
    story["labels"] = ReaderSensor(keyword="labels")

    question[story_contain, "question", "story", "relation", "id", "label"] = JointSensor(
        story["questions"],
        story["stories"],
        story["relations"],
        story["question_ids"],
        story["labels"],
        forward=make_question,
        device=device,
    )

    # Parameterized learner
    question[answer_class] = FrSpecificDummyLearner(
        story_contain,
        num_labels=NUM_LABELS,
        predictions=predictions,
        logit_weight_conclusion=log_conclusion_weight,
        device=device,
    )

    transitive[tran_quest1.reversed, tran_quest2.reversed, tran_quest3.reversed] = CompositionCandidateSensor(
        relations=(tran_quest1.reversed, tran_quest2.reversed, tran_quest3.reversed),
        forward=check_transitive,
        device=device,
    )

    poi_list = [question, answer_class, transitive]
    program = SolverPOIProgram(graph=graph, poi=poi_list, device=device)

    for datanode in program.populate(dataset=synthetic_dataset):
        print(f"\n=== BEFORE ILP INFERENCE (predictions={predictions}) ===")

        for i, q_node in enumerate(datanode.getChildDataNodes()):
            print(f"\nQuestion {i}:")
            print(f"  Dummy Prediction: {q_node.getAttribute(answer_class, 'local/softmax')}")

        print("\n=== RUNNING ILP INFERENCE ===")
        datanode.inferILPResults()
        print("\n=== AFTER ILP INFERENCE ===")
        print("\nILP predictions (after constraint enforcement):")
        if vacuously_true:
            ilp_predictions = []
        for i, q_node in enumerate(datanode.getChildDataNodes()):
            print(f"\nQuestion {i}:")
            print(f"Inferred constraint: {q_node.getAttribute(answer_class, 'ILP')}")
            if vacuously_true:
                ilp_predictions.append(q_node.getAttribute(answer_class, "ILP"))
            else:
                assert_ilp_result(q_node, answer_class, torch.tensor(expected_ilp[i], device=device), device=device)
        if vacuously_true:
            ilp_prediction_matrix = np.array([result for result in ilp_predictions])
            unwanted_matrix = np.array([expected_ilp[0], expected_ilp[1]])
            print(f"ILP predictions:\n{ilp_prediction_matrix}")
            for idx in range(0, NUM_LABELS):
                if idx > 0:
                    unwanted_conclusion = [0.0] * NUM_LABELS
                    unwanted_conclusion[idx] = 1.0
                    unwanted_matrix = np.array([expected_ilp[0], expected_ilp[1], unwanted_conclusion])
                    print(f"Unwanted matrix violating constraints:\n{unwanted_matrix}")
                    assert np.any(ilp_prediction_matrix != unwanted_matrix), (
                        f"ILP prediction matrix \n{ilp_prediction_matrix}\n must not be equal to the unwanted matrix\n {unwanted_matrix}."
                    )
