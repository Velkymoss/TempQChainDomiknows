# import torch
# from domiknows.program import SolverPOIProgram
# from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
# from domiknows.sensor.pytorch.sensors import JointSensor, ReaderSensor

# from tests.graphs.conftest import (
#     FrSpecificDummyLearner,
#     assert_ilp_result,
#     assert_local_softmax,
#     check_symmetric,
#     make_question,
# )
# from tests.graphs.graph import get_graph


# def test_symmetric(device):
#     (
#         graph,
#         story,
#         question,
#         transitive,
#         inverse,
#         answer_class,
#         story_contain,
#         tran_quest1,
#         tran_quest2,
#         tran_quest3,
#         inv_quest1,
#         inv_quest2,
#     ) = get_graph(symmetric=True)

#     synthetic_dataset = [
#         {
#             "questions": "When did t21 happen in time compared to e1?@@When did e1 happen in time compared to t21?",
#             "stories": "story@@story",
#             "relation": "@@symmetric,0",
#             "question_ids": "0@@1",
#             "labels": "4@@4",
#         }
#     ]

#     story["questions"] = ReaderSensor(keyword="questions")
#     story["stories"] = ReaderSensor(keyword="stories")
#     story["relations"] = ReaderSensor(keyword="relation")
#     story["question_ids"] = ReaderSensor(keyword="question_ids")
#     story["labels"] = ReaderSensor(keyword="labels")

#     question[story_contain, "question", "story", "relation", "id", "label"] = JointSensor(
#         story["questions"],
#         story["stories"],
#         story["relations"],
#         story["question_ids"],
#         story["labels"],
#         forward=make_question,
#         device=device,
#     )

#     question[answer_class] = FrSpecificDummyLearner(story_contain, num_labels=6, predictions=[4, -1], device=device)

#     inverse[inv_quest1.reversed, inv_quest2.reversed] = CompositionCandidateSensor(
#         relations=(inv_quest1.reversed, inv_quest2.reversed),
#         forward=check_symmetric,
#         device=device,
#     )

#     poi_list = [question, answer_class, inverse]

#     program = SolverPOIProgram(graph=graph, poi=poi_list, device=device)

#     for datanode in program.populate(dataset=synthetic_dataset):
#         print("\n=== BEFORE ILP INFERENCE ===")
#         print(f"Number of questions: {len(datanode.getChildDataNodes())}")
#         for i, q_node in enumerate(datanode.getChildDataNodes()):
#             print(f"\nQuestion {i}:")
#             print(f"  Dummy Prediction: {q_node.getAttribute(answer_class, 'local/softmax')}")
#             if i == 0:
#                 assert_local_softmax(
#                     q_node, answer_class, torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device), device=device
#                 )
#             else:
#                 assert_local_softmax(
#                     q_node,
#                     answer_class,
#                     torch.tensor([0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667], device=device),
#                     device=device,
#                 )

#         print("\n=== RUNNING ILP INFERENCE ===")
#         datanode.inferILPResults()

#         print("\n=== AFTER ILP INFERENCE ===")
#         print("\nILP predictions (after constraint enforcement):")
#         for i, q_node in enumerate(datanode.getChildDataNodes()):
#             print(f"\nQuestion {i}:")
#             print(f"Inferred constraint: {q_node.getAttribute(answer_class, 'ILP')}")

#             assert_ilp_result(
#                 q_node, answer_class, torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device), device=device
#             )

import numpy as np
import pytest
import torch
from domiknows.program import SolverPOIProgram
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import JointSensor, ReaderSensor

from tests.graphs.conftest import (
    FrSpecificDummyLearner,
    assert_ilp_result,
    check_symmetric,
    make_question,
)
from tests.graphs.graph import get_graph


@pytest.mark.parametrize(
    "predictions,log_conclusion_weight,expected_ilp,vacuously_true",
    [
        # Case 1: uniform logits distribution for second question
        (
            [4, -1],
            1.0,
            [
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            False,
        ),
        # Case 2: Second question predicts label 0, logits of second question smaller
        (
            [4, 0],
            0.005,
            [
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            False,
        ),
        # Case 3: Test vacuous satisfaction of constraint
        # Second question predicts label 0, logits have equal size for both questions
        (
            [4, 0],
            1.0,
            [
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            True,
        ),
    ],
)
def test_symmetric(
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
    ) = get_graph(symmetric=True)

    synthetic_dataset = [
        {
            "questions": "When did t21 happen in time compared to e1?@@When did e1 happen in time compared to t21?",
            "stories": "story@@story",
            "relation": "@@symmetric,0",
            "question_ids": "0@@1",
            "labels": "4@@4",
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

    inverse[inv_quest1.reversed, inv_quest2.reversed] = CompositionCandidateSensor(
        relations=(inv_quest1.reversed, inv_quest2.reversed),
        forward=check_symmetric,
        device=device,
    )

    poi_list = [question, answer_class, inverse]
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
                if idx != 4:
                    unwanted_conclusion = [0.0] * NUM_LABELS
                    unwanted_conclusion[idx] = 1.0
                    unwanted_matrix = np.array([expected_ilp[0], unwanted_conclusion])
                    print(f"Unwanted matrix violating constraints:\n{unwanted_matrix}")
                    assert np.any(ilp_prediction_matrix != unwanted_matrix), (
                        f"ILP prediction matrix \n{ilp_prediction_matrix}\n must not be equal to the unwanted matrix\n {unwanted_matrix}."
                    )
