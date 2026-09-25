import torch
from domiknows.program import SolverPOIProgram
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import JointSensor, ReaderSensor

from tests.graphs.conftest import (
    FrSpecificDummyLearner,
    assert_ilp_result,
    assert_local_softmax,
    check_transitive,
    make_question,
)
from tests.graphs.graph import get_graph


def test_transitive(device):
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

    question[answer_class] = FrSpecificDummyLearner(story_contain, num_labels=6, predictions=[0, 0, -1], device=device)

    transitive[tran_quest1.reversed, tran_quest2.reversed, tran_quest3.reversed] = CompositionCandidateSensor(
        relations=(tran_quest1.reversed, tran_quest2.reversed, tran_quest3.reversed),
        forward=check_transitive,
        device=device,
    )

    poi_list = [question, answer_class, transitive]

    program = SolverPOIProgram(graph=graph, poi=poi_list, device=device)

    for datanode in program.populate(dataset=synthetic_dataset):
        print("\n=== BEFORE ILP INFERENCE ===")
        print(f"Number of questions: {len(datanode.getChildDataNodes())}")
        for i, q_node in enumerate(datanode.getChildDataNodes()):
            print(f"\nQuestion {i}:")
            print(f"  Dummy Prediction: {q_node.getAttribute(answer_class, 'local/softmax')}")
            if i == 0 or i == 1:
                assert_local_softmax(
                    q_node, answer_class, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device), device=device
                )
            else:
                assert_local_softmax(
                    q_node,
                    answer_class,
                    torch.tensor([0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667], device=device),
                    device=device,
                )

        print("\n=== RUNNING ILP INFERENCE ===")
        datanode.inferILPResults()

        print("\n=== AFTER ILP INFERENCE ===")
        print("\nILP predictions (after constraint enforcement):")
        for i, q_node in enumerate(datanode.getChildDataNodes()):
            print(f"\nQuestion {i}:")
            print(f"Inferred constraint: {q_node.getAttribute(answer_class, 'ILP')}")

            assert_ilp_result(
                q_node, answer_class, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device), device=device
            )
