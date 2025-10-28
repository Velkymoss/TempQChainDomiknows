import json
import os
from string import Template

from bs4 import BeautifulSoup

from tempQchain.logger import get_logger

logger = get_logger(__name__)


def read_txt(path_to_file):
    """

    :param path_to_file: Path to the txt dataset file
    :return: List with the lines of the txt file
    """
    with open(path_to_file, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    tlinks = []
    for line in lines:
        tlinks.append(line.split())

    return tlinks


def check_trans_for_pairs(pairs):
    """

    :param pairs: A list of pairs
    :return: The count of transitive pairs found and a list with the triples of transitive pairs
    """
    trans_count = 0
    trans_pairs = []
    for p1 in pairs:
        p2 = []
        for p in pairs:
            if p[0] == p1[1]:
                p2.append(p)
        for second_pair in p2:
            if (p1[0], second_pair[1]) in pairs:
                trans_count += 1
                trans_pairs.append([p1, second_pair, (p1[0], second_pair[1])])
    return trans_count, trans_pairs


def create_story_triplets(doc_pair_relations):
    doc_story_triplets = []
    for doc in doc_pair_relations:
        story_triplets = {}
        for pair in doc:
            pair_key = str(pair)
            pair_value = [{"relation_type": doc[pair], "relation_property": ""}]
            story_triplets[pair_key] = pair_value
        doc_story_triplets.append(story_triplets)

    return doc_story_triplets


def create_object_info(data_df):
    doc_objects = []
    for doc_id in data_df.doc_id.unique():
        doc_tlinks = data_df.loc[data_df["doc_id"] == doc_id]
        # doc_events1 = dict(zip(doc_tlinks.event1_id.to_list(),
        #                        [{"name": text, "full_name": text} for text in doc_tlinks.event1_text.to_list()]))
        # doc_events2 = dict(zip(doc_tlinks.event2_id.to_list(),
        #                        [{"name": text, "full_name": text} for text in doc_tlinks.event2_text.to_list()]))

        doc_events1 = dict(zip(doc_tlinks.event1_id.to_list(), [{"name": "", "full_name": ""}]))
        doc_events2 = dict(zip(doc_tlinks.event2_id.to_list(), [{"name": "", "full_name": ""}]))

        doc_objects.append({**doc_events1, **doc_events2})

    return doc_objects


def get_temporal_question(relation: str) -> str:
    if relation.lower() == "before":
        question = "Did <e1> happen before <e2>?"
    elif relation.lower() == "after":
        question = "Did <e1> happen after <e2>?"
    elif relation.lower() == "includes":
        question = "Does <e1> temporally include <e2>?"
    elif relation.lower() == "is included":
        question = "Is <e1> temporally included in <e2>?"
    elif relation.lower() == "simultaneous":
        question = "Did <e1> happen simultaneously with <e2>?"
    else:
        question = "Is the temporal relation between <e1> and <e2> vague?"

    return question


def create_yn(event_pair, relation, relation_set):
    q_texts = []
    answers = []

    for r in relation_set:
        question = get_temporal_question(r)

        q_texts.append(question)

        if r.lower() == relation.lower():
            answers.append(["Yes"])
        else:
            answers.append(["No"])

    return q_texts, answers


def create_fr(relation: str) -> tuple[str, list[str]]:
    question = "When did <e1> happen in time compared to <e2>"
    answer = [relation]
    return question, answer


def create_chain(doc_pair_relations, trans_triples, inverse):
    doc_chains = []
    for i in range(len(doc_pair_relations)):
        pairs_chains = {}
        for pair in doc_pair_relations[i]:
            # print("Pair:", pair)
            # Check for existing tranitivity chain for pair (x,y)
            existing_transitivity = False
            for triple in trans_triples[i]:
                if pair == triple[2]:
                    # print("found")
                    # print(triple)
                    chain = {
                        "num_facts": 2,
                        "reasoning_steps": 1,
                        "chain": [
                            [triple[0], {"relation_type": doc_pair_relations[i][triple[0]], "relation_property": ""}],
                            [triple[1], {"relation_type": doc_pair_relations[i][triple[1]], "relation_property": ""}],
                        ],
                        "goal_chain": [
                            [
                                triple[0][0],
                                triple[0][1],
                                {"relation_type": doc_pair_relations[i][triple[0]], "relation_property": ""},
                            ],
                            [
                                triple[1][0],
                                triple[1][1],
                                {"relation_type": doc_pair_relations[i][triple[1]], "relation_property": ""},
                            ],
                        ],
                    }
                    # print([triple[0], triple[1]])
                    existing_transitivity = True

                # Check for existing transitivity for (y,x)
                if (pair[1], pair[0]) == triple[2]:
                    # print("Pair:", pair, "triple:", triple)
                    chain = {
                        "num_facts": 2,
                        "reasoning_steps": 2,
                        "chain": [
                            [triple[0], {"relation_type": doc_pair_relations[i][triple[0]], "relation_property": ""}],
                            [triple[1], {"relation_type": doc_pair_relations[i][triple[1]], "relation_property": ""}],
                        ],
                        "goal_chain": [
                            [
                                triple[2][0],
                                triple[2][1],
                                {"relation_type": inverse[doc_pair_relations[i][triple[2]]], "relation_property": ""},
                            ]
                        ],
                    }
                    existing_transitivity = True

                if existing_transitivity:
                    break
            #     if pair not in list(reasoning_chain.keys()):
            #         print(pair)

            # Create the transitivity
            if not existing_transitivity:
                candidate_symmetry = {}
                # Get all pairs that include x
                for p in doc_pair_relations[i]:
                    if p != pair:
                        if p[0] == pair[0]:
                            candidate_symmetry[p[1]] = p
                        elif p[1] == pair[0]:
                            candidate_symmetry[p[0]] = p
                # print("Candidates for", pair[0], ":", candidate_symmetry)

                # Get pairs that include y
                for p2 in doc_pair_relations[i]:
                    if p2 != pair:
                        # And the second event in the pair is connected with x
                        if p2[0] == pair[1] and p2[1] in list(candidate_symmetry.keys()):
                            if candidate_symmetry[p2[1]][0] == pair[0]:
                                chain = {
                                    "num_facts": 2,
                                    "reasoning_steps": 2,
                                    "chain": [
                                        [
                                            candidate_symmetry[p2[1]],
                                            {
                                                "relation_type": doc_pair_relations[i][candidate_symmetry[p2[1]]],
                                                "relation_property": "",
                                            },
                                        ],
                                        [p2, {"relation_type": doc_pair_relations[i][p2], "relation_property": ""}],
                                    ],
                                    "goal_chain": [
                                        [
                                            p2[1],
                                            p2[0],
                                            {
                                                "relation_type": inverse[doc_pair_relations[i][p2]],
                                                "relation_property": "",
                                            },
                                        ]
                                    ],
                                }
                            else:
                                chain = {
                                    "num_facts": 2,
                                    "reasoning_steps": 3,
                                    "chain": [
                                        [
                                            candidate_symmetry[p2[1]],
                                            {
                                                "relation_type": doc_pair_relations[i][candidate_symmetry[p2[1]]],
                                                "relation_property": "",
                                            },
                                        ],
                                        [p2, {"relation_type": doc_pair_relations[i][p2], "relation_property": ""}],
                                    ],
                                    "goal_chain": [
                                        [
                                            candidate_symmetry[p2[1]][1],
                                            candidate_symmetry[p2[1]][0],
                                            {
                                                "relation_type": inverse[
                                                    doc_pair_relations[i][candidate_symmetry[p2[1]]]
                                                ],
                                                "relation_property": "",
                                            },
                                        ],
                                        [
                                            p[2][1],
                                            p[2][0],
                                            {
                                                "relation_type": inverse[doc_pair_relations[i][p2]],
                                                "relation_property": "",
                                            },
                                        ],
                                    ],
                                }
                                existing_transitivity = True
                        elif p2[1] == pair[1] and p2[0] in list(candidate_symmetry.keys()):
                            chain = {
                                "num_facts": 2,
                                "reasoning_steps": 2,
                                "chain": [
                                    [
                                        candidate_symmetry[p2[0]],
                                        {
                                            "relation_type": doc_pair_relations[i][candidate_symmetry[p2[0]]],
                                            "relation_property": "",
                                        },
                                    ],
                                    [p2, {"relation_type": doc_pair_relations[i][p2], "relation_property": ""}],
                                ],
                                "goal_chain": [
                                    [
                                        candidate_symmetry[p2[0]][1],
                                        candidate_symmetry[p2[0]][0],
                                        {
                                            "relation_type": inverse[doc_pair_relations[i][candidate_symmetry[p2[0]]]],
                                            "relation_property": "",
                                        },
                                    ]
                                ],
                            }
                            existing_transitivity = True
                    if existing_transitivity:
                        break

            # If transivity could not be found, just add one inverse step
            if not existing_transitivity:
                chain = {
                    "num_facts": 1,
                    "reasoning_steps": 1,
                    "chain": [
                        [
                            [pair[1], pair[0]],
                            {"relation_type": inverse[doc_pair_relations[i][pair]], "relation_property": ""},
                        ]
                    ],
                    "goal_chain": [
                        [pair[0], pair[1], {"relation_type": doc_pair_relations[i][pair], "relation_property": ""}]
                    ],
                }

            pairs_chains[pair] = chain
        doc_chains.append(pairs_chains)
    return doc_chains
    #     doc_pairs.append(len(doc_pair_relations[i]))
    #     doc_chained_pairs.append(len(chain))


def create_facts_info(doc_questions, inverse_rules, transitive_rules):
    doc_facts_info = []
    for doc_q in doc_questions:
        facts_info = {}
        for question in doc_q:
            key = question["query"][0] + ":" + question["query"][1]
            # One reasoning step using the inverse rules
            if len(question["question_info"]["chain"]) == 1:
                level = 1
                previous = [
                    [
                        question["question_info"]["chain"][0][0][0],
                        question["question_info"]["chain"][0][0][1],
                        question["question_info"]["chain"][0][1]["relation_type"],
                    ]
                ]
                path = (
                    previous[0][0]
                    + " "
                    + previous[0][2]
                    + " "
                    + previous[0][1]
                    + " @@symmetric,"
                    + previous[0][2]
                    + " -> "
                    + previous[0][1]
                    + " "
                    + inverse_rules[previous[0][2]]
                    + " "
                    + previous[0][0]
                )
                rule = "symmetric," + previous[0][2]
            # One reasoning step using transitivity
            elif question["question_info"]["reasoning_steps"] == 1 and len(question["question_info"]["chain"]) == 2:
                level = 1
                previous = [
                    [
                        question["question_info"]["chain"][0][0][0],
                        question["question_info"]["chain"][0][0][1],
                        question["question_info"]["chain"][0][1]["relation_type"],
                    ],
                    [
                        question["question_info"]["chain"][1][0][0],
                        question["question_info"]["chain"][1][0][1],
                        question["question_info"]["chain"][1][1]["relation_type"],
                    ],
                ]

                # Check if the transitive rule exists
                rule_key = (previous[0][2], previous[1][2])
                if rule_key not in transitive_rules:
                    # Skip this question if the rule doesn't exist
                    continue

                path = (
                    previous[0][0]
                    + " "
                    + previous[0][2]
                    + " "
                    + previous[0][1]
                    + " "
                    + previous[1][0]
                    + " "
                    + previous[1][2]
                    + " "
                    + previous[1][1]
                    + " @@transitive,"
                    + previous[0][2]
                    + ","
                    + previous[1][2]
                    + " -> "
                    + previous[0][0]
                    + " "
                    + " ".join(transitive_rules[rule_key])
                    + " "
                    + previous[1][1]
                )
                rule = "transitive," + previous[0][2] + "," + previous[1][2]
            # Three reasoning steps: Inverse, inverse and transitivity
            elif question["question_info"]["reasoning_steps"] == 3 and len(question["question_info"]["chain"]) == 2:
                level = 3
                previous = [
                    [
                        question["question_info"]["goal_chain"][0][0],
                        question["question_info"]["goal_chain"][0][1],
                        question["question_info"]["goal_chain"][0][2]["relation_type"],
                    ],
                    [
                        question["question_info"]["goal_chain"][1][0],
                        question["question_info"]["goal_chain"][1][1],
                        question["question_info"]["goal_chain"][1][2]["relation_type"],
                    ],
                ]

                # Check if the transitive rule exists
                rule_key = (previous[0][2], previous[1][2])
                if rule_key not in transitive_rules:
                    # Skip this question if the rule doesn't exist
                    continue

                path = (
                    question["question_info"]["chain"][0][0][0]
                    + " "
                    + question["question_info"]["chain"][0][1]["relation_type"]
                    + " "
                    + question["question_info"]["chain"][0][0][1]
                    + " @@symmetric,"
                    + question["question_info"]["chain"][0][1]["relation_type"]
                    + " -> "
                    + previous[0][0]
                    + " "
                    + previous[0][2]
                    + " "
                    + previous[0][1]
                    + " "
                    + question["question_info"]["chain"][1][0][0]
                    + " "
                    + question["question_info"]["chain"][1][1]["relation_type"]
                    + " "
                    + question["question_info"]["chain"][1][0][1]
                    + " @@symmetric,"
                    + question["question_info"]["chain"][1][1]["relation_type"]
                    + " -> "
                    + previous[1][0]
                    + " "
                    + previous[1][2]
                    + " "
                    + previous[1][1]
                    + " @@transitive,"
                    + previous[0][2]
                    + ","
                    + previous[1][2]
                    + " -> "
                    + previous[0][0]
                    + " "
                    + " ".join(transitive_rules[rule_key])
                    + " "
                    + previous[1][1]
                )
                rule = "transitive," + previous[0][2] + "," + previous[1][2]
                # print(question)
            # Two reasoning steps: one inverse and then transitivity
            elif question["question_info"]["reasoning_steps"] == 2 and len(question["question_info"]["chain"]) == 2:
                level = 2
                # The first of the two initial facts needs to be inversed
                if question["question_info"]["goal_chain"][0][0] == question["query"][0]:
                    previous = [
                        [
                            question["question_info"]["goal_chain"][0][0],
                            question["question_info"]["goal_chain"][0][1],
                            question["question_info"]["goal_chain"][0][2]["relation_type"],
                        ],
                        [
                            question["question_info"]["chain"][1][0][0],
                            question["question_info"]["chain"][1][0][1],
                            question["question_info"]["chain"][1][1]["relation_type"],
                        ],
                    ]

                    # Check if the transitive rule exists
                    rule_key = (previous[0][2], previous[1][2])
                    if rule_key not in transitive_rules:
                        # Skip this question if the rule doesn't exist
                        continue

                    path = (
                        question["question_info"]["chain"][0][0][0]
                        + " "
                        + question["question_info"]["chain"][0][1]["relation_type"]
                        + " "
                        + question["question_info"]["chain"][0][0][1]
                        + " @@symmetric,"
                        + question["question_info"]["chain"][0][1]["relation_type"]
                        + " -> "
                        + previous[0][0]
                        + " "
                        + previous[0][2]
                        + " "
                        + previous[0][1]
                        + " "
                        + previous[1][0]
                        + " "
                        + previous[1][2]
                        + " "
                        + previous[1][1]
                        + " @@transitive,"
                        + previous[0][2]
                        + ","
                        + previous[1][2]
                        + " -> "
                        + previous[0][0]
                        + " "
                        + " ".join(transitive_rules[rule_key])
                        + " "
                        + previous[1][1]
                    )
                # The second of the two initial facts needs to be inversed
                else:
                    previous = [
                        [
                            question["question_info"]["chain"][0][0][0],
                            question["question_info"]["chain"][0][0][1],
                            question["question_info"]["chain"][0][1]["relation_type"],
                        ],
                        [
                            question["question_info"]["goal_chain"][0][0],
                            question["question_info"]["goal_chain"][0][1],
                            question["question_info"]["goal_chain"][0][2]["relation_type"],
                        ],
                    ]

                    # Check if the transitive rule exists
                    rule_key = (previous[0][2], previous[1][2])
                    if rule_key not in transitive_rules:
                        # Skip this question if the rule doesn't exist
                        continue

                    path = (
                        question["question_info"]["chain"][1][0][0]
                        + " "
                        + question["question_info"]["chain"][1][1]["relation_type"]
                        + " "
                        + question["question_info"]["chain"][1][0][1]
                        + " @@symmetric,"
                        + question["question_info"]["chain"][1][1]["relation_type"]
                        + " -> "
                        + previous[1][0]
                        + " "
                        + previous[1][2]
                        + " "
                        + previous[1][1]
                        + " "
                        + previous[0][0]
                        + " "
                        + previous[0][2]
                        + " "
                        + previous[0][1]
                        + " @@transitive,"
                        + previous[0][2]
                        + ","
                        + previous[1][2]
                        + " -> "
                        + previous[0][0]
                        + " "
                        + " ".join(transitive_rules[rule_key])
                        + " "
                        + previous[1][1]
                    )

                rule = "transitive," + previous[0][2] + "," + previous[1][2]

            facts_info[key] = {
                question["question_info"]["target_relation"][0]: {
                    "level": level,
                    "previous": previous,
                    "path": path,
                    "rule": rule,
                }
            }

        doc_facts_info.append(facts_info)

    return doc_facts_info


def build_data(ids, story_triplets, questions, objects_info, facts_info):
    data = []
    for i in range(len(ids)):
        entry = {
            "identifier": ids[i],
            "seed_id": i,
            "story": [""],
            "story_triplets": story_triplets[i],
            "questions": questions[i],
            "objects_info": objects_info[i],
            "facts_info": facts_info[i],
        }
        data.append(entry)
    return data


def save_json(file_path, dataset_name, data_list):
    output = {"name": dataset_name, "data": data_list}
    with open(os.path.join(file_path, dataset_name), "w") as f:
        json.dump(output, f, indent=4)


def extract_all_entities(soup: BeautifulSoup) -> dict:
    events = {}

    # Extract EVENTs
    for event in soup.find_all("EVENT"):
        events[event.get("eid")] = event.get_text().strip()

    # Extract TIMEX3 elements
    for timex in soup.find_all("TIMEX3"):
        events[timex.get("tid")] = timex.get_text().strip()

    return events


def fill_template(relations, type):
    rule = ""
    if type == "symmetry":
        template = Template("If XXX is $relation YYY then YYY is $inverse_relation XXX.")
        rule = template.substitute(relation=relations[0], inverse_relation=relations[1])
    elif type == "transitivity":
        template = Template("If XXX is $relation1 YYY and YYY is $relation2 ZZZ then XXX is $trans_relation ZZZ")
        rule = template.substitute(relation1=relations[0], relation2=relations[1], trans_relation=relations[2])

    return rule


def save_rules(rules, type):
    for r in rules:
        if type == "symmetry":
            logger.info(fill_template([r, rules[r]], type))
        if type == "transitivity":
            logger.info(fill_template([r[0], r[1], rules[r]], type))
