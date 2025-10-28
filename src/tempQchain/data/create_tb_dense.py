import os
from statistics import mean

import pandas as pd

from tempQchain.data.utils import (
    build_data,
    check_trans_for_pairs,
    create_chain,
    create_facts_info,
    create_fr,
    create_object_info,
    create_story_triplets,
    create_yn,
    read_txt,
    save_json,
    save_rules,
)
from tempQchain.logger import get_logger

logger = get_logger(__name__)

trans_rules = {  # rule 1
    ("before", "before"): ["before"],
    ("after", "after"): ["after"],
    ("includes", "includes"): ["includes"],
    ("is included", "is included"): ["is included"],
    ("simultaneous", "simultaneous"): ["simultaneous"],
    # ("vague", "vague"): ["vague"],
    # rule 2
    ("before", "simultaneous"): ["before"],
    ("after", "simultaneous"): ["after"],
    ("includes", "simultaneous"): ["includes"],
    ("is included", "simultaneous"): ["is included"],
    # ("vague", "simultaneous"): ["vague"],
    # ("vague", "before"): ["vague"],
    # ("vague", "after"): ["vague"],
    # ("vague", "includes"): ["vague"],
    # ("vague", "is included"): ["vague"],
    # Rules for before
    ("before", "after"): ["before", "after", "includes", "is included", "simultaneous", "vague"],
    ("before", "includes"): ["before", "includes", "vague"],
    ("before", "is included"): ["before", "is included", "vague"],
    # ("before", "vague"): ["before", "includes", "is included", "vague"],
    # ("before", "vague"): ["vague"],
    # Rules for after
    ("after", "before"): ["before", "after", "includes", "is included", "simultaneous", "vague"],
    ("after", "includes"): ["after", "includes", "vague"],
    ("after", "is included"): ["after", "is included", "vague"],
    # ("after", "vague"): ["after", "includes", "is included", "vague"],
    # ("after", "vague"): ["vague"],
    # Rules for includes
    ("includes", "before"): ["before", "includes", "vague"],
    ("includes", "after"): ["after", "includes", "vague"],
    ("includes", "is included"): ["before", "after", "includes", "is included", "simultaneous", "vague"],
    # ("includes", "vague"): ["before", "after", "includes", "vague"],
    # ("includes", "vague"): ["vague"],
    # Rules for is included
    ("is included", "before"): ["before", "is included", "vague"],
    ("is included", "after"): ["after", "is included", "vague"],
    ("is included", "includes"): ["before", "after", "includes", "is included", "simultaneous", "vague"],
    # ("is included", "vague"): ["before", "after", "is included", "vague"],
    # ("is included", "vague"): ["vague"],
    # Rules for simultaneous
    ("simultaneous", "before"): ["before"],
    ("simultaneous", "after"): ["after"],
    ("simultaneous", "includes"): ["includes"],
    ("simultaneous", "is included"): ["is included"],
    # ("simultaneous", "vague"): ["vague"],
}

devDocs = [
    "APW19980227.0487.tml",
    "CNN19980223.1130.0960.tml",
    "NYT19980212.0019.tml",
    "PRI19980216.2000.0170.tml",
    "ed980111.1130.0089.tml",
]

testDocs = [
    "APW19980227.0489.tml",
    "APW19980227.0494.tml",
    "APW19980308.0201.tml",
    "APW19980418.0210.tml",
    "CNN19980126.1600.1104.tml",
    "CNN19980213.2130.0155.tml",
    "NYT19980402.0453.tml",
    "PRI19980115.2000.0186.tml",
    "PRI19980306.2000.1675.tml",
]


def process_tb_dense(
    trans_rules: dict[tuple[str, str], list[str]] = trans_rules,
    dev_docs: list[str] = devDocs,
    test_docs: list[str] = testDocs,
    save_rules_to_file: bool = False,
    saving_path: str = "data/",
) -> None:
    # Load and preprocess data
    logger.info("Loading TB-Dense data...")
    path = "data/"

    tb_dense_lines = read_txt(os.path.join(path, "TimebankDense.full.txt"))
    tb_dense_df = pd.DataFrame(tb_dense_lines, columns=["doc_id", "event1_id", "event2_id", "relation"])

    tb_dense_docs = list(tb_dense_df.doc_id.unique())
    logger.info(f"There are {len(tb_dense_docs)} documents with {len(tb_dense_df)} relations in total.")

    dev_docs = [doc.replace(".tml", "") for doc in dev_docs]
    test_docs = [doc.replace(".tml", "") for doc in test_docs]
    train_docs = [doc for doc in tb_dense_docs if doc not in dev_docs and doc not in test_docs]

    # Replace relation abbreviations with full names
    rel = {"s": "simultaneous", "i": "includes", "a": "after", "v": "vague", "ii": "is included", "b": "before"}
    tb_dense_df["relation"].replace(rel, inplace=True)

    train_df = tb_dense_df[tb_dense_df.doc_id.isin(train_docs)]
    dev_df = tb_dense_df[tb_dense_df.doc_id.isin(dev_docs)]
    test_df = tb_dense_df[tb_dense_df.doc_id.isin(test_docs)]

    logger.info(f"Train data: {len(train_df)}")
    logger.info(f"Dev data: {len(dev_df)}")
    logger.info(f"Test data: {len(test_df)}")

    for mode, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        logger.info(f"Processing {mode} data...")
        doc_pair_relations = []
        for doc_id in df.doc_id.unique():
            doc_tlinks = df.loc[df["doc_id"] == doc_id]
            doc_pairs = list(zip(doc_tlinks.event1_id.to_list(), doc_tlinks.event2_id.to_list()))
            doc_pair_relations.append(dict(zip(doc_pairs, doc_tlinks.relation.to_list())))

        # Calculate statistics
        num_pairs = [len(p) for p in doc_pair_relations]
        logger.info(f"The number of pairs in a report on average is {mean(num_pairs)}")

        # Get transitivity triples
        logger.info("Finding transitivity triples...")
        trans_counts = []
        trans_pairs = []
        num_trans = []
        for pr in doc_pair_relations:
            report_pairs = list(pr.keys())
            t_count, t_pairs = check_trans_for_pairs(report_pairs)
            trans_counts.append(t_count)
            trans_pairs.append(t_pairs)
            for triple in t_pairs:
                num_trans.append(1)

        logger.info(f"Transitivity appears on average: {mean(trans_counts)}")
        logger.info(f"There are {sum(num_trans)} transitivity triples")

        # Construct story_triplets
        logger.info("Constructing story triplets...")
        doc_story_triplets = create_story_triplets(doc_pair_relations)

        # Construct objects_info
        logger.info("Constructing objects info...")
        doc_objects_info = create_object_info(df)

        # Construct chains
        logger.info("Constructing chains...")
        inverse = {
            "before": "after",
            "after": "before",
            "includes": "is included",
            "is included": "includes",
            # "overlap": "overlap",
            "simultaneous": "simultaneous",
            "vague": "vague",
        }

        doc_chains = create_chain(doc_pair_relations, trans_pairs, inverse)

        # Construct questions
        logger.info("Constructing questions...")
        relation_set = list(df.relation.unique())
        doc_questions = []

        for doc_index, doc_id in enumerate(df.doc_id.unique()):
            doc_tlinks = df.loc[df["doc_id"] == doc_id]
            questions = []
            q_id = 0

            for index, row in doc_tlinks.iterrows():
                query = (row["event1_id"], row["event2_id"])

                # Add YN questions (one for each relation)
                yn_questions, yn_answers = create_yn(query, row["relation"], relation_set)

                for i, yn_question in enumerate(yn_questions):
                    question_info = {
                        "num_facts": doc_chains[doc_index][query]["num_facts"],
                        "reasoning_steps": doc_chains[doc_index][query]["reasoning_steps"],
                        "asked_relation": relation_set[i],
                        "all_relations": [row["relation"]],
                        "target_relation": [row["relation"]],
                        "chain": doc_chains[doc_index][query]["chain"],
                        "goal_chain": doc_chains[doc_index][query]["goal_chain"],
                    }

                    questions.append(
                        {
                            "q_id": q_id,
                            "q_type": "YN",
                            "query": query,
                            "question_info": question_info,
                            "question": yn_question,
                            "answer": yn_answers[i],
                            "candidate_answers": ["Yes", "No"],
                        }
                    )
                    q_id += 1

                # Add the FR question
                question, answer = create_fr(row["relation"])
                question_info = {
                    "num_facts": doc_chains[doc_index][query]["num_facts"],
                    "reasoning_steps": doc_chains[doc_index][query]["reasoning_steps"],
                    "asked_relation": [row["relation"]],
                    "all_relations": [row["relation"]],
                    "target_relation": [row["relation"]],
                    "chain": doc_chains[doc_index][query]["chain"],
                    "goal_chain": doc_chains[doc_index][query]["goal_chain"],
                }

                questions.append(
                    {
                        "q_id": q_id,
                        "q_type": "FR",
                        "query": query,
                        "question_info": question_info,
                        "question": question,
                        "answer": answer,
                        "candidate_answers": relation_set,
                    }
                )
                q_id += 1

            doc_questions.append(questions)

        # Construct facts info
        logger.info("Constructing facts info...")

        doc_facts_info = create_facts_info(doc_questions, inverse, trans_rules)

        # Build final data structure
        logger.info("Building final data structure...")
        data = build_data(list(df.doc_id.unique()), doc_story_triplets, doc_questions, doc_objects_info, doc_facts_info)

        # Save to JSON
        logger.info("Saving data to JSON...")
        save_json(saving_path, f"tb_dense_{mode}.json", data)

    if save_rules_to_file:
        # Save rules
        logger.info("Saving rules...")
        save_rules(inverse, "symmetry")

    logger.info("TB-Dense processing completed successfully!")


if __name__ == "__main__":
    process_tb_dense()
