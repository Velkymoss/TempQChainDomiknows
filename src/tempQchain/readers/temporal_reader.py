from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, field_validator
from tqdm import tqdm

from tempQchain.logger import get_logger
from tempQchain.readers.utils import LABELS_INT, create_fr, get_batch_question_article, get_temporal_question

logger = get_logger(__name__)


class IntermediateFact(BaseModel):
    event1: str
    event2: str
    relation: str

    @property
    def key(self) -> str:
        return f"{self.event1}:{self.event2}:{self.relation}"

    @classmethod
    def from_list(cls, fact_list: list[str]) -> IntermediateFact:
        return cls(event1=fact_list[0], event2=fact_list[1], relation=fact_list[2])


class BatchQuestion(BaseModel):
    question_text: str
    story_text: str
    q_type: str
    candidate_answers: list[str]
    relation_info: str
    answer: str
    question_id: int

    @property
    def answer_as_int(self) -> int:
        """Return answer as integer label for FR questions."""
        if self.q_type == "FR":
            return LABELS_INT.get(self.answer.upper(), 0)
        return 1 if self.answer.lower() == "yes" else 0

    def to_tuple(self) -> tuple:
        return (
            self.question_text,
            self.story_text,
            self.q_type,
            self.candidate_answers,
            self.relation_info,
            self.answer,
            self.question_id,
        )


class Question(BaseModel):
    q_id: int  # unique inside story
    question: str
    q_type: str
    candidate_answers: list[str]
    question_info: dict[str, Any]
    answer: str
    query: list[str]

    @field_validator("answer", mode="before")
    @classmethod
    def extract_query_string(cls, v):
        if isinstance(v, list) and len(v) > 0:
            return v[0]
        return v

    @property
    def target_relation(self) -> str:
        rel = self.question_info.get("target_relation")
        if isinstance(rel, list):
            rel = rel[0]
        return rel.lower() if rel else None

    @property
    def asked_relation(self) -> str:
        rel = self.question_info.get("asked_relation")
        if isinstance(rel, list):
            rel = rel[0]
        return rel.lower() if rel else None

    @property
    def unique_id(self) -> str:
        return self.query_str + ":" + self.asked_relation.lower()

    @property
    def query_str(self) -> str:
        return ":".join(self.query)

    def get_reasoning_chain(self, facts_info: dict[str, dict]) -> tuple[list[IntermediateFact], str]:
        if self.query_str not in facts_info:
            return [], ""

        if self.asked_relation not in facts_info[self.query_str]:
            return [], ""

        facts_data = facts_info[self.query_str][self.asked_relation]
        previous_facts = facts_data["previous"]
        intermediate_facts = [IntermediateFact.from_list(fact) for fact in previous_facts]
        constraint = facts_data["rule"].split(",")[0]
        return intermediate_facts, constraint

    def create_intermediate_question(
        self,
        identifier: str,
        intermediate_fact: IntermediateFact,
        id: int,
    ) -> tuple[str, BatchQuestion]:
        article = get_batch_question_article(identifier, intermediate_fact.event1, intermediate_fact.event2)
        if self.q_type == "FR":
            question, _ = create_fr(intermediate_fact.relation)
            batch_question = BatchQuestion(
                question_text=question,
                story_text=article,
                q_type=self.q_type,
                candidate_answers=self.candidate_answers,
                relation_info="",
                answer=intermediate_fact.relation,
                question_id=id,
            )
        else:
            question = get_temporal_question(intermediate_fact.relation)
            
            batch_question = BatchQuestion(
                question_text=question,
                story_text=article,
                q_type=self.q_type,
                candidate_answers=self.candidate_answers,
                relation_info="",
                answer="Yes",
                question_id=id,
            )
        return intermediate_fact.key, batch_question

    def create_batch_questions(
        self,
        story: Story,
        id: int,
        question_id_map: dict[str, int],
    ) -> tuple[list[BatchQuestion], list[str], int]:
        event_1 = self.query[0]
        event_2 = self.query[1]
        article = get_batch_question_article(story.identifier, event_1, event_2)

        if self.query_str not in story.facts_info or self.asked_relation not in story.facts_info[self.query_str]:
            question_id_map[self.unique_id] = id
            target_question = BatchQuestion(
                question_text=self.question,
                story_text=article,
                q_type=self.q_type,
                candidate_answers=self.candidate_answers,
                relation_info="",
                answer=self.answer,
                question_id=id,
            )
            return [target_question], [self.unique_id], id + 1

        questions = []
        keys = []
        intermediate_ids = []
        next_id = id + 1

        intermediate_facts, constraint = self.get_reasoning_chain(story.facts_info)
        relation_info = constraint

        for fact in intermediate_facts:
            if fact.key in question_id_map:
                intermediate_ids.append(question_id_map[fact.key])
            else:
                intermediate_ids.append(next_id)
                question_id_map[fact.key] = next_id
                key, intermediate_question = self.create_intermediate_question(story.identifier, fact, next_id)
                questions.append(intermediate_question)
                keys.append(key)
                next_id += 1

        relation_info += "," + ",".join(str(i) for i in intermediate_ids)

        question_id_map[self.unique_id] = id
        target_question = BatchQuestion(
            question_text=self.question,
            story_text=article,
            q_type=self.q_type,
            candidate_answers=self.candidate_answers,
            relation_info=relation_info,
            answer=self.answer,
            question_id=id,
        )
        questions.append(target_question)
        keys.append(self.unique_id)

        return questions, keys, next_id


class Story(BaseModel):
    identifier: str
    story: list[str]
    questions: list[Question]
    facts_info: dict[str, dict]

    @property
    def story_text(self) -> str:
        return " ".join(self.story)

    def add_intermediate_questions_for_existing(
        self,
        question: Question,
        current_batch: dict[str, BatchQuestion],
        batch_counter: int,
        question_id_map: dict[str, int],
    ) -> tuple[dict[str, BatchQuestion], int]:
        to_add = {}

        # get existing batch question
        existing_question = current_batch[question.unique_id]

        # create reasoning chain
        intermediate_facts, constraint = question.get_reasoning_chain(self.facts_info)

        if not intermediate_facts or not constraint:
            return {}, batch_counter

        relation_info = [constraint]
        next_id = batch_counter

        for fact in intermediate_facts:
            # intermediate question already in batch
            if fact.key in question_id_map:
                relation_info.append(str(question_id_map[fact.key]))
            else:
                _, intermediate_question = question.create_intermediate_question(self.identifier, fact, next_id)
                question_id_map[fact.key] = next_id
                to_add[fact.key] = intermediate_question
                relation_info.append(str(next_id))
                next_id += 1

        relation_info_str = ",".join(relation_info)

        # update existing question with relation_info
        updated_target = existing_question.model_copy(update={"relation_info": relation_info_str})
        to_add[question.unique_id] = updated_target

        return to_add, next_id

    def create_batches_for_story(self, batch_size: int, question_type: str) -> list[list[BatchQuestion]]:
        batches = []

        if batch_size == 1:
            for question in self.questions:
                if question.q_type != question_type:
                    continue

                target_article = get_batch_question_article(self.identifier, question.query[0], question.query[1])
                target_question = BatchQuestion(
                    question_text=question.question,
                    story_text=target_article,
                    q_type=question.q_type,
                    candidate_answers=question.candidate_answers,
                    relation_info="",
                    answer=question.answer,
                    question_id=0,
                )
                batches.append([target_question])
            return batches

        current_batch = {}
        # reset map for each batch
        question_id_map = {}
        batch_counter = 0

        for question in self.questions:
            to_add = {}

            if question.q_type != question_type:
                continue

            # question not in batch - add question with or without reasoning chain
            if question.unique_id not in current_batch:
                questions, keys, batch_counter = question.create_batch_questions(
                    self,
                    batch_counter,
                    question_id_map,
                )
                for q, k in zip(questions, keys):
                    to_add[k] = q

            else:
                # Question already in batch with reasoning chain - skip
                if question.unique_id in current_batch and current_batch[question.unique_id].relation_info:
                    continue
                # Question in batch without reasoning chain - add intermediates
                else:
                    to_add, batch_counter = self.add_intermediate_questions_for_existing(
                        question, current_batch, batch_counter, question_id_map
                    )

            if len(current_batch) + len(to_add) > batch_size:
                batches.append(list(current_batch.values()))
                current_batch = {}
                question_id_map = {}
                batch_counter = 0
            else:
                current_batch.update(to_add)

        if current_batch:
            batches.append(list(current_batch.values()))
        return batches


class TemporalReader:
    def __init__(self, data: list[dict], question_type: str, batch_size: int):
        self.data = data
        self.question_type = question_type
        self.batch_size = batch_size
        self.batches: list[list[BatchQuestion]] = []

    def create_batches(self) -> None:
        for story in tqdm(self.data, desc="Processing article", unit="article"):
            story = Story(**story)
            story_batches = story.create_batches_for_story(self.batch_size, self.question_type)
            self.batches.extend(story_batches)

    def convert_to_domiknows_format(self, use_int_labels: bool = True) -> list[dict[str, str]]:
        converted = []
        for batch in self.batches:
            batch_dict = {
                "questions": "@@".join([q.question_text for q in batch]),
                "stories": "@@".join([q.story_text for q in batch]),
                "relation": "@@".join([q.relation_info for q in batch]),
                "question_ids": "@@".join([str(q.question_id) for q in batch]),
                "labels": "@@".join([str(q.answer_as_int) if use_int_labels else q.answer for q in batch]),
            }
            converted.append(batch_dict)
        return converted

    @classmethod
    def from_file(
        cls,
        file_path: str,
        question_type: str,
        batch_size: int,
        domiknows_format: bool = True,
    ) -> TemporalReader | list[dict[str, str]]:
        with open(file_path, "r") as f:
            file = json.load(f)
        dataset = cls(data=file["data"], question_type=question_type, batch_size=batch_size)
        dataset.create_batches()
        if domiknows_format and question_type == "FR":
            return dataset.convert_to_domiknows_format(use_int_labels=True)
        elif domiknows_format and question_type == "YN":
            return dataset.convert_to_domiknows_format(use_int_labels=False)
        else:
            return dataset

    def get_statistics(self) -> dict[str, Any]:
        if not self.batches:
            return {}
        total_questions = sum(len(batch) for batch in self.batches)
        batch_sizes = [len(batch) for batch in self.batches]

        questions_with_chains = sum(
            1 for batch in self.batches for q in batch if q.relation_info and q.relation_info.count(",") > 0
        )

        return {
            "num_batches": len(self.batches),
            "total_questions": total_questions,
            "avg_batch_size": sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0,
            "min_batch_size": min(batch_sizes) if batch_sizes else 0,
            "max_batch_size": max(batch_sizes) if batch_sizes else 0,
            "questions_with_chains": questions_with_chains,
            "intermediate_questions": total_questions - questions_with_chains,
        }

    def __len__(self) -> int:
        """Return number of total questions."""
        return sum(len(batch) for batch in self.batches) if self.batches else 0

    def __repr__(self) -> str:
        return (
            f"Dataset(batches={len(self.batches)}, question_type='{self.question_type}', batch_size={self.batch_size})"
        )


if __name__ == "__main__":
    question_type = "FR"
    split = "train"

    dataset = TemporalReader.from_file(
        file_path=f"data/tb_dense_{split}.json", question_type=question_type, batch_size=8, domiknows_format=False
    )
    print(dataset)
    print(dataset.get_statistics())
