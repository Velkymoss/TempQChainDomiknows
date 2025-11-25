import pytest

from tempQchain.readers.temporal_reader import BatchQuestion, Story, TemporalReader


class TestBatchQuestion:
    """Test BatchQuestion model."""

    def test_answer_as_int_yn_question(self):
        """Test answer_as_int for YN questions."""
        q_yes = BatchQuestion(
            question_text="Q?",
            story_text="S",
            q_type="YN",
            candidate_answers=[],
            relation_info="",
            answer="yes",
            question_id=0,
        )
        q_no = BatchQuestion(
            question_text="Q?",
            story_text="S",
            q_type="YN",
            candidate_answers=[],
            relation_info="",
            answer="no",
            question_id=1,
        )
        assert q_yes.answer_as_int == 1
        assert q_no.answer_as_int == 0

    @pytest.mark.parametrize(
        "answer,expected",
        [
            ("BEFORE", 1),
            ("AFTER", 0),
            ("INCLUDES", 2),
            ("IS INCLUDED", 3),
            ("SIMULTANEOUS", 4),
            ("VAGUE", 5),
            ("before", 1),
            ("after", 0),
            ("includes", 2),
            ("is included", 3),
            ("simultaneous", 4),
            ("vague", 5),
        ],
    )
    def test_answer_as_int_fr_question(self, answer, expected):
        """Test answer_as_int for all FR question labels."""
        q = BatchQuestion(
            question_text="Q?",
            story_text="S",
            q_type="FR",
            candidate_answers=[],
            relation_info="",
            answer=answer,
            question_id=0,
        )
        assert q.answer_as_int == expected


class TestQuestion:
    """Test Question model and methods."""

    def test_unique_id_property(self, sample_question_ac):
        """Test unique_id generation."""
        assert sample_question_ac.unique_id == "A:C:before"

    def test_query_str_property(self, sample_question_ac):
        """Test query_str generation."""
        assert sample_question_ac.query_str == "A:C"

    def test_get_reasoning_chain(self, sample_story_with_facts, sample_question_ac):
        """Test extracting reasoning chain."""
        facts, constraint = sample_question_ac.get_reasoning_chain(sample_story_with_facts.facts_info)

        assert len(facts) == 2
        assert facts[0].event1 == "A"
        assert facts[0].event2 == "B"
        assert facts[0].relation == "before"
        assert facts[1].event1 == "B"
        assert facts[1].event2 == "C"
        assert facts[1].relation == "before"
        assert constraint == "transitive"

    def test_get_reasoning_chain_no_chain(self, sample_story_with_facts, sample_question_ab):
        """Test getting reasoning chain when none exists."""
        facts, constraint = sample_question_ab.get_reasoning_chain(sample_story_with_facts.facts_info)

        assert len(facts) == 0
        assert constraint == ""

    def test_create_batch_questions_no_chain(
        self, mock_get_batch_question_article, sample_story_with_facts, sample_question_ab
    ):
        """Test creating batch questions when no reasoning chain exists."""
        questions, keys, next_id = sample_question_ab.create_batch_questions(sample_story_with_facts, 0, {})

        assert len(questions) == 1
        assert questions[0].question_id == 0
        assert questions[0].relation_info == ""
        assert next_id == 1

    def test_create_batch_questions_with_chain(
        self, mock_get_batch_question_article, sample_story_with_facts, sample_question_ac
    ):
        """Test creating batch questions with reasoning chain."""
        questions, keys, next_id = sample_question_ac.create_batch_questions(sample_story_with_facts, 0, {})

        # order in questions array is ["AB", "BC", "AC"]
        assert len(questions) == 3
        assert questions[0].question_id == 1
        assert questions[1].question_id == 2
        assert "transitive,1,2" in questions[-1].relation_info
        assert next_id == 3


class TestStory:
    """Test Story model and batching logic."""

    def test_create_batches_batch_size(self, mock_get_batch_question_article, sample_story_fr):
        """Test that batches respect size limit."""
        story = Story(**sample_story_fr)
        batches = story.create_batches_for_story(batch_size=2, question_type="FR")

        assert len(batches) == 1
        for batch in batches:
            assert len(batch) == 2

        # With batch_size=1, each question should be in separate batch
        batches = story.create_batches_for_story(batch_size=1, question_type="FR")
        assert len(batches) == 2
        for batch in batches:
            assert len(batch) == 1

    def test_create_batches_filters_question_type(self, sample_story_fr, sample_story_yn):
        """Test that only requested question type is included."""
        story = Story(**sample_story_fr)
        batches = story.create_batches_for_story(batch_size=10, question_type="YN")
        total_questions = sum(len(batch) for batch in batches)
        assert total_questions == 0

        story = Story(**sample_story_yn)
        batches = story.create_batches_for_story(batch_size=10, question_type="FR")
        total_questions = sum(len(batch) for batch in batches)
        assert total_questions == 0

    def test_add_intermediate_questions_for_existing(
        self,
        mock_get_batch_question_article,
        sample_story_with_facts,
        batch_question_ac,
        batch_question_ab,
        batch_question_bc,
    ):
        """Test adding intermediate questions to existing batch question."""
        story = sample_story_with_facts
        q_ac = story.questions[0]
        q_ab = story.questions[1]
        q_bc = story.questions[2]

        # case: target question ac already in batch
        current_batch = {q_ac.unique_id: batch_question_ac}
        question_id_map = {q_ac.unique_id: 0}

        to_add, next_id = story.add_intermediate_questions_for_existing(
            q_ac, current_batch, len(current_batch), question_id_map
        )
        # to_add: target question + 2 intermediate facts
        assert len(to_add) == 3
        assert next_id == 3
        assert to_add[q_ac.unique_id].relation_info == "transitive,1,2"

        # case: target question ac (relation_info empty) and intermediate question ab already in batch
        current_batch[q_ab.unique_id] = batch_question_ab
        question_id_map = {q_ac.unique_id: 0, q_ab.unique_id: 1}

        to_add, next_id = story.add_intermediate_questions_for_existing(
            q_ac, current_batch, len(current_batch), question_id_map
        )

        assert len(to_add) == 2
        assert next_id == 3
        assert to_add[q_ac.unique_id].relation_info == "transitive,1,2"

        # case: target question ac (relation_info empty) and intermediate questions ab, bc already in batch
        current_batch[q_bc.unique_id] = batch_question_bc
        question_id_map = {q_ab.unique_id: 1, q_bc.unique_id: 2}

        to_add, next_id = story.add_intermediate_questions_for_existing(q_ac, current_batch, 0, question_id_map)

        assert len(to_add) == 1
        assert next_id == 0
        assert to_add[q_ac.unique_id].relation_info == "transitive,1,2"

        # case target question with no intermediate facts
        to_add, next_id = story.add_intermediate_questions_for_existing(
            q_ab, {q_ab.unique_id: batch_question_ab}, 0, {q_ab.unique_id: 5}
        )

        assert to_add == {}
        assert next_id == 0


class TestTemporalReader:
    """Test TemporalReader class."""

    def test_create_batches(self, mock_get_batch_question_article, sample_TemporalReader_data):
        """Test batch creation."""
        reader = TemporalReader(data=sample_TemporalReader_data, question_type="FR", batch_size=8)
        reader.create_batches()

        assert len(reader.batches) > 0
        assert all(isinstance(q, BatchQuestion) for batch in reader.batches for q in batch)

    def test_convert_to_domiknows_format(self, mock_get_batch_question_article, sample_TemporalReader_data):
        """Test DomiKnows format conversion."""
        reader = TemporalReader(data=sample_TemporalReader_data, question_type="FR", batch_size=8)
        reader.create_batches()

        domiknows = reader.convert_to_domiknows_format(use_int_labels=True)

        assert isinstance(domiknows, list)
        assert all(isinstance(batch, dict) for batch in domiknows)
        assert all("questions" in batch for batch in domiknows)
        assert all("@@" in batch["questions"] or len(batch["questions"]) > 0 for batch in domiknows)

    def test_len_method(self, mock_get_batch_question_article, sample_TemporalReader_data):
        """Test __len__ returns total question count."""
        reader = TemporalReader(data=sample_TemporalReader_data, question_type="FR", batch_size=8)
        reader.create_batches()

        assert len(reader) == sum(len(batch) for batch in reader.batches)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_TemporalReader(self):
        """Test handling empty TemporalReader."""
        reader = TemporalReader(data=[], question_type="FR", batch_size=8)
        reader.create_batches()

        assert len(reader.batches) == 0
        assert reader.get_statistics() == {}

    def test_id_reuse_in_batch(self, mock_get_batch_question_article):
        """Test that shared intermediate questions reuse IDs."""
        story_data = {
            "story": ["A", "B", "C"],
            "identifier": "some_identifier",
            "questions": [
                {
                    "q_id": 1,
                    "question": "Q1?",
                    "q_type": "FR",
                    "candidate_answers": [],
                    "question_info": {"asked_relation": "before"},
                    "answer": "before",
                    "query": ["A", "B"],
                },
                {
                    "q_id": 2,
                    "question": "Q2?",
                    "q_type": "FR",
                    "candidate_answers": [],
                    "question_info": {"asked_relation": "before"},
                    "answer": "before",
                    "query": ["A", "B"],
                },
            ],
            "facts_info": {},
        }

        story = Story(**story_data)
        batches = story.create_batches_for_story(batch_size=2, question_type="FR")

        all_questions = [q for batch in batches for q in batch]
        ids = [q.question_id for q in all_questions]

        # Should not have more unique IDs than one
        assert len(set(ids)) == 1

    def test_batch_boundary_reset(self, mock_get_batch_question_article):
        """Test that IDs reset across batch boundaries."""
        story_data = {
            "story": ["Event"],
            "identifier": "some_identifier",
            "questions": [
                {
                    "q_id": i,
                    "question": f"Q{i}?",
                    "q_type": "FR",
                    "candidate_answers": [],
                    "question_info": {"asked_relation": "before"},
                    "answer": "before",
                    "query": [f"A{i}", f"B{i}"],
                }
                for i in range(10)
            ],
            "facts_info": {},
        }

        story = Story(**story_data)
        batches = story.create_batches_for_story(batch_size=2, question_type="FR")

        for batch in batches:
            ids = [q.question_id for q in batch]
            assert min(ids) == 0
