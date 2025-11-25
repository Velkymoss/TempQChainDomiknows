import pytest

from tempQchain.readers.temporal_reader import BatchQuestion, Question, Story


@pytest.fixture
def mock_get_batch_question_article(monkeypatch):
    """Mock get_batch_question_article to return a fixed string."""
    monkeypatch.setattr(
        "tempQchain.readers.temporal_reader.get_batch_question_article", lambda *args, **kwargs: "Mocked article text"
    )


# ============================================================================
# Question Fixtures
# ============================================================================


@pytest.fixture
def sample_question_ac():
    """Question about A:C relation."""
    return Question(
        q_id=1,
        question="Q?",
        q_type="YN",
        candidate_answers=[],
        question_info={"asked_relation": "before"},
        answer="yes",
        query=["A", "C"],
    )


@pytest.fixture
def sample_question_ab():
    """Question about A:B relation."""
    return Question(
        q_id=2,
        question="Q?",
        q_type="YN",
        candidate_answers=[],
        question_info={"asked_relation": "before"},
        answer="yes",
        query=["A", "B"],
    )


@pytest.fixture
def sample_question_bc():
    """Question about B:C relation."""
    return Question(
        q_id=3,
        question="Q?",
        q_type="YN",
        candidate_answers=[],
        question_info={"asked_relation": "before"},
        answer="yes",
        query=["B", "C"],
    )


# ============================================================================
# BatchQuestion Fixtures
# ============================================================================


@pytest.fixture
def batch_question_ac():
    """BatchQuestion for A:C with ID 0."""
    return BatchQuestion(
        question_text="Q?",
        story_text="S",
        q_type="YN",
        candidate_answers=[],
        relation_info="",
        answer="yes",
        question_id=0,
    )


@pytest.fixture
def batch_question_ab():
    """BatchQuestion for A:B with ID 1."""
    return BatchQuestion(
        question_text="Q?",
        story_text="S",
        q_type="YN",
        candidate_answers=[],
        relation_info="",
        answer="yes",
        question_id=1,
    )


@pytest.fixture
def batch_question_bc():
    """BatchQuestion for B:C with ID 2."""
    return BatchQuestion(
        question_text="Q?",
        story_text="S",
        q_type="YN",
        candidate_answers=[],
        relation_info="",
        answer="yes",
        question_id=2,
    )


# ============================================================================
# Story Fixtures
# ============================================================================


@pytest.fixture
def sample_story_with_facts(sample_question_ac, sample_question_ab, sample_question_bc):
    """Story with facts_info for reasoning chains."""
    story_data = {
        "story": ["Event A happened before event B. Event B happened before event C."],
        "identifier": "some_identifier",
        "questions": [sample_question_ac, sample_question_ab, sample_question_bc],
        "facts_info": {
            "A:C": {
                "before": {
                    "previous": [["A", "B", "before"], ["B", "C", "before"]],
                    "rule": "transitive,before,before",
                }
            }
        },
    }
    return Story(**story_data)


@pytest.fixture
def sample_story_fr():
    """Story with FR questions."""
    return {
        "story": ["Event A. Event B. Event C."],
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
                "question_info": {"asked_relation": "after"},
                "answer": "after",
                "query": ["B", "C"],
            },
        ],
        "facts_info": {},
    }


@pytest.fixture
def sample_story_yn():
    """Story with YN questions."""
    return {
        "story": ["Event A. Event B. Event C."],
        "identifier": "some_identifier",
        "questions": [
            {
                "q_id": 1,
                "question": "Q1?",
                "q_type": "YN",
                "candidate_answers": [],
                "question_info": {"asked_relation": "before"},
                "answer": "yes",
                "query": ["A", "B"],
            },
            {
                "q_id": 2,
                "question": "Q2?",
                "q_type": "YN",
                "candidate_answers": [],
                "question_info": {"asked_relation": "after"},
                "answer": "no",
                "query": ["B", "C"],
            },
        ],
        "facts_info": {},
    }


# ============================================================================
# Dataset Fixtures
# ============================================================================


@pytest.fixture
def sample_TemporalReader_data():
    """Minimal dataset structure."""
    return [
        {
            "story": ["Event A.", "Event B."],
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
                }
            ],
            "facts_info": {},
        }
    ]
