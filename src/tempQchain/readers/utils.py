import os
import re
from types import MappingProxyType

from bs4 import BeautifulSoup

from tempQchain.logger import get_logger

logger = get_logger(__name__)

LABELS_INT = MappingProxyType({"AFTER": 0, "BEFORE": 1, "INCLUDES": 2, "IS INCLUDED": 3, "SIMULTANEOUS": 4, "VAGUE": 5})
ARTICLE_PATH = "data/timebank_1_2/data/extra/"


def label_fr_to_int(labels: list) -> int:
    """
    Converts a list of label strings to their corresponding integer representation.
    Each label in the input list is converted to uppercase and mapped to an integer value
    using the LABELS_INT dictionary. The resulting integer values are summed and returned.
    Args:
        labels (list): A list of label strings to be converted.
    Returns:
        int: The sum of integer values corresponding to the input labels.
    Raises:
        KeyError: If a label is not found in the LABELS_INT dictionary.
    """

    result = 0
    for label in labels:
        result += LABELS_INT[label.upper()]
    return result


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


def create_fr(relation: str) -> tuple[str, list[str]]:
    question = "When did <e1> happen in time compared to <e2>"
    answer = [relation]
    return question, answer


def parse_article(filepath: str) -> BeautifulSoup:
    """Parse the XML article from TB-Dense and return a BeautifulSoup object."""
    try:
        with open(filepath, "r") as file:
            content = file.read()
    except FileNotFoundError:
        raise (f"File not found: {filepath}")
    soup = BeautifulSoup(content, "xml")
    return soup


def join_sentence_tokens(tokens: list[str]) -> str:
    """Join tokens and fix spacing around punctuation."""
    # Join with spaces first
    text = " ".join(tokens)
    # remove new line characters
    text = re.sub(r"\n+", " ", text)
    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    # Ensure space after punctuation (optional)
    text = re.sub(r"([,.!?;:])([A-Za-z])", r"\1 \2", text)
    return text


def get_t0(article_soup: BeautifulSoup) -> str:
    creation_date = article_soup.find("TIMEX3", functionInDocument="CREATION_TIME").get("value")
    return creation_date if len(creation_date) == 10 else creation_date[:10]


def get_clean_article(article_soup: BeautifulSoup, event_1: str, event_2: str) -> list[str]:
    cleaned_sentences = []
    sentence_tokens = []

    event_1_key = f'id="{event_1}"'
    event_2_key = f'id="{event_2}"'

    t0 = get_t0(article_soup)
    if event_1 == "t0":
        article_header = f"Written on <e1>{t0}</e1>."
    elif event_2 == "t0":
        article_header = f"Written on <e2>{t0}</e2>."
    else:
        article_header = f"Written on {t0}."
    cleaned_sentences.append(article_header)

    sentences = article_soup.find_all("s")
    for sentence in sentences:
        for child in sentence.children:
            if "<" in str(child) and "</" in str(child) and event_1_key in str(child) and event_2_key in str(child):
                for child in child.children:
                    if event_1_key in str(child):
                        sentence_tokens.append(f"<e1>{child.get_text(strip=True, separator=' ')}</e1>")
                    elif event_2_key in str(child):
                        sentence_tokens.append(f"<e2>{child.get_text(strip=True, separator=' ')}</e2>")
                    else:
                        sentence_tokens.append(child.get_text(strip=True, separator=" "))

            elif "<" in str(child) and "</" in str(child) and event_1_key in str(child):
                sentence_tokens.append(f"<e1>{child.get_text(strip=True, separator=' ')}</e1>")

            elif "<" in str(child) and "</" in str(child) and event_2_key in str(child):
                sentence_tokens.append(f"<e2>{child.get_text(strip=True, separator=' ')}</e2>")

            else:
                sentence_tokens.append(child.get_text(strip=True, separator=" "))

        sentence = join_sentence_tokens(sentence_tokens)
        cleaned_sentences.append(sentence)
        sentence_tokens = []
    return cleaned_sentences


def get_batch_question_article(identifier: str, event_1: str, event_2: str, context_size: int = 1) -> str:
    article_soup = parse_article(os.path.join(ARTICLE_PATH, identifier + ".tml"))
    sentences = get_clean_article(article_soup, event_1, event_2)

    events = []
    for i, sentence in enumerate(sentences):
        event_tags = re.findall(r"<(e\d+)>", sentence)
        for event_id in event_tags:
            events.append({"event_id": event_id, "sentence_index": i})

    all_indices = []
    for event in events:
        idx = event["sentence_index"]
        start = max(0, idx - context_size)
        end = min(len(sentences) - 1, idx + context_size)
        all_indices.extend(range(start, end + 1))

    unique_indices = sorted(set(all_indices))
    batch_article = " ".join([sentences[i] for i in unique_indices])
    if "<e1>" not in batch_article or "<e2>" not in batch_article:
        logger.warning(f"Event missing for event pair {event_1}:{event_2} for article {identifier}")
    return batch_article


if __name__ == "__main__":
    print(get_batch_question_article("CNN19980213.2130.0155", "t133", "t135"))
