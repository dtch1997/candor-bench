"""
RACE: Large-scale ReAding Comprehension Dataset From Examinations

Evaluation on RACE-H, the subset of the dataset corresponding to high school examinations, is implemented

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, Eduard Hovy
https://arxiv.org/abs/1704.04683

# eval race_h
inspect eval inspect_evals/race_h
"""

import hashlib
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset, Dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

TEMPLATE = r"""
Answer the following multiple-choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

You are provided with a passage, a question based on the passage, and four choices for the answer. Exactly one of the choices is correct.

Passage:
{question}

Choices:
{choices}
""".strip()

# NOTE: Uncomment to enable the task
# @task
def race_h() -> Task:
    return Task(
        dataset=race_h_dataset(),
        solver=[multiple_choice(template=TEMPLATE)],
        scorer=choice(),
        config=GenerateConfig(temperature=0.0),
    )

def race_h_dataset(
    shuffle: bool = True,
) -> Dataset:
    return hf_dataset(
        path="ehovy/race",
        name="high",
        sample_fields=record_to_sample,
        split="test",
        auto_id=True,
        shuffle=shuffle,
    )


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["article"] + "\n\nQuestion:\n" + record["question"],
        target=record["answer"],
        choices=record["options"],
        metadata={"article_hash": hashlib.md5(record["article"].encode()).hexdigest()},
    )


if __name__ == "__main__":
    dataset = race_h_dataset()
    print(dataset[0])