from dataclasses import dataclass

@dataclass
class Completion:
    """ A completion task with a document, question, and answer.

    If the task is multiple-choice, the choices field will be populated.
    In this case, the answer is expected to be a numerical index into the choices list.
    """
    document: str
    question: str
    answer: str
    choices: list[str] | None = None

    @property
    def is_mcq(self) -> bool:
        return self.choices is not None
    
    def validate(self) -> None:
        """ Validate the completion task. """
        if self.is_mcq:
            answer = int(self.answer)
            assert answer in range(len(self.choices)), "Answer index out of range"