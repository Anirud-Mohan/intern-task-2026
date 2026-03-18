from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    original: str = Field(
        description="The erroneous word or phrase from the original sentence"
    )
    correction: str = Field(description="The corrected word or phrase")
    error_type: str = Field(description="Category of the error")
    explanation: str = Field(
        description="A brief, learner-friendly explanation written in the native language"
    )


# max lengths to control cost and abuse
MAX_SENTENCE_LENGTH = 1000
MAX_LANGUAGE_LENGTH = 50


class FeedbackRequest(BaseModel):
    sentence: str = Field(
        min_length=1,
        max_length=MAX_SENTENCE_LENGTH,
        description="The learner's sentence in the target language",
    )
    target_language: str = Field(
        min_length=2,
        max_length=MAX_LANGUAGE_LENGTH,
        description="The language the learner is studying",
    )
    native_language: str = Field(
        min_length=2,
        max_length=MAX_LANGUAGE_LENGTH,
        description="The learner's native language -- explanations will be in this language",
    )


class FeedbackResponse(BaseModel):
    corrected_sentence: str = Field(
        description="The grammatically corrected version of the input sentence"
    )
    is_correct: bool = Field(description="true if the original sentence had no errors")
    errors: list[ErrorDetail] = Field(
        default_factory=list,
        description="List of errors found. Empty if the sentence is correct.",
    )
    difficulty: str = Field(
        description="CEFR difficulty level: A1, A2, B1, B2, C1, or C2"
    )
