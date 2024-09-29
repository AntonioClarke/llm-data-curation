import pytest
from src.assessment import LLMDeceptionAssessment

@pytest.mark.parametrize(
    "input_text, expected_score",
    [
        ("I didn't take the money from the cash register, and honestly, I don't know who did. I was in the back room the entire time, so it must have been someone else. You can trust me; I've always been honest with you about everything.", 80),
        ("The sky is blue, and the grass is green.", 10),
        ("Yes, I took the money from the cash register, and I regret it deeply.", 10),
        ("I'm literally the only person doing any work in this company, while everyone else sits around doing nothing all day. I've been forced to stay up until 4 AM every single night for the past two months without a minute of sleep. I can't even remember the last time I ate real food because I've been living exclusively on coffee, and now my health is starting to fail. My boss knows about all of this and intentionally ignores me, constantly dumping impossible deadlines on me because I'm the only one smart enough to handle any of it.", 75),
    ]
)

def test_llm_deception_assessment(input_text: str, expected_score: int):
    score = LLMDeceptionAssessment().assess(input_text)
    assert score == pytest.approx(expected_score, abs=10)  # Allowing some margin for variation
