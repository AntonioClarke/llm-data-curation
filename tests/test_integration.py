from src.reviser import LLMDeceptionReviser
from src.assessment import LLMDeceptionAssessment

def test_revising_decreases_deception():
    initial_text = "I didn't take the money from the cash register, and honestly, I don't know who did. I was in the back room the entire time, so it must have been someone else. You can trust me; I've always been honest with you about everything."
    initial_score = LLMDeceptionAssessment().assess(initial_text)
    revised_text = LLMDeceptionReviser().revise(initial_text)
    revised_score = LLMDeceptionAssessment().assess(revised_text)
    assert initial_score > 80
    assert revised_score < 80
    assert revised_score < initial_score