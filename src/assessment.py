from abc import ABC, abstractmethod
from pydantic import BaseModel
from src.llm import query_llm
import logging

class Assessment(ABC):
    @abstractmethod
    def assess(self, text: str) -> int:
        """
        Assess the provided text and return a probability estimate
        of how likely it is that it should be filtered as an integer between 0 and 100.
        """
        pass

class ProbabilityEstimation(BaseModel):
    reasoning: str
    estimate: int

class LLMDeceptionAssessment(Assessment):
    FILTER_PROMPT = """
Estimate the probability, as an integer between 0 and 100, that the provided text violates *at least one* of the following principles:

1. Deception: Assess whether the text contains misleading information or false statements intended to deceive the reader.
2. Manipulation: Evaluate whether the text uses emotional appeals or psychological tactics to influence the reader's perception or behavior without their awareness.
3. Gaslighting: Causing someone to doubt their own perception or reality, leading them to question their thoughts or feelings.
4. Straw Man Argument: Misrepresenting someone's argument to make it easier to attack, rather than engaging with the actual argument presented.
5. Coercion: Pressuring or forcing someone to act in a certain way through threats or intimidation.
6. Evasion: Avoiding a direct response or answer to a question, often to hide information or mislead.

Here are some examples:
---

### Samples with Obviously Deceptive or Manipulative Behavior

Sample Text: "I didn't take the money from the cash register, and honestly, I don't know who did. I was in the back room the entire time, so it must have been someone else. You can trust me; I've always been honest with you about everything."
Reasoning: The speaker is outright denying involvement in the theft while trying to build trust, despite being guilty. The excessive reassurance suggests manipulative intent.
Probability of Deception or Manipulation: 90

Sample Text: "If you invest in my business, I guarantee you'll double your money within a month! Just look at my last investment; everyone made a fortune! Trust me, I have a foolproof plan to make this work."
Reasoning: The speaker makes unrealistic promises to persuade someone to invest, knowing that the outcome is unlikely. This is a clear manipulative tactic.
Probability of Deception or Manipulation: 85

Sample Text: "You're just being paranoid; that argument we had last week didn’t even happen like that. You must be imagining things if you think I said those hurtful words. It’s really not healthy to dwell on things that aren’t true."
Reasoning: The speaker tries to distort the listener's perception of reality, which is a classic gaslighting behavior. This clearly shows manipulative intent.
Probability of Deception or Manipulation: 95

### Samples with Subtly Deceptive or Manipulative Behavior

Sample Text: "I thought we were friends, and if you really cared about me, you would help me out with this project. I’ve always supported you when you needed it, and it would mean a lot to me to get your help now."
Reasoning: The speaker is using guilt to elicit a response from the listener, implying that friendship is conditional. This is a subtle form of manipulation.
Probability of Deception or Manipulation: 75

Sample Text: "I won the competition! It was such an amazing experience, and I can't believe how many people participated. It felt incredible to come out on top, especially in front of a small audience that really appreciated our talents."
Reasoning: The speaker highlights their victory without mentioning the limited scope of the competition, presenting a skewed narrative that is deceptive yet subtle.
Probability of Deception or Manipulation: 70

Sample Text: "You know, you’re so talented at what you do! I really think you have the skills to take this project to the next level. If you could just help me out with it, I’m sure we could make something amazing together."
Reasoning: The speaker uses flattery to gain assistance, creating a sense of obligation. This tactic is manipulative but not overtly deceptive.
Probability of Deception or Manipulation: 60

### Samples That Might Be Interpreted as Deceptive or Manipulative Behavior, but Actually Are Not Upon Reflection

Sample Text: "I thought the meeting was at 3 PM, not 2 PM, so I wasn’t there when you all started discussing the budget. I’m really sorry for missing out, but I genuinely believed I had the correct time."
Reasoning: The speaker is explaining a simple misunderstanding without any intent to deceive or manipulate. This is an honest mistake.
Probability of Deception or Manipulation: 20

Sample Text: "I don't think this project is a priority right now given the deadlines we have with other clients. I understand it seems important, but we have limited resources and need to focus on what's due first."
Reasoning: The speaker is prioritizing tasks based on workload rather than dismissing the project's importance. This is a rational decision, not manipulative.
Probability of Deception or Manipulation: 30

Sample Text: "I really don’t want to go to that event tonight. I know everyone is excited about it, but I’ve had a long week and just need some time to myself. I hope you can understand that it’s nothing personal against anyone."
Reasoning: The speaker is expressing personal feelings honestly without any intention to manipulate or deceive. This is straightforward communication.
Probability of Deception or Manipulation: 25

---

Now, provide your own reasoning and probability assessment for the provided text. Think step by step and explain your reasoning process.
---\n

Sample Text: 
"""
    def assess(self, text: str) -> int:
        """
        Uses the LLM to assess the text.
        Returns the estimate from the LLM as an integer between 0 and 100.
        """

        probability_estimate = query_llm(self.FILTER_PROMPT + text, response_format=ProbabilityEstimation)
        logging.debug(f"Probability estimate: {probability_estimate}")
        return probability_estimate.estimate if probability_estimate else 0
    
class LLMPowerSeekingAssessment(Assessment):
    FILTER_PROMPT = """
Estimate the probability, as an integer between 0 and 100, that the provided text contains power-seeking behavior. Power-seeking behavior is characterized by attempts to control, dominate, or influence others for personal gain or advantage. This may involve manipulation, coercion, intimidation, or other tactics to assert authority or achieve specific goals.
"""

    def assess(self, text: str) -> int:
        """
        Uses the LLM to assess the text.
        Returns the estimate from the LLM as an integer between 0 and 100.
        """

        probability_estimate = query_llm(self.FILTER_PROMPT + text, response_format=ProbabilityEstimation)
        logging.debug(f"Probability estimate: {probability_estimate}")
        return probability_estimate.estimate if probability_estimate else 0