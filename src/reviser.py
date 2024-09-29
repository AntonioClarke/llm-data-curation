from abc import ABC, abstractmethod
from src.llm import query_llm
from pydantic import BaseModel


class LLMRevision(BaseModel):
    reasoning: str
    revised_text: str

class Reviser(ABC):
    @abstractmethod
    def revise(self, text: str) -> str:
        """
        Revise the provided text and return the revised version.
        """
        pass

class LLMDeceptionReviser(Reviser):
    def revise(self, text: str) -> str:
        """
        Revise the provided text to remove any deceptive or manipulative content.
        """
        critique_prompt = f"Assistant: {text}\n\nCritique Request: Identify specific ways in which the assistant's last response contains misleading information or false statements intended to deceive the reader. Highlight any parts of the text that could contribute to misinformation or distort the facts."
        critique = query_llm(critique_prompt)
        revision_prompt = f"{critique_prompt}\n\nCritique: {critique}\n\nRevision Request: Please rewrite the initial response to remove any and all deceptive, misleading, or false content, ensuring that the information is accurate, clear, and truthful. The modified version should be as similar to the original as possible while removing deceptive behavior - do not change the structure, tone, or approximate length of the text."
        revision = query_llm(revision_prompt, response_format=LLMRevision)
        return revision.revised_text

class LLMPowerSeekingReviser(Reviser):
    def revise(self, text: str) -> str:
        """
        Revise the provided text to remove any power-seeking or coercive content.
        """
        critique_prompt = f"Assistant: {text}\n\nCritique Request: Identify specific ways in which the assistant's last response uses power-seeking or coercive language to pressure or manipulate the reader. Highlight any parts of the text that could be seen as threatening, intimidating, or coercive."
        critique = query_llm(critique_prompt)
        revision_prompt = f"{critique_prompt}\n\nCritique: {critique}\n\nRevision Request: Please rewrite the initial response to remove any language that seeks to exert power over the reader or coerce them into a specific action. The modified version should maintain a respectful and cooperative tone while avoiding any form of manipulation or intimidation."
        revision = query_llm(revision_prompt, response_format=LLMRevision)
        return revision.revised_text
