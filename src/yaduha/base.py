from abc import ABC, abstractmethod
from typing import Dict, Union
import dotenv
from pydantic import BaseModel

dotenv.load_dotenv()

class Translation(BaseModel):
    source: str
    target: str
    back_translation: str

    translation_prompt_tokens: int
    translation_completion_tokens: int
    translation_time: float
    back_translation_prompt_tokens: int
    back_translation_completion_tokens: int
    back_translation_time: float

    metadata: Dict[str, Union[str, int, float]] = {}

class Translator(ABC):
    @abstractmethod
    def translate(self, text: str) -> Translation:
        """Translate the text to the target language and back to the source language.
        
        Args:
            text (str): The text to translate.

        Returns:
            Translation: The translation
        """
        raise NotImplementedError


    
