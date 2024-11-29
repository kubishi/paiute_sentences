import time
from ..base import Translation, Translator
import openai
import os

class FinetunedTranslator(Translator):
    DEFAULT_SYSTEM_PROMPT = (
        "You are a translator for translating text from English to OVP. "
        "For any word that does not have an equivalent in OVP, "
        "leave the word untranslated and place it inside brackets. "
    )
    DEFAULT_EXAMPLES = [
        ("The sleet climbs that rafter.", "[sleet]-uu [rafter]-noka u-dsibui-dü"),
        ("The pouch has smelled this ledge.", "[pouch]-uu [ledge]-neika a-gwana-pü"),
        ("The jackrabbit is eating the pinenuts.", "kamü-uu tüba-neika a-düka-ti")
    ]

    @classmethod
    def get_default_messages(cls):
        messages = [
            {
                "role": "system",
                "content": cls.DEFAULT_SYSTEM_PROMPT
            }
        ]
        for (source, target) in cls.DEFAULT_EXAMPLES:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": source
                    }
                ]
            })
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": target
                    }
                ]
            })
        return messages

    def __init__(self,
                 model: str,
                 system_prompt: str = None,
                 examples: list = None):
        self.model = model
        self.client = openai.Client(api_key=os.getenv('OPENAI_API_KEY'))
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def translate(self, sentence: str) -> Translation:
        start_time = time.time()
        messages = self.get_default_messages()
        messages += [
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": sentence
                    }
                ]
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        target = response.choices[0].message.content
        translation_time = time.time() - start_time
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        translation = Translation(
            source=sentence,
            target=target,
            back_translation="",
            translation_prompt_tokens=prompt_tokens,
            translation_completion_tokens=completion_tokens,
            translation_time=translation_time,
            back_translation_prompt_tokens=0,
            back_translation_completion_tokens=0,
            back_translation_time=0
        )
        return translation
