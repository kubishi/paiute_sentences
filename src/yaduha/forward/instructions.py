import time
import openai
import os

from yaduha.base import Translation, Translator
from yaduha.sentence_builder import LENIS_MAP, NOUNS, Verb, Subject, Object

class InstructionsTranslator(Translator):
    def __init__(self, model: str):
        self.model = model
        self.client = openai.Client(api_key=os.environ.get('OPENAI_API_KEY'))

    def translate(self, text: str) -> str:
        start_time = time.time()
        examples = [
            ("The dog is running.", "isha'pugu-ii poyoha-ti."),
            ("That cat will eat this rice.", "kidi'-uu wai-neika a-düka-wei."),
            ("Jared has eaten", "[Jared]-ii tüka-pü."),
            ( # Missing Vocabulary Example
                "The boy and the girl are eating a quesadilla.",
                "[boy]-ii [quesadilla]-neika a-düka-ti. [girl]-ii [quesadilla]-neika a-düka-ti."
            ),
            ( # Nominalization Example
                "This cook saw the ones who walked by the house.",
                "sawa-dü-ii hukaw̃ia-doka ui-buni-ku."
            ),
            ( # Nominalization Example
                "This one who will cook drank the water.",
                "sawa-weidü-ii paya-neika a-hibi-wei."
            ),
            ( # Simplified Sentence Structure Example
                "Jane went to the store and bought some rice and juice.",
                "[Jane]-uu mia-ku. Uhu wai-noka u-[buy]-ku. Uhu [juice]-noka u-[buy]-ku."
            )
        ]
        example_messages = []
        for eng, ovp in examples:
            example_messages.extend([
                {"role": "user", "content": eng},
                {"role": "assistant", "content": ovp}
            ])
        messages = [
            {
                "role": "system",
                "content": (
                    "You use the following grammar rules to translate user input sentences from English to Owens Valley Paiute.\n" + 
                    "Use the vocabulary and sentence structures available to translate the input sentence as best as possible.\n" +
                    "It doesn't need to be perfect and you can leave English words untranslated if necessary.\n" +
                    # section on vocabulary
                    "# Vocabulary\n" +
                    "## Nouns: \n" + "\n".join([f"{ovp}: {eng}" for ovp, eng in NOUNS.items()]) + "\n" +
                    "## Transitive Verbs: \n" + "\n".join([f"{ovp}: {eng}" for ovp, eng in Verb.TRANSITIVE_VERBS.items()]) + "\n" +
                    "## Intransitive Verbs: \n" + "\n".join([f"{ovp}: {eng}" for ovp, eng in Verb.INTRANSITIVE_VERBS.items()]) + "\n" +
                    "## Object Suffixes: \n" + "\n".join([f"{ovp}: {eng}" for ovp, eng in Object.SUFFIXES.items()]) + "\n" +
                    "## Object Pronouns: \n" + "\n".join([f"{ovp}: {eng}" for ovp, eng in Object.PRONOUNS.items()]) + "\n" +
                    "## Subject Suffixes: \n" + "\n".join([f"{ovp}: {eng}" for ovp, eng in Subject.SUFFIXES.items()]) + "\n" +
                    "## Subject Pronouns: \n" + "\n".join([f"{ovp}: {eng}" for ovp, eng in Subject.PRONOUNS.items()]) + "\n" +
                    "## Verb Nominalizer Tenses: \n" + "\n".join([f"{ovp}: {eng}" for ovp, eng in Verb.NOMINALIZER_TENSES.items()]) + "\n" +
                    "\n# Sentence Structure\n" + # next section on sentence structure
                    "## Simple Sentence Structure: \n" +
                    "Subject-Object-Verb: [object noun]-[object suffix] [subject noun]-[subject suffix] [object pronoun]-[verb]-[verb tense]\n" +
                    "Subject Pronoun-Object-Verb: [object noun]-[object suffix] [subject pronoun] [object pronoun]-[verb]-[verb tense]\n" +
                    "Subject-Verb: [verb]-[verb tense] [subject noun]-[subject suffix]\n" +
                    "## Verb Nominalization Sentence Structure: \n" +
                    "Subject Nominalizer: [verb]-[verb nominalizer tense]-[subject suffix] [verb nominalizer]-[verb nominalizer tense]\n" +
                    "Object Nominalizer: [verb]-[verb nominalizer tense]-[object suffix] [subject noun]-[subject suffix] [object pronoun]-[verb]-[verb tense]\n" +
                    "Subject&Object Nominalizer: [verb]-[verb nominalizer tense]-[object suffix] [verb nominalizer]-[verb nominalizer tense]-[subject suffix] [subject noun]-[subject suffix] [object pronoun]-[verb]-[verb tense]\n" +
                    "\n# Fortis/Lenis Transformations\n" + # next section on fortis/lenis transformations
                    ", ".join([f"{f}->{l}" for f, l in LENIS_MAP.items()]) + "\n"
                )
            },
            *example_messages,
            {
                "role": "user",
                "content": text
            }
        ]

        res = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        prompt_tokens = res.usage.prompt_tokens
        completion_tokens = res.usage.completion_tokens
        target = res.choices[0].message.content
        translation_time = time.time() - start_time

        # TODO: Back Translation
        back_translation = ""
        back_translation_prompt_tokens = 0
        back_translation_completion_tokens = 0
        back_translation_time = 0.0

        return Translation(
            source=text,
            target=target,
            back_translation=back_translation,
            translation_prompt_tokens=prompt_tokens,
            translation_completion_tokens=completion_tokens,
            translation_time=translation_time,
            back_translation_prompt_tokens=back_translation_prompt_tokens,
            back_translation_completion_tokens=back_translation_completion_tokens,
            back_translation_time=back_translation_time
        )