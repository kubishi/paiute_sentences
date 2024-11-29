import logging
import time
import openai
import json
from typing import Dict, List, Optional, Union
import openai
import pathlib
import os
import tempfile
from openai.types.chat import ChatCompletion

from ..sentence_builder import format_sentence, get_all_choices, sentence_to_str
from ..back_translate import translate as translate_ovp2eng
from ..base import Translator, Translation


class AgenticTranslator(Translator):
    """Translator that takes an agentive approach to translation.
    
    In each step, the translator uses the 
    """
    
    def __init__(self,
                 model: Optional[str] = None, 
                 savepath: Optional[pathlib.Path] = None,
                 max_iterations: int = 30,
                 max_sentences: int = 4,
                 auto_choices: List[str] = []):
        self.openai_model = model
        self.openai_client = None
        self.savepath = savepath
        self.max_iterations = max_iterations
        self.max_sentences = max_sentences
        self.auto_choices = auto_choices

        # create temporary directory for examples that doesn't delete automatically
        self._examples_dir = None
        if model is not None:
            self.openai_client = openai.Client(api_key=os.environ.get("OPENAI_API_KEY"))
            self._examples_dir = tempfile.TemporaryDirectory()
            self._build_examples()

    def __del__(self):
        if self._examples_dir is not None:
            self._examples_dir.cleanup()

    def _build_examples(self):
        savedir = pathlib.Path(self._examples_dir.name)
        target_sentence = "This dog and cat are running."
        sentence = AgenticTranslator(
            savepath=savedir / "messages-dog-cat.json",
            auto_choices=[
                "subject_noun",
                "isha'pugu",
                "subject_suffix",
                "ii",
                "verb",
                "poyoha",
                "verb_tense",
                "ti",
                "n", # new sentence
                "subject_noun",
                "kidi'",
                "subject_suffix",
                "ii",
                "verb",
                "poyoha",
                "verb_tense",
                "ti",
                "t", # terminate
            ]
        ).translate(target_sentence)
        logging.info(f"{target_sentence} => {sentence}")

        target_sentence = "Jared will eat an apple."
        sentence = AgenticTranslator(
            savepath= savedir / "messages=jared-apple.json",
            auto_choices=[
                "subject_noun",
                "[Jared]",
                "subject_suffix",
                "ii",
                "verb",
                "tüka",
                "verb_tense",
                "wei",
                "c", # continue this sentence
                "object_noun",
                "aaponu'",
                "object_suffix",
                "eika",
                "object_pronoun",
                "a",
                "t", # terminate
            ]
        ).translate(target_sentence)
        logging.info(f"{target_sentence} => {sentence}")

        target_sentence = "That frog drank that juice."
        sentence = AgenticTranslator(
            savepath=savedir / "messages-frog-water.json",
            auto_choices=[
                "subject_noun",
                "[frog]",
                "subject_suffix",
                "uu",
                "verb",
                "hibi",
                "verb_tense",
                "ku",
                "c", # new sentence
                "object_noun",
                "[juice]",
                "object_suffix",
                "oka",
                "object_pronoun",
                "u",
                "t", # terminate
            ]
        ).translate(target_sentence)
        logging.info(f"{target_sentence} => {sentence}")

        target_sentence = "The runner saw the one who will eat."
        sentence = AgenticTranslator(
            savepath=savedir / "messages-nominalization.json",
            auto_choices=[
                "subject_noun",
                "poyoha",
                "subject_noun_nominalizer",
                "dü",
                "subject_suffix",
                "uu",
                "verb",
                "puni",
                "verb_tense",
                "ku",
                "c", # continue this sentence
                "object_noun",
                "tüka",
                "object_noun_nominalizer",	
                "weidü",
                "object_suffix",
                "oka",
                "object_pronoun",
                "u",
                "t", # terminate
            ]
        ).translate(target_sentence)
        logging.info(f"{target_sentence} => {sentence}")

    @property
    def example_paths(self) -> List[pathlib.Path]:
        if self._examples_dir is None:
            return []
        return list(pathlib.Path(self._examples_dir.name).glob("*.json"))

    def translate(self, text: str) -> Translation:
        start_time = time.time()
        if self.auto_choices and self.openai_model is not None:
            raise Exception("Cannot use auto_choices with openai_model=True")
        
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an assistant trying to build a sentence in Paiute. "
                    "The user will provide you with parts of speech and vocabulary options one at a time. "
                    "Make choices to best approximate the meaning of Input Sentence. "
                    "Do not make choices that are not provided by the user. "
                    "This may mean you can't build the sentence you want, but that's okay. "
                    "Whenever you've chosen enough parts of speech and vocabulary to form a grammatically correct sentence, "
                    "The user will ask you if you want to continue. "
                    "If you're happy with the sentence you've built, you can choose to stop. "
                    "If not, continue selecting optional parts of speech and vocabulary until you're satisfied."
                )
            }
        ]
        word_choices = {}

        for example_path in self.example_paths:
            example_messages = json.loads(example_path.read_text())
            messages.extend(example_messages[1:])

        choice_idx: int = 0
        prompt_tokens: int = 0
        completion_tokens: int = 0
        model_calls: int = 0
        if self.openai_model is not None:
            def get_choice(choices: Union[Dict[str, str],List[str]],
                           prompt: str = "Options: ",
                           allow_wild: bool = False) -> str:
                nonlocal prompt_tokens, completion_tokens, model_calls
                messages.append({"role": "user", "content": prompt})
                logging.info(f"{self.openai_model}: {prompt}")
                model_calls += 1
                res = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    temperature=0.0
                )
                choice = res.choices[0].message.content
                prompt_tokens += res.usage.prompt_tokens
                completion_tokens += res.usage.completion_tokens
                logging.info(f"{self.openai_model}: {choice}")
                messages.append({"role": "assistant", "content": choice})
                try_count = 0
                while choice not in choices and not (allow_wild and choice.startswith('[') and choice.endswith(']')):
                    messages.append({"role": "user", "content": f"Invalid choice. Please try again.\n{prompt}"})
                    model_calls += 1
                    res = self.openai_client.chat.completions.create(
                        model=self.openai_model,
                        messages=messages,
                        temperature=0.0
                    )
                    choice = res.choices[0].message.content
                    prompt_tokens += res.usage.prompt_tokens
                    completion_tokens += res.usage.completion_tokens
                    logging.info(f"{self.openai_model} [Retry]: {choice}")
                    messages.append({"role": "assistant", "content": choice})
                    try_count += 1
                    if try_count > 3:
                        raise Exception("Max retries reached." + json.dumps(messages[-6:], indent=2, ensure_ascii=False))
                return choice

        else:
            def get_choice(choices: Union[Dict[str, str], List[str]],
                           prompt: str = "Options: ",
                           allow_wild: bool = False) -> str:
                nonlocal choice_idx    
                messages.append({"role": "user", "content": prompt})
                if choice_idx < len(self.auto_choices):
                    choice = self.auto_choices[choice_idx]
                    choice_idx += 1
                else:
                    choice = input(prompt + "\nChoice: ")
                messages.append({"role": "assistant", "content": choice})
                try_count = 0
                while choice not in choices and not (allow_wild and choice.startswith('[') and choice.endswith(']')):
                    messages.append({"role": "user", "content": f"Invalid choice. Please try again.\n{prompt}"})
                    if choice_idx < len(self.auto_choices):
                        choice = self.auto_choices[choice_idx]
                        choice_idx += 1
                    else:
                        choice = input(f"Invalid choice. Please try again.\n{prompt}\nChoice: ")
                    messages.append({"role": "assistant", "content": choice})
                    try_count += 1
                    if try_count > 3:
                        raise Exception("Max retries reached." + json.dumps(messages[-6:], indent=2, ensure_ascii=False))
                return choice

        iteration = 0
        sentences = []
        all_word_choices = []
        sentence = "" # current sentence
        while True:
            iteration += 1
            if iteration > self.max_iterations:
                if self.savepath is not None:
                    self.savepath.parent.mkdir(parents=True, exist_ok=True)
                    self.savepath.write_text(json.dumps(messages, indent=4, ensure_ascii=False))
                raise Exception("Max iterations reached.")
            if len(sentences) >= self.max_sentences:
                if self.savepath is not None:
                    self.savepath.parent.mkdir(parents=True, exist_ok=True)
                    self.savepath.write_text(json.dumps(messages, indent=4, ensure_ascii=False))
                raise Exception("Max sentences reached.")
            choices = get_all_choices(**word_choices)
            word_choices = {k: v['value'] for k, v in choices.items()}
            try:
                sentence = sentence_to_str(format_sentence(**word_choices)).strip()
                continue_choice = get_choice(
                    ["c", "n", "t"],
                    "Input Sentence: " + text + "\n" +
                    "Current Translation: " + ". ".join([*sentences, sentence]) + ".\n" +
                    "Enter one of the following choices:\n" +
                    "c: Continue building the last sentence\n" +
                    "n: Add and build a new Paiute sentence for this translation\n" +
                    "t: Terminate and return the current translation. "
                    "Respond only with your choices and no other text."
                )
                if not continue_choice == "c":
                    sentences.append(sentence)
                    all_word_choices.append(word_choices)
                    sentence = "" # reset sentence
                    if continue_choice == "t": # terminate and return sentences
                        if self.savepath is not None:
                            self.savepath.parent.mkdir(parents=True, exist_ok=True)
                            self.savepath.write_text(json.dumps(messages, indent=4, ensure_ascii=False))
                        
                        backwards_prompt_tokens = 0
                        backwards_completion_tokens = 0
                        backwards_model_calls = 0
                        def count_tokens(completion: ChatCompletion):
                            nonlocal backwards_prompt_tokens, backwards_completion_tokens, backwards_model_calls
                            backwards_prompt_tokens += completion.usage.prompt_tokens
                            backwards_completion_tokens += completion.usage.completion_tokens
                            backwards_model_calls += 1
        
                        translation = ". ".join(sentences) + "."
                        translation_time = time.time() - start_time
                        back_translation_start_time = time.time()
                        back_translation = " ".join([
                            translate_ovp2eng(**_word_choices, res_callback=count_tokens)
                            for _word_choices in all_word_choices
                        ])
                        back_translation_time = time.time() - back_translation_start_time
                        return Translation(
                            source=text,
                            target=translation,
                            back_translation=back_translation,
                            translation_prompt_tokens=prompt_tokens,
                            translation_completion_tokens=completion_tokens,
                            translation_time=translation_time,
                            back_translation_prompt_tokens=backwards_prompt_tokens,
                            back_translation_completion_tokens=backwards_completion_tokens,
                            back_translation_time=back_translation_time,
                            metadata={
                                "model_calls": model_calls,
                                "back_model_calls": backwards_model_calls
                            }
                        )
                    elif continue_choice == "n": # start a new sentence
                        word_choices = {}
                        iteration = 0
                        continue
                else:
                    pass # continue building sentence
            except Exception as e:
                sentence = "" # reset sentence
                pass

            required_parts_of_speech = [
                part_of_speech
                for part_of_speech, details in choices.items()
                if details["requirement"] == "required" and not word_choices.get(part_of_speech)
            ]
            if required_parts_of_speech:
                # ask user to select which part of speech they want to choose next
                try:
                    part_of_speech = get_choice(
                        required_parts_of_speech,
                        "Input Sentence: " + text + "\n" +
                        "Current Translation: " + ". ".join([*sentences, sentence]) + ".\n" +
                        f"Current Choices: {word_choices}\n" +
                        "Please select one of the following part of speeches to choose next: " +
                        ", ".join(required_parts_of_speech) + ". " +
                        "Respond only with your choices and no other text."
                    )
                except Exception as e:
                    logging.warning(f"Error: {e}")
                    continue
            else:
                non_disabled_parts_of_speech = [
                    part_of_speech
                    for part_of_speech, details in choices.items()
                    if details["requirement"] != "disabled"
                ]
                # let user select any part of speech
                try:
                    part_of_speech = get_choice(
                        non_disabled_parts_of_speech,
                        "Input Sentence: " + text + "\n" +
                        "Current Translation: " + ". ".join([*sentences, sentence]) + ".\n" +
                        f"Current Choices: {word_choices}\n" +
                        "Please select one of the following part of speeches to choose next: " +
                        ", ".join(non_disabled_parts_of_speech) + ". " +
                        "Respond only with your choices and no other text."
                    )
                except Exception as e:
                    logging.warning(f"Error: {e}")
                    continue

            # ask user to select which word they want to choose for the selected part of speech
            allow_wild = part_of_speech in ["subject_noun", "object_noun", "verb"]
            try:
                choice = get_choice(
                    choices[part_of_speech]["choices"],
                    "Input Sentence: " + text + "\n" +
                    "Current Translation: " + ". ".join([*sentences, sentence]) + ".\n" +
                    f"Current Choices: {word_choices}\n" +
                    f"Please select a word for {part_of_speech}: " + ", ".join(
                        [f"{k} ({v})" for k, v in choices[part_of_speech]["choices"].items()]
                    ) + (
                        f"\nBecause this is a {part_of_speech} word, you can also choose to use a wildcard " + 
                        "by putting the word in brackets. For example: [wildcard]"
                        if allow_wild else ""
                    ) + ". Respond only with your choices and no other text.",
                    allow_wild=allow_wild
                )
            except Exception as e:
                logging.warning(f"Error: {e}")
                continue
            word_choices[part_of_speech] = choice



        