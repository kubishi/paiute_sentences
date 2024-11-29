"""Functions for translating simple sentences from English to Paiute."""
from dataclasses import dataclass
from functools import partial
import os
import random
import time
from typing import Dict, List, Optional, Tuple, Union

from openai.types.chat import ChatCompletion

from ..sentence_builder import NOUNS, Object, Subject, Verb
from ..segment import make_sentence, semantic_similarity_transformers, semantic_similarity_openai, split_sentence
from ..back_translate import translate as translate_ovp_to_english
from ..base import Translator, Translation

SS_MODE = os.getenv('SS_MODE', 'sentence-transformers')

if SS_MODE == 'openai':
    semantic_similarity = partial(semantic_similarity_openai, model='text-embedding-ada-002')
else:
    semantic_similarity = partial(semantic_similarity_transformers, model='sentence-transformers/all-MiniLM-L6-v2')

R_TRANSITIVE_VERBS = {v: k for k, v in Verb.TRANSITIVE_VERBS.items()}
R_INTRANSITIVE_VERBS = {v: k for k, v in Verb.INTRANSITIVE_VERBS.items()}
R_NOUNS = {v: k for k, v in NOUNS.items()}

R_OBJECT_PRONOUNS = {
    'me': ['i'],
    'you': ['ü'],
    'him': ['u', 'a', 'ma'],
    'her': ['u', 'a', 'ma'],
    'it': ['u', 'a', 'ma'],
    'us': ['tei', 'ni'],
    'them': ['ui', 'ai', 'mai'],
    'you all': ['üi'],
}
R_VERB_TENSES = {
    'present': 'dü',
    'past': 'ku',
    'future': 'wei',
    'past_continuous': 'ti',
    'present_continuous': 'ti'
}
R_SUBJECT_PRONOUNS = {
    'i': ['nüü'],
    'you': ['üü'],
    'he': ['uhu', 'mahu'],
    'she': ['uhu', 'mahu'],
    'it': ['uhu', 'mahu'],
    'we': ['nüügwa', 'taagwa'],
    'they': ['uhuw̃a', 'mahuw̃a'],
    'you all': ['üügwa'],
    'this': ['ihi']
}
R_VERB_NOMINALIZERS = {
    'present': 'dü',
    'past': 'pü',
    'future': 'weidü',
}

def translate_simple(sentence: Dict[str, str]) -> Tuple[Subject, Verb, Object]:
    """Translate a simple English sentence to Paiute.

    Args:
        sentence (Dict[str, str]): A simple English sentence.

    Returns:
        List[Union[Subject, Verb, Object]]: A list of Paiute words.
    """
    sentence = {k: v.strip().lower() for k, v in sentence.items() if v}
    if sentence.get('object'):
        verb_stem = R_TRANSITIVE_VERBS.get(sentence['verb'], f"[{sentence['verb']}]")
    else:
        verb_stem = R_INTRANSITIVE_VERBS.get(sentence['verb'], R_TRANSITIVE_VERBS.get(sentence['verb'], f"[{sentence['verb']}]"))

    verb_tense = R_VERB_TENSES.get(sentence['verb_tense'], f"[{sentence['verb_tense']}]")
    verb = Verb(verb_stem, verb_tense, object_pronoun_prefix=None)

    _object = None
    if (sentence.get('object') or '').strip(): # if there is an object
        if sentence['object'] in R_OBJECT_PRONOUNS:
            object_pronoun = random.choice(R_OBJECT_PRONOUNS.get(sentence['object'], [sentence['object']]))
            verb = Verb(verb_stem, verb_tense, object_pronoun_prefix=object_pronoun)
        elif sentence.get('object_nominalizer'):
            _object = {**R_TRANSITIVE_VERBS, **R_INTRANSITIVE_VERBS}.get(sentence['object'], f"[{sentence['object']}]")
            object_suffix = random.choice(['eika', 'oka'])
            object_nominalizer = R_VERB_NOMINALIZERS.get(sentence['object_nominalizer'], "dü")
            _object = Object(_object, object_nominalizer, object_suffix)
        else:
            _object = R_NOUNS.get(sentence['object'], f"[{sentence['object']}]")
            object_pronoun = random.choice(R_OBJECT_PRONOUNS['it'])
            object_suffix = Object.get_matching_suffix(object_pronoun)
            _object = Object(_object, None, object_suffix)
            # verb = f"{object_pronoun}-{verb}"
            verb = Verb(verb_stem, verb_tense, object_pronoun_prefix=object_pronoun)

    if sentence.get('subject_nominalizer'):
        subject = {**R_TRANSITIVE_VERBS, **R_INTRANSITIVE_VERBS}.get(sentence['subject'], f"[{sentence['subject']}]")
        subject_suffix = random.choice(['ii', 'uu'])
        subject_nominalizer = R_VERB_NOMINALIZERS.get(sentence['subject_nominalizer'], "dü")
        # subject = f"{subject}-{subject_suffix}"
        subject = Subject(subject, subject_nominalizer, subject_suffix)
    elif sentence['subject'] in R_SUBJECT_PRONOUNS:
        subject = random.choice(R_SUBJECT_PRONOUNS.get(sentence['subject'], [sentence['subject']]))
        subject = Subject(subject, subject_noun_nominalizer=None, subject_suffix=None)
    else:
        subject = R_NOUNS.get(sentence['subject'], f"[{sentence['subject']}]")
        subject_suffix = random.choice(['ii', 'uu'])
        # subject = f"{subject}-{subject_suffix}"
        subject = Subject(subject, None, subject_suffix)

    return subject, verb, _object

def order_sentence(subject: Subject, verb: Verb, _object: Optional[Object] = None) -> List[Union[Subject, Verb, Object]]:
    if subject.noun in Subject.PRONOUNS:
        sentence = [subject, verb] if _object is None else [subject, _object, verb]
    else:
        sentence = [verb, subject] if _object is None else [_object, subject, verb]
    return sentence

def comparator_sentence(simple_sentence: Dict[str, str]) -> str:
    simple_sentence = {k: v.strip().lower() for k, v in simple_sentence.items() if v}
    subject_nominalizer = simple_sentence.get('subject_nominalizer')
    object_nominalizer = simple_sentence.get('object_nominalizer')
    if subject_nominalizer is None:
        if simple_sentence['subject'] not in {*R_SUBJECT_PRONOUNS, *R_NOUNS}:
            simple_sentence['subject'] = '[SUBJECT]'
    else:
        if simple_sentence['subject'] not in {*R_TRANSITIVE_VERBS, *R_INTRANSITIVE_VERBS}:
            simple_sentence['subject'] = '[SUBJECT]'
    if simple_sentence['verb'] not in R_TRANSITIVE_VERBS and simple_sentence['verb'] not in R_INTRANSITIVE_VERBS:
        simple_sentence['verb'] = '[VERB]'
    
    if object_nominalizer is None:
        if (simple_sentence.get('object') or '').strip() and simple_sentence['object'] not in {*R_NOUNS, *R_OBJECT_PRONOUNS}:
            simple_sentence['object'] = '[OBJECT]'
    else:
        if (simple_sentence.get('object') or '').strip() and simple_sentence['object'] not in R_NOUNS:
            simple_sentence['object'] = '[OBJECT]'

    return simple_sentence

class PipelineTranslator(Translator):
    def __init__(self, model: str):
        self.model = model

    def translate(self, sentence: str) -> Translation:
        start_time = time.time()
        prompt_tokens = 0
        completion_tokens = 0
        model_calls = 0
        def res_callback(res: ChatCompletion):
            nonlocal prompt_tokens, completion_tokens, model_calls
            prompt_tokens += res.usage.prompt_tokens
            completion_tokens += res.usage.completion_tokens
            model_calls += 1

        prompt_tokens_back = 0
        completion_tokens_back = 0
        model_calls_back = 0
        def res_callback_backwards(res: ChatCompletion):
            nonlocal prompt_tokens_back, completion_tokens_back, model_calls_back
            prompt_tokens_back += res.usage.prompt_tokens
            completion_tokens_back += res.usage.completion_tokens
            model_calls_back += 1

        simple_sentences = split_sentence(sentence, model=self.model, res_callback=res_callback)
        comparator_sentences = []
        target_simple_sentences = []
        backwards_translations = []
        back_translation_time = 0
        for simple_sentence in simple_sentences:
            comparator_sentences.append(comparator_sentence(simple_sentence))
            subject, verb, _object = translate_simple(simple_sentence)
            target_simple_sentence = order_sentence(subject, verb, _object)
            target_simple_sentences.append(" ".join(map(str, target_simple_sentence)))
            back_translation_start_time = time.time()
            backwards_translations.append(
                translate_ovp_to_english(
                    subject_noun=subject.noun,
                    subject_noun_nominalizer=subject.subject_noun_nominalizer,
                    subject_suffix=subject.subject_suffix,
                    verb=verb.verb_stem,
                    verb_tense=verb.tense_suffix,
                    object_pronoun=verb.object_pronoun_prefix,
                    object_noun=_object.noun if _object else None,
                    object_noun_nominalizer=_object.object_noun_nominalizer if _object else None,
                    object_suffix=_object.object_suffix if _object else None,
                    res_callback=res_callback_backwards
                ).strip(".")
            )
            back_translation_time += time.time() - back_translation_start_time

        simple_sentences_nl = ". ".join([make_sentence(sentence, model=self.model, res_callback=res_callback) for sentence in simple_sentences]) + '.'
        comparator_sentence_nl = ". ".join([make_sentence(sentence, model=self.model, res_callback=res_callback_backwards) for sentence in comparator_sentences]) + '.'
        target_simple_sentence_nl = ". ".join(target_simple_sentences) + '.'
        backwards_translation_nl = ". ".join(backwards_translations) + '.'

        translation_time = (time.time() - start_time) - back_translation_time
        return Translation(
            source=sentence,
            target=target_simple_sentence_nl,
            back_translation=backwards_translation_nl,
            translation_prompt_tokens=prompt_tokens,
            translation_completion_tokens=completion_tokens,
            translation_time=translation_time,
            back_translation_prompt_tokens=prompt_tokens_back,
            back_translation_completion_tokens=completion_tokens_back,
            back_translation_time=back_translation_time,
            metadata={
                'simple': simple_sentences_nl,
                'comparator': comparator_sentence_nl,
                'model_calls': model_calls,
                'back_model_calls': model_calls_back,
            }
        )
