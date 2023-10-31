from functools import lru_cache
import json
import logging
import os
import pathlib
import pprint
from typing import Optional

import dotenv
import openai
import pandas as pd

from main import (NOUNS, Object, Subject, Verb, format_sentence,
                  get_random_sentence, sentence_to_str)

dotenv.load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
thisdir = pathlib.Path(__file__).parent.absolute()

def get_english_structure(subject_noun: str,
                          subject_suffix: Optional[str],
                          verb: Optional[str],
                          verb_tense: Optional[str],
                          object_pronoun: Optional[str],
                          object_noun: Optional[str],
                          object_suffix: Optional[str]) -> str:
    sentence_details = []

    subject_info = {'part_of_speech': 'subject'}
    if subject_noun in NOUNS:
        subject_info['word'] = NOUNS[subject_noun]
        subject_info['positional'] = Subject.SUFFIXES[subject_suffix]
    elif subject_noun in Subject.PRONOUNS:
        subject_info['word'] = Subject.PRONOUNS[subject_noun]
    else: # unknown word
        subject_info['word'] = subject_noun
    sentence_details.append(subject_info)
    
    object_info = {'part_of_speech': 'object'}
    
    plural_keywords = ['plural', 'you all', 'they', 'them', 'we', 'us']
    if object_pronoun and any(kw in Object.PRONOUNS[object_pronoun] for kw in plural_keywords):
        object_info['plural'] = True
    if object_noun in NOUNS:
        object_info['word'] = NOUNS[object_noun]
        object_info['positional'] = Object.SUFFIXES[object_suffix]
        sentence_details.append(object_info)
    elif object_pronoun and not object_noun: # object pronoun
        object_info['word'] = Object.PRONOUNS[object_pronoun]
        sentence_details.append(object_info)
    elif (object_noun or '').strip(): # unknown word
        object_info['word'] = object_noun
        object_info['positional'] = Object.SUFFIXES[object_suffix]
        sentence_details.append(object_info)

    verb_info = {'part_of_speech': 'verb'}
    verb_info['word'] = Verb.TRANSIITIVE_VERBS.get(verb, Verb.INTRANSITIVE_VERBS.get(verb))
    if verb_info['word'] is None:
        # raise Exception(f"Invalid verb: {verb}")
        verb_info['word'] = verb
    verb_info['tense'] = Verb.TENSES[verb_tense]
    sentence_details.append(verb_info)

    return sentence_details

@lru_cache(maxsize=1000)
def translate(subject_noun: str,
              subject_suffix: Optional[str],
              verb: Optional[str],
              verb_tense: Optional[str],
              object_pronoun: Optional[str],
              object_noun: Optional[str],
              object_suffix: Optional[str]) -> str:
    structure = get_english_structure(
        subject_noun, subject_suffix,
        verb, verb_tense,
        object_pronoun, object_noun, object_suffix
    )

    examples = [
        {
            'role': 'user', 
            'content': json.dumps(
                [{'part_of_speech': 'subject', 'positional': 'proximal', 'word': 'wood'},
                {'part_of_speech': 'object', 'positional': 'proximal', 'word': 'dog'},
                {'part_of_speech': 'verb', 'tense': 'present ongoing (-ing)', 'word': 'see'}]
            )
        },
        {'role': 'assistant', 'content': 'This wood is seeing this dog.'},
        {
            'role': 'user',
            'content': json.dumps(
                [{'part_of_speech': 'subject', 'positional': 'proximal', 'word': 'cup'},
                 {'part_of_speech': 'object', 'positional': 'distal', 'word': 'cup', 'plural': True},
                 {'part_of_speech': 'verb', 'tense': 'future (will)', 'word': 'eat'}]
            )
        },
        {'role': 'assistant', 'content': 'This cup will eat those cups.'},
        {
            'role': 'user',
            'content': json.dumps(
                [{'part_of_speech': 'subject', 'positional': 'distal', 'word': 'pinenuts'},
                 {'part_of_speech': 'object', 'positional': 'distal', 'word': 'horse'},
                 {'part_of_speech': 'verb', 'tense': 'future (will)', 'word': 'see'}]
            )
        },
        {'role': 'assistant', 'content': 'Those pinenuts will see that horse.'},
    ]
    messages = [
        {'role': 'system', 'content': 'You are an assistant for translating structured sentences into simple natural English sentences.'},
        *examples,
        {'role': 'user', 'content': json.dumps(structure)}
    ]
    res = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages,
        request_timeout=10,
    )

    # pprint.pprint(messages)

    num_tokens = res['usage']['total_tokens']
    logging.info(f"Used {num_tokens} tokens")
    translation = res['choices'][0]['message']['content']
    return translation

def main():
    choices = get_random_sentence()
    sentence_details = format_sentence(**{key: value['value'] for key, value in choices.items()})
    print(f"Sentence: {sentence_to_str(sentence_details)}")
    translation = translate(**{key: value['value'] for key, value in choices.items()})
    print(f"Translation: {translation}")

def evaluate(num: int, savepath: pathlib.Path):
    rows = []
    if savepath.exists():
        df = pd.read_csv(savepath)
        rows = df.to_dict('records')
    for i in range(len(rows), num):
        print(f"Generating sentence {i+1}/{num}")
        choices = get_random_sentence()
        sentence_details = format_sentence(**{key: value['value'] for key, value in choices.items()})
        translation = translate(**{key: value['value'] for key, value in choices.items()})
        rows.append({
            'sentence': sentence_to_str(sentence_details),
            'translation': translation,
        })

        df = pd.DataFrame(rows)
        df.to_csv(savepath, index=False, encoding='utf-8')

if __name__ == '__main__':
    main()
    # evaluate(100, thisdir / '.data' / 'translations.csv')