from typing import Optional, Callable
import pathlib
import argparse
import random
import numpy as np
import pandas as pd

from yaduha.back_translate import translate
from yaduha.sentence_builder import format_sentence, get_random_sentence, get_random_sentence_big, sentence_to_str



def translate_random(_get_random_sentence: Optional[Callable[[], dict]] = get_random_sentence):
    choices = _get_random_sentence()
    sentence_details = format_sentence(**{key: value['value'] for key, value in choices.items()})
    translation = translate(**{key: value['value'] for key, value in choices.items()})
    print(f"Sentence: {sentence_to_str(sentence_details)}")
    print(f"Translation: {translation}")

def evaluate(num: int, savepath: pathlib.Path, _get_random_sentence: Optional[Callable[[], dict]] = get_random_sentence):
    rows = []
    if savepath.exists():
        df = pd.read_csv(savepath)
        rows = df.to_dict('records')
    for i in range(len(rows), num):
        print(f"Generating sentence {i+1}/{num}")
        choices = _get_random_sentence()
        sentence_details = format_sentence(**{key: value['value'] for key, value in choices.items()})
        translation = translate(**{key: value['value'] for key, value in choices.items()})
        rows.append({
            'sentence': sentence_to_str(sentence_details),
            'translation': translation,
            'correct': None
        })

        df = pd.DataFrame(rows)
        df.to_csv(savepath, index=False, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description='Translate OVP sentences to English')
    parser.add_argument('--seed', type=int, help='Seed for random sentence generation')
    parser.add_argument('--big', action='store_true', help='Generate a random "big" sentence to translate')

    subparsers = parser.add_subparsers(dest='subparser_name')

    translate_parser = subparsers.add_parser('translate-random', help='Translate a randomly generated sentence')
    translate_parser.set_defaults(func='translate-random')

    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate the translation of a number of randomly generated sentences')
    evaluate_parser.add_argument('num', type=int, help='Number of sentences to evaluate')
    evaluate_parser.add_argument('savepath', type=pathlib.Path, help='Path to save the evaluation results')
    evaluate_parser.set_defaults(func='evaluate')
    
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    _get_random_sentence = get_random_sentence_big if args.big else get_random_sentence

    if not hasattr(args, 'func') or getattr(args, 'func') is None:
        parser.print_help()
        return
    elif args.func == 'translate-random':
        translate_random(_get_random_sentence)
    elif args.func == 'evaluate':
        evaluate(args.num, args.savepath, _get_random_sentence)

if __name__ == '__main__':
    main()