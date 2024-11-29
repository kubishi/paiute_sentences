import logging
import pathlib
import pandas as pd
import random
import dotenv
import os

from yaduha.sentence_builder import (
    NOUNS, Verb, Subject, Object,
    format_sentence, get_all_choices, get_random_sentence, get_random_simple_sentence, sentence_to_str
)
from yaduha.back_translate import translate as translate_ovp2eng
from yaduha.forward.pipeline import PipelineTranslator

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()
datadir = thisdir / "data"

uncommon_nouns = pd.read_csv(datadir / "uncommon_nouns.csv")
uncommon_verbs = pd.read_csv(datadir / "uncommon_verbs.csv")

def random_translations(savepath: pathlib.Path,
                        n: int = None,
                        overwrite: bool = False,
                        replace_subject_noun: bool = False,
                        replace_object_noun: bool = False,
                        replace_verb: bool = False):
    _get_random_sentence = get_random_sentence
    if replace_object_noun or replace_subject_noun or replace_verb:
        _get_random_sentence = get_random_simple_sentence

    df = pd.DataFrame(columns=["ovp", "eng"])
    if savepath.exists():
        if not overwrite:
            df = pd.read_csv(savepath)
        else:
            logging.info(f"random_translations: Overwriting {savepath}")
    
    # shuffle nouns
    random_nouns = pd.Series([
        *Subject.PRONOUNS.keys(),
        *NOUNS.keys(),
        *Verb.TRANSITIVE_VERBS.keys(), 
        *Verb.INTRANSITIVE_VERBS.keys()
    ]).sample(frac=1)
    random_verbs = pd.Series([*Verb.TRANSITIVE_VERBS.keys(), *Verb.INTRANSITIVE_VERBS.keys()]).sample(frac=1)

    if n is None:
        n = 3*max(len(random_nouns), len(random_verbs))

    if len(df) >= n:
        logging.info(f"random_translations: Already have at least {n} sentences.")
        return df
    
    rows = []
    for i in range(n):
        print(f"random_translations: {(i+1)}/{n} ({(i+1)/n*100:.2f}%)", end="\r")
        choices = get_all_choices()
        choices['subject_noun']['value'] = random_nouns[i % len(random_nouns)]
        choices['verb']['value'] = random_verbs[i % len(random_verbs)]
        if Verb._is_transitive(choices['verb']['value']):
            choices['object_noun']['value'] = random_nouns[random.randint(0, len(random_nouns) - 1)]
            
        choices = _get_random_sentence(choices)
        word_choices = {k: v['value'] for k, v in choices.items()}

        if replace_subject_noun and word_choices["subject_noun"] in NOUNS:
            word_choices["subject_noun"] = f'[{random.choice(uncommon_nouns["noun"])}]'
        if replace_object_noun and word_choices["object_noun"] in NOUNS:
            word_choices["object_noun"] = f'[{random.choice(uncommon_nouns["noun"])}]'
        if replace_verb:
            word_choices["verb"] = f'[{random.choice(uncommon_verbs["verb"])}]'

        eng = translate_ovp2eng(**word_choices)
        ovp = sentence_to_str(format_sentence(**word_choices))

        if eng is not None:
            rows.append({"ovp": ovp, "eng": eng})
            logging.info(f"random_translations: {len(df)} / {n}")
        else:
            logging.info(f"random_translations: Failed to translate {ovp}")

        _df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        _df.to_csv(savepath, index=False)

    print(" "*100, end="\r")
    print(f"random_translations: Done! Saved to {savepath}")


complex_sentences = {
    "The dog barked loudly, as the cat ran away.": "The dog barked. The cat ran.",
    "She sat calmly, on the wooden bench.": "She sat.",
    "She smiled brightly, despite the rain outside.": "She smiled. It rained.",
    "The car stopped suddenly, near the old tree.": "The car stopped.",
    "He spoke softly, while holding her hand.": "He spoke. He held a hand.",
    "They danced happily, on the wooden floor.": "They danced.",
    "The wind blew gently, opening the window.": "The wind blew. The window opened.",
    "She laughed quietly, at his silly joke.": "He told a joke. She laughed.",
    "The flowers bloomed brightly, in the morning sun.": "The flowers bloomed. The sun shined.",
    "The bell rang loudly, over the silent hall.": "The bell rang.",
    "The car skidded wildly and crashed into the fence.": "The car skidded. The car crashed.",
    "She grabbed the phone and called for help.": "She grabbed the phone. She sought help.",
    "He laughed loudly but dropped his keys.": "He laughed. He dropped the keys.",
    "The dog chased the ball and barked with joy.": "The dog chased the ball. The dog barked.",
    "The kids ran outside and played in the rain.": "The kids ran. The kids played.",
    "She opened the book and found a hidden note.": "She opened the book. She found a note.",
    "He shouted for help and waved his arms.": "He shouted. He waved his arms.",
    "The rain poured heavily and soaked the ground.": "It rained. The rain soaked the ground.",
    "She stirred the soup and tasted it cautiously.": "She stirred the soup. She tasted it.",
    "He climbed the ladder and painted the wall.": "He climbed the ladder. He painted the wall.",
    "The wind howled loudly and shook the windows.": "The wind howled. It shook the windows.",
    "She folded the clothes and placed them neatly.": "She folded the clothes. She placed them.",
    "He picked the flowers and arranged them in a vase.": "He picked the flowers. He arranged them.",
    "The child giggled softly and hid behind the curtain.": "The child giggled. The child hid.",
    "The birds fluttered around and chirped in the tree.": "The birds fluttered. The birds chirped.",
}
def random_complex_translations(savepath: pathlib.Path,
                                overwrite: bool = False):
    translator = PipelineTranslator(model="gpt-4o-mini")
    df = pd.DataFrame(columns=["ovp", "eng", "simple"])
    if savepath.exists():
        if not overwrite:
            df = pd.read_csv(savepath)
        else:
            logging.info(f"random_translations: Overwriting {savepath}")

    # remove any rows where ovp['eng'] is not in the complex_sentences
    df = df[~df["eng"].isin(complex_sentences)]

    rows = []
    for sentence, simple_sentence in complex_sentences.items():
        if simple_sentence in df["eng"].values:
            rows.append([df.loc[df["eng"] == simple_sentence, "ovp"].values[0], sentence, simple_sentence])
        else:
            translation = translator.translate(simple_sentence)
            rows.append([translation.target, sentence, translation.simple])

        _df = pd.DataFrame(rows, columns=["ovp", "eng", "simple"])
        _df.to_csv(savepath, index=False)

def main():
    overwrite = False
    random_translations(
        datadir / "random_good_translations.csv",
        overwrite=overwrite
    )
    # random_translations(
    #     datadir / "random_no_subject_noun.csv",
    #     n=25,
    #     overwrite=overwrite,
    #     replace_subject_noun=True
    # )
    # random_translations(
    #     datadir / "random_no_object_noun.csv",
    #     n=25,
    #     overwrite=overwrite,
    #     replace_object_noun=True
    # )
    # random_translations(
    #     datadir / "random_no_verb.csv",
    #     n=25,
    #     overwrite=overwrite,
    #     replace_verb=True
    # )
    # random_translations(
    #     datadir / "random_no_nouns.csv",
    #     n=25,
    #     overwrite=overwrite,
    #     replace_subject_noun=True,
    #     replace_object_noun=True
    # )
    # random_translations(
    #     datadir / "random_no_vocab.csv",
    #     n=25,
    #     overwrite=overwrite,
    #     replace_subject_noun=True,
    #     replace_object_noun=True,
    #     replace_verb=True
    # )
    # random_complex_translations(
    #     datadir / "random_complex_translations.csv",
    #     overwrite=overwrite
    # )



if __name__ == "__main__":
    main()
