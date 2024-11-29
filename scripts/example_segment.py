import json
import os
import functools
import numpy as np
import pandas as pd
from yaduha.rbo import RankingSimilarity
import pathlib

from yaduha.segment import (
    semantic_similarity_spacy, semantic_similarity_bert, 
    semantic_similarity_sentence_transformers, semantic_similarity_openai,
    split_sentence, make_sentence, nlp
)

thisdir = pathlib.Path(__file__).parent.absolute()

def main(): # pylint: disable=missing-function-docstring
    source_sentences = [
        "The dog fell.",
        "The dog fell yesterday.",
        "The dog was running yesterday and fell.",
        "The dog was running yesterday and fell while chasing a cat.",
        "The dog sat in the house.",
        "I gave him bread.",
        "The dog and the cat were running."
    ]
    for source_sentence in source_sentences:
        simple_sentences = split_sentence(source_sentence, model=os.environ['OPENAI_MODEL'])
        print(simple_sentences)
        simple_nl_sentence = '. '.join([make_sentence(sentence) for sentence in simple_sentences]) + '.'

        print(f"Source sentence: {source_sentence}")
        print(f"Simple sentences: {simple_nl_sentence}")
        similarity = semantic_similarity_spacy(source_sentence, simple_nl_sentence)
        print(f"Semantic similarity: {similarity:0.3f}")
        print()

def avg_displacement(truth: np.ndarray, arr: np.ndarray) -> float:
    """Compute the average displacement between two arrays.
    
    Computes the distance of each element to its proper position in the truth array
    and returns the average of these distances.
    """
    return np.mean(np.abs(np.argsort(truth) - np.argsort(arr)))    

def test_similarity():
    sentences = json.loads((thisdir.parent / 'data' / 'semantic_sentences.json').read_text())
    similarity_funcs = {
        "spacy": semantic_similarity_spacy,
        "bert": semantic_similarity_bert,
        "all-MiniLM-L6-v2": functools.partial(semantic_similarity_sentence_transformers, model='all-MiniLM-L6-v2'),
        "paraphrase-MiniLM-L6-v2": functools.partial(semantic_similarity_sentence_transformers, model='paraphrase-MiniLM-L6-v2'),
        # "SFR-Embedding-Mistral": functools.partial(semantic_similarity_sentence_transformers, model='Salesforce/SFR-Embedding-Mistral'),
        "text-embedding-3-large": functools.partial(semantic_similarity_openai, model='text-embedding-3-large'),
        "text-embedding-3-small": functools.partial(semantic_similarity_openai, model='text-embedding-3-small'),
        "text-embedding-ada-002": functools.partial(semantic_similarity_openai, model='text-embedding-ada-002'),
    }
    
    rows = []
    for sentence in sentences:
        base_sentence = sentence['base']
        sentences = sentence['sentences']
        for similarity_func_name, similarity_func in similarity_funcs.items():
            similarities = np.array([similarity_func(base_sentence, s) for s in sentences])
            dist = np.mean(np.abs(np.argsort(-similarities) - np.arange(len(similarities))))

            sorted_sentences = [sentences[i] for i in np.argsort(-similarities)]
            rbo_similarity = RankingSimilarity(sorted_sentences, sentences).rbo()

            rows.append([base_sentence, similarity_func_name, dist, rbo_similarity])

    df = pd.DataFrame(rows, columns=['sentence', 'similarity_func', 'avg_displacement', 'rbo'])
    print(df)

    # compute stats for each similarity function
    stats = df.groupby('similarity_func').agg({'avg_displacement': ['mean', 'std'], 'rbo': ['mean', 'std']})
    stats = stats.sort_values(by=('rbo', 'mean'), ascending=False)
    print(stats.round(3))
    print(stats.to_latex(float_format="%.3f", bold_rows=True, column_format="lcccc"))
        

def test_split_sentence():
    sentence = "The writer is writing a book."
    simple_sentences = split_sentence(sentence, model=os.environ['OPENAI_MODEL'])
    print(simple_sentences)
    simple_nl_sentence = '. '.join([make_sentence(sentence) for sentence in simple_sentences]) + '.'
    print(simple_nl_sentence)

if __name__ == '__main__':
    # main()
    test_similarity()
    test_split_sentence()