"""Functions for segmenting complex sentences into sets of simple SVO or SV sentences."""
import functools
import json
import os
import pathlib
import pprint
from typing import Dict, List

import dotenv
import numpy as np
import openai
import pandas as pd
import spacy
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import BertModel, BertTokenizer
import numpy as np

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()


openai.api_key = os.getenv('OPENAI_API_KEY')
MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')

nlp = spacy.load("en_core_web_md")
@functools.lru_cache(maxsize=1000)
def semantic_similarity_spacy(sentence1: str, sentence2: str) -> float:
    """Compute the semantic similarity between two sentences using spaCy.

    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.

    Returns:
        float: The semantic similarity between the two sentences.
    """
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)
    similarity = doc1.similarity(doc2)
    return similarity

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
@functools.lru_cache(maxsize=1000)
def semantic_similarity_bert(sentence1: str, sentence2: str) -> float:
    with torch.no_grad():
        inputs1 = tokenizer(sentence1, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs1 = model(**inputs1)
        inputs2 = tokenizer(sentence2, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs2 = model(**inputs2)
        
        # Use the average of the last hidden states as sentence embeddings
        emb1 = outputs1.last_hidden_state.mean(dim=1)
        emb2 = outputs2.last_hidden_state.mean(dim=1)
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()
        return (similarity + 1) / 2  # Scale to 0-1 range

@functools.lru_cache(maxsize=1000)
def get_model(model: str) -> SentenceTransformer:
    return SentenceTransformer(model)

def semantic_similarity_sentence_transformers(sentence1: str, sentence2: str, model: str) -> float:
    embedder = get_model(model)
    emb1 = embedder.encode(sentence1, convert_to_tensor=True)
    emb2 = embedder.encode(sentence2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return (similarity + 1) / 2  # Scale to 0-1 range


sentence_schema = {
  "type": "object",
  "properties": {
    "subject": {
      "type": "string",
      "description": "The subject of the sentence. Must be a single word and singular (not plural)."
    },
    "verb": {
      "type": "string",
      "description": "The present-tense verb of the sentence. Must be a single word (infinitive without 'to')."
    },
    "verb_tense": {
        "type": "string",
        "description": "The tense of the verb. Must be one of: past, present, future.",
        "enum": ["past", "present", "future", "past_continuous", "present_continuous"]
    },
    "object": {
      "type": "string",
      "description": "The object of the sentence (optional). Must be a single word and singular (not plural)."
    }
  },
  "required": ["subject", "verb", "verb_tense"]
}

@functools.lru_cache(maxsize=1000)
def split_sentence(sentence: str) -> List[Dict[str, str]]:
    """Split a sentence into a set of simple SVO or SV sentences.

    Args:
        sentence (str): The sentence to split.

    Returns:
        list: A list of simple sentences.
    """
    functions = [
        {
            'name': 'set_sentences',
            'description': 'Set the simple sentences.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'sentences': sentence_schema
                },
                'required': ['sentences']
            }
        }
    ]

    messages = [
        {'role': 'system', 'content': "".join([
            'You are an assistant that splits user input sentences into a set of simple SVO or SV sentences. ',
            'The set of simple sentences should be as semantically equivalent as possible to the user input sentence. ',
            'No adjectives, adverbs, prepositions, or conjunctions should be added to the simple sentences. ',
            'Indirect objects and objects of prepositions should not be included in the simple sentences. ',
        ])},
        {'role': 'user', 'content': 'I am sitting in a chair.'},
        {
            "role": "assistant",
            "content": None,
            "function_call": {
                "arguments": json.dumps({
                    'sentences': [
                        {'subject': 'I', 'verb': 'sit', 'verb_tense': 'present_continuous', 'object': None},
                    ]
                }),
                "name": "set_sentences"
            },
        },
        {'role': 'user', 'content': 'The dogs were chasing their tails.'},
        {
            "role": "assistant",
            "content": None,
            "function_call": {
                "arguments": json.dumps({
                    'sentences': [
                        {'subject': 'dog', 'verb': 'chase', 'verb_tense': 'past_continuous', 'object': 'tail'},
                    ]
                }),
                "name": "set_sentences"
            },
        },
        {'role': 'user', 'content': 'I saw two men walking their dogs yesterday at Starbucks while drinking a cup of coffee'},
        {
            "role": "assistant",
            "content": None,
            "function_call": {
                "arguments": json.dumps({
                    'sentences': [
                        {'subject': 'I', 'verb': 'see', 'verb_tense': 'past', 'object': 'man'},
                        {'subject': 'man', 'verb': 'walk', 'verb_tense': 'past_continuous', 'object': 'dog'},
                        {'subject': 'man', 'verb': 'drink', 'verb_tense': 'past_continuous', 'object': 'coffee'}
                    ]
                }),
                "name": "set_sentences"
            },
        },
        {'role': 'user', 'content': sentence},
    ]
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        functions=functions,
        function_call={'name': 'set_sentences'},
        temperature=0.0,
        request_timeout=10,
    )
    response_message = response["choices"][0]["message"]
    function_args = json.loads(response_message["function_call"]["arguments"])
    return function_args.get('sentences')

def hash_dict(func):
    """Transform mutable dictionnary
    Into immutable
    Useful to be compatible with cache
    """
    class HDict(dict):
        def __hash__(self):
            return hash(frozenset(self.items()))

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple([HDict(arg) if isinstance(arg, dict) else arg for arg in args])
        kwargs = {k: HDict(v) if isinstance(v, dict) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapped

@hash_dict
@functools.lru_cache(maxsize=1000)
def make_sentence(sentence: Dict) -> str:
    """Generate a simple SVO or SV sentence from a schema.

    Args:
        sentence (dict): The sentence schema.

    Returns:
        str: The generated sentence.
    """
    functions = [
        {
            'name': 'make_sentence',
            'description': 'Write a simple natural language sentence.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'sentence': {'type': 'string'}
                },
                'required': ['sentence']
            }
        }
    ]
    messages = [
        {
            'role': 'system',
            'content': 'You are an assistant takes structured data and generates simple SVO or SV natural language sentence. Only add add necessary articles and conjugations. Do not add any other words.'
        },
        {
            'role': 'system',
            'content': "{'subject': 'I', 'verb': 'see', 'verb_tense': 'past', 'object': 'man'}"
        },
        {
            'role': 'assistant',
            'content': None,
            'function_call': {
                'arguments': json.dumps({'sentence': 'I saw a man'}),
                'name': 'make_sentence'
            }
        },
        {
            'role': 'user',
            'content': json.dumps(sentence)
        }
    ]
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        functions=functions,
        function_call={'name': 'make_sentence'},
        temperature=0.0,
        request_timeout=10,
    )
    response_message = response["choices"][0]["message"]
    function_args = json.loads(response_message["function_call"]["arguments"])
    return function_args.get('sentence')


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
        simple_sentences = split_sentence(source_sentence)
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
    sentences = json.loads((thisdir / '.data' / 'semantic_sentences.json').read_text())
    similarity_funcs = {
        "spacy": semantic_similarity_spacy,
        "bert": semantic_similarity_bert,
        "all-MiniLM-L6-v2": functools.partial(semantic_similarity_sentence_transformers, model='all-MiniLM-L6-v2'),
        "paraphrase-MiniLM-L6-v2": functools.partial(semantic_similarity_sentence_transformers, model='paraphrase-MiniLM-L6-v2'),
    }
    
    rows = []
    for sentence in sentences:
        base_sentence = sentence['base']
        sentences = sentence['sentences']
        for similarity_func_name, similarity_func in similarity_funcs.items():
            similarities = np.array([similarity_func(base_sentence, s) for s in sentences])
            dist = np.mean(np.abs(np.argsort(-similarities) - np.arange(len(similarities))))
            rows.append([base_sentence, similarity_func_name, dist])

    df = pd.DataFrame(rows, columns=['sentence', 'similarity_func', 'avg_displacement'])
    print(df)

    # compute stats for each similarity function
    stats = df.groupby('similarity_func')['avg_displacement'].agg(['mean', 'std'])
    print(stats)
        

if __name__ == '__main__':
    # main()
    test_similarity()
