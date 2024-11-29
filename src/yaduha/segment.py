"""Functions for segmenting complex sentences into sets of simple SVO or SV sentences."""
import functools
import hashlib
import json
import logging
import os
import pathlib
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import openai
from openai.types.chat import ChatCompletion
import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

cachedir = pathlib.Path.home() / '.cache' / 'yaduha'
cachedir.mkdir(exist_ok=True, parents=True)

def get_openai_client():
    try:
        return openai.Client(api_key=os.environ['OPENAI_API_KEY'])
    except KeyError:
        raise ValueError("OPENAI_API_KEY environment variable not set")

nlp = None
@functools.lru_cache(maxsize=1000)
def semantic_similarity_spacy(sentence1: str, sentence2: str) -> float:
    """Compute the semantic similarity between two sentences using spaCy.

    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.

    Returns:
        float: The semantic similarity between the two sentences.
    """
    global nlp
    import spacy
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_md")
        except OSError:
            from spacy.cli import download
            logging.info("Downloading spaCy model 'en_core_web_md'")
            download('en_core_web_md')
            nlp = spacy.load("en_core_web_md")
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)
    similarity = doc1.similarity(doc2)
    return similarity

tokenizer = None
model = None
@functools.lru_cache(maxsize=1000)
def semantic_similarity_bert(sentence1: str, sentence2: str) -> float:
    global tokenizer, model
    import torch
    from transformers import BertModel, BertTokenizer

    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if model is None:
        model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
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
def get_model(model: str) -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model)

def semantic_similarity_sentence_transformers(sentence1: str, sentence2: str, model: str) -> float:
    from sentence_transformers import util
    embedder = get_model(model)
    emb1 = embedder.encode(sentence1, convert_to_tensor=True)
    emb2 = embedder.encode(sentence2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return (similarity + 1) / 2  # Scale to 0-1 range

def semantic_similarity_sentence_transforms_all_combinations(sentences: List[str], model: str) -> np.ndarray:
    from sentence_transformers import util
    embedder = get_model(model)
    embeddings = embedder.encode(sentences, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(embeddings, embeddings)
    similarities = similarities.cpu().numpy()
    # scale to 0-1 range
    similarities = (similarities + 1) / 2
    return similarities

tf_tokenizer = None
tf_model = None
def semantic_similarity_transformers_all_combinations(sentences: List[str], model: str) -> np.ndarray:
    global tf_tokenizer, tf_model
    from transformers import AutoTokenizer, AutoModel
    import torch
    import torch.nn.functional as F

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Load model from HuggingFace Hub
    if tf_tokenizer is None:
        tf_tokenizer = AutoTokenizer.from_pretrained(model)
    if tf_model is None:
        tf_model = AutoModel.from_pretrained(model)

    # Tokenize sentences
    encoded_input = tf_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = tf_model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    # Compute cosine similarity
    similarities = F.cosine_similarity(sentence_embeddings.unsqueeze(1), sentence_embeddings.unsqueeze(0), dim=2)
    similarities = similarities.cpu().numpy()
    # scale to 0-1 range
    similarities = (similarities + 1) / 2
    return similarities

def semantic_similarity_transformers(sentence1: str, sentence2: str, model: str) -> float:
    # use semantic_similarity_transformers_all_combinations to compute similarity
    similarities = semantic_similarity_transformers_all_combinations([sentence1, sentence2], model)
    return similarities[0, 1]

def _get_openai_embeddings(model: str, *sentences: str) -> Dict[str, np.ndarray]:
    savedir = cachedir / '.results' / 'embeddings' / model
    savedir.mkdir(exist_ok=True, parents=True)
    # load cached embeddings from disk
    embeddings = {}
    for sentence in sentences:
        sentence_id = hashlib.md5(sentence.encode()).hexdigest()
        try:
            with open(savedir / f'{sentence_id}.npy', 'rb') as f:
                embeddings[sentence] = np.load(f)
        except FileNotFoundError:
            pass

    new_sentences = [s for s in sentences if s not in embeddings]
    if new_sentences:
        res = get_openai_client().embeddings.create(
            input=new_sentences,
            model=model,
            encoding_format="float"
        )
        # save embeddings to disk
        for sentence, embedding in zip(new_sentences, res.data):
            emb  = np.array(embedding.embedding)
            sentence_id = hashlib.md5(sentence.encode()).hexdigest()
            with open(savedir / f'{sentence_id}.npy', 'wb') as f:
                np.save(f, emb)
            embeddings[sentence] = emb

    return embeddings

def semantic_similarity_openai(sentence1: str, sentence2: str, model: str) -> float:
    from sentence_transformers import util
    embeddings = _get_openai_embeddings(model, sentence1, sentence2)
    # convert python lists to numpy arrays
    emb1 = np.array(embeddings[sentence1])
    emb2 = np.array(embeddings[sentence2])
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return (similarity + 1) / 2  # Scale to 0-1 range

def semantic_similarity_openai_all_combinations(sentences: List[str], model: str) -> np.ndarray:
    from sentence_transformers import util
    embeddings = _get_openai_embeddings(model, *sentences)
    # convert python lists to numpy arrays
    embeddings = np.array([embeddings[s] for s in sentences])
    similarities = util.pytorch_cos_sim(embeddings, embeddings)
    similarities = similarities.cpu().numpy()
    # scale to 0-1 range
    similarities = (similarities + 1) / 2
    return similarities

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

# @functools.lru_cache(maxsize=1000)
def split_sentence(sentence: str, model: str, res_callback: Optional[Callable[[ChatCompletion], None]] = None) -> List[Dict]:
    """Split a sentence into a set of simple SVO or SV sentences.

    Args:
        sentence (str): The sentence to split.
        res_callback (Optional[Callable[[ChatCompletion], None]]): Callback function to be called with the completion response.

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
            'Subjects and objects can be verbs IF they are "nominalized" as "present" or "future" ',
            '(e.g., "run" -> "the runner", "the one who ran", "the one who will run"). ',
            'The present nominalizer should be used to describe those who always do the action (runner, drinker, cook(er), etc.). ',
            'nominalizer_tense must be either None or one of "present" or "future". ',
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
        {'role': 'user', 'content': 'The one who ran is sitting.'},
        {
            "role": "assistant",
            "content": None,
            "function_call": {
                "arguments": json.dumps({
                    'sentences': [
                        {'subject': 'run', 'subject_nominalizer': 'past', 'verb': 'sit', 'verb_tense': 'present_continuous', 'object': None},
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
        {'role': 'user', 'content': 'The drinker is eating.'},
        {
            "role": "assistant",
            "content": None,
            "function_call": {
                "arguments": json.dumps({
                    'sentences': [
                        {'subject': 'drink', 'subject_nominalizer': 'present', 'verb': 'eat', 'verb_tense': 'present', 'object': None},
                    ]
                }),
                "name": "set_sentences"
            },
        },
        {'role': 'user', 'content': 'The book sits on the table.'},
        {
            "role": "assistant",
            "content": None,
            "function_call": {
                "arguments": json.dumps({
                    'sentences': [
                        {'subject': 'book', 'verb': 'sit', 'verb_tense': 'present', 'object': None},
                    ]
                }),
                "name": "set_sentences"
            },
        },
        {'role': 'user', 'content': 'The boy talked about the girl.'},
        {
            "role": "assistant",
            "content": None,
            "function_call": {
                "arguments": json.dumps({
                    'sentences': [
                        {'subject': 'boy', 'verb': 'talk', 'verb_tense': 'past', 'object': None},
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
        {'role': 'user', 'content': 'That runner down the street will eat the one that fell.'},
        {
            "role": "assistant",
            "content": None,
            "function_call": {
                "arguments": json.dumps({
                    'sentences': [
                        {'subject': 'run', 'subject_nominalizer': 'present', 'verb': 'eat', 'verb_tense': 'future', 'object': 'fall', 'object_nominalizer': 'past'},
                    ]
                }),
                "name": "set_sentences"
            },
        },
        {'role': 'user', 'content': sentence},
    ]
    response = get_openai_client().chat.completions.create(
        model=model,
        messages=messages,
        functions=functions,
        function_call={'name': 'set_sentences'},
        temperature=0.0,
        timeout=10,
    )
    if res_callback:
        res_callback(response)
    response_message = response.choices[0].message
    function_args = json.loads(response_message.function_call.arguments)
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
# @functools.lru_cache(maxsize=1000)
def make_sentence(sentence: Dict, model: str, res_callback: Optional[Callable[[ChatCompletion], None]] = None) -> str:
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
            'content': (
                'You are an assistant takes structured data and generates simple SVO or SV natural language sentence. '
                'Only add add necessary articles and conjugations. '
                'Do not add any other words.'
                'When a subject_nominalizer or object_nominalizer is present, subjects are "nominalized" verbs as "past", "present", or "future" '
                '(e.g., "run" -> "the runner", "the one who ran", "the one who will run"; "drink" -> "the drinker", "the one who drank", "the one who will drink"). '

            )
        },
        {
            'role': 'user',
            'content': "{'subject': 'He', 'verb': '[VERB]', 'verb_tense': 'past', 'object': 'dog'}"
        },
        {
            'role': 'assistant',
            'content': None,
            'function_call': {
                'arguments': json.dumps({'sentence': 'He [VERB]-ed a dog'}),
                'name': 'make_sentence'
            }
        },
        {
            'role': 'user',
            'content': "{'subject': '[SUBJECT]', 'verb': 'drink', 'verb_tense': 'present_continuous'}"
        },
        {
            'role': 'assistant',
            'content': None,
            'function_call': {
                'arguments': json.dumps({'sentence': '[SUBJECT] was drinking'}),
                'name': 'make_sentence'
            }
        },
        {
            'role': 'user',
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
            'content': "{'subject': 'drink', 'subject_nominalizer': 'past', 'verb': 'stand', 'verb_tense': 'present', 'object': None}"
        },
        {
            'role': 'assistant',
            'content': None,
            'function_call': {
                'arguments': json.dumps({'sentence': 'The one who drank stood'}),
                'name': 'make_sentence'
            }
        },
        {
            'role': 'user',
            'content': "{'subject': 'walk', 'subject_nominalizer': 'present', 'verb': 'drink', 'verb_tense': 'past', 'object': None}"
        },
        {
            'role': 'assistant',
            'content': None,
            'function_call': {
                'arguments': json.dumps({'sentence': 'The walker drank'}),
                'name': 'make_sentence'
            }
        },
        {
            'role': 'user',
            'content': json.dumps(sentence)
        }
    ]
    response = get_openai_client().chat.completions.create(
        model=model,
        messages=messages,
        functions=functions,
        function_call={'name': 'make_sentence'},
        temperature=0.0,
        timeout=10,
    )
    if res_callback:
        res_callback(response)
    response_message = response.choices[0].message
    function_args = json.loads(response_message.function_call.arguments)
    return function_args.get('sentence')
