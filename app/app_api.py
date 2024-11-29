"""API routes for the app"""
import logging
import os
import traceback
from typing import Dict

from yaduha.forward.pipeline import PipelineTranslator, translate_ovp_to_english
from yaduha.sentence_builder import NOUNS, Object, Subject, Verb, get_all_choices, format_sentence, get_random_simple_sentence
from yaduha.segment import semantic_similarity_sentence_transformers as plot_semantic_similarity

from flask import jsonify, make_response, request, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from app_base import app
from app_oauth import get_app_metadata

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=os.environ.get('REDIS_URL'),
    storage_options={"socket_connect_timeout": 30},
    strategy="fixed-window",
    default_limits=["10/second"], # to prevent abuse
)
LIMITS = {
    "translate": "1/second;30/minute;200/month",
}

TRANSLATION_QUALITY_THRESHOLD = 0.8

# API Routes
@app.errorhandler(429)
def ratelimit_handler(exp):
    message = 'Add your OpenAI API key in your account settings for unlimited access'
    return make_response(jsonify({"rate_limit_message": f"{message}."}), 429)

def get_restricted_choices(data: Dict) -> Dict:
    choices = get_all_choices(
        subject_noun=data.get('subject_noun') or None,
        subject_noun_nominalizer=None,
        subject_suffix=data.get('subject_suffix') or None,
        verb=data.get('verb') or None,
        verb_tense=data.get('verb_tense') or None,
        object_pronoun=data.get('object_pronoun') or None,
        object_noun=data.get('object_noun') or None,
        object_noun_nominalizer=None,
        object_suffix=data.get('object_suffix') or None,
    )
    
    # remove TRANSITIVE_VERB and INTRANSITIVE_VERB choices from subject_noun and object_noun
    verbs = {*Verb.TRANSITIVE_VERBS, *Verb.INTRANSITIVE_VERBS}
    subject_nouns = {*NOUNS, *Subject.PRONOUNS}
    object_nouns = {*NOUNS, *Object.PRONOUNS}

    choices['subject_noun']['choices'] = {
        c: v
        for c, v in choices['subject_noun']['choices'].items()
        if c is None or (c in subject_nouns and c not in verbs)
    }
    choices['object_noun']['choices'] = {
        c: v
        for c, v in choices['object_noun']['choices'].items()
        if c is None or (c in object_nouns and c not in verbs)
    }

    sentence = []
    try:
        sentence = format_sentence(**{k: v['value'] for k, v in choices.items()})
    except Exception as e:
        pass

    return choices, sentence

def format_choices(choices: Dict) -> Dict:
    return {
        k: {
            'choices': sorted(list(v['choices'].items())),
            'value': v['value'],
            'requirement': v['requirement'],
        } for k, v in choices.items()
    }

@app.route('/api/builder/choices', methods=['POST'])
def get_choices():
    data: Dict = request.get_json()
    choices, sentence = get_restricted_choices(data)
    return jsonify(choices=format_choices(choices), sentence=sentence)

@app.route('/api/builder/sentence', methods=['POST'])
def build_sentence():
    data: Dict = request.get_json()
    try:
        sentence = format_sentence(
            subject_noun=data.get('subject_noun') or None,
            subject_suffix=data.get('subject_suffix') or None,
            verb=data.get('verb') or None,
            verb_tense=data.get('verb_tense') or None,
            object_pronoun=data.get('object_pronoun') or None,
            object_noun=data.get('object_noun') or None,
            object_suffix=data.get('object_suffix') or None,
        )
        return jsonify(sentence=sentence)
    except Exception as e:
        return jsonify(sentence=[], error=str(e)), 400

@app.route('/api/builder/translate', methods=['POST'])
@limiter.limit(LIMITS['translate'], key_func = lambda: session.get('profile', {}).get('sub', 'anonymous'))
def get_translation():
    data: Dict = request.get_json()
    try:
        translation = translate_ovp_to_english(
            subject_noun=data.get('subject_noun') or None,
            subject_noun_nominalizer=None,
            subject_suffix=data.get('subject_suffix') or None,
            verb=data.get('verb') or None,
            verb_tense=data.get('verb_tense') or None,
            object_pronoun=data.get('object_pronoun') or None,
            object_noun=data.get('object_noun') or None,
            object_noun_nominalizer=None,
            object_suffix=data.get('object_suffix') or None,
            model='gpt-3.5-turbo'
        )
        return jsonify(translation=translation)
    except Exception as e:
        return jsonify(sentence=[], error=str(e)), 400
    

# route to get random sentence
@app.route('/api/builder/random', methods=['POST'])
def get_random():
    data: Dict = request.get_json() 
    try:
        choices, sentence = get_restricted_choices(data)
        choices = get_random_simple_sentence(choices)
        sentence = format_sentence(**{k: v['value'] for k, v in choices.items()})
        return jsonify(choices=format_choices(choices), sentence=sentence)
    except Exception as e:
        traceback.print_exc()
        return jsonify(sentence=[], error=str(e)), 400
    
@app.route('/api/translator/translate', methods=['POST'])
@limiter.limit(LIMITS['translate'], key_func = lambda: session.get('profile', {}).get('sub', 'anonymous'))
def translate_sentence():
    data: Dict = request.get_json()
    try:
        translator = PipelineTranslator(model='gpt-4o-mini')
        translation = translator.translate(data.get('english'))
        response = {
            'simple': translation.simple,
            'comparator': translation.comparator,
            'target': translation.target,
            'backwards': translation.back_translation,
            'sim_simple': plot_semantic_similarity(
                translation.source, translation.simple, 
                model='all-MiniLM-L6-v2'
            ),
            'sim_backwards': plot_semantic_similarity(
                translation.source, translation.back_translation, 
                model='all-MiniLM-L6-v2'
            ),
            'sim_comparator': plot_semantic_similarity(
                translation.source, translation.comparator, 
                model='all-MiniLM-L6-v2'
            )
        }
        logging.info(response)
        if response['sim_simple'] < TRANSLATION_QUALITY_THRESHOLD:
            response['warning'] = (
                'The input sentence is complex, so alot of meaning may have been lost in breaking ' +
                'it down into simple sentences.'
            )
        elif response['sim_backwards'] < TRANSLATION_QUALITY_THRESHOLD:
            response['warning'] = 'The translation doesn\'t seem to be very accurate.'
        elif response['sim_comparator'] < TRANSLATION_QUALITY_THRESHOLD:
            response['warning'] = (
                'The translator doesn\'t know some of the words in your input sentence. ' +
                'It left the english words as placeholders and gave you the best translation it could.'
            )
        elif all([response[k] >= TRANSLATION_QUALITY_THRESHOLD for k in ['sim_simple', 'sim_backwards', 'sim_comparator']]):
            response['message'] = 'The translation is probably pretty good!'
        return jsonify(
            english=response['backwards'],
            paiute=response['target'],
            message=response.get('message', ''),
            warning=response.get('warning', '')
        )
    except Exception as e:
        return jsonify(error=str(e)), 400

# health check
@app.route('/api/healthz', methods=['GET'])
def health_check():
    return jsonify(status='ok')


if __name__ == '__main__':
    app.run(debug=True)