from copy import deepcopy
import logging
import random
from typing import Any, Dict, List, Optional


NOUNS = {
    "isha'": "coyote",
    "isha'pugu": "dog",
    "kidi'": "cat",
    "pugu": "horse",
    "wai": "rice",
    "tüba": "pinenuts",
    "maishibü": "corn",
    "paya": "water",
    "payahuupü": "river",
    "katünu": "chair",
    "toyabi": "mountain",
    "tuunapi": "food",
    "pasohobü": "tree",
    "nobi": "house",
    "toni": "wickiup",
    "apo": "cup",
    "küna": "wood",
    "tübbi": "rock",
    "tabuutsi'": "cottontail",
    "kamü": "jackrabbit",
    "aaponu'": "apple",
    "tüsüga": "weasle",
    "mukita": "lizard",
    "wo'ada": "mosquito",
    "wükada": "bird snake",
    "wo'abi": "worm",
    "aingwü": "squirrel",
    "tsiipa": "bird",
    "tüwoobü": "earth",
    "koopi'": "coffee",
    "pahabichi": "bear",
    "pagwi": "fish",
    "kwadzi": "tail",
}

class Subject:
    SUFFIXES = {
        'ii': 'proximal',
        'uu': 'distal',
    }
    PRONOUNS = {
        "nüü": "I",
        "uhu": "he/she/it",
        "uhuw̃a": "they",
        "mahu": "he/she/it",
        "mahuw̃a": "they",
        "ihi": "this",
        "ihiw̃a": "these",
        "taa": "you and I",
        "nüügwa": "we (exclusive)",
        "taagwa": "we (inclusive)",
        "üü": "you",
        "üügwa": "you (plural)",
    }
    def __init__(self, noun: str, subject_noun_nominalizer: Optional[str], subject_suffix: Optional[str]):
        self.noun = noun
        self.subject_noun_nominalizer = subject_noun_nominalizer
        self.subject_suffix = subject_suffix

        if self.noun in Subject.PRONOUNS and self.subject_suffix is not None:
            raise ValueError("Subject suffix is not allowed with pronouns")
        
        if self.noun in NOUNS and self.subject_noun_nominalizer is not None:
            raise ValueError("Subject noun nominalizer is not allowed with nouns")
        elif self.noun in {*Verb.TRANSITIVE_VERBS.keys(), *Verb.INTRANSITIVE_VERBS.keys()} and self.subject_noun_nominalizer is None:
            raise ValueError("Subject noun nominalizer is required with verbs")
        
        if self.noun not in Subject.PRONOUNS:
            if self.subject_suffix is None:
                raise ValueError("Subject suffix is required with non-pronoun subjects")
        
            if subject_suffix not in self.SUFFIXES:
                raise ValueError(f"Subject suffix must be one of {self.SUFFIXES} (not {subject_suffix})")
        
    def __str__(self) -> str:
        if self.subject_suffix is None:
            return self.noun
        elif self.subject_noun_nominalizer is not None:
            return f"{self.noun}-{self.subject_noun_nominalizer}-{self.subject_suffix}"
        else:
            return f"{self.noun}-{self.subject_suffix}"
        
    @property
    def details(self) -> Dict:
        data = {
            'type': 'subject',
            'text': str(self),
            'parts': []
        }
        if self.noun in Subject.PRONOUNS:
            data['parts'].append({
                'type': 'pronoun',
                'text': self.noun,
                'definition': Subject.PRONOUNS[self.noun]
            })
        elif self.subject_noun_nominalizer is not None:
            data['parts'].append({
                'type': 'noun',
                'text': self.noun,
                'definition': {**Verb.TRANSITIVE_VERBS, **Verb.INTRANSITIVE_VERBS}.get(self.noun, f"[{self.noun}]")
            })
            data['parts'].append({
                'type': 'nominalizer',
                'text': self.subject_noun_nominalizer,
                'definition': Verb.NOMINALIZER_TENSES[self.subject_noun_nominalizer]
            })
            data['parts'].append({
                'type': 'subject_suffix',
                'text': self.subject_suffix,
                'definition': self.SUFFIXES[self.subject_suffix]
            })
        else:
            data['parts'].append({
                'type': 'noun',
                'text': self.noun,
                'definition': NOUNS.get(self.noun, f"[{self.noun}]")
            })
            data['parts'].append({
                'type': 'subject_suffix',
                'text': self.subject_suffix,
                'definition': self.SUFFIXES[self.subject_suffix]
            })
            
        return data


LENIS_MAP = {
    'p': 'b',
    't': 'd',
    'k': 'g',
    's': 'z',
    'm': 'w̃'
}       
def to_lenis(word: str) -> str:
    """Convert a word to its lenis form
    """
    first_letter = word[0]
    if first_letter in LENIS_MAP:
        return LENIS_MAP[first_letter] + word[1:]
    else:
        return word

class Verb:
    TENSES = {
        'ku': 'completive (past)',
        'ti': 'present ongoing (-ing)',
        'dü': 'present',
        'wei': 'future (will)',
        'gaa-wei': 'future (going to)',
        'pü': 'have x-ed, am x-ed',
    }
    NOMINALIZER_TENSES = {
        'dü': 'present',
        'weidü': 'future (will)',
    }
    TRANSITIVE_VERBS = {
        'tüka': 'eat',
        'puni': 'see',
        'hibi': 'drink',
        'naka': 'hear',
        'kwana': 'smell',
        'kwati': 'hit',
        'yadohi': 'talk to',
        'naki': 'chase',
        'tsibui': 'climb',
        'sawa': 'cook',
        'tama\'i': 'find',
        'nia': 'read',
        'mui': 'write',
        'nobini': 'visit',
    }
    INTRANSITIVE_VERBS = {
        'katü': 'sit',
        'üwi': 'sleep',
        'kwisha\'i': 'sneeze',
        'poyoha': 'run',
        'mia': 'go',
        'hukaw̃ia': 'walk',
        'wünü': 'stand',
        'habi': 'lie down',
        'yadoha': 'talk',
        'kwatsa\'i': 'fall',
        'waakü': 'work',
        'wükihaa': 'smile',
        'hubiadu': 'sing',
        'nishua\'i': 'laugh',
        'tsibui': 'climb',
        'tübinohi': 'play',
        'yotsi': 'fly',
        'nüga': 'dance',
        'pahabi': 'swim',
        'tünia': 'read',
        'tümui': 'write',
        'tsiipe\'i': 'chirp',
    }
    def __init__(self, 
                 verb_stem: str, 
                 tense_suffix: str, 
                 object_pronoun_prefix: Optional[str]):
        self.verb_stem = verb_stem
        self.tense_suffix = tense_suffix
        self.object_pronoun_prefix = object_pronoun_prefix

        if tense_suffix not in self.TENSES:
            raise ValueError(f"Tense must be one of {self.TENSES} (not {tense_suffix})")
        
        if self.is_intransitive:
            if self.object_pronoun_prefix is not None:
                raise ValueError("Intransitive verbs cannot have object pronouns")
        
    def __str__(self) -> str:
        if self.object_pronoun_prefix is None:
            return f"{self.verb_stem}-{self.tense_suffix}"
        else:
            verb_stem = to_lenis(self.verb_stem)
            return f"{self.object_pronoun_prefix}-{verb_stem}-{self.tense_suffix}"
        
    @classmethod
    def _is_transitive(cls, verb_stem: str) -> bool:
        return not cls._is_intransitive(verb_stem) # by default, all verbs are considered transitive
    
    @classmethod
    def _is_intransitive(cls, verb_stem: str) -> bool:
        return verb_stem in cls.INTRANSITIVE_VERBS and verb_stem not in cls.TRANSITIVE_VERBS

    @property
    def is_transitive(self) -> bool:
        return Verb._is_transitive(self.verb_stem)
    
    @property
    def is_intransitive(self) -> bool:
        return Verb._is_intransitive(self.verb_stem)
    
    @property
    def details(self) -> Dict:
        data = {
            'type': 'verb',
            'text': str(self),
            'parts': []
        }
        if self.object_pronoun_prefix is not None:
            data['parts'].append({
                'type': 'object_pronoun',
                'text': self.object_pronoun_prefix,
                'definition': Object.PRONOUNS[self.object_pronoun_prefix]
            })
        data['parts'].append({
            'type': 'verb_stem',
            'text': self.verb_stem,
            'definition': (
                Verb.TRANSITIVE_VERBS.get(self.verb_stem, f'[{self.verb_stem}]')
                if self.is_transitive else
                Verb.INTRANSITIVE_VERBS.get(self.verb_stem, f'[{self.verb_stem}]')
            )
        })
        data['parts'].append({
            'type': 'tense',
            'text': self.tense_suffix,
            'definition': self.TENSES[self.tense_suffix]
        })
        return data


class Object:
    SUFFIXES = {
        'eika': 'proximal',
        'oka': 'distal',
    }
    PRONOUNS = {
        'i': 'me',
        'u': 'him/her/it (distal)',
        'ui': 'them (distal)',
        'ma': 'him/her/it (proximal)',
        'mai': 'them (proximal)',
        'a': 'him/her/it (proximal)',
        'ai': 'them (proximal)',
        'ni': 'us (plural, exclusive)',
        'tei': 'us (plural, inclusive)',
        'ta': 'us (dual), you and I',
        'ü': 'you (singular)',
        'üi': 'you (plural), you all',
    }
    def __init__(self, noun: str, object_noun_nominalizer: Optional[str], object_suffix: Optional[str]):
        self.noun = noun
        self.object_noun_nominalizer = object_noun_nominalizer
        self.object_suffix = object_suffix

        if self.object_suffix is None:
            raise ValueError("Object suffix is required")
        elif self.object_suffix not in self.SUFFIXES:
            raise ValueError(f"Object suffix must be one of {self.SUFFIXES} (not {object_suffix})")
        
        if self.noun in NOUNS and self.object_noun_nominalizer is not None:
            raise ValueError("Object noun nominalizer is not allowed with nouns")
        elif self.noun in {*Verb.TRANSITIVE_VERBS.keys(), *Verb.INTRANSITIVE_VERBS.keys()} and self.object_noun_nominalizer is None:
            raise ValueError("Object noun nominalizer is required with verbs")
        
    def __str__(self) -> str:
        object_suffix = self.object_suffix
        if self.object_noun_nominalizer is None:
            if "'" not in self.noun[-2:]: # noun does not end in glottal stop
                if object_suffix == 'eika':
                    object_suffix = 'neika'
                elif object_suffix == 'oka':
                    object_suffix = 'noka'
        else:
            if object_suffix == 'eika':
                object_suffix = f'{self.object_noun_nominalizer[:-1]}eika'
            elif object_suffix == 'oka':
                object_suffix = f'{self.object_noun_nominalizer[:-1]}oka'
        return f"{self.noun}-{object_suffix}"
    
    @classmethod
    def check_agreement(self, object_suffix: str, object_pronoun: str) -> bool:
        """Check whether the object suffix and object pronoun agree
        
        Either both must contain 'proximal', both contain 'distal', or neither contain either.
        """
        if 'proximal' in object_suffix and 'proximal' in object_pronoun:
            return True
        elif 'distal' in object_suffix and 'distal' in object_pronoun:
            return True
        elif 'proximal' not in object_suffix and 'proximal' not in object_pronoun and 'distal' not in object_suffix and 'distal' not in object_pronoun:
            return True
        else:
            return False
        
    @classmethod
    def get_matching_suffix(self, object_pronoun: str) -> str:
        """Get the object suffix that matches the object pronoun
        
        If the object pronoun is proximal, return the proximal suffix.
        If the object pronoun is distal, return the distal suffix.
        """
        if object_pronoun is None:
            return None
        elif 'proximal' in Object.PRONOUNS[object_pronoun]:
            return 'eika'
        elif 'distal' in Object.PRONOUNS[object_pronoun]:
            return 'oka'
        else:
            return None
        
    @classmethod
    def get_matching_third_person_pronouns(self, object_suffix: str) -> List[str]:
        """Get the object pronouns that match the object suffix
        
        If the object suffix is proximal, return the proximal pronouns.
        If the object suffix is distal, return the distal pronouns.
        """
        proximal_pronouns = [pronoun for pronoun in self.PRONOUNS if 'proximal' in self.PRONOUNS[pronoun]]
        distal_pronouns = [pronoun for pronoun in self.PRONOUNS if 'distal' in self.PRONOUNS[pronoun]]
        third_person_pronouns = [*proximal_pronouns, *distal_pronouns]
        if not object_suffix:
            return third_person_pronouns
        elif object_suffix == 'eika':
            return proximal_pronouns
        elif object_suffix == 'oka':
            return distal_pronouns
        else:
            raise ValueError(f"Object suffix must be one of {self.SUFFIXES}")
        
    @property
    def details(self) -> Dict:
        data = {
            'type': 'object',
            'text': str(self),
            'parts': []
        }
        if self.object_noun_nominalizer is None:
            data['parts'].append({
                'type': 'noun',
                'text': self.noun,
                'definition': NOUNS.get(self.noun, f"[{self.noun}]")
            })
        else:
            data['parts'].append({
                'type': 'noun',
                'text': self.noun,
                'definition': {**Verb.TRANSITIVE_VERBS, **Verb.INTRANSITIVE_VERBS}.get(self.noun, f"[{self.noun}]")
            })
            data['parts'].append({
                'type': 'nominalizer',
                'text': self.object_noun_nominalizer,
                'definition': Verb.NOMINALIZER_TENSES[self.object_noun_nominalizer]
            })
        data['parts'].append({
            'type': 'object_suffix',
            'text': self.object_suffix,
            'definition': self.SUFFIXES[self.object_suffix]
        })
        return data

def get_all_choices(subject_noun: Optional[str] = None,
                    subject_noun_nominalizer: Optional[str] = None,
                    subject_suffix: Optional[str] = None,
                    verb: Optional[str] = None,
                    verb_tense: Optional[str] = None,
                    object_pronoun: Optional[str] = None,
                    object_noun: Optional[str] = None,
                    object_noun_nominalizer: Optional[str] = None,
                    object_suffix: Optional[str] = None) -> Dict[str, Any]:
    is_verb_wild = verb is not None and verb.startswith('[') and verb.endswith(']')
    is_subject_noun_wild = subject_noun is not None and subject_noun.startswith('[') and subject_noun.endswith(']')
    is_object_noun_wild = object_noun is not None and object_noun.startswith('[') and object_noun.endswith(']')

    choices = {}
    # Validate inputs
    if subject_noun not in {*Subject.PRONOUNS.keys(), *NOUNS.keys(), *Verb.TRANSITIVE_VERBS.keys(), *Verb.INTRANSITIVE_VERBS.keys()} and not is_subject_noun_wild:
        subject_noun = None

    if subject_noun_nominalizer not in Verb.NOMINALIZER_TENSES.keys():
        subject_noun_nominalizer = None

    if subject_suffix not in Subject.SUFFIXES.keys():
        subject_suffix = None

    if verb not in [*Verb.TRANSITIVE_VERBS.keys(), *Verb.INTRANSITIVE_VERBS.keys()] and not is_verb_wild:
        verb = None

    if verb_tense not in Verb.TENSES.keys():
        verb_tense = None

    if object_pronoun not in Object.PRONOUNS.keys():
        object_pronoun = None

    if object_noun not in [*NOUNS.keys(), *Verb.TRANSITIVE_VERBS.keys(), *Verb.INTRANSITIVE_VERBS.keys()] and not is_object_noun_wild:
        object_noun = None

    if object_suffix not in Object.SUFFIXES.keys():
        object_suffix = None

    if object_noun_nominalizer not in Verb.NOMINALIZER_TENSES.keys():
        object_noun_nominalizer = None

    # Check object_pronoun and object_suffix match
    # if mismatch, set to None (will be corrected below)
    if object_pronoun is not None and object_suffix is not None:
        if object_pronoun not in Object.get_matching_third_person_pronouns(object_suffix):
            object_suffix = None

    # Subject
    choices['subject_noun'] = {
        'choices': {
            **Subject.PRONOUNS,
            **NOUNS,
            **Verb.TRANSITIVE_VERBS,
            **Verb.INTRANSITIVE_VERBS
        },
        'value': subject_noun,
        'requirement': "required"
    }
    if subject_noun is None or subject_noun in Subject.PRONOUNS:
        choices['subject_suffix'] = {
            'choices': {},
            'value': None,
            'requirement': "disabled"
        }
        subject_suffix = None
    else:
        choices['subject_suffix'] = {
            'choices': Subject.SUFFIXES,
            'value': subject_suffix,
            'requirement': "required"
        }

    if subject_noun in [*Verb.TRANSITIVE_VERBS.keys(), *Verb.INTRANSITIVE_VERBS.keys()]:
        choices['subject_noun_nominalizer'] = {
            'choices': Verb.NOMINALIZER_TENSES,
            'value': subject_noun_nominalizer,
            'requirement': "required"
        }
    else:
        choices['subject_noun_nominalizer'] = {
            'choices': {},
            'value': None,
            'requirement': "disabled"
        }
        subject_noun_nominalizer = None

    # Verb
    if object_noun is not None: # verb must be transitive
        if not Verb._is_transitive(verb):
            verb = None
        choices['verb'] = {
            'choices': Verb.TRANSITIVE_VERBS,
            'value': verb if Verb._is_transitive(verb) else None,
            'requirement': "required"
        }
    else:
        choices['verb'] = {
            'choices': {
                **Verb.TRANSITIVE_VERBS,
                **Verb.INTRANSITIVE_VERBS
            },
            'value': verb,
            'requirement': "required"
        }

    # Verb tense
    if verb is None:
        choices['verb_tense'] = {
            'choices': {},
            'value': None,
            'requirement': "disabled"
        }
        verb_tense = None
    else:
        choices['verb_tense'] = {
            'choices': Verb.TENSES,
            'value': verb_tense,
            'requirement': "required"
        }

    # Object pronoun
    if verb is None or Verb._is_intransitive(verb): 
        choices['object_pronoun'] = {
            'choices': {},
            'value': None,
            'requirement': "disabled"
        }
        object_pronoun = None
    elif object_noun is not None: # object pronoun must match object suffix
        choices['object_pronoun'] = {
            'choices': {
                p: Object.PRONOUNS[p] for p in Object.get_matching_third_person_pronouns(object_suffix)
            },
            'value': object_pronoun,
            'requirement': "required"
        }
    else:
        choices['object_pronoun'] = {
            'choices': Object.PRONOUNS,
            'value': object_pronoun,
            'requirement': "optional"
        }


    # Object noun
    if (verb is not None and Verb._is_intransitive(verb)) or object_pronoun not in [None, *Object.get_matching_third_person_pronouns(None)]:
        choices['object_noun'] = {
            'choices': {},
            'value': None,
            'requirement': "disabled"
        }
        object_noun = None
    else: # verb is not selected or is transitive
        choices['object_noun'] = {
            'choices': {
                **NOUNS,
                **Verb.TRANSITIVE_VERBS,
                **Verb.INTRANSITIVE_VERBS
            },
            'value': object_noun,
            'requirement': "optional"
        }
    
    # if object_noun is None and it's either a transitive or intransitive verb, require an object_noun_nominalizer
    if object_noun in [*Verb.TRANSITIVE_VERBS.keys(), *Verb.INTRANSITIVE_VERBS.keys()]:
        choices['object_noun_nominalizer'] = {
            'choices': Verb.NOMINALIZER_TENSES,
            'value': object_noun_nominalizer,
            'requirement': "required"
        }
    else:
        choices['object_noun_nominalizer'] = {
            'choices': {},
            'value': None,
            'requirement': "disabled"
        }
        object_noun_nominalizer = None

    # Object suffix
    if object_noun is None:
        choices['object_suffix'] = {
            'choices': {},
            'value': None,
            'requirement': "disabled"
        }
        object_suffix = None
    elif object_pronoun is not None:
        matching_suffix = Object.get_matching_suffix(object_pronoun)
        choices['object_suffix'] = {
            'choices': {} if matching_suffix is None else {matching_suffix: Object.SUFFIXES[matching_suffix]},
            'value': None if object_suffix != Object.get_matching_suffix(object_pronoun) else object_suffix,
            'requirement': "required"
        }
    else:
        choices['object_suffix'] = {
            'choices': Object.SUFFIXES,
            'value': object_suffix,
            'requirement': "required"
        }

    return choices

def format_sentence(subject_noun: Optional[str],
                    subject_noun_nominalizer: Optional[str],
                    subject_suffix: Optional[str],
                    verb: Optional[str],
                    verb_tense: Optional[str],
                    object_pronoun: Optional[str],
                    object_noun: Optional[str],
                    object_noun_nominalizer: Optional[str],
                    object_suffix: Optional[str]) -> List[Dict]:
    subject = Subject(subject_noun, subject_noun_nominalizer, subject_suffix)
    _verb = Verb(verb, verb_tense, object_pronoun)

    # check object_pronoun and object_suffix match
    if object_suffix is not None:
        if object_pronoun not in Object.get_matching_third_person_pronouns(object_suffix):
            raise ValueError("Object pronoun and suffix do not match")

    object = None
    try:
        object = Object(object_noun, object_noun_nominalizer, object_suffix)
    except ValueError as e: # could not create object
        if object_noun is not None:
            raise e
        else: # okay, since object is optional
            pass
    
    if subject.noun in Subject.PRONOUNS:
        if object:
            return [object.details, subject.details, _verb.details]
        else:
            return [_verb.details, subject.details]
    else:
        if object:
            return [subject.details, object.details, _verb.details]
        else:
             return [subject.details, _verb.details]

def get_random_sentence(choices: Dict[str, Dict[str, Any]] = {}):
    choices = get_all_choices(**{k: v['value'] for k, v in choices.items()})
    all_pos = list(choices.keys())
    i = 0
    while True:
        random.shuffle(all_pos)
        for pos in all_pos:
            if not choices[pos]['choices'] or choices[pos]['value'] is not None:
                continue
            choices[pos]['value'], _ = random.choice(list(choices[pos]['choices'].items()))
            choices = get_all_choices(**{k: v['value'] for k, v in choices.items()})
            i = 0

        try:
            format_sentence(**{k: v['value'] for k, v in choices.items()})
            return choices
        except ValueError as e:
            i += 1
            if i > 20:
                raise e
            continue

def get_random_simple_sentence(choices: Dict[str, Dict[str, Any]] = {}):
    choices = deepcopy(choices)
    choices = get_all_choices(**{k: v['value'] for k, v in choices.items()})
    word_choices = {pos: v.get('value') for pos, v in choices.items()}
    choices['subject_noun']['value'] = word_choices.get('subject_noun') or random.choice(list(NOUNS.keys()))
    choices['verb']['value'] = word_choices.get('verb') or random.choice([*Verb.TRANSITIVE_VERBS.keys(), *Verb.INTRANSITIVE_VERBS.keys()])
    if Verb._is_transitive(choices['verb']['value']):
        choices['object_noun']['value'] = word_choices.get('object_noun') or random.choice(list(NOUNS.keys()))
    choices = get_all_choices(**{k: v['value'] for k, v in choices.items()})
    return get_random_sentence(choices)


def get_random_sentence_big():
    subject_noun = random.choice([*Verb.TRANSITIVE_VERBS.keys(), *Verb.INTRANSITIVE_VERBS.keys()])
    subject_noun_nominalizer = random.choice(list(Verb.NOMINALIZER_TENSES.keys()))
    subject_suffix = random.choice(list(Subject.SUFFIXES.keys()))
    verb = random.choice(list(Verb.TRANSITIVE_VERBS.keys()))
    verb_tense = random.choice(list(Verb.TENSES.keys()))
    object_pronoun = random.choice(list(Object.get_matching_third_person_pronouns(None)))
    object_noun = random.choice([*Verb.TRANSITIVE_VERBS.keys(), *Verb.INTRANSITIVE_VERBS.keys()])
    object_noun_nominalizer = random.choice(list(Verb.NOMINALIZER_TENSES.keys()))
    object_suffix = Object.get_matching_suffix(object_pronoun)

    choices = get_all_choices(
        subject_noun=subject_noun,
        subject_noun_nominalizer=subject_noun_nominalizer,
        subject_suffix=subject_suffix,
        verb=verb,
        verb_tense=verb_tense,
        object_pronoun=object_pronoun,
        object_noun=object_noun,
        object_noun_nominalizer=object_noun_nominalizer,
        object_suffix=object_suffix
    )

    return choices

def sentence_to_str(sentence: List[Dict]):
    text = ""
    for word in sentence:
        text += word['text'] + " "
    return text

def print_sentence(sentence: List[Dict]):
    print(sentence_to_str(sentence))


    