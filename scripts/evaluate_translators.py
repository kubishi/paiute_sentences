import json
import traceback
from typing import Dict, List, Set, Tuple, Type
from yaduha.base import Translation, Translator
from yaduha.forward import (
    PipelineTranslator,
    InstructionsTranslator,
    AgenticTranslator
)
import pandas as pd
import pathlib
from itertools import product

from pydantic import BaseModel
import dotenv
import logging

from yaduha.forward.finetuned import FinetunedTranslator

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.resolve()

# logging.basicConfig(level=logging.INFO)

class Result(BaseModel):
    translator: str
    model: str
    sentence_type: str
    translation: Translation

class Results(BaseModel):
    results: List[Result]

def main():
    sentences = [
        (sentence_type, sentence)
        for _, sentence, sentence_type in
        pd.read_csv(thisdir / 'data/evaluation_sentences.csv').itertuples()
    ]
    translators: Dict[str, Dict[str, Translator]] = {
        'gpt-4o-mini': {
            # 'pipeline': PipelineTranslator(model='gpt-4o-mini'),
            # 'instructions': InstructionsTranslator(model='gpt-4o-mini'),
            # 'agentic': AgenticTranslator(model='gpt-4o-mini'),
            # 'finetuned-simple': FinetunedTranslator(model='ft:gpt-4o-mini-2024-07-18:kubishi:brackets-plus-prompt-merged:AFvrmkic'),
            # 'finetuned-old': FinetunedTranslator(model='ft:gpt-4o-mini-2024-07-18:kubishi::AHxZJd3A')
            'finetuned': FinetunedTranslator(model='ft:gpt-4o-mini-2024-07-18:kubishi::AInrzzLW')
        },
        'gpt-4o': {
            # 'pipeline': PipelineTranslator(model='gpt-4o'),
            # 'instructions': InstructionsTranslator(model='gpt-4o'),
            # 'agentic': AgenticTranslator(model='gpt-4o'),
            # 'finetuned-simple': FinetunedTranslator(model='ft:gpt-4o-2024-08-06:kubishi:brackets-plus-prompt-merged-4o:AGCTD0Ao'),
            # 'finetuned-old': FinetunedTranslator(model='ft:gpt-4o-2024-08-06:kubishi::AHxjqcsb')
            'finetuned': FinetunedTranslator(model='ft:gpt-4o-2024-08-06:kubishi::AInyiTpj')
        },
    }

    # load results from disk
    resultspath = thisdir / 'results/evaluation_results.json'
    resultspath.parent.mkdir(exist_ok=True, parents=True)
    results = Results(results=[])
    if resultspath.exists():
        results = Results.model_validate_json(resultspath.read_text())
    finished = {
        (r.translator, r.model, r.translation.source)
        for r in results.results
    }
    for i, (model, _translators) in enumerate(translators.items(), start=1):
        print(f"[RUNNING] model={model} ({i}/{len(translators)})")
        for j, (translator_name, translator) in enumerate(_translators.items(), start=1):
            print(f"  [RUNNING] translator={translator_name} ({j}/{len(_translators)})")
            for k, (sentence_type, sentence) in enumerate(sentences, start=1):
                print(f"    [RUNNING] sentence={k}/{len(sentences)} ({k/len(sentences)*100:.2f}%)", end='\r')
                if (translator_name, model, sentence) in finished:
                    continue
                try:
                    translation = translator.translate(sentence)
                except Exception as e:
                    print(traceback.format_exc())
                    print(f"    [ERROR] translator={translator_name} sentence={sentence} error={e}")
                    return
                result = Result(
                    translator=translator_name,
                    model=model,
                    sentence_type=sentence_type,
                    translation=translation
                )
                results.results.append(result)
                finished.add((translator_name, model, sentence))
                resultspath.write_text(results.model_dump_json(indent=2))
            print(" "*100, end='\r')
            print(f"  [FINISHED] translator={translator_name}")
        print(f"[FINISHED] model={model}")

if __name__ == '__main__':
    main()