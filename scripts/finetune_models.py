import json
import os
import openai
import dotenv
import pathlib

import pandas as pd

from yaduha.forward.finetuned import FinetunedTranslator


dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.resolve()

def main():
    datadir = thisdir / "data"
    sources = [
        'random_good_translations.csv',
        'random_no_subject_noun.csv',
        'random_no_object_noun.csv',
        'random_no_verb.csv',
        'random_no_nouns.csv',
        'random_no_vocab.csv',
        'random_complex_translations.csv',
    ]

    df = pd.concat(
        [pd.read_csv(datadir / source) for source in sources],
        ignore_index=True
    )
    jsonl_lines = []
    for _, row in df.iterrows():
        messages = FinetunedTranslator.get_default_messages()
        messages += [
            {
                'role': 'user',
                'content': row['eng']
            },
            {
                'role': 'assistant',
                'content': row['ovp']
            }
        ]
        jsonl_lines.append(json.dumps({'messages': messages}, ensure_ascii=False))

    finetune_dataset_path = thisdir / 'data/finetune_dataset.jsonl'
    finetune_dataset_path.write_text('\n'.join(jsonl_lines))

    client = openai.Client(api_key=os.getenv('OPENAI_API_KEY'))
    with finetune_dataset_path.open('rb') as f:
        res_file = client.files.create(file=f, purpose="fine-tune")

    for model in ['gpt-4o-2024-08-06', 'gpt-4o-mini-2024-07-18']:
        client.fine_tuning.jobs.create(
            training_file=res_file.id,
            model=model
        )


if __name__ == '__main__':
    main()