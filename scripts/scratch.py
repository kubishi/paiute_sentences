import json
import pathlib
from typing import Dict, List, Union

import pandas as pd

thisdir = pathlib.Path(__file__).parent.resolve()

def remove_old_translators(to_remove: List[str]):
    resultspath = thisdir / 'results/evaluation_results.json'
    results = json.loads(resultspath.read_text())
    results['results'] = [
        res for res in results['results']
        if res['translator'] not in to_remove
    ]
    resultspath.write_text(json.dumps(results, indent=2, ensure_ascii=False))

def rename_translator(old: str, new: str):
    resultspath = thisdir / 'results/evaluation_results.json'
    results = json.loads(resultspath.read_text())
    for res in results['results']:
        if res['translator'] == old:
            res['translator'] = new
    resultspath.write_text(json.dumps(results, indent=2, ensure_ascii=False))

def remove_old_sentences():
    resultspath = thisdir / 'results/evaluation_results.json'
    datapath = thisdir / 'data/evaluation_sentences.csv'

    results = json.loads(resultspath.read_text())
    sentences = set()
    for _, sentence, _ in pd.read_csv(datapath).itertuples():
        sentences.add(sentence)

    results['results'] = [
        res for res in results['results']
        if res['translation']['source'] in sentences
    ]

    resultspath.write_text(json.dumps(results, indent=2, ensure_ascii=False))


def combine_results():
    path_new = thisdir / "results" / "evaluation_results.json"
    path_old = thisdir / "results" / "evaluation_results_old.json"

    new: Dict[str, List[Dict[str, Union[Dict,str,float,int]]]] = json.loads(path_new.read_text())
    old: Dict[str, List[Dict[str, Union[Dict,str,float,int]]]] = json.loads(path_old.read_text())
    metadata = {
        (det['translator'], det['model'], det['translation']['source']): det
        for det in old['results']
    }
    print(metadata)
    combined = {'results': []}
    for det in new['results']:
        key = (det['translator'], det['model'], det['translation']['source'])
        print(key)
        if key in metadata:
            print('here')
            det.update(metadata[key])

        combined['results'].append(det)

    path_new.write_text(json.dumps(combined, indent=2, ensure_ascii=False))

def main():
    # remove_old_sentences()
    combine_results()
    # remove_old_translators(['finetuned-simple'])
    # rename_translator('finetuned', 'finetuned-old')

if __name__ == "__main__":
    main()

