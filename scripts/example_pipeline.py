import argparse
import os
import pathlib
from yaduha.forward.pipeline import PipelineTranslator
import dotenv

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()

def main():
    parser = argparse.ArgumentParser(description="Translate English to Paiute")

    parser.add_argument('sentence', help="The English sentence to translate (if not provided, will enter interactive mode)", nargs='?')
    parser.add_argument('--model', help="The model to use for translation", default='gpt-4o-mini')
    parser.set_defaults(func="translate")

    args = parser.parse_args()
    translator = PipelineTranslator(model=args.model)
    translation = translator.translate(args.sentence)
    print(translation.model_dump_json())

if __name__ == '__main__':
    main()
