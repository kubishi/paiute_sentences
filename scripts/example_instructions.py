from yaduha.forward.instructions import InstructionsTranslator


def main():
    translator = InstructionsTranslator(model='gpt-4o')
    sentences = [
        "The dog fell.",
        "That horse heard the runner.",
        "The frog is standing by the door."
    ]
    for sentence in sentences:
        translation = translator.translate(sentence)
        print(translation, end='\n\n')

if __name__ == '__main__':
    main()