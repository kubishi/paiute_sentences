from yaduha.sentence_builder import get_random_sentence_big, get_random_sentence, get_random_simple_sentence, format_sentence, sentence_to_str


def main():
    funcs = {
        "random": get_random_sentence,
        "simple": get_random_simple_sentence,
        "big": get_random_sentence_big
    }
    for type, func in funcs.items():
        choices = func()
        sentence = format_sentence(**{k: v['value'] for k, v in choices.items()})
        print(f"{type}: {sentence_to_str(sentence)}")

if __name__ == "__main__":
    main()