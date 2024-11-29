from yaduha.forward import FinetunedTranslator

def main():
    models = [
        # 'ft:gpt-4o-mini-2024-07-18:kubishi:brackets-plus-prompt-merged:AFvrmkic',
        # 'ft:gpt-4o-2024-08-06:kubishi:brackets-plus-prompt-merged-4o:AGCTD0Ao',
        'ft:gpt-4o-mini-2024-07-18:kubishi::AGvA83su',
        'ft:gpt-4o-2024-08-06:kubishi::AGvLgN91'
    ]
    for model in models:
        translator = FinetunedTranslator(model=model)
        translation = translator.translate("The cat is sitting and the dog is running.")
        print("Model:", model)
        print(translation)
        print()

if __name__ == "__main__":
    main()