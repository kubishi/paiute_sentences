import pathlib
from yaduha.forward.agentic import AgenticTranslator
import logging
import json
import dotenv

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent.absolute()

# set logging level to INFO
# logging.basicConfig(level=logging.INFO)

def main():
    savedir = thisdir / "agentic"
    savedir.mkdir(exist_ok=True, parents=True)

    sentences = {
        "simple": "The dog is running.",
        # "complex": "The boy and the girl are eating a quesadilla.",
        # "nominalization": "The cook saw the ones who walked."
    }

    results = []
    for model in ["gpt-4o-mini"]: #, "gpt-4o"]:	
        print(f"=== Using model: {model} ===")
        for i, (name, target_sentence) in enumerate(sentences.items()):
            translation = AgenticTranslator(
                savepath=savedir / f"messages-openai-{name}-{model}.json",
                model=model
            ).translate(target_sentence)
            results.append({
                "model": model,
                "sentence": name,
                "translation": translation.model_dump_json()
            })
            print(translation, end="\n\n")

    costs = { # price per million tokens
        "gpt-4o": {
            "prompt": 2.50,
            "completion": 10.00,
        },
        "gpt-4o-mini": {
            "prompt": 0.150,
            "completion": 0.600
        }	
    }

    for result in results:
        model = result["model"]
        translation = result["translation"]
        cost = sum([
            translation["translation_prompt_tokens"] / 1_000_000 * costs[model]["prompt"],
            translation["translation_completion_tokens"] / 1_000_000 * costs[model]["completion"],
            translation["back_translation_prompt_tokens"] / 1_000_000 * costs[model]["prompt"],
            translation["back_translation_completion_tokens"] / 1_000_000 * costs[model]["completion"]
        ])
        result["cost"] = cost
        print(f"Cost for {model}/{result['sentence']}: ${cost:.2f}")

    total_costs = {
        model: sum([r["cost"] for r in results if r["model"] == model])
        for model in ["gpt-4o-mini", "gpt-4o"]
    }
    average_costs = {
        model: total_costs[model] / 3
        for model in ["gpt-4o-mini", "gpt-4o"]
    }
    print(f"Total costs: {total_costs}")
    print(f"Average costs: {average_costs}")

    (savedir / "results.json").write_text(json.dumps(results, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
