import argparse
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from coherence import *

parser = argparse.ArgumentParser()

parser.add_argument('-model', required=True, help='Path to model')

opts = parser.parse_args()


def main():
    archive = load_archive(opts.model)
    predictor = Predictor.from_archive(archive, 'dialogue_context-predictor')

    context = response = ""

    while True:
        context = input("Enter Context (q to exit): ").strip()
        if context == 'q':
            break
        response = input("Enter Response (q to exit): ").strip()
        if response == 'q':
            break
        inputs = {
            "context": context,
            "response": response
        }

        result = predictor.predict_json(inputs)
        label = result.get("label")
        prob = max(result.get("class_probabilities"))
        print("Predicted label: '{}' with probability: {}".format(label, prob))


if __name__ == "__main__":
    main()
