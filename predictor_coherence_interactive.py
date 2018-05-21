import argparse
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from coherence import *


# Monkey-patch because I trained with a newer version.
# This can be removed once PyTorch 0.4.x is out.
# See https://discuss.pytorch.org/t/question-about-rebuild-tensor-v2/14560
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


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
