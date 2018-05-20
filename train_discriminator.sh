#!/bin/bash

CLASSIFIER=$1
EXPERIMENT=experiments/dialogue_context_${CLASSIFIER}_classifier.json
MODEL=trained_models/${CLASSIFIER}${2}

# rm -fr $MODEL

allennlp train \
${EXPERIMENT} \
-s ${MODEL} \
--include-package coherence

#--file-friendly-logging \

