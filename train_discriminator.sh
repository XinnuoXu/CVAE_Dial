#!/bin/bash

CLASSIFIER=$1
EXPERIMENT=experiments/dialogue_context_coherence_${CLASSIFIER}_classifier.json
MODEL=trained_models/${CLASSIFIER}

rm -fr $MODEL

allennlp train \
${EXPERIMENT} \
-s ${MODEL} \
--include-package coherence

