# pylint: disable=no-self-use,invalid-name,unused-import
from unittest import TestCase
from pytest import approx
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from allennlp.common.testing import AllenNlpTestCase

import coherence


class TestWikiContextClassifierPredictor(AllenNlpTestCase):
    def test_predict_use_case(self):
        inputs = {
            "context": "that was fun . <u2> oh , that was great ! <u1> oh , time for a break ?",
            "response": "dad , i 'm hungry .",
        }

        archive = load_archive('tests/fixtures/model.tar.gz')
        predictor = Predictor.from_archive(archive, 'dialogue_context-predictor')

        result = predictor.predict_json(inputs)

        label = result.get("label")
        assert label in ['pos', 'neg']

        class_probabilities = result.get("class_probabilities")
        assert class_probabilities is not None
        assert all(cp > 0 for cp in class_probabilities)
        assert sum(class_probabilities) == approx(1.0)