import torch
import torch.nn
import torch.nn.functional as F
import numpy

from typing import Optional, Dict
from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure


@Model.register("dialogue_context_coherence_classifier")
class DialogueContextCoherenceClassifier(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 context_encoder: Seq2VecEncoder,
                 response_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(DialogueContextCoherenceClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = vocab.get_vocab_size("labels")
        self.context_encoder = context_encoder
        self.response_encoder = response_encoder
        self.classifier_feedforward = classifier_feedforward
        labels = self.vocab.get_index_to_token_vocabulary('labels')
        pos_label_index = list(labels.keys())[list(labels.values()).index('neg')]
        self.metrics = {
            "accuracy": CategoricalAccuracy()
            # "f1": F1Measure(positive_label=pos_label_index)
        }
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    @overrides
    def forward(self,
                context: Dict[str, torch.LongTensor],
                response: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        embedded_context = self.text_field_embedder(context)
        encoded_context = self.context_encoder(
            embedded_context,
            util.get_text_field_mask(context))

        encoded_response = self.response_encoder(
            self.text_field_embedder(response),
            util.get_text_field_mask(response))

        logits = self.classifier_feedforward(torch.cat([encoded_context, encoded_response], dim=-1))

        class_probs = F.softmax(logits, dim=-1)

        output_dict = {"class_probabilities": class_probs}

        if label is not None:
            loss = self.loss(logits, label.squeeze(-1))
            for metric in self.metrics.values():
                metric(logits, label.squeeze(-1))
            output_dict['loss'] = loss

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # precision, recall, f1 = self.metrics["f1"].get_metric(reset)
        # metrics = {"accuracy": self.metrics["accuracy"].get_metric(reset),
        #            "precision:": precision,
        #            "recall": recall,
        #            "f1": f1}
        # return metrics
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = output_dict["class_probabilities"].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels") for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'DialogueContextCoherenceClassifier':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        context_encoder = Seq2VecEncoder.from_params(params.pop("context_encoder"))
        response_encoder = Seq2VecEncoder.from_params(params.pop("response_encoder"))
        classifier_feedforward = FeedForward.from_params(params.pop("classifier_feedforward"))

        initializer = InitializerApplicator.from_params(params.pop("initializer", []))
        regularizer = RegularizerApplicator.from_params(params.pop("regularizer", []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   context_encoder=context_encoder,
                   response_encoder=response_encoder,
                   classifier_feedforward=classifier_feedforward,
                   initializer=initializer,
                   regularizer=regularizer)
