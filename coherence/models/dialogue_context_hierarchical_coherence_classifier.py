import torch
import torch.nn
import torch.nn.functional as F
import numpy

from typing import Optional, Dict
from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder, TimeDistributed
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure


@Model.register("dialogue_context_hierarchical_coherence_classifier")
class DialogueContextHierarchicalCoherenceClassifier(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 utterance_encoder: Seq2VecEncoder,
                 context_encoder: Seq2VecEncoder,
                 response_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(DialogueContextHierarchicalCoherenceClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = vocab.get_vocab_size("labels")
        self.utterances_encoder = TimeDistributed(utterance_encoder)
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
        # batch_size x sents_length (3) x max_context_seq_length x word_emb_dim
        embedded_context = self.text_field_embedder(context)
        # batch_size x sents_length (3) x max_context_seq_length [0,1s]
        utterances_mask = util.get_text_field_mask(context, 1)
        # batch_size x sents_length (3) [0,1s]
        context_mask = util.get_text_field_mask(context)
        # batch_size x sents_length x utt_emb_dim
        encoded_utterances = self.utterances_encoder(embedded_context, utterances_mask)
        # batch_size x context_emb_dim
        encoded_context = self.context_encoder(encoded_utterances, context_mask)

        # batch_size x max_response_seq_length x context_emb_dim
        embedded_response = self.text_field_embedder(response)
        # batch_size x max_response_seq_length [0,1s]
        response_mask = util.get_text_field_mask(response)
        # batch_size x response_emb_dim
        encoded_response = self.response_encoder(embedded_response, response_mask)

        # batch_size x (context_emb_dim + response_emb_dim)
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
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'DialogueContextHierarchicalCoherenceClassifier':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        utterance_encoder = Seq2VecEncoder.from_params(params.pop("utterance_encoder"))
        context_encoder = Seq2VecEncoder.from_params(params.pop("context_encoder"))
        response_encoder = Seq2VecEncoder.from_params(params.pop("response_encoder"))
        classifier_feedforward = FeedForward.from_params(params.pop("classifier_feedforward"))

        initializer = InitializerApplicator.from_params(params.pop("initializer", []))
        regularizer = RegularizerApplicator.from_params(params.pop("regularizer", []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   utterance_encoder=utterance_encoder,
                   context_encoder=context_encoder,
                   response_encoder=response_encoder,
                   classifier_feedforward=classifier_feedforward,
                   initializer=initializer,
                   regularizer=regularizer)
