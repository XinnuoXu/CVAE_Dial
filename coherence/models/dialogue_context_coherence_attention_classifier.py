from typing import Optional, Dict

import numpy
import torch
import torch.nn
import torch.nn.functional as F
from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2VecEncoder, \
    TextFieldEmbedder, Seq2SeqEncoder, SimilarityFunction, \
    TimeDistributed, MatrixAttention
from allennlp.nn import InitializerApplicator, RegularizerApplicator

from allennlp.nn.util import get_text_field_mask, last_dim_softmax, weighted_sum
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("dialogue_context_coherence_attention_classifier")
class DialogueContextCoherenceAttentionClassifier(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 attend_feedforward: FeedForward,
                 similarity_function: SimilarityFunction,
                 compare_feedforward: FeedForward,
                 classifier_feedforward: FeedForward,
                 context_encoder: Optional[Seq2SeqEncoder] = None,
                 response_encoder: Optional[Seq2SeqEncoder] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(DialogueContextCoherenceAttentionClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = vocab.get_vocab_size("labels")
        self.context_encoder = context_encoder
        self.response_encoder = response_encoder
        self.attend_feedforward = TimeDistributed(attend_feedforward)
        self.matrix_attention = MatrixAttention(similarity_function)
        self.compare_feedforward = TimeDistributed(compare_feedforward)
        self.classifier_feedforward = classifier_feedforward
        labels = self.vocab.get_index_to_token_vocabulary('labels')
        pos_label_index = list(labels.keys())[list(labels.values()).index('neg')]

        check_dimensions_match(text_field_embedder.get_output_dim(), attend_feedforward.get_input_dim(),
                               "text field embedding dim", "attend feedforward input dim")
        check_dimensions_match(classifier_feedforward.get_output_dim(), self.num_classes,
                               "final output dimension", "number of labels")

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
        context_mask = get_text_field_mask(context).float()

        embedded_response = self.text_field_embedder(response)
        response_mask = get_text_field_mask(response).float()

        if self.context_encoder:
            embedded_context = self.context_encoder(embedded_context, context_mask)
        if self.response_encoder:
            embedded_response = self.response_encoder(embedded_response, response_mask)

        projected_context = self.attend_feedforward(embedded_context)
        projected_response = self.attend_feedforward(embedded_response)
        # batch x context_length x response_length
        similarity_matrix = self.matrix_attention(projected_context, projected_response)

        # batch x context_length x response_length
        c2r_attention = last_dim_softmax(similarity_matrix, response_mask)
        # batch x context_length x embedded_context_dim
        attended_response = weighted_sum(embedded_response, c2r_attention)

        r2c_attention = last_dim_softmax(similarity_matrix.transpose(1, 2).contiguous(), context_mask)
        attended_context = weighted_sum(embedded_context, r2c_attention)

        # batch x context_length x embedded_context_dim + attended_response_dim
        context_compare_input = torch.cat([embedded_context, attended_response], dim=-1)
        response_compare_input = torch.cat([embedded_response, attended_context], dim=-1)

        compared_context = self.compare_feedforward(context_compare_input)
        compared_context = compared_context * context_mask.unsqueeze(-1)
        # batch x compare_dim
        compared_context = compared_context.sum(dim=1)

        compared_response = self.compare_feedforward(response_compare_input)
        compared_response = compared_response * response_mask.unsqueeze(-1)
        compared_response = compared_response.sum(dim=1)


        # batch x compare_context_dim + compared_response_dim
        aggregate_input = torch.cat([compared_context, compared_response], dim=-1)

        class_logits = self.classifier_feedforward(aggregate_input)

        class_probs = F.softmax(class_logits, dim=-1)

        output_dict = {"class_logits": class_logits, "class_probabilities": class_probs}

        if label is not None:
            loss = self.loss(class_logits, label.squeeze(-1))
            for metric in self.metrics.values():
                metric(class_logits, label.squeeze(-1))
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
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'DialogueContextCoherenceAttentionClassifier':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)

        context_encoder_params = params.pop("context_encoder", None)
        if context_encoder_params is not None:
            context_encoder = Seq2SeqEncoder.from_params(context_encoder_params)
        else:
            context_encoder = None

        response_encoder_params = params.pop("response_encoder", None)
        if response_encoder_params is not None:
            response_encoder = Seq2SeqEncoder.from_params(response_encoder_params)
        else:
            response_encoder = None

        attend_feedforward = FeedForward.from_params(params.pop('attend_feedforward'))
        similarity_function = SimilarityFunction.from_params(params.pop("similarity_function"))
        compare_feedforward = FeedForward.from_params(params.pop('compare_feedforward'))
        classifier_feedforward = FeedForward.from_params(params.pop("classifier_feedforward"))

        initializer = InitializerApplicator.from_params(params.pop("initializer", []))
        regularizer = RegularizerApplicator.from_params(params.pop("regularizer", []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   attend_feedforward=attend_feedforward,
                   similarity_function=similarity_function,
                   compare_feedforward=compare_feedforward,
                   classifier_feedforward=classifier_feedforward,
                   context_encoder=context_encoder,
                   response_encoder=response_encoder,
                   initializer=initializer,
                   regularizer=regularizer)
