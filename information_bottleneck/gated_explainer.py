'''
This file contains models that do independent masking strategy over constituents/sentences
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions import Bernoulli
from torch.distributions import kl_divergence
from torch.distributions.one_hot_categorical import OneHotCategorical
from modeling_distilbert import DistilBertPreTrainedModel, DistilBertForSequenceClassification
from modeling_roberta import RobertaForSequenceClassification
from transformers import DistilBertModel, BertModel, RobertaModel
from modeling_bert import (BertPreTrainedModel, BertForSequenceClassification)
from transformers.modeling_utils import PreTrainedModel
from transformers import WEIGHTS_NAME
from ib_utils import idxtobool
from copy import deepcopy as copy
import pdb, os
import logging
import math
import numpy as np
from kuma import HardBinary, HardKuma, StretchedVariable, RelaxedBinary, Kuma

logger = logging.getLogger(__name__)
'''
It is not clear how masking will affect the performance of models for QA since the Q part, 
even in unsupervised settings should not be masked, so there is defintitely a need for required input mask.
Some tokens have to always be turned on even when sampling * not easy to achieve
'''
EPS = 1e-16
# torch.set_anomaly_enabled(True)


def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.

    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.

    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


class TransformerExplainer(PreTrainedModel):
    # from pretrained should be
    def __init__(self, config, model_params=None):
        super(TransformerExplainer, self).__init__(config)

    def tie_weights(self):
        raise NotImplementedError

    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_pretrained_default(cls, pretrained_model_name_or_path, config, model_params, state_dict):
        # Load model
        archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        logger.info("loading weights file {}".format(archive_file))
        # Instantiate model.
        model = cls(config, model_params)

        if state_dict is None:
            try:
                state_dict = torch.load(archive_file, map_location='cpu')
            except:
                raise OSError("Unable to load weights from pytorch checkpoint file. "
                              "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. ")
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
        # so we need to apply the function recursively.
        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ''
        model_to_load = model
        #if not hasattr(model, cls.base_model_prefix) and any(
        #        s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
        #    start_prefix = cls.base_model_prefix + '.'
        #if hasattr(model, cls.base_model_prefix) and not any(
        #        s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
        #    model_to_load = getattr(model, cls.base_model_prefix)
        load(model_to_load, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                model.__class__.__name__, "\n\t".join(error_msgs)))

        # TODO: Implement weight tying (input and output embeddings similar to the way Seojin
        # model.tie_weights()  # make sure word embedding weights are still tied if needed
        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()
        # if output_loading_info:
        #     loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
        #     return model, loading_info

        return model


class GatedExplainer(TransformerExplainer):
    def __init__(self, config, model_params=None):
        super(GatedExplainer, self).__init__(config)

        # Check and see if you can share parameters i.e. a single encoder architecture over
        self.dropout = nn.Dropout(p=0.2)

        self.chunk_size = model_params["chunk_size"]
        self.tau = model_params["tau"]
        self.K = model_params["K"]
        self.beta = model_params["beta"]
        self.num_sample = model_params["num_avg"]
        self.fixed_mask_length = model_params["max_query_len"]
        self.threshold = model_params["threshold"]

        self.explainer_model = nn.Sequential(
            nn.Linear(2 * config.dim, config.dim),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(config.dim, 1),
        )

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, p_mask=None,
                inputs_embeds=None, labels=None, num_sample=1, evaluate=False):

        # TODO: Instead of gating apply IB using sigmoid activation and gumbel sampling
        # Call explain model
        # Call reparameterize
        # Add essential input mask
        # Call classify model, which returns label loss
        # Compute classification and IB loss and return
        batch_size = input_ids.shape[0]
        outputs = self.encoder(input_ids,
                               attention_mask=attention_mask,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)

        # Add dropout over encoder
        sequence_output = outputs[0]
        sequence_output = sequence_output * attention_mask.unsqueeze(-1)
        sequence_output = self.dropout(sequence_output)

        # Chunk enumeration only for the portion which can be variably masked

        # Enumerate all possible chunks using their start and end indices
        seq_len = input_ids.shape[1] - self.fixed_mask_length
        # Number of chunks created : seeq_len - chunk_size + 1
        candidate_starts = torch.tensor(range(seq_len - self.chunk_size + 1), dtype=torch.long, device=input_ids.device)
        candidate_ends = candidate_starts + self.chunk_size - 1
        candidate_starts += self.fixed_mask_length
        candidate_ends += self.fixed_mask_length
        # TODO: There are many variations like max and average pooling over spans
        chunk_representations = torch.cat((torch.index_select(sequence_output, 1, candidate_starts), \
                                           torch.index_select(sequence_output, 1, candidate_ends)), dim=-1)
        chunk_mask = (torch.index_select(attention_mask, 1, candidate_starts) + \
                      torch.index_select(attention_mask, 1, candidate_ends) > 1).long()

        # non overlapping chunks of 5 words
        #

        logits = F.logsigmoid(self.explainer_model(chunk_representations).squeeze(-1))

        # Draw a gumbel 2 category sample for every logit
        Z_hat, Z_hat_fixed = self.reparameterize(logits, tau=self.tau, k=self.K, num_sample=self.num_sample)

        # Output classifier
        Z_hat = Z_hat * chunk_mask
        Z_hat = Z_hat.unsqueeze(1)
        class_loss, class_logit = self.classify(input_ids, attention_mask, p_mask, Z_hat, labels, num_sample)

        # Compute loss (since calling function needs to directly receive the loss function itself?)
        # if evaluate:
        #     Z_hat_fixed = torch.exp(logits).unsqueeze(1)
        #     # since you dont have to draw a differentiable sample maybe sample from the multinomial?
        #     # we want to see how useful the distributionis with different beta
        #     _, class_fixed_logits = self.classify(input_ids, attention_mask, p_mask, Z_hat_fixed, labels)
        # else:
        if evaluate:
           q_z = Bernoulli(logits=logits)
           Z_hat_fixed = q_z.sample().unsqueeze(1) # Suggestion given by Xiang Lisa Li
        # since you dont have to draw a differentiable sample maybe sample from the multinomial?
           _, class_fixed_logits = self.classify(input_ids, attention_mask, p_mask, Z_hat_fixed, labels)
        else:
            class_fixed_logits = class_logit
        # if labels is not None:
        #     # contiguity_loss = self.contiguity_loss()
        #     sparsity_loss = torch.norm(Z_hat, 1)/(batch_size*seq_len)
        #     total_loss = class_loss + sparsity_loss #+ 0.5 * contiguity_loss
        #     return total_loss, class_loss, sparsity_loss, class_logit, Z_hat
        # else:
        #     # torch where Z_hat > Threshold is returned as rationale_idx
        #     # rationale_idx = Z_hat > 0.5
        #     return class_logit, rationale_idx, Z_hat


        if labels is not None:
            p_i_prior = self.prior(var_size=logits.size(), device=logits.device, threshold=self.threshold)
            # Generate Bernoulli from prior and logits
            q_z = Bernoulli(logits=logits)
            p_z = Bernoulli(probs=p_i_prior)
            info_loss = (torch.distributions.kl_divergence(q_z, p_z).sum()) / batch_size
            # class_loss = class_loss.div(math.log(2))
            total_loss = class_loss + self.beta * info_loss
            return total_loss, class_loss, info_loss, class_logit, logits, Z_hat
        else:
            # Find top - K chunks and translate them back into a binary mask over tokens
            # TODO: Logits still need to be masked out appropriately
            # logits.masked_fill_(chunk_mask.bool(), -float('-inf'))
            # _, index_chunk = logits.unsqueeze(1).topk(self.K, dim=-1) # this is exactly the operation that got us fixed class logits
            q_z = Bernoulli(logits=logits)
            index_chunk = (q_z.sample() * chunk_mask)
            # logic of how these chunks actually come from just part of the input
            newadd = torch.tensor(range(self.chunk_size), dtype=torch.long, device=logits.device)\
                .unsqueeze(0).unsqueeze(0).unsqueeze(0)
            new_size_col = candidate_starts.shape[0]
            rationale_idx = torch.add(index_chunk, torch.mul(torch.div(index_chunk, new_size_col), self.chunk_size - 1))
            rationale_idx = torch.add(rationale_idx.unsqueeze(-1).expand(-1, -1, -1, self.chunk_size), newadd)
            newsize = rationale_idx.size()
            # rationale_idx = rationale_idx.view(newsize[0], newsize[1], -1, 1).squeeze(-1)
            rationale_idx = rationale_idx.squeeze(1)
            rationale_idx = index_chunk + self.fixed_mask_length
            class_logit = class_fixed_logits
            return class_logit, rationale_idx, logits, Z_hat

    def classify(self, input_ids, attention_mask, p_mask, z_hat, labels=None, num_sample=1):
        # Resize z_hat accordingly so that is it once again batch_size * num_sentences * num_words_per_sentence
        # Apply mask to the encoder

        # Apply a required mask where CLS token is not considered

        # Apply a different instance of BERT once again on this masked version and predict class logits
        z_hat = nn.Sequential(
            nn.ConstantPad1d(self.chunk_size - 1, 0),
            nn.MaxPool1d(kernel_size=self.chunk_size, stride=1, padding=0)
        )(z_hat)

        # TODO: Squeezing works only if num_samples = 1
        z_hat = z_hat.squeeze(1)
        # reweighing each token and making the weights as close to one hot as possible

        # TODO: Another catch is that when the sampling is being done the attention masked positions should
        # be disregarded, currently not sure if this should be done before hand
        fixed_mask = torch.ones((z_hat.shape[0], self.fixed_mask_length), dtype=torch.float, device=z_hat.device)
        z_hat = torch.cat((fixed_mask, z_hat), dim=-1) * attention_mask.float()

        # Manipulating z_hat will break differentiability assumption across the board

        classification_output = self.classifier_model(input_ids=input_ids,
                                                      attention_mask=z_hat,
                                                      labels=labels)
        # creates a very very sparse mask on the output of the transformer.
        # Instead what you should be doing is implementing the sparsity on the embeddings
        # and them making use of the deep model to do its job and arrive at a good representation in the CLS token
        # distilbert_output = distilbert_output * z_hat
        # hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        # pooled_output = hidden_state[:, 0]
        # logits = self.classifier_model(pooled_output)
        if labels is not None:
            logits = classification_output[1]
            classification_loss = classification_output[0]
            # attention_weights = classification_output[2]
        else:
            classification_loss = 0
            logits = classification_output[0]
            # attention_weights = classification_output[1]
        return classification_loss, logits

    def reparameterize(self, p_i, tau, k, num_sample=1):

        # Generating k-hot samples  *

        ## sampling
        p_i_ = p_i.view(p_i.size(0), 1, -1)
        p_i_ = p_i_.expand(p_i_.size(0), num_sample, p_i_.size(-1)) # Batch size, Feature size,num_samples
        C_dist = RelaxedBernoulli(tau, logits=p_i_)
        V = C_dist.sample().mean(dim=1)
        # TODO: Logic that you draw 10 samples and average across runs?
        ## without sampling
        ## prob are already part of the
        # V_fixed_size = p_i.unsqueeze(1).size()
        # _, V_fixed_idx = p_i.unsqueeze(1).topk(k, dim=-1)  # batch * 1 * k
        # V_fixed = idxtobool(V_fixed_idx, V_fixed_size, device=p_i.device)
        # V_fixed = V_fixed.type(torch.float) # this should

        # unlike V which is still soft weight matrix, V_fixed is a binary mask, so it should probably being doing horribly
        return V, V

    def prior(self, var_size, device, threshold=0.5):
        # TODO: prior will be influenced by the actualy sparsity for the dataset?
        p = torch.tensor([threshold], device=device)
        # p = p / var_size[1]
        # based on sparsity this is a 0/1 choice
        p = p.view(1, 1)
        p_prior = p.expand(var_size)  # [batch-size, k, feature dim]

        return p_prior


class GatedSentenceExplainer(TransformerExplainer):
    def __init__(self, config, model_params=None):
        super(GatedSentenceExplainer, self).__init__(config)

        # Check and see if you can share parameters i.e. a single encoder architecture over
        self.dropout = nn.Dropout(p=0.2)

        self.chunk_size = model_params["chunk_size"]
        self.tau = model_params["tau"]
        self.K = model_params["K"]
        self.beta = model_params["beta"]
        self.beta_norm = model_params["beta_norm"]
        self.num_sample = model_params["num_avg"]
        self.fixed_mask_length = model_params["max_query_len"] - 1  # The first SEP take a different mask weight
        self.threshold = model_params["threshold"]
        self.distribution = model_params["distribution"]
        self.norm = model_params["norm"]
        self.soft_evaluation = model_params["soft_eval"]
        self.semi_supervised = model_params["semi_supervised"]
        self.sampled_evaluation = model_params["sampled_eval"]

        self.pi = nn.Parameter(torch.tensor(model_params["threshold"], requires_grad=False))

        # EMA and lagrangian learning rate over loss
        # lagrange buffers
        self.lagrange_alpha = 0.5
        self.lambda_init = 0.1
        self.lambda_min = 1e-6
        self.lambda_max = 1.0
        self.lagrange_lr = 0.05
        self.register_buffer('lambda0', torch.full((1,), self.lambda_init))
        self.register_buffer('c0_ma', torch.full((1,), 0.))  # moving average

        hidden_dim = config.dim if hasattr(config, "dim") else config.hidden_size
        if self.distribution == "binary":
            self.explainer_model = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(True),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.explainer_model = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(True),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_dim, 2),
                nn.Softplus()
            )

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, p_mask=None,
                inputs_embeds=None, labels=None, num_sample=1, evaluate=False, evaluate_faithfulness = False,
                sentence_starts=None, sentence_ends=None, sentence_mask=None, evidence_labels=None):

        batch_size = input_ids.shape[0]
        outputs = self.encoder(input_ids,
                               attention_mask=attention_mask,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)
        # Add dropout over encoder
        sequence_output = outputs[0]
        # Original attention mask that ignores all the padded tokens is reinforced
        sequence_output = sequence_output * attention_mask.unsqueeze(-1).float()
        sequence_output = self.dropout(sequence_output)

        sentence_rep_shape = (sequence_output.shape[0], sentence_starts.shape[1], sequence_output.shape[-1])
        sentence_representations = torch.cat((sequence_output.gather(dim=1, index=sentence_starts.unsqueeze(-1).expand(sentence_rep_shape)), \
                                   sequence_output.gather(dim=1, index=sentence_ends.unsqueeze(-1).expand(sentence_rep_shape))), dim=-1)
        sentence_mask = sentence_mask.float()
        if self.distribution == "binary":
            logits = self.explainer_model(sentence_representations).squeeze(-1)
            distribution = F.logsigmoid(logits)
            # TODO: Logits need to be sentence mask
            # Draw a gumbel 2 category sample for every logit
            p_i = distribution.exp()
            Z_hat, Z_hat_fixed = self.reparameterize(distribution, tau=self.tau, k=self.K, sentence_mask=sentence_mask, num_sample=self.num_sample)
        else:
            distribution_parameters = self.explainer_model(sentence_representations)
            distribution_parameters = distribution_parameters.clamp(1e-6, 100.)
            distribution = Kuma([distribution_parameters[:,:,0], distribution_parameters[:,:,1]])
            Z_hat = distribution.sample()
            V_fixed_size = Z_hat.size()
            p_i = distribution.mean()
            p_i_ = distribution.mean()
            p_i_.masked_fill_((1 - sentence_mask).bool(), float('-inf'))
            # V_fixed_idx = torch.LongTensor(p_i.shape[0], k).random_(0, p_i.shape[-1])
            active_top_k = (sentence_mask.sum(-1)*self.threshold).ceil().long()
            _, V_fixed_idx = p_i_.topk(p_i_.size(-1), dim=-1)  # batch * 1 * k
            # only self.threshold % of sentence mask needs to have >0 value;
            Z_hat_fixed = torch.zeros(V_fixed_size, dtype=torch.long, device=p_i.device)
            for i in range(V_fixed_size[0]):
                subidx = V_fixed_idx[i, :active_top_k[i]]
                Z_hat_fixed[i, subidx] = float(1)
            Z_hat_fixed = Z_hat_fixed.type(torch.float) 

        # Output classifier
        Z_hat = Z_hat * sentence_mask

        # for full context evaluation
        # Z_hat = sentence_mask.float()
        # Z_hat_fixed = sentence_mask.float()

        # for gold context evaluation
        # Z_hat = evidence_labels.float()
        # Z_hat_fixed = evidence_labels.float()

        class_loss, class_logit = self.classify(input_ids, attention_mask, p_mask, Z_hat, labels, num_sample)
        if evaluate:
            # Deterministic evaluation
            if self.soft_evaluation:
                _, class_fixed_logits = self.classify(input_ids, attention_mask, p_mask, p_i, labels)
                if evaluate_faithfulness:
                    _, class_all_logits = self.classify(input_ids, attention_mask, p_mask, sentence_mask.float(),
                                                        labels)
                    _, class_residual_logits = self.classify(input_ids, attention_mask, p_mask, 1 - distribution.exp(), labels)
            else:
                _, class_fixed_logits = self.classify(input_ids, attention_mask, p_mask, Z_hat_fixed, labels)
                if evaluate_faithfulness:
                    _, class_all_logits = self.classify(input_ids, attention_mask, p_mask, sentence_mask.float(), labels)
                    _, class_residual_logits = self.classify(input_ids, attention_mask, p_mask, 1 - Z_hat_fixed, labels)
            if self.sampled_evaluation:
                # obtain a sample from distribution
                Z_hat_fixed = Bernoulli(logits=logits).sample() * sentence_mask
                _, class_fixed_logits = self.classify(input_ids, attention_mask, p_mask, Z_hat_fixed, labels)
        else:
            class_fixed_logits = class_logit

        # Compute loss (since calling function needs to directly receive the loss function itself?)
        if labels is not None:

            if self.semi_supervised > 0.0:
                batch_mask = torch.cat((evidence_labels.new_ones((int(np.ceil(self.semi_supervised * batch_size)),)),
                                    evidence_labels.new_zeros((batch_size - int(np.ceil(self.semi_supervised * batch_size)),))))
                unsupervised_batch_mask = 1 - batch_mask
                sentence_mask = sentence_mask * unsupervised_batch_mask.unsqueeze(-1)

            if self.norm == "L1":
                if self.distribution == "binary":
                    # How much should it be if all of them are 0.2
                    # pdf0 = torch.where(sentence_mask.bool(), pdf0, pdf0.new_zeros([1]))
                    pdf_nonzero = Z_hat #torch.sigmoid(distribution)
                    pdf_nonzero = torch.where(sentence_mask.bool(), pdf_nonzero, pdf_nonzero.new_zeros([1]))
                    # norm_info_loss = torch.max(distribution.new_zeros([1]), torch.sum(torch.norm(pdf_nonzero, p=1, dim=-1) / torch.sum(sentence_mask, dim=-1)) / batch_size - self.threshold)
                    norm_info_loss = torch.abs(torch.sum(torch.norm(pdf_nonzero, p=1, dim=-1) / torch.sum(sentence_mask, dim=-1)) / batch_size - self.threshold)

                    """
                    norm_info_loss = torch.sum(torch.norm(pdf_nonzero, p=1, dim=-1) / torch.sum(sentence_mask, dim=-1)) / batch_size
                    c0_hat = (norm_info_loss - self.threshold)
                    self.c0_ma = self.lagrange_alpha * self.c0_ma + (1 - self.lagrange_alpha) * c0_hat.item()
                    # compute smoothed constraint (equals moving average c0_ma)
                    c0 = c0_hat + (self.c0_ma.detach() - c0_hat.detach())
                    # update lambda
                    self.lambda0 = self.lambda0 * torch.exp(self.lagrange_lr * c0.detach())
                    self.lambda0 = self.lambda0.clamp(self.lambda_min, self.lambda_max)
                    norm_info_loss = self.lambda0.detach() * c0
                    """

                else:
                    pdf_nonzero = Z_hat
                    pdf_nonzero = torch.where(sentence_mask.bool(), pdf_nonzero, pdf_nonzero.new_zeros([1]))
                    norm_info_loss = torch.max(Z_hat.new_zeros([1]), torch.sum(torch.norm(pdf_nonzero, p=1, dim=-1) / torch.sum(sentence_mask, dim=-1)) / batch_size - self.threshold)
            if self.distribution == "binary":
                pdf_nonzero = self.kldivergence(distribution)
                pdf_nonzero = torch.where(sentence_mask.bool(), pdf_nonzero, pdf_nonzero.new_zeros([1]))
                # when sentence maskis completely zeroed out for some of the examples in semi-supervised state
                kl_info_loss = torch.sum( pdf_nonzero.sum(-1) / (torch.sum(sentence_mask, dim=-1) + 1e-9)) / batch_size
            else:
                pdf_nonzero = self.kldivergence_kuma(distribution)
                pdf_nonzero = torch.where(sentence_mask.bool(), pdf_nonzero, pdf_nonzero.new_zeros([1]))
                kl_info_loss = torch.sum( pdf_nonzero.sum(-1) / torch.sum(sentence_mask, dim=-1)) / batch_size
            info_loss = self.beta * kl_info_loss  + self.beta_norm * norm_info_loss
            total_loss = class_loss + info_loss

            # Semisupervised setting; For X% of the batch compute BCE Loss
            if self.semi_supervised > 0.0:
                #batch_mask = torch.cat((evidence_labels.new_ones((int(np.ceil(self.semi_supervised * batch_size)),)),
                #                    evidence_labels.new_zeros((batch_size - int(np.ceil(self.semi_supervised * batch_size)),))))
                batch_sentence_mask = sentence_mask * batch_mask.unsqueeze(-1)
                loss_fct = nn.BCEWithLogitsLoss()
                active_loss = batch_sentence_mask.view(-1) == 1
                active_logits = logits.view(-1)[active_loss]
                active_labels = evidence_labels.float().view(-1)[active_loss]
                rationale_loss = loss_fct(active_logits, active_labels)
                total_loss += rationale_loss

            return total_loss, class_loss, info_loss, class_logit, Z_hat, self.lambda0
        else:
            # TODO: Shouldn't you be sampling here?
            # Find top - K chunks and translate them back into a binary mask over tokens
            # logits.masked_fill_((1 - sentence_mask).bool(), float('-inf'))
            # pdf
            avg_nnz = 0
            for t in range(10):
                if self.distribution == "binary":
                    nnz = Bernoulli(logits=distribution).sample()
                else:
                    nnz = distribution.sample()
                nnz = torch.where(sentence_mask.bool(), nnz, nnz.new_zeros([1]))
                nnz_std = nnz.clone()
                nnz = nnz.sum(1) / (sentence_mask.sum(-1) + 1e-9)  # [B]
                nnz = nnz.sum() / batch_size
                avg_nnz += nnz
            nnz = avg_nnz/10.0

            index_chunk = Z_hat_fixed.nonzero()
            rationale_idx = -1*torch.ones(sentence_mask.size(), dtype=torch.long, device=sentence_mask.device)
            for tup in index_chunk:
                rationale_idx[tup[0]][tup[1]] = 1

            # rationale_idx = index_chunk # Logic if exactly k are chosen
            class_logit = class_fixed_logits
            if evaluate_faithfulness:
                return class_logit, rationale_idx, p_i, class_all_logits, class_residual_logits, nnz_std, nnz
            else:
                return class_logit, rationale_idx, p_i, nnz_std, nnz

    def classify(self, input_ids, attention_mask, p_mask, z_hat, labels=None, num_sample=1):
        # TODO: Squeezing works only if num_samples = 1
        z_hat = z_hat.squeeze(1).gather(dim=1, index=p_mask)


        # reweighing each token and making the weights as close to one hot as possible
        fixed_mask = torch.ones((z_hat.shape[0], self.fixed_mask_length), dtype=torch.float, device=z_hat.device)
        z_hat = torch.cat((fixed_mask, z_hat), dim=-1) * attention_mask.float()

        # Manipulating z_hat will break differentiability assumption across the board
        z_hat = z_hat + EPS
        classification_output = self.classifier_model(input_ids=input_ids,
                                                      attention_mask=z_hat,
                                                      labels=labels)
        if labels is not None:
            logits = classification_output[1]
            classification_loss = classification_output[0]
            # attention_weights = classification_output[2]
        else:
            classification_loss = 0
            logits = classification_output[0]
            # attention_weights = classification_output[1]
        return classification_loss, logits

    def reparameterize_kuma(self, p_i, tau, k, num_sample=1):
        pass

    def prior_kuma(self, var_size, device, threshold=0.5):
        pass

    def Beta_fn(self, a, b):
        return torch.exp(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))

    def kldivergence_kuma(self, distribution):
        prior_alpha = torch.tensor([1.0], device=distribution.a.device)
        prior_beta = torch.tensor([4.0], device=distribution.a.device)
        kl = 1. / (1 + distribution.a * distribution.b) * self.Beta_fn(distribution.a.reciprocal(), distribution.b)
        kl += 1. / (2 + distribution.a * distribution.b) * self.Beta_fn(2.0 * distribution.a.reciprocal(), distribution.b)
        kl += 1. / (3 + distribution.a * distribution.b) * self.Beta_fn(3. * distribution.a.reciprocal(), distribution.b)
        kl += 1. / (4 + distribution.a * distribution.b) * self.Beta_fn(4. * distribution.a.reciprocal(), distribution.b)
        kl += 1. / (5 + distribution.a * distribution.b) * self.Beta_fn(5. * distribution.a.reciprocal(), distribution.b)
        kl += 1. / (6 + distribution.a * distribution.b) * self.Beta_fn(6. * distribution.a.reciprocal(), distribution.b)
        kl += 1. / (7 + distribution.a * distribution.b) * self.Beta_fn(7. * distribution.a.reciprocal(), distribution.b)
        kl += 1. / (8 + distribution.a * distribution.b) * self.Beta_fn(8. * distribution.a.reciprocal(), distribution.b)
        kl += 1. / (9 + distribution.a * distribution.b) * self.Beta_fn(9. * distribution.a.reciprocal(), distribution.b)
        kl += 1. / (10 + distribution.a * distribution.b) * self.Beta_fn(10. * distribution.a.reciprocal(), distribution.b)
        kl *= (prior_beta - 1) * distribution.b

        # use another taylor approx for Digamma function
        psi_b_taylor_approx = torch.log(distribution.b) - 1. / (2 * distribution.b) - 1. / (12 * distribution.b ** 2)
        kl += (distribution.a - prior_alpha) / distribution.a * (
                    -0.57721 - psi_b_taylor_approx - 1 / distribution.b)  # T.psi(self.posterior_b)

        # add normalization constants
        kl += torch.log(distribution.a * distribution.b) + torch.log(self.Beta_fn(prior_alpha, prior_beta))

        # final term
        kl += -(distribution.b - 1) / distribution.b

        return kl

    def reparameterize(self, p_i, tau, k, sentence_mask, num_sample=1):

        # Generating k-hot samples  *

        ## sampling
        p_i_ = p_i.view(p_i.size(0), 1, -1)
        p_i_ = p_i_.expand(p_i_.size(0), num_sample, p_i_.size(-1)) # Batch size, Feature size,num_samples
        C_dist = RelaxedBernoulli(tau, logits=p_i_)
        V = C_dist.rsample().squeeze(1)

        V_fixed_size = p_i.size()
        p_i.masked_fill_((1 - sentence_mask).bool(), float('-inf'))
        active_top_k = (sentence_mask.sum(-1)*self.threshold).ceil().long()
        _, V_fixed_idx = p_i.topk(p_i.size(-1), dim=-1)  # batch * 1 * k
        # only self.threshold % of sentence mask needs to have >0 value;
        V_fixed = torch.zeros(V_fixed_size, dtype=torch.long, device=p_i.device)
        for i in range(V_fixed_size[0]):
            subidx = V_fixed_idx[i, :active_top_k[i]]
            V_fixed[i, subidx] = float(1)
        # V_fixed_idx = torch.LongTensor(p_i.shape[0], k).random_(0, p_i.shape[-1])
        # V_fixed = idxtobool(V_fixed_idx, V_fixed_size, device=p_i.device)
        V_fixed = V_fixed.type(torch.float) # this should

        # unlike V which is still soft weight matrix, V_fixed is a binary mask, so it should probably being doing horribly
        return V, V_fixed

    def prior(self, var_size, device, threshold=0.5):
        # TODO: prior will be influenced by the actualy sparsity for the dataset?
        p = torch.tensor([threshold], device=device)
        # p = p / var_size[1]
        # based on sparsity this is a 0/1 choice
        p = p.view(1, 1)
        p_prior = p.expand(var_size)  # [batch-size, k, feature dim]

        return p_prior

    def kldivergence(self, logits):
        p_i_prior = self.prior(var_size=logits.size(), device=logits.device, threshold=self.threshold)
        q_z = Bernoulli(logits=logits)
        p_z = Bernoulli(probs=p_i_prior)
        return torch.distributions.kl_divergence(q_z, p_z)
        '''
        p = Bernoulli(logits=logits)
        t1 = p.probs * (p.probs / self.pi).log()
        t1[self.pi == 0] = math.inf
        t1[p.probs == 0] = 0
        t2 = (1 - p.probs) * ((1 - p.probs) / (1 - self.pi)).log()
        t2[self.pi == 1] = math.inf
        t2[p.probs == 1] = 0
        kl_divergence = t1 + t2
        return kl_divergence
        '''


class HardGatedSentenceExplainer(TransformerExplainer):
    def __init__(self, config, model_params=None):
        super(HardGatedSentenceExplainer, self).__init__(config)

        # Check and see if you can share parameters i.e. a single encoder architecture over
        self.dropout = nn.Dropout(p=0.2)

        self.chunk_size = model_params["chunk_size"]
        self.tau = model_params["tau"]
        self.K = model_params["K"]
        self.beta = model_params["beta"]
        self.beta_norm = model_params["beta_norm"]
        self.num_sample = model_params["num_avg"]
        self.fixed_mask_length = model_params["max_query_len"] - 1  # The first SEP take a different mask weight
        self.threshold = model_params["threshold"]
        support = [-0.1, 1.1]
        self.register_buffer('support', torch.tensor(support))
        self.distribution = model_params["distribution"]
        self.soft_evaluation = model_params["soft_eval"]
        self.semi_supervised = model_params["semi_supervised"]
        self.sampled_evaluation = model_params["sampled_eval"]


        # EMA and lagrangian learning rate over loss
        # lagrange buffers
        self.lagrange_alpha = 0.5
        self.lambda_init = 1e-5
        self.lambda_min = 1e-12
        self.lambda_max = 5
        self.lagrange_lr = 10.0
        self.register_buffer('lambda0', torch.full((1,), self.lambda_init))
        self.register_buffer('c0_ma', torch.full((1,), 0.))  # moving average

        hidden_dim = config.dim if hasattr(config, "dim") else config.hidden_size
        if self.distribution == "binary":
            self.explainer_model = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(True),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.explainer_model = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(True),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_dim, 2),
                nn.Softplus()
            )

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, p_mask=None,
                inputs_embeds=None, labels=None, num_sample=1, evaluate=False, evaluate_faithfulness = False,
                sentence_starts=None, sentence_ends=None, sentence_mask=None, evidence_labels=None):

        batch_size = input_ids.shape[0]
        outputs = self.encoder(input_ids,
                               attention_mask=attention_mask,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)
        # Add dropout over encoder
        sequence_output = outputs[0]
        # Original attention mask that ignores all the padded tokens is reinforced
        sequence_output = sequence_output * attention_mask.unsqueeze(-1).float()
        sequence_output = self.dropout(sequence_output)

        sentence_rep_shape = (sequence_output.shape[0], sentence_starts.shape[1], sequence_output.shape[-1])
        sentence_representations = torch.cat((sequence_output.gather(dim=1, index=sentence_starts.unsqueeze(-1).expand(sentence_rep_shape)), \
                                   sequence_output.gather(dim=1, index=sentence_ends.unsqueeze(-1).expand(sentence_rep_shape))), dim=-1)

        # logits = F.logsigmoid(self.explainer_model(sentence_representations).squeeze(-1))
        hard_distribution_parameters  = self.explainer_model(sentence_representations).squeeze(-1)
        # TODO: Logits need to be sentence mask
        # Draw a gumbel 2 category sample for every logit
        # Z_hat, Z_hat_fixed = self.reparameterize(logits, tau=self.tau, k=self.K, num_sample=self.num_sample)

        # logits are drawn from a relaxed hard Binary variable and there are two?
        if self.distribution == "binary":
            hard_distribution_parameters = hard_distribution_parameters.clamp(-100, 100.)
            logits = hard_distribution_parameters
            hard_distribution = HardBinary(StretchedVariable(RelaxedBinary(hard_distribution_parameters, self.tau), support=self.support))
        else:
            hard_distribution_parameters = hard_distribution_parameters.clamp(1e-6, 100.)
            hard_distribution = HardKuma([hard_distribution_parameters[:,:,0], hard_distribution_parameters[:,:,1]], support=self.support)

        Z_hat = hard_distribution.sample()
        # Output classifier
        Z_hat = Z_hat * sentence_mask.float()
        class_loss, class_logit = self.classify(input_ids, attention_mask, p_mask, Z_hat, labels, num_sample)
        # logit_fixed = self.classify(input_ids, attention_mask, p_mask, Z_hat_fixed, labels)
        # Call another classify only during evaluation since batch has to fit into memory
        if evaluate:
            # Bastings method of deterministic evaluation
            # How to get a deterministic top-k from a hard distribution
            # p0 = hard_distribution.pdf(sentence_representations.new_zeros(()))
            # p1 = hard_distribution.pdf(sentence_representations.new_ones(()))
            # pc = 1. - p0 - p1  # prob. of sampling a continuous value that is neither exactly 1 or 0 [B, M]
            # zero_one = torch.where(
            #     p0 > p1, sentence_representations.new_zeros([1]), sentence_representations.new_ones([1]))
            # Z_hat_fixed = torch.where((pc > p0) & (pc > p1),
            #                   hard_distribution.mean(), zero_one)  # [B, M]
            # code to count how many nnz
            nz = hard_distribution.pdf(0.)
            nz = torch.where(sentence_mask.bool(), nz, nz.new_zeros([1]))
            nnz = 1. - nz  # [B, T]
            nnz = torch.where(sentence_mask.bool(), nnz, nnz.new_zeros([1]))
            nnz = nnz.sum(1) / (sentence_mask.sum(-1) + 1e-9)  # [B]
            nnz = nnz.sum() / batch_size

            p_i = hard_distribution.mean()
            p_i_ = F.hardtanh( hard_distribution.mean(), min_val=0., max_val=1.)
            p_i.masked_fill_((1 - sentence_mask).bool(), float('-inf'))
            V_fixed_size = p_i.size()
            active_top_k = (sentence_mask.sum(-1) * self.threshold).ceil().long()
            _, V_fixed_idx = p_i.topk(p_i.size(-1), dim=-1)  # batch * 1 * k
            # only self.threshold % of sentence mask needs to have >0 value;
            Z_hat_fixed = torch.zeros(V_fixed_size, dtype=torch.long, device=p_i.device)
            for i in range(V_fixed_size[0]):
                subidx = V_fixed_idx[i, :active_top_k[i]]
                Z_hat_fixed[i, subidx] = float(1)
            Z_hat_fixed = Z_hat_fixed.type(torch.float)  # this should
            _, class_fixed_logits = self.classify(input_ids, attention_mask, p_mask, Z_hat_fixed, labels)

            if self.soft_evaluation:
                _, class_fixed_logits = self.classify(input_ids, attention_mask, p_mask, p_i_, labels)
                if evaluate_faithfulness:
                    _, class_all_logits = self.classify(input_ids, attention_mask, p_mask, sentence_mask.float(),
                                                        labels)
                    _, class_residual_logits = self.classify(input_ids, attention_mask, p_mask, 1 - p_i_, labels)
            else:
                _, class_fixed_logits = self.classify(input_ids, attention_mask, p_mask, Z_hat_fixed, labels)
                if evaluate_faithfulness:
                    _, class_all_logits = self.classify(input_ids, attention_mask, p_mask, sentence_mask.float(), labels)
                    _, class_residual_logits = self.classify(input_ids, attention_mask, p_mask, 1 - Z_hat_fixed, labels)
            if self.sampled_evaluation:
                Z_hat_fixed = Bernoulli(logits=hard_distribution.params()[0]).sample() * sentence_mask
                _, class_fixed_logits = self.classify(input_ids, attention_mask, p_mask, Z_hat_fixed, labels)

        else:
            class_fixed_logits = class_logit
        # Compute loss (since calling function needs to directly receive the loss function itself?)
        if labels is not None:
            pdf0 = hard_distribution.pdf(0.)
            pdf0 = torch.where(sentence_mask.bool(), pdf0, pdf0.new_zeros([1]))
            pdf_nonzero = 1. - pdf0  # [B, T]
            pdf_nonzero = torch.where(sentence_mask.bool(), pdf_nonzero, pdf_nonzero.new_zeros([1]))
            l0 = pdf_nonzero.sum(1) / (sentence_mask.sum(-1) + 1e-9)  # [B]
            norm_info_loss = l0.sum() / batch_size
            norm_info_loss = torch.max(hard_distribution_parameters.new_zeros([1]), norm_info_loss - self.threshold)

            '''
            c0_hat = (norm_info_loss - self.threshold)
            self.c0_ma = self.lagrange_alpha * self.c0_ma + (1 - self.lagrange_alpha) * c0_hat.item()
            # compute smoothed constraint (equals moving average c0_ma)
            c0 = c0_hat + (self.c0_ma.detach() - c0_hat.detach())
            # update lambda
            self.lambda0 = self.lambda0 * torch.exp(self.lagrange_lr * c0.detach())
            self.lambda0 = self.lambda0.clamp(self.lambda_min, self.lambda_max)
            norm_info_loss = self.lambda0.detach() * c0
            '''

            # KL Divergence (for the binary variable and not the hard binary variable)
            # kl_loss = hard_distribution._dist.cdf(0.) * (hard_distribution._dist.log_cdf(0.) - prior_distribution.log_cdf(0.))
            # kl_loss can be computed over s in this work and for s' (left to future work?)
            p_i_prior = self.prior(var_size=Z_hat.size(), device=Z_hat.device, threshold=self.threshold)
            q_z = Bernoulli(logits=hard_distribution._dist._dist.logits)
            p_z = Bernoulli(probs=p_i_prior)
            pdf_nonzero = torch.distributions.kl_divergence(q_z, p_z)
            pdf_nonzero = torch.where(sentence_mask.bool(), pdf_nonzero, pdf_nonzero.new_zeros([1]))
            kl_info_loss = torch.sum(pdf_nonzero.sum(-1) / torch.sum(sentence_mask, dim=-1)) / batch_size
            info_loss = self.beta * kl_info_loss + self.beta_norm * norm_info_loss
            total_loss = class_loss + info_loss

            if self.semi_supervised > 0.0:
                batch_mask = torch.cat((evidence_labels.new_ones((int(np.ceil(self.semi_supervised * batch_size)),)),
                                    evidence_labels.new_zeros((batch_size - int(np.ceil(self.semi_supervised * batch_size)),))))
                batch_sentence_mask = sentence_mask * batch_mask.unsqueeze(-1)
                loss_fct = nn.BCEWithLogitsLoss()
                active_loss = batch_sentence_mask.view(-1) == 1
                active_logits = logits.view(-1)[active_loss]
                active_labels = evidence_labels.float().view(-1)[active_loss]
                rationale_loss = loss_fct(active_logits, active_labels)
                total_loss += rationale_loss


            return total_loss, class_loss, info_loss, class_logit, hard_distribution_parameters, Z_hat
        else:
            # TODO: Think shouldn't you be sampling here?
            # Find top - K chunks and translate them back into a binary mask over tokens
            # Z_hat_fixed.masked_fill_((1 - sentence_mask).bool(), float('-inf'))
            # _, index_chunk = Z_hat_fixed.topk(self.K,
            #                                dim=-1)  # this is exactly the operation that got us fixed class logits
            index_chunk = Z_hat_fixed.nonzero()
            rationale_idx = -1 * torch.ones(sentence_mask.size(), dtype=torch.long, device=sentence_mask.device)
            for tup in index_chunk:
                rationale_idx[tup[0]][tup[1]] = 1
            # you may not need to translate this into the chunk that was selected (in token space since we can retireve sent from index)
            # rationale_idx = index_chunk
            class_logit = class_fixed_logits
            if evaluate_faithfulness:
                return class_logit, rationale_idx, p_i_, class_all_logits, class_residual_logits, nnz
            else:
                return class_logit, rationale_idx, p_i_, nnz

    def classify(self, input_ids, attention_mask, p_mask, z_hat, labels=None, num_sample=1):
        # TODO: Squeezing works only if num_samples = 1
        z_hat = z_hat.squeeze(1).gather(dim=1, index=p_mask)

        fixed_mask = torch.ones((z_hat.shape[0], self.fixed_mask_length), dtype=torch.float, device=z_hat.device)
        z_hat = torch.cat((fixed_mask, z_hat), dim=-1) * attention_mask.float()

        # Manipulating z_hat will break differentiability assumption but why? Especially the zero (something like the transformer); And Nanning happening to the
        # class Variable
        z_hat = z_hat + EPS
        classification_output = self.classifier_model(input_ids=input_ids,
                                                      attention_mask=z_hat,
                                                      labels=labels)
        if labels is not None:
            logits = classification_output[1]
            classification_loss = classification_output[0]
            # attention_weights = classification_output[2]
        else:
            classification_loss = 0
            logits = classification_output[0]
            # attention_weights = classification_output[1]
        return classification_loss, logits

    def prior(self, var_size, device, threshold=0.5):
        # TODO: prior will be influenced by the actualy sparsity for the dataset?
        p = torch.tensor([threshold], device=device)
        # p = p / var_size[1]
        # based on sparsity this is a 0/1 choice
        p = p.view(1, 1)
        p_prior = p.expand(var_size)  # [batch-size, k, feature dim]

        return p_prior


# DistilBert classes
class DistilBertGatedSentenceExplainer(GatedSentenceExplainer, DistilBertPreTrainedModel):
    base_model_prefix  = ""
    def __init__(self, config, model_params=None):
        super(DistilBertGatedSentenceExplainer, self).__init__(config, model_params)
        self.encoder = DistilBertModel(config)
        self.classifier_model = DistilBertForSequenceClassification(config)

        self.init_weights()

    def tie_weights(self):
        # Tie word embeddings
        self._tie_or_clone_weights(self.classifier_model.distilbert.embeddings.word_embeddings,
                                   self.encoder.embeddings.word_embeddings)

        # Tie positional embeddings
        self._tie_or_clone_weights(self.classifier_model.distilbert.embeddings.position_embeddings,
                                   self.encoder.embeddings.position_embeddings)

    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        model_params = kwargs.pop('model_params', None)
        state_dict = kwargs.pop('state_dict', None)
        if os.path.isdir(pretrained_model_name_or_path):
            logger.warning("Loading model from a recursive state dict loading utility")
            return cls.from_pretrained_default(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                               config=config, model_params=model_params, state_dict=state_dict)
        model = cls(config, model_params)
        model.encoder = DistilBertModel.from_pretrained(pretrained_model_name_or_path)
        classifier_config = copy(config)
        classifier_config.output_attentions = True
        kwargs.update({"config": classifier_config})
        model.classifier_model = DistilBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path,
                                                                                     config=classifier_config)
        return model

class DistilBertHardGatedSentenceExplainer(HardGatedSentenceExplainer, DistilBertPreTrainedModel):
    def __init__(self, config, model_params=None):
        super(DistilBertHardGatedSentenceExplainer, self).__init__(config, model_params)
        self.encoder = DistilBertModel(config)
        self.classifier_model = DistilBertForSequenceClassification(config)

        self.init_weights()

    def tie_weights(self):
        # Tie word embeddings
        self._tie_or_clone_weights(self.classifier_model.distilbert.embeddings.word_embeddings,
                                   self.encoder.embeddings.word_embeddings)

        # Tie positional embeddings
        self._tie_or_clone_weights(self.classifier_model.distilbert.embeddings.position_embeddings,
                                   self.encoder.embeddings.position_embeddings)

    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        model_params = kwargs.pop('model_params', None)
        state_dict = kwargs.pop('state_dict', None)
        if os.path.isdir(pretrained_model_name_or_path):
            logger.warning("Loading model from a recursive state dict loading utility")
            return cls.from_pretrained_default(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                               config=config, model_params=model_params, state_dict=state_dict)
        model = cls(config, model_params)
        model.encoder = DistilBertModel.from_pretrained(pretrained_model_name_or_path)
        classifier_config = copy(config)
        classifier_config.output_attentions = True
        kwargs.update({"config": classifier_config})
        model.classifier_model = DistilBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path,
                                                                                     config=classifier_config)
        return model


# Bert classes
class BertGatedSentenceExplainer(GatedSentenceExplainer, BertPreTrainedModel):
    def __init__(self, config, model_params=None):
        super(BertGatedSentenceExplainer, self).__init__(config, model_params)
        self.encoder = BertModel(config)
        self.classifier_model = BertForSequenceClassification(config)

        self.init_weights()

    def tie_weights(self):
        # Tie word embeddings
        self._tie_or_clone_weights(self.classifier_model.bert.embeddings.word_embeddings,
                                   self.encoder.embeddings.word_embeddings)

        # Tie positional embeddings
        self._tie_or_clone_weights(self.classifier_model.bert.embeddings.position_embeddings,
                                   self.encoder.embeddings.position_embeddings)

    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        model_params = kwargs.pop('model_params', None)
        state_dict = kwargs.pop('state_dict', None)
        if os.path.isdir(pretrained_model_name_or_path):
            logger.warning("Loading model from a recursive state dict loading utility")
            return cls.from_pretrained_default(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                               config=config, model_params=model_params, state_dict=state_dict)
        model = cls(config, model_params)
        model.encoder = BertModel.from_pretrained(pretrained_model_name_or_path)
        classifier_config = copy(config)
        classifier_config.output_attentions = True
        kwargs.update({"config": classifier_config})
        model.classifier_model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path,
                                                                                     config=classifier_config)
        return model


class BertHardGatedSentenceExplainer(HardGatedSentenceExplainer, BertPreTrainedModel):
    def __init__(self, config, model_params=None):
        super(BertHardGatedSentenceExplainer, self).__init__(config, model_params)
        self.encoder = BertModel(config)
        self.classifier_model = BertForSequenceClassification(config)

        self.init_weights()

    def tie_weights(self):
        # Tie word embeddings
        self._tie_or_clone_weights(self.classifier_model.bert.embeddings.word_embeddings,
                                   self.encoder.embeddings.word_embeddings)

        # Tie positional embeddings
        self._tie_or_clone_weights(self.classifier_model.bert.embeddings.position_embeddings,
                                   self.encoder.embeddings.position_embeddings)

    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        model_params = kwargs.pop('model_params', None)
        state_dict = kwargs.pop('state_dict', None)
        if os.path.isdir(pretrained_model_name_or_path):
            logger.warning("Loading model from a recursive state dict loading utility")
            return cls.from_pretrained_default(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                               config=config, model_params=model_params, state_dict=state_dict)
        model = cls(config, model_params)
        model.encoder = BertModel.from_pretrained(pretrained_model_name_or_path)
        classifier_config = copy(config)
        classifier_config.output_attentions = True
        kwargs.update({"config": classifier_config})
        model.classifier_model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path,
                                                                                     config=classifier_config)
        return model


# Roberta classes
class RobertaGatedSentenceExplainer(GatedSentenceExplainer, BertPreTrainedModel):
    def __init__(self, config, model_params=None):
        super(RobertaGatedSentenceExplainer, self).__init__(config, model_params)
        self.encoder = RobertaModel(config)
        self.classifier_model = RobertaForSequenceClassification(config)

    def tie_weights(self):
        # Tie word embeddings
        self._tie_or_clone_weights(self.classifier_model.roberta.embeddings.word_embeddings,
                                   self.encoder.embeddings.word_embeddings)

        # Tie positional embeddings
        self._tie_or_clone_weights(self.classifier_model.roberta.embeddings.position_embeddings,
                                   self.encoder.embeddings.position_embeddings)

    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        model_params = kwargs.pop('model_params', None)
        state_dict = kwargs.pop('state_dict', None)
        if os.path.isdir(pretrained_model_name_or_path):
            logger.warning("Loading model from a recursive state dict loading utility")
            return cls.from_pretrained_default(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                               config=config, model_params=model_params, state_dict=state_dict)
        model = cls(config, model_params)
        model.encoder = RobertaModel.from_pretrained(pretrained_model_name_or_path)
        classifier_config = copy(config)
        classifier_config.output_attentions = True
        kwargs.update({"config": classifier_config})
        model.classifier_model = RobertaForSequenceClassification.from_pretrained(pretrained_model_name_or_path,
                                                                                     config=classifier_config)
        return model


class RobertaHardGatedSentenceExplainer(HardGatedSentenceExplainer, BertPreTrainedModel):
    def __init__(self, config, model_params=None):
        super(RobertaHardGatedSentenceExplainer, self).__init__(config, model_params)
        self.encoder = RobertaModel(config)
        self.classifier_model = RobertaForSequenceClassification(config)

    def tie_weights(self):
        # Tie word embeddings
        self._tie_or_clone_weights(self.classifier_model.roberta.embeddings.word_embeddings,
                                   self.encoder.embeddings.word_embeddings)

        # Tie positional embeddings
        self._tie_or_clone_weights(self.classifier_model.roberta.embeddings.position_embeddings,
                                   self.encoder.embeddings.position_embeddings)

    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        model_params = kwargs.pop('model_params', None)
        state_dict = kwargs.pop('state_dict', None)
        if os.path.isdir(pretrained_model_name_or_path):
            logger.warning("Loading model from a recursive state dict loading utility")
            return cls.from_pretrained_default(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                               config=config, model_params=model_params, state_dict=state_dict)
        model = cls(config, model_params)
        model.encoder = RobertaModel.from_pretrained(pretrained_model_name_or_path)
        classifier_config = copy(config)
        classifier_config.output_attentions = True
        kwargs.update({"config": classifier_config})
        model.classifier_model = RobertaForSequenceClassification.from_pretrained(pretrained_model_name_or_path,
                                                                                     config=classifier_config)
        return model
