'''
This file contains models that do categorical k-subset selection over constituents/sentences
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from modeling_distilbert import (DistilBertPreTrainedModel, DistilBertForSequenceClassification)
from transformers import DistilBertModel
from modeling_bert import (BertPreTrainedModel, BertModel, \
                           BertForSequenceClassification)
from transformers import WEIGHTS_NAME
from ib_utils import idxtobool
from copy import deepcopy as copy
import pdb,os
import logging
import math
from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)
EPS = 1e-16
'''
It is not clear how masking will affect the performance of models for QA since the Q part, 
even in unsupervised settings should not be masked, so there is defintitely a need for required input mask.
Some tokens have to always be turned on even when sampling * not easy to achieve
'''

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



class DistilBertSentenceClassifier(DistilBertPreTrainedModel):
    def __init__(self, config, model_params=None):
        super(DistilBertSentenceClassifier, self).__init__(config)

        # Check and see if you can share parameters i.e. a single encoder architecture over
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(p=0.2)

        self.fixed_mask_length = model_params["max_query_len"] - 1  # The first SEP take a different mask weight

        self.explainer_model = nn.Sequential(
            nn.Linear(2 * config.dim, config.dim),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(config.dim, 1),
        )

        # This is included just so that the model can be easily loaded for warm starting
        self.classifier_model = DistilBertForSequenceClassification(config)
        self.init_weights()

    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        model_params = kwargs.pop('model_params', None)
        state_dict = kwargs.pop('state_dict', None)
        if os.path.isdir(pretrained_model_name_or_path):
            logger.warning("Loading model from a recursive state dict loading utility")
            return cls.from_pretrained_default(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                               config=config, model_params=model_params, state_dict=state_dict)
        model = cls(config, model_params)
        model.distilbert = DistilBertModel.from_pretrained(pretrained_model_name_or_path)
        return model

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
        if not hasattr(model, cls.base_model_prefix) and any(
                s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            start_prefix = cls.base_model_prefix + '.'
        if hasattr(model, cls.base_model_prefix) and not any(
                s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            model_to_load = getattr(model, cls.base_model_prefix)
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

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, p_mask=None,
                inputs_embeds=None, labels=None, num_sample=1, evaluate=False, sentence_starts=None,
                sentence_ends=None, sentence_mask=None, evidence_labels=None):

        batch_size = input_ids.shape[0]
        outputs = self.distilbert(input_ids,
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

        logits = self.explainer_model(sentence_representations).squeeze(-1)

        outputs = (logits,)
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            class_loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if sentence_mask is not None:
                active_loss = sentence_mask.view(-1) == 1
                active_logits = logits.view(-1)[active_loss]
                active_labels = evidence_labels.float().view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
                z_hat = torch.sigmoid(logits)*sentence_mask
                z_hat = z_hat.squeeze(1).gather(dim=1, index=p_mask)

                # reweighing each token and making the weights as close to one hot as possible
                fixed_mask = torch.ones((z_hat.shape[0], self.fixed_mask_length), dtype=torch.float,
                                        device=z_hat.device)
                z_hat = torch.cat((fixed_mask, z_hat), dim=-1) * attention_mask.float()
                z_hat = z_hat + EPS
                classification_output = self.classifier_model(input_ids=input_ids,
                                                              attention_mask=z_hat,
                                                              labels=labels)
                # I can do a bert to bert pipeline or I could just use losses at different levels and weigh them?
                class_loss = classification_output[0]
                classification_logits = classification_output[1]
                loss += class_loss
            else:
                # Ignored since mask is always defined
                loss = loss_fct(logits.view(-1), evidence_labels.float().view(-1))
            outputs = outputs + (classification_logits,)
            outputs = (loss,) + outputs
        return outputs


# The Seojin Bang incorrect categorical implementation for sentences
class DistilBertSentenceExplainer(DistilBertPreTrainedModel):
    def __init__(self, config, model_params=None):
        super(DistilBertSentenceExplainer, self).__init__(config)

        # Check and see if you can share parameters i.e. a single encoder architecture over
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(p=0.2)

        self.chunk_size = model_params["chunk_size"]
        self.tau = model_params["tau"]
        self.K = model_params["K"]
        self.beta = model_params["beta"]
        # self.fixed_mask_length = 64
        self.fixed_mask_length = model_params["max_query_len"] - 1  # The first SEP take a different mask weight

        self.explainer_model = nn.Sequential(
            nn.Linear(2 * config.dim, config.dim),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(config.dim, 1),
        )

        self.classifier_model = DistilBertForSequenceClassification(config)

        self.info_criterion = nn.KLDivLoss(reduction='sum')

        # This function already calls tie weight function
        self.init_weights()

    def tie_weights(self):
        # Tie word embeddings
        self._tie_or_clone_weights(self.classifier_model.distilbert.embeddings.word_embeddings,
                                   self.distilbert.embeddings.word_embeddings)

        # Tie positional embeddings
        self._tie_or_clone_weights(self.classifier_model.distilbert.embeddings.position_embeddings,
                                   self.distilbert.embeddings.position_embeddings)

    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        model_params = kwargs.pop('model_params', None)
        state_dict = kwargs.pop('state_dict', None)
        if os.path.isdir(pretrained_model_name_or_path):
            logger.warning("Loading model from a recursive state dict loading utility")
            return cls.from_pretrained_default(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                               config=config, model_params=model_params, state_dict=state_dict)
        if model_params["warm_start"]:
            classifier_config = copy(config)
            model = cls.from_pretrained_default(pretrained_model_name_or_path=model_params["warm_start_model_name_or_path"], 
                                               config=config, model_params=model_params, state_dict=state_dict)
            model.classifier_model = DistilBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, config=classifier_config)
            return model
        model = cls(config, model_params)
        model.distilbert = DistilBertModel.from_pretrained(pretrained_model_name_or_path)
        classifier_config = copy(config)
        classifier_config.output_attentions = True
        kwargs.update({"config": classifier_config})
        model.classifier_model = DistilBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path,
                                                                                     config=classifier_config)
        return model

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
        if not hasattr(model, cls.base_model_prefix) and any(
                s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            start_prefix = cls.base_model_prefix + '.'
        if hasattr(model, cls.base_model_prefix) and not any(
                s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            model_to_load = getattr(model, cls.base_model_prefix)
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

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, p_mask=None,
                inputs_embeds=None, labels=None, num_sample=1, evaluate=False, sentence_starts=None,
                sentence_ends=None, sentence_mask=None, evidence_labels=None):

        batch_size = input_ids.shape[0]
        outputs = self.distilbert(input_ids,
                                  attention_mask=attention_mask,
                                  head_mask=head_mask,
                                  inputs_embeds=inputs_embeds)

        # Add dropout over encoder
        sequence_output = outputs[0]
        # Original attention mask that ignores all the padded tokens is reinforced
        sequence_output = sequence_output * attention_mask.unsqueeze(-1).float()
        sequence_output = self.dropout(sequence_output)

        # Chunk enumeration only for the portion which can be variably masked

        # Enumerate all possible chunks using their start and end indices
        # seq_len = input_ids.shape[1] - self.fixed_mask_length
        # # Number of chunks created : seeq_len - chunk_size + 1
        # candidate_starts = torch.tensor(range(seq_len - self.chunk_size + 1), dtype=torch.long, device=input_ids.device)
        # candidate_ends = candidate_starts + self.chunk_size - 1
        # candidate_starts += self.fixed_mask_length
        # candidate_ends += self.fixed_mask_length
        # # TODO: There are many variations like max and average pooling over spans
        # chunk_representations = torch.cat((torch.index_select(sequence_output, 1, candidate_starts), \
        #                                    torch.index_select(sequence_output, 1, candidate_ends)), dim=-1)
        # chunk_mask = (torch.index_select(attention_mask, 1, candidate_starts) + \
        #               torch.index_select(attention_mask, 1, candidate_ends) > 0).long()

        # sentence representations are collected using index select
        sentence_rep_shape = (sequence_output.shape[0], sentence_starts.shape[1], sequence_output.shape[-1])
        sentence_representations = torch.cat((sequence_output.gather(dim=1, index=sentence_starts.unsqueeze(-1).expand(sentence_rep_shape)), \
                                   sequence_output.gather(dim=1, index=sentence_ends.unsqueeze(-1).expand(sentence_rep_shape))), dim=-1)

        # TODO: Apply an attention mask over this sampler: This is what breaks the assumption of a fixed number of relevant features
        # Convincing argument : p(z|x) when x has some bad/useless features will have a predecided structure
        # Pass the cadidates over to the classifier
        log_p_i = masked_log_softmax(self.explainer_model(sentence_representations).squeeze(-1), sentence_mask, dim=1)

        # Carry out reparameterization
        Z_hat, Z_hat_fixed = self.reparameterize(log_p_i, tau=self.tau, k=self.K, num_sample=num_sample)

        # returns torch.Size([batch-size, num-samples for multishot prediction, d])
        # Now that Z_hat has been created put back the fixed mask that you need

        # Output classifier
        class_loss, class_logit = self.classify(input_ids, attention_mask, p_mask, Z_hat, labels, num_sample)
        # logit_fixed = self.classify(input_ids, attention_mask, p_mask, Z_hat_fixed, labels)
        # Call another classify only during evaluation since batch has to fit into memory
        if evaluate:
            # Z_hat_fixed = torch.exp(log_p_i).unsqueeze(1)
            _, class_fixed_logits = self.classify(input_ids, attention_mask, p_mask, Z_hat_fixed, labels)
        else:
            class_fixed_logits = class_logit
        # Compute loss (since calling function needs to directly receive the loss function itself?)
        if labels is not None:
            p_i_prior = self.prior(var_size=log_p_i.size(), device=log_p_i.device)
            # TODO : is batch size division required for class_loss (mistake while copying over from L2X code)
            # TODO: Check this formulation why multiplication by self.K is important ?
            # info_loss = self.K * self.info_criterion(torch.log(p_i_prior), torch.exp(log_p_i)) / batch_size
            info_loss = self.info_criterion(torch.log(p_i_prior), torch.exp(log_p_i)) / batch_size
            # class_loss = class_loss.div(math.log(2))
            total_loss = class_loss + self.beta * info_loss
            return total_loss, class_loss, info_loss, class_logit, log_p_i, Z_hat
        else:
            # TODO: Think shouldn't you be sampling here?
            # Find top - K chunks and translate them back into a binary mask over tokens
            _, index_chunk = log_p_i.topk(self.K, dim=-1) # this is exactly the operation that got us fixed class logits
            # logic of how these chunks actually come from just part of the input
            # newadd = torch.tensor(range(self.chunk_size), dtype=torch.long, device=log_p_i.device) \
            #     .unsqueeze(0).unsqueeze(0).unsqueeze(0)
            # new_size_col = candidate_starts.shape[0]
            # rationale_idx = torch.add(index_chunk, torch.mul(torch.div(index_chunk, new_size_col), self.chunk_size - 1))
            # rationale_idx = torch.add(rationale_idx.unsqueeze(-1).expand(-1, -1, -1, self.chunk_size), newadd)
            # newsize = rationale_idx.size()
            # rationale_idx = rationale_idx.view(newsize[0], newsize[1], -1, 1).squeeze(-1)
            # rationale_idx = rationale_idx.squeeze(1)
            # rationale_idx += self.fixed_mask_length


            # you may not need to translate this into the chunk that was selected (in token space since we can retireve sent from index)
            rationale_idx = index_chunk
            class_logit = class_fixed_logits
            return class_logit, rationale_idx, log_p_i, Z_hat

    def classify(self, input_ids, attention_mask, p_mask, z_hat, labels=None, num_sample=1):
        # Resize z_hat accordingly so that is it once again batch_size * num_sentences * num_words_per_sentence
        # Apply mask to the encoder

        # Apply a required mask where CLS token is not considered

        # Apply a different instance of BERT once again on this masked version and predict class logits

        # Compute label loss (since labels are available at train time)
        # z_hat = nn.Sequential(
        #     nn.ConstantPad1d(self.chunk_size - 1, 0),
        #     nn.MaxPool1d(kernel_size=self.chunk_size, stride=1, padding=0)
        # )(z_hat)

        # TODO: Squeezing works only if num_samples = 1
        z_hat = z_hat.squeeze(1).gather(dim=1, index=p_mask)
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
        p_i_ = p_i.view(p_i.size(0), 1, 1, -1)
        p_i_ = p_i_.expand(p_i_.size(0), num_sample, k, p_i_.size(-1))
        C_dist = RelaxedOneHotCategorical(tau, logits=p_i_)
        V = torch.max(C_dist.sample(), -2)[0]  # [batch-size, multi-shot, d]

        ## without sampling
        V_fixed_size = p_i.unsqueeze(1).size()
        _, V_fixed_idx = p_i.unsqueeze(1).topk(k, dim=-1)  # batch * 1 * k
        V_fixed = idxtobool(V_fixed_idx, V_fixed_size, device=p_i.device)
        V_fixed = V_fixed.type(torch.float)

        # unlike V which is still soft weight matrix, V_fixed is a binary mask, so it should probably being doing horribly
        return V, V_fixed

    def prior(self, var_size, device):
        # TODO: prior will be influenced by the actualy sparsity for the dataset?
        p = torch.ones(var_size[1], device=device)
        p = p / var_size[1]
        p = p.view(1, var_size[1])
        p_prior = p.expand(var_size)  # [batch-size, k, feature dim]

        return p_prior


# The Seojin Bang incorrect categorical implementation
class DistilBertExplainer(DistilBertPreTrainedModel):
    def __init__(self, config, model_params=None):
        super(DistilBertExplainer, self).__init__(config)

        # Check and see if you can share parameters i.e. a single encoder architecture over
        self.distilbert = DistilBertModel(config)
        # self.distilbert_classifier = DistilBertModel(config)
        self.dropout = nn.Dropout(p=0.2)

        self.chunk_size = model_params["chunk_size"]
        self.tau = model_params["tau"]
        self.K = model_params["K"]
        self.beta = model_params["beta"]
        self.num_samples = model_params["num_avg"]
        self.fixed_mask_length = model_params["max_query_len"]

        # Could eventually be replaced with DistilBertForTokenClassification
        self.explainer_model = nn.Sequential(
            nn.Linear(2*config.dim, config.dim),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(config.dim, 1),
            # Which dimension you are working to improve
            # TODO: Are you independently scoring each with a Sigmoid or are you scoring all of them relatively
            # with a softmax (similar to the argument made in my BOW training obj paper)
        )

        # Even here, we will have to enforce that the CLS token is never masked out
        # Config will have to be altered for changing number of classes
        self.classifier_model = DistilBertForSequenceClassification(config)
        # self.classifier_model = nn.Sequential(
        #     nn.Linear(config.dim, config.dim),
        #     nn.ReLU(True),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(config.dim, config.num_labels)
        # )

        self.info_criterion = nn.KLDivLoss(reduction='sum')

        # This function already calls tie weight function
        self.init_weights()

    def tie_weights(self):
        # Tie word embeddings
        self._tie_or_clone_weights(self.classifier_model.distilbert.embeddings.word_embeddings,
                                        self.distilbert.embeddings.word_embeddings)

        # Tie positional embeddings
        self._tie_or_clone_weights(self.classifier_model.distilbert.embeddings.position_embeddings,
                                        self.distilbert.embeddings.position_embeddings)

    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        model_params = kwargs.pop('model_params', None)
        state_dict = kwargs.pop('state_dict', None)
        if os.path.isdir(pretrained_model_name_or_path):
            logger.warning("Loading model from a recursive state dict loading utility")
            return cls.from_pretrained_default(pretrained_model_name_or_path=pretrained_model_name_or_path, config=config, model_params=model_params, state_dict=state_dict)
        model = cls(config, model_params)
        model.distilbert = DistilBertModel.from_pretrained(pretrained_model_name_or_path)
        classifier_config = copy(config)
        classifier_config.output_attentions = True
        kwargs.update({"config" : classifier_config})
        model.classifier_model = DistilBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, config=classifier_config)
        return model
    
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
        if not hasattr(model, cls.base_model_prefix) and any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            start_prefix = cls.base_model_prefix + '.'
        if hasattr(model, cls.base_model_prefix) and not any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            model_to_load = getattr(model, cls.base_model_prefix)
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

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, p_mask=None,
                inputs_embeds=None, labels=None, num_sample = 1, evaluate=False):
        # Call explain model
        # Call reparameterize
        # Add essential input mask
        # Call classify model, which returns label loss
        # Compute classification and IB loss and return
        batch_size = input_ids.shape[0]
        outputs = self.distilbert(input_ids,
                            attention_mask=attention_mask,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        # Add dropout over encoder
        sequence_output = outputs[0]
        sequence_output = sequence_output * attention_mask.unsqueeze(-1).float()
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

        # TODO: Apply an attention mask over this sampler: This is what breaks the assumption of a fixed number of relevant features
        # Convincing argument : p(z|x) when x has some bad/useless features will have a predecided structure
        # Pass the cadidates over to the classifier
        log_p_i = masked_log_softmax(self.explainer_model(chunk_representations).squeeze(-1), chunk_mask, dim=1)

        # Carry out reparameterization
        Z_hat, Z_hat_fixed = self.reparameterize(log_p_i, tau=self.tau, k=self.K, num_sample=num_sample)

        # returns torch.Size([batch-size, num-samples for multishot prediction, d])
        # Now that Z_hat has been created put back the fixed mask that you need

        # Output classifier
        class_loss, class_logit = self.classify(input_ids, attention_mask, p_mask, Z_hat, labels, num_sample)
        # logit_fixed = self.classify(input_ids, attention_mask, p_mask, Z_hat_fixed, labels)
        # Call another classify only during evaluation since batch has to fit into memory
        if evaluate:
            Z_hat_fixed = torch.exp(log_p_i).unsqueeze(1)
            # since you dont have to draw a differentiable sample maybe sample from the multinomial?
            # we want to see how useful the distributionis with different beta
            _, class_fixed_logits = self.classify(input_ids, attention_mask, p_mask, Z_hat_fixed, labels)
        else:
            class_fixed_logits = class_logit
        # Compute loss (since calling function needs to directly receive the loss function itself?)
        if labels is not None:
            p_i_prior = self.prior(var_size=log_p_i.size(), device=log_p_i.device)
            # TODO : is batch size division required for class_loss (mistake while copying over)
            # TODO: Check this formulation why multiplication by self.K is important ?
            # info_loss = self.K * self.info_criterion(torch.log(p_i_prior), torch.exp(log_p_i)) / batch_size
            info_loss = self.info_criterion(torch.log(p_i_prior), torch.exp(log_p_i)) / batch_size
            # class_loss = class_loss.div(math.log(2))
            total_loss = class_loss + self.beta * info_loss
            return total_loss, class_loss, info_loss, class_logit, log_p_i, Z_hat
        else:
            # Find top - K chunks and translate them back into a binary mask over tokens
            _, index_chunk = log_p_i.unsqueeze(1).topk(self.K, dim=-1) # this is exactly the operation that got us fixed class logits
            # logic of how these chunks actually come from just part of the input
            newadd = torch.tensor(range(self.chunk_size), dtype=torch.long, device=log_p_i.device)\
                .unsqueeze(0).unsqueeze(0).unsqueeze(0)
            new_size_col = candidate_starts.shape[0]
            rationale_idx = torch.add(index_chunk, torch.mul(torch.div(index_chunk, new_size_col), self.chunk_size - 1))
            rationale_idx = torch.add(rationale_idx.unsqueeze(-1).expand(-1, -1, -1, self.chunk_size), newadd)
            newsize = rationale_idx.size()
            # rationale_idx = rationale_idx.view(newsize[0], newsize[1], -1, 1).squeeze(-1)
            rationale_idx = rationale_idx.squeeze(1)
            rationale_idx += self.fixed_mask_length
            class_logit = class_fixed_logits
            return class_logit, rationale_idx, log_p_i, Z_hat


    def classify(self, input_ids, attention_mask, p_mask, z_hat, labels=None, num_sample = 1):
        # Resize z_hat accordingly so that is it once again batch_size * num_sentences * num_words_per_sentence
        # Apply mask to the encoder

        # Apply a required mask where CLS token is not considered

        # Apply a different instance of BERT once again on this masked version and predict class logits

        # Compute label loss (since labels are available at train time)
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
            #attention_weights = classification_output[2]
        else:
            classification_loss = 0
            logits = classification_output[0]
            #attention_weights = classification_output[1]
        return classification_loss, logits

    def reparameterize(self, p_i, tau, k, num_sample = 1):

        # Generating k-hot samples  *
        # TODO: With or without replacement

        ## sampling
        p_i_ = p_i.view(p_i.size(0), 1, 1, -1)
        p_i_ = p_i_.expand(p_i_.size(0), num_sample, k, p_i_.size(-1))
        C_dist = RelaxedOneHotCategorical(tau, logits=p_i_)
        V = torch.max(C_dist.sample(), -2)[0] # [batch-size, multi-shot, d]

        ## without sampling
        V_fixed_size = p_i.unsqueeze(1).size()
        _, V_fixed_idx = p_i.unsqueeze(1).topk(k, dim = -1) # batch * 1 * k
        V_fixed = idxtobool(V_fixed_idx, V_fixed_size, device=p_i.device)
        V_fixed = V_fixed.type(torch.float)

        # unlike V which is still soft weight matrix, V_fixed is a binary mask, so it should probably being doing horribly
        return V, V_fixed

    def prior(self, var_size, device):
        # TODO: prior will be influenced by the actualy sparsity for the dataset?
        p = torch.ones(var_size[1], device=device)
        p = p/var_size[1]
        p = p.view(1, var_size[1])
        p_prior = p.expand(var_size)  # [batch-size, k, feature dim]

        return p_prior


# BERT Based Explainer when eventually we move towards that;
class BertExplainer(BertPreTrainedModel):
    def __init__(self, config, model_params=None):
        super(BertExplainer, self).__init__(config)

        # Check and see if you can share parameters i.e. a single encoder architecture over
        self.bert = BertModel(config)
        # self.distilbert_classifier = DistilBertModel(config)
        self.dropout = nn.Dropout(p=0.2)

        self.chunk_size = model_params["chunk_size"]
        self.tau = model_params["tau"]
        self.K = model_params["K"]
        self.beta = model_params["beta"]

        # Could eventually be replaced with DistilBertForTokenClassification
        self.explainer_model = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(config.hidden_size, 1),
            # Which dimension you are working to improve
            # TODO: Are you independently scoring each with a Sigmoid or are you scoring all of them relatively
            # with a softmax (similar to the argument made in my BOW training obj paper)
            nn.LogSoftmax(1)
        )

        # Even here, we will have to enforce that the CLS token is never masked out
        # Config will have to be altered for changing number of classes
        self.classifier_model = BertForSequenceClassification(config)
        # self.classifier_model = nn.Sequential(
        #     nn.Linear(config.dim, config.dim),
        #     nn.ReLU(True),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(config.dim, config.num_labels)
        # )

        self.info_criterion = nn.KLDivLoss(reduction='sum')

        # This function already calls tie weight function
        self.init_weights()

    def tie_weights(self):
        # Tie word embeddings
        self._tie_or_clone_weights(self.classifier_model.bert.embeddings.word_embeddings,
                                   self.bert.embeddings.word_embeddings)

        # Tie positional embeddings
        self._tie_or_clone_weights(self.classifier_model.bert.embeddings.position_embeddings,
                                   self.bert.embeddings.position_embeddings)

    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        model_params = kwargs.pop('model_params', None)
        state_dict = kwargs.pop('state_dict', None)
        if os.path.isdir(pretrained_model_name_or_path):
            logger.warning("Loading model from a recursive state dict loading utility")
            return cls.from_pretrained_default(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                               config=config, model_params=model_params, state_dict=state_dict)
        model = cls(config, model_params)
        model.distilbert = BertModel.from_pretrained(pretrained_model_name_or_path)
        classifier_config = copy(config)
        classifier_config.output_attentions = True
        kwargs.update({"config": classifier_config})
        model.classifier_model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path,
                                                                                     config=classifier_config)
        return model

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
        if not hasattr(model, cls.base_model_prefix) and any(
                s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            start_prefix = cls.base_model_prefix + '.'
        if hasattr(model, cls.base_model_prefix) and not any(
                s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            model_to_load = getattr(model, cls.base_model_prefix)
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

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, p_mask=None,
                inputs_embeds=None, labels=None, num_sample=1):
        # Call explain model
        # Call reparameterize
        # Add essential input mask
        # Call classify model, which returns label loss
        # Compute classification and IB loss and return
        batch_size = input_ids.shape[0]
        outputs = self.bert(input_ids,
                                  attention_mask=attention_mask,
                                  head_mask=head_mask,
                                  inputs_embeds=inputs_embeds)

        # Add dropout over encoder
        sequence_output = outputs[0]
        sequence_output = sequence_output * attention_mask.unsqueeze(-1)
        sequence_output = self.dropout(sequence_output)

        # Enumerate all possible chunks using their start and end indices
        seq_len = input_ids.shape[1]
        # Number of chunks created : seeq_len - chunk_size + 1
        candidate_starts = torch.tensor(range(seq_len - self.chunk_size + 1), dtype=torch.long, device=input_ids.device)
        candidate_ends = candidate_starts + self.chunk_size - 1
        chunk_representations = torch.cat((torch.index_select(sequence_output, 1, candidate_starts), \
                                           torch.index_select(sequence_output, 1, candidate_ends)), dim=-1)
        # Pass the cadidates over to the classifier
        log_p_i = self.explainer_model(chunk_representations).squeeze(-1)

        # Carry out reparameterization
        Z_hat, Z_hat_fixed = self.reparameterize(log_p_i, tau=self.tau, k=self.K, num_sample=num_sample)

        # returns torch.Size([batch-size, num-samples for multishot prediction, d])

        # Output classifier
        class_loss, class_logit = self.classify(input_ids, attention_mask, p_mask, Z_hat, labels, num_sample)
        # Compute loss (since calling function needs to directly receive the loss function itself?)
        if labels is not None:
            p_i_prior = self.prior(var_size=log_p_i.size(), device=log_p_i.device)
            # TODO : is batch size division required for class_loss
            # TODO: Check this formulation once again
            info_loss = self.K * self.info_criterion(torch.log(p_i_prior), torch.exp(log_p_i)) / batch_size
            # TODO: May be needed to ensure that we are still deadline with nats?
            # class_loss = class_loss.div(math.log(2))
            total_loss = class_loss + self.beta * info_loss
            return total_loss, class_loss, info_loss, class_logit, log_p_i, Z_hat
        else:
            # Find top - K chunks and translate them back into a binary mask over tokens
            _, index_chunk = log_p_i.unsqueeze(1).topk(self.K, dim=-1)
            newadd = torch.tensor(range(self.chunk_size), dtype=torch.long, device=log_p_i.device) \
                .unsqueeze(0).unsqueeze(0).unsqueeze(0)
            new_size_col = input_ids.shape[1] - self.chunk_size + 1
            rationale_idx = torch.add(index_chunk, torch.mul(torch.div(index_chunk, new_size_col), self.chunk_size - 1))
            rationale_idx = torch.add(rationale_idx.unsqueeze(-1).expand(-1, -1, -1, self.chunk_size), newadd)
            newsize = rationale_idx.size()
            rationale_idx = rationale_idx.view(newsize[0], newsize[1], -1, 1).squeeze(-1)
            rationale_idx = rationale_idx.squeeze(1)
            # return the fixed version
            return class_logit, rationale_idx, log_p_i, Z_hat

    def classify(self, input_ids, attention_mask, p_mask, z_hat, labels=None, num_sample=1):
        # Resize z_hat accordingly so that is it once again batch_size * num_sentences * num_words_per_sentence
        # Apply mask to the encoder

        # Apply a required mask where CLS token is not considered

        # Apply a different instance of BERT once again on this masked version and predict class logits

        # Compute label loss (since labels are available at train time)
        z_hat = nn.Sequential(
            nn.ConstantPad1d(self.chunk_size - 1, 0),
            nn.MaxPool1d(kernel_size=self.chunk_size, stride=1, padding=0)
        )(z_hat)

        # TODO: Squeezing works only if num_samples = 1
        z_hat = z_hat.squeeze(1)
        # reweighing each token and making the weights as close to one hot as possible

        # TODO: Another catch is that when the sampling is being done the attention masked positions should
        # be disregarded
        z_hat = z_hat * attention_mask  # + (1-p_mask)

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
        else:
            classification_loss = 0
            logits = classification_output[0]
        return classification_loss, logits

    def reparameterize(self, p_i, tau, k, num_sample=1):

        # Generating k-hot samples

        ## sampling
        p_i_ = p_i.view(p_i.size(0), 1, 1, -1)
        p_i_ = p_i_.expand(p_i_.size(0), num_sample, k, p_i_.size(-1))
        C_dist = RelaxedOneHotCategorical(tau, logits=p_i_)
        V = torch.max(C_dist.sample(), -2)[0]  # [batch-size, multi-shot, d]

        ## without sampling
        V_fixed_size = p_i.unsqueeze(1).size()
        _, V_fixed_idx = p_i.unsqueeze(1).topk(k, dim=-1)  # batch * 1 * k
        V_fixed = idxtobool(V_fixed_idx, V_fixed_size, device=p_i.device)
        V_fixed = V_fixed.type(torch.float)

        return V, V_fixed

    def prior(self, var_size, device):
        # TODO: prior will be influenced by the actualy sparsity for the dataset?
        p = torch.ones(var_size[1], device=device)
        p = p / var_size[1]
        p = p.view(1, var_size[1])
        p_prior = p.expand(var_size)  # [batch-size, k, feature dim]

        return p_prior
