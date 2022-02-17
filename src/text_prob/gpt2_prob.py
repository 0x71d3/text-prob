from typing import Optional

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM


class GPT2Prob:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        do_segmentation: Optional[bool] = False,
        use_mecab: Optional[bool] = False
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path
        )

        self.do_segmentation = do_segmentation
        self.use_mecab = use_mecab

        if do_segmentation:
            if use_mecab:
                try:
                    from fugashi import GenericTagger

                except ImportError:
                    raise ImportError('fugashi is required for segmentation.')

                try:
                    import jumandic
                
                except ImportError:
                    raise ImportError('JumanDIC is required for segmentation.')

                self.tagger = GenericTagger('-Owakati ' + jumandic.MECAB_ARGS)

            else:
                try:
                    import zenhan
                
                except ImportError:
                    raise ImportError('zenhan is required for normalization.')
                
                self.zenhan = zenhan

                try:
                    from pyknp import Juman
                
                except ImportError:
                    raise ImportError('PyKNP is required for segmentation.')
                
                self.jumanpp = Juman()

    def _segment(self, text: str) -> str:
        if self.use_mecab:
            segment_text = self.tagger.parse(text)

        else:
            result = self.jumanpp.analysis(self.zenhan.h2z(text))
            segment_text = ' '.join(mrph.midasi for mrph in result.mrph_list())
        return segment_text

    def _get_loss(
        self,
        lm_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return loss
    
    def _get_log_prob(
        self,
        loss: torch.Tensor,
        length_penalty: float
    ) -> torch.Tensor:
        log_prob = -loss.sum() / (loss.numel() ** length_penalty)
        return log_prob

    def __call__(
        self,
        text: str,
        return_log_prob: Optional[bool] = False,
        length_penalty: Optional[float] = 1.0
    ) -> float:
        if self.do_segmentation:
            text = self._segment(text)

        inputs  = self.tokenizer(text, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(**inputs)

        lm_logits = outputs.logits
        labels = inputs['input_ids']

        loss = self._get_loss(lm_logits, labels)
        log_prob = self._get_log_prob(loss, length_penalty)

        if return_log_prob:
            return log_prob.item()

        prob = log_prob.exp()
        return prob.item()
