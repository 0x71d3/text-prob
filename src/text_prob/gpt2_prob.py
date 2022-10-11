from typing import Optional

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM


class GPT2Prob:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        do_segmentation: Optional[bool] = False,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
        )

        self.do_segmentation = do_segmentation
        if do_segmentation:
            try:
                import zenhan
            except ImportError:
                raise ImportError('zenhan is required for segmentation.')

            self.zenhan = zenhan

            try:
                from pyknp import Juman
            except ImportError:
                raise ImportError('PyKNP is required for segmentation.')
            
            self.jumanpp = Juman()

    def _segment_text(self, text: str) -> str:
        result = self.jumanpp.analysis(self.zenhan.h2z(text))
        segmented_text = ' '.join(mrph.midasi for mrph in result.mrph_list())
        return segmented_text

    def _calc_loss_from_logits(
        self,
        lm_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return loss

    def __call__(
        self,
        text: str,
        length_penalty: Optional[float] = 1.0,
        return_log_prob: Optional[bool] = False,
    ) -> float:
        if self.do_segmentation:
            text = self._segment_text(text)

        inputs  = self.tokenizer(text, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(**inputs)

        lm_logits = outputs.logits
        labels = inputs['input_ids']

        loss = self._calc_loss_from_logits(lm_logits, labels)

        log_prob = -loss.sum() / (loss.numel() ** length_penalty)
        if return_log_prob:
            return log_prob.item()

        prob = log_prob.exp()
        return prob.item()
