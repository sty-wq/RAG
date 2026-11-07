"""Qwen's official yes/no causal-LM scoring protocol, without generation."""
from __future__ import annotations

from common import check

MODELS = {
    '0.6b': ('Qwen/Qwen3-Reranker-0.6B', 'e61197ed45024b0ed8a2d74b80b4d909f1255473'),
    '4b': ('Qwen/Qwen3-Reranker-4B', '22e683669bc0f0bd69640a1354a6d0aebcfeede5'),
}
INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
          'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
          '<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def format_pair(query, document):
    return f'<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {document}'


def encode_pairs(tokenizer, query, documents, max_length):
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    budget = max_length - len(prefix) - len(suffix)
    check(budget > 0, 'max_length 太小，放不下评分模板')
    query_tokens = tokenizer.encode(format_pair(query, ''), add_special_tokens=False)
    check(len(query_tokens) < budget, '问题和指令已经超长，请提高 max_length')
    bodies = tokenizer([format_pair(query, d) for d in documents], add_special_tokens=False,
                       padding=False, truncation=False, return_attention_mask=False)['input_ids']
    lengths = [len(prefix) + len(b) + len(suffix) for b in bodies]
    # Right truncation of the formatted body matches the official example.
    # Preserve the system prefix and the complete assistant suffix.
    sequences = [prefix + b[:budget] + suffix for b in bodies]
    return sequences, lengths


class QwenScorer:
    def __init__(self, model_key, max_length=2048, batch_size=4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        check(torch.cuda.is_available(), '没有检测到 CUDA GPU')
        check(torch.cuda.is_bf16_supported(), 'GPU 不支持 BF16')
        self.max_length, self.batch_size = max_length, batch_size
        model_id, revision = MODELS[model_key]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side='left', trust_remote_code=False)
        self.tokenizer.truncation_side = 'right'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.yes, self.no = [self.tokenizer.convert_tokens_to_ids(s) for s in ('yes', 'no')]
        check(self.tokenizer.encode('yes', add_special_tokens=False) == [self.yes], 'yes 不是单 token')
        check(self.tokenizer.encode('no', add_special_tokens=False) == [self.no], 'no 不是单 token')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=torch.bfloat16, use_safetensors=True,
            attn_implementation='sdpa', trust_remote_code=False).to('cuda:0').eval()

    def score(self, query, documents):
        torch = self.torch
        sequences, lengths = encode_pairs(self.tokenizer, query, documents, self.max_length)
        scores = []
        with torch.inference_mode():
            for start in range(0, len(sequences), self.batch_size):
                inputs = self.tokenizer.pad({'input_ids': sequences[start:start + self.batch_size]},
                                            padding=True, return_tensors='pt').to('cuda:0')
                logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
                # sigmoid(yes-no) equals softmax([no,yes])[yes]. Rank raw logits
                # to avoid 0/1 saturation creating artificial ties.
                scores.extend((logits[:, self.yes] - logits[:, self.no]).cpu().tolist())
        return scores, lengths
