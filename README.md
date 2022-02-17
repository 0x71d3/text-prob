# text-prob

## Requirements

- Python 3.8.7

## Installation

```bash
git clone https://github.com/nlp-waseda/text-prob
cd text-prob
pip install --editable ./
```

## Examples

```python
>>> from text_prob import GPT2Prob

>>> text = '人間 と 同 程度 に 言語 を 理解 する こと の できる 人工 知能 システム に ついて 研究 して い ます 。'
>>> prob = GPT2Prob('nlp-waseda/gpt2-small-japanese-wikipedia')

>>> prob(text)
0.04528525471687317
```
