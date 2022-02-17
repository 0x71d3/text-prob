import pytest
from text_prob import GPT2Prob


def test_gpt2_prob():
    text = '人間 と 同 程度 に 言語 を 理解 する こと の できる 人工 知能 システム に ついて 研究 して い ます 。'
    prob = GPT2Prob('nlp-waseda/gpt2-small-japanese-wikipedia')
    assert prob(text) == 0.04528525471687317


@pytest.mark.parametrize('use_mecab', [False, True])
def test_gpt2_prob_segmentation(use_mecab):
    text = '人間と同程度に言語を理解することのできる人工知能システムについて研究しています。'
    prob = GPT2Prob('nlp-waseda/gpt2-small-japanese-wikipedia', do_segmentation=True, use_mecab=use_mecab)
    assert prob(text) == 0.04528525471687317
