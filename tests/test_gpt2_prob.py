from text_prob import GPT2Prob


def test_gpt2_prob():
    text = '人間 と 同 程度 に 言語 を 理解 する こと の できる 人工 知能 システム に ついて 研究 して い ます 。'
    prob = GPT2Prob(
        pretrained_model_name_or_path='nlp-waseda/gpt2-small-japanese',
    )
    assert prob(text) == 0.06074552610516548
