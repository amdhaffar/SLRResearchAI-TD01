from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from slrresearch.config import resolve_decoding
from slrresearch.research_questions import build_zero_shot_messages, generate_questions


class FakeModel:
    def __init__(self):
        self.calls = 0
        self.received = None

    def invoke(self, messages):
        self.calls += 1
        self.received = messages
        return SimpleNamespace(
            content="  RQ1. Comment évaluer les LLM pour le screening ?  ",
            usage_metadata={"input_tokens": 42, "output_tokens": 18},
            response_metadata={"finish_reason": "stop"},
        )


def test_prompt_systeme_et_utilisateur():
    topic = "LLM et screening des SLR"
    messages = build_zero_shot_messages(topic)
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert topic not in str(messages[0].content)
    assert topic in str(messages[1].content)
    for section in ("[OBJECTIF]", "[CONTEXTE]", "[ENTRÉE]",
                    "[CONTRAINTES]", "[SORTIE ATTENDUE]"):
        assert str(messages[1].content).count(section) == 1


def test_un_seul_appel_et_texte_libre():
    model = FakeModel()
    result = generate_questions(model, "LLM et SLR", "fake")
    assert model.calls == 1
    assert result.text.startswith("RQ1.")
    assert result.provider == "fake"
    assert result.input_tokens == 42
    assert result.finish_reason == "stop"


def test_sujet_vide_refuse_avant_appel():
    model = FakeModel()
    with pytest.raises(ValueError):
        generate_questions(model, " ")
    assert model.calls == 0


def test_greedy_et_sampling_donnent_des_parametres_distincts():
    greedy = resolve_decoding("greedy", 0.8, 0.9, 400)
    sampling = resolve_decoding("sampling", 0.8, 0.9, 400)
    assert greedy.temperature == 0 and greedy.top_k == 1
    assert sampling.temperature == 0.8 and sampling.top_p == 0.9

