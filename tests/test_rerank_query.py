"""召回与重排问题分离；防止 Qasper 标题修复把 CRUD 难题换成易题。"""
from contextlib import nullcontext
from dataclasses import replace
import json

import pytest

from evaluation.run_retrieval import evaluate, main, resolve_rerank_query
from retrieval import RetrievalConfig, retrieve
from vector_store import SearchHit


QUESTION = 'In the paper "EEG recognition", what EEG features were used?'
ORIGINAL = 'what EEG features were used?'


class Store:
    def __init__(self):
        self.calls = []
        self.hits = [SearchHit(0.9, 'abstract', 'paper.md', '/paper.md', 'EEG recognition > Abstract',
                               0, 'We use EEG features.'),
                     SearchHit(0.8, 'evidence', 'paper.md', '/paper.md', 'EEG recognition > Features',
                               1, 'Root mean square and zero crossing rate.')]

    def read_meta(self):
        return {'embed_model': 'test', 'chunk_count': 2}

    def search(self, vector, k, **kwargs):
        self.calls.append(('dense', vector, k, kwargs))
        return self.hits[:k]

    def lexical_search(self, question, k, **kwargs):
        self.calls.append(('bm25', question, k, kwargs))
        return [replace(h, score_type='bm25') for h in self.hits[:k]]


class Embedder:
    model, base_url = 'test', 'http://example.test'

    def __init__(self):
        self.inputs = []

    def embed(self, questions):
        self.inputs.append(questions)
        return [[0.25] for _ in questions]


class Scorer:
    model, revision = 'controlled', 'test-revision'

    def __init__(self):
        self.inputs = []
        self.loaded = False

    def warmup(self):
        self.loaded = True

    def score(self, question, documents):
        self.inputs.append((question, documents))
        return [0.1, 0.9] if question == ORIGINAL else [0.9, 0.1]

    def describe(self):
        return {'backend': 'test', 'model': 'controlled'}


def question(**kwargs):
    return {'qid': 'q1', 'type': 'answerable', 'question': QUESTION, 'question_original': ORIGINAL,
            'gold_docs': ['paper.md'], 'evidence': [['Root mean square and zero crossing rate.']], **kwargs}


@pytest.mark.parametrize('mode', ['dense', 'bm25', 'hybrid'])
def test_original_question_only_changes_rerank_not_recall_or_candidate_content(mode):
    store, embedder, scorer = Store(), Embedder(), Scorer()
    summary, rows = evaluate([question()], embedder, store, k_max=5,
                             config=RetrievalConfig(mode), reranker=scorer)
    assert embedder.inputs == ([] if mode == 'bm25' else [[QUESTION]])
    for route, query, depth, filters in store.calls:
        assert query == (QUESTION if route == 'bm25' else [0.25])
        assert depth == 50 and not filters
    assert scorer.inputs == [(ORIGINAL, [f'{h.heading_path}\n{h.text}' for h in store.hits])]
    assert rows[0]['retrieval_query'] == QUESTION
    assert rows[0]['rerank_query'] == ORIGINAL and rows[0]['rerank_query_source'] == 'question_original'
    assert summary['overall']['hit@1'] == 1


def test_retrieval_policy_reproduces_legacy_input_and_ranking():
    scorer = Scorer()
    summary, rows = evaluate([question()], Embedder(), Store(), reranker=scorer,
                             rerank_query_policy='retrieval')
    assert scorer.inputs[0][0] == QUESTION
    assert rows[0]['rerank_query_source'] == 'question'
    assert summary['overall']['hit@1'] == 0


def test_crud_hard_question_is_not_silently_replaced_by_original_easy_question():
    q = question(question='口语化的难版题目', question_original='原始长易版题目')
    del q['evidence']
    scorer = Scorer()
    _, rows = evaluate([q], Embedder(), Store(), reranker=scorer)
    assert scorer.inputs[0][0] == q['question']
    assert rows[0]['rerank_query_source'] == 'question'
    assert resolve_rerank_query(q, 'original') == (q['question_original'], 'question_original')


@pytest.mark.parametrize('original', [None, '', '   ', 123])
def test_invalid_original_falls_back_without_guessing_or_stripping_title(original):
    q = question(question_original=original)
    assert resolve_rerank_query(q, 'auto') == (QUESTION, 'question')
    del q['question_original']
    assert resolve_rerank_query(q, 'original') == (QUESTION, 'question')


def test_no_reranker_retains_dense_baseline_and_records_no_rerank_input():
    store, embedder = Store(), Embedder()
    summary, rows = evaluate([question()], embedder, store, k_max=5)
    assert store.calls == [('dense', [0.25], 5, {})]
    assert embedder.inputs == [[QUESTION]] and summary['overall']['hit@1'] == 0
    assert rows[0]['rerank_query'] is None and rows[0]['rerank_query_source'] is None


def test_shared_api_default_keeps_existing_search_and_answer_callers_compatible():
    scorer = Scorer()
    hits = retrieve(QUESTION, [0.25], Store(), 1, reranker=scorer)
    assert scorer.inputs[0][0] == QUESTION and hits[0].chunk_id == 'abstract'


@pytest.mark.parametrize('value', ['', ' ', 5])
def test_invalid_explicit_query_fails_before_retrieval(value):
    store = Store()
    with pytest.raises(ValueError, match='rerank_question'):
        retrieve(QUESTION, [0.25], store, reranker=Scorer(), rerank_question=value)
    assert store.calls == []


def test_invalid_policy_fails_before_embedding():
    embedder = Embedder()
    with pytest.raises(ValueError, match='rerank_query_policy'):
        evaluate([question()], embedder, Store(), rerank_query_policy='invalid')
    assert embedder.inputs == []


@pytest.mark.parametrize('policy,expected', [('auto', ORIGINAL), ('retrieval', QUESTION)])
def test_cli_records_policy_and_actual_queries(tmp_path, monkeypatch, policy, expected):
    import embedder as embedding_module
    import evaluation.run_retrieval as cli
    import reranker as reranker_module
    import vector_store as store_module

    data = tmp_path / 'qasper'
    data.mkdir()
    (data / 'questions.jsonl').write_text(json.dumps(question()) + '\n')
    (data / 'manifest.json').write_text(json.dumps({'corpus_docs': 1}))
    embedder, store, scorer = Embedder(), Store(), Scorer()
    monkeypatch.setattr(embedding_module, 'Embedder', lambda: embedder)
    monkeypatch.setattr(store_module, 'VectorStore', lambda **kwargs: nullcontext(store))
    monkeypatch.setattr(reranker_module, 'create_reranker', lambda backend: scorer)
    monkeypatch.setattr(cli, 'git_commit', lambda: 'controlled')
    monkeypatch.setattr(cli, 'PROJECT_DIR', tmp_path)
    monkeypatch.setattr(cli, 'RESULTS_DIR', tmp_path / 'results')
    assert main(['--data', str(data), '--label', policy, '--reuse-index', '--rerank', 'local',
                 '--rerank-query', policy]) == 0
    result = json.loads(next((tmp_path / 'results').glob('*.json')).read_text())
    row = json.loads(next((data / 'runs').glob('*.jsonl')).read_text())
    assert result['config']['rerank_query_policy'] == policy
    assert scorer.loaded and result['rerank_model_load_ms'] is not None
    assert result['retrieval_latency_ms']['local_model_preloaded'] is True
    assert result['retrieval_latency_ms']['includes_cold_start'] is False
    assert row['retrieval_query'] == QUESTION and row['rerank_query'] == expected
    assert scorer.inputs[0][0] == expected and embedder.inputs == [[QUESTION]]


def test_cli_bad_model_fails_before_cloud_embedding_or_opening_database(tmp_path, monkeypatch, capsys):
    import embedder as embedding_module
    import reranker as reranker_module
    import vector_store as store_module

    (tmp_path / 'questions.jsonl').write_text(json.dumps(question()) + '\n')
    (tmp_path / 'manifest.json').write_text(json.dumps({'corpus_docs': 1}))
    embedder = Embedder()

    class BrokenScorer(Scorer):
        def warmup(self):
            raise reranker_module.RerankError('model loading failed')

    def must_not_open(**kwargs):
        pytest.fail('database opened before model validation')

    monkeypatch.setattr(embedding_module, 'Embedder', lambda: embedder)
    monkeypatch.setattr(reranker_module, 'create_reranker', lambda backend: BrokenScorer())
    monkeypatch.setattr(store_module, 'VectorStore', must_not_open)
    assert main(['--data', str(tmp_path), '--label', 'bad', '--reuse-index', '--rerank', 'local']) == 1
    assert embedder.inputs == [] and not (tmp_path / 'runs').exists()
    assert 'model loading failed' in capsys.readouterr().out
