import unittest
from common import metrics, rank_indices, validate_prediction
from score import PREFIX, SUFFIX, encode_pairs


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode('utf-8'))

    def __call__(self, texts, **kwargs):
        return {'input_ids': [self.encode(t) for t in texts]}


class BaselineTests(unittest.TestCase):
    def test_cross_partial_evidence_is_not_complete(self):
        label = {'coverage': {'a': [0], 'b': [1], 'c': [2]}, 'groups': [[0, 1, 2]]}
        r = metrics(['a', 'b', 'x', 'y', 'z', 'c'], label)
        self.assertEqual(r['hit@1'], 1)
        self.assertEqual(r['recall@5'], 2 / 3)
        self.assertEqual(r['all_evidence@5'], 0)
        self.assertEqual(r['all_evidence@10'], 1)
        self.assertEqual(r['mrr@5'], 1)

    def test_annotator_groups_and_duplicate_evidence(self):
        label = {'coverage': {'a': [0], 'b': [0]}, 'groups': [[0], [1, 2]]}
        r = metrics(['x', 'a', 'b'], label)
        self.assertEqual(r['recall@5'], 1)
        self.assertEqual(r['mrr@5'], .5)

    def test_ties_and_invalid_scores(self):
        self.assertEqual(rank_indices([0, 2, 2, -3]), [1, 2, 0, 3])
        for bad in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                rank_indices([bad])

    def test_cached_ranking_must_match_scores(self):
        ids = [str(i) for i in range(50)]
        q = {'qid': 'q', 'candidates': ids}
        row = {'qid': 'q', 'candidate_ids': ids, 'scores': [0] * 50, 'ranked_ids': ids[::-1]}
        with self.assertRaises(ValueError):
            validate_prediction(row, q)

    def test_truncation_keeps_complete_prompt_suffix(self):
        tok = CharacterTokenizer()
        size = len(tok.encode(PREFIX + SUFFIX)) + 150
        seqs, lengths = encode_pairs(tok, 'short', ['d' * 1000], size)
        self.assertGreater(lengths[0], size)
        self.assertEqual(len(seqs[0]), size)
        self.assertEqual(seqs[0][-len(tok.encode(SUFFIX)):], tok.encode(SUFFIX))
        with self.assertRaises(ValueError):
            encode_pairs(tok, 'q' * 1000, ['doc'], size)


if __name__ == '__main__':
    unittest.main()
