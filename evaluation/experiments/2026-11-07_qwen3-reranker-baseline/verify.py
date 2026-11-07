"""Offline dataset validation against previously computed historical metrics."""
import math
from common import ROOT, check, load_inputs, read_json, summarize


def main():
    queries, docs = load_inputs()
    actual, _ = summarize(queries)
    expected = read_json(ROOT / 'data/expected_dense.json')
    check(set(actual) == set(expected), '分组不一致')
    for group in expected:
        check(actual[group]['n'] == expected[group]['n'], f'题数不一致: {group}')
        for key, value in expected[group]['metrics'].items():
            check(math.isclose(actual[group]['metrics'][key], value, rel_tol=0, abs_tol=1e-12), f'Dense 历史指标不一致: {group}/{key}')
    print(f'验证通过：{len(queries)}题，{len(docs)}个去重候选块；Dense指标与历史结果一致。')
    for group, values in actual.items():
        m = values['metrics']
        print(f"{group}: n={values['n']} hit@5={m['hit@5']:.4f} recall@5={m['recall@5']:.4f} recall@50={m['recall@50']:.4f}")


if __name__ == '__main__':
    main()
