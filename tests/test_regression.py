"""检查本次重构最容易改变结果的边界、路径和文本计数。"""
import importlib
import io
import json
from pathlib import Path
import unittest

import pandas as pd
from amazon_project.common import classify_scalar, classify_series

ROOT = Path(__file__).resolve().parents[1]


class RegressionTests(unittest.TestCase):
    def test_strict_threshold(self):
        scores = pd.Series([-0.2, -0.1, 0.0, 0.1, 0.10000000000000005, 0.2])
        expected = ['Negative', 'Near-neutral', 'Near-neutral',
                    'Near-neutral', 'Positive', 'Positive']
        self.assertEqual(classify_series(scores, .1).tolist(), expected)
        self.assertEqual([classify_scalar(x, .1) for x in scores], expected)

    def test_round_trip_boundary(self):
        df = pd.read_csv(io.StringIO('polarity\n0.10000000000000005\n'),
                         float_precision='round_trip')
        self.assertEqual(classify_scalar(df.loc[0, 'polarity']), 'Positive')

    def test_tasks_import_without_running(self):
        tasks = json.loads((ROOT / 'config/tasks.json').read_text(encoding='utf-8'))
        for task in tasks.values():
            with self.subTest(task=task['name']):
                self.assertTrue(callable(importlib.import_module(task['module']).run))

    def test_business_reference_counts(self):
        table = pd.read_csv(ROOT / 'reports/week4/tables/product_review_actions.csv')
        self.assertEqual(len(table), 15)
        unique = table.drop_duplicates('review_text_hash')
        self.assertEqual(len(unique), 14)
        actual = unique['issue_types'].str.split('|', regex=False).explode().value_counts().to_dict()
        self.assertEqual(actual, {'功能表现与耐用性':12, '商品说明与适配':9,
                                 '操作与结构设计':6, '售后与评价核查':2})


if __name__ == '__main__':
    unittest.main()
