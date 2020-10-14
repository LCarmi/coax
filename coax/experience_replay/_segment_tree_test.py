# ------------------------------------------------------------------------------------------------ #
# MIT License                                                                                      #
#                                                                                                  #
# Copyright (c) 2020, Microsoft Corporation                                                        #
#                                                                                                  #
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software    #
# and associated documentation files (the "Software"), to deal in the Software without             #
# restriction, including without limitation the rights to use, copy, modify, merge, publish,       #
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the    #
# Software is furnished to do so, subject to the following conditions:                             #
#                                                                                                  #
# The above copyright notice and this permission notice shall be included in all copies or         #
# substantial portions of the Software.                                                            #
#                                                                                                  #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING    #
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND       #
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,     #
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.          #
# ------------------------------------------------------------------------------------------------ #

r"""

This module was copied from the `OpenAI Baseline <https://github.com/openai/baselines>`_ project. It
contains the data structures needed for efficient sampling and updating in `Prioritized Experience
Replay <https://arxiv.org/abs/1511.05952>`_.

"""

import pytest
import numpy as onp
import pandas as pd

from ._segment_tree import SumTree, MinTree


@pytest.fixture
def sum_tree():
    return SumTree(capacity=14)


@pytest.fixture
def min_tree():
    tr = MinTree(capacity=8)
    tr.array = onp.array([13, 7, 11, 17, 19, 5, 3, 23])
    return tr


def test_sum_tree_basic(sum_tree):
    assert sum_tree.array.shape == (14,)
    assert sum_tree.depth == 5
    assert sum_tree.array.sum() == sum_tree.root_value == 0


def test_min_tree_basic(min_tree):
    assert min_tree.array.shape == (8,)
    assert min_tree.depth == 4
    assert min_tree.array.min() == min_tree.root_value == 3


def test_set_values_emptyslice(sum_tree):
    values = onp.arange(1, 15)
    sum_tree[:] = values
    assert sum_tree.array.sum() == sum_tree.root_value == values.sum() > 0


def test_set_values_none(sum_tree):
    values = onp.arange(1, 15)
    sum_tree[None] = values
    assert sum_tree.array.sum() == sum_tree.root_value == values.sum() > 0


def test_set_values_ellipsis(sum_tree):
    values = onp.arange(1, 15)
    sum_tree[...] = values
    assert sum_tree.array.sum() == sum_tree.root_value == values.sum() > 0


def test_set_values_array(sum_tree):
    values = onp.arange(1, 15)
    sum_tree.array = values
    assert sum_tree.array.sum() == sum_tree.root_value == values.sum() > 0


def test_set_values_with_idx(sum_tree):
    idx = onp.array([2, 6, 5, 12, 13])
    values = onp.array([7., 13., 11., 17., 5.])
    sum_tree[idx] = values
    assert sum_tree.array.sum() == sum_tree.root_value == values.sum() > 0


def test_partial_reduce_empty_range(sum_tree):
    msg = r'inconsistent ranges detected from \(start, stop\) = \(1, 1\)'
    with pytest.raises(IndexError, match=msg):
        sum_tree.partial_reduce(1, 1)


def test_partial_reduce_all(sum_tree):
    values = onp.arange(1, 15)
    sum_tree.array = values
    assert sum_tree.partial_reduce() == sum_tree.array.sum() == sum_tree.root_value == values.sum()


@pytest.mark.parametrize('i,j', [(1, 2), (13, 14), (3, 8), (0, None), (0, 3), (7, None)])
def test_partial_reduce(sum_tree, i, j):
    values = onp.arange(100, 114)
    sum_tree.array = values
    assert sum_tree.partial_reduce(i, j) == sum_tree.array[i:j].sum() == values[i:j].sum() > 0


def test_partial_reduce_array_sum(sum_tree):
    i, j = onp.array([0, 8, 3, 0, 0]), onp.array([1, 13, 14, 5, -1])
    values = onp.arange(100, 114)
    sum_tree.array = values
    expected = onp.vectorize(lambda i, j: values[i:j].sum())
    onp.testing.assert_allclose(sum_tree.partial_reduce(i, j), expected(i, j))


def test_partial_reduce_array_min(min_tree):
    i, j = onp.array([1, 6, 0]), onp.array([8, 7, 5])
    expected = onp.vectorize(lambda i, j: min_tree.array[i:j].min())
    onp.testing.assert_allclose(min_tree.partial_reduce(i, j), expected(i, j))


@pytest.mark.parametrize('s', [slice(0, 1), [0, -1], slice(6, 9), slice(None), 0, 10, -1])
def test_inverse_cdf(s):
    tr = SumTree(capacity=8)
    tr.array = onp.array([13, 7, 11, 17, 19, 5, 3, 23])

    df = pd.DataFrame(
        columns=['uniform', 'idx', 'value'],
        data=onp.array([
            [0, 0, 13],
            [5, 0, 13],
            [12, 0, 13],
            [13, 1, 7],
            [14, 1, 7],
            [19, 1, 7],
            [20, 2, 11],
            [25, 2, 11],
            [30, 2, 11],
            [31, 3, 17],
            [40, 3, 17],
            [47, 3, 17],
            [48, 4, 19],
            [50, 4, 19],
            [66, 4, 19],
            [67, 5, 5],
            [70, 5, 5],
            [71, 5, 5],
            [72, 6, 3],
            [73, 6, 3],
            [74, 6, 3],
            [75, 7, 23],
            [80, 7, 23],
            [97, 7, 23],
            [98, 7, 23],
        ]))
    df['uniform'] /= df['uniform'].max()  # normalize to unit interval [0, 1]

    actual = tr.inverse_cdf(df['uniform'].values[s])
    expected = df['idx'].values[s]

    onp.testing.assert_allclose(actual, expected)


def test_sample_distribution():
    tr = SumTree(capacity=8, random_seed=13)
    tr.array = onp.array([13, 7, 11, 17, 19, 5, 3, 23])

    # this also demonstrates how fast the batched implementation is
    idx = tr.sample(n=1000000)

    empirical = pd.Series(idx).value_counts(normalize=True).sort_index()
    expected = pd.Series(tr.array / tr.array.sum())

    print(pd.concat((empirical.rename('empirical'), expected.rename('expected')), axis=1))
    onp.testing.assert_allclose(empirical, expected, rtol=1e-2)
