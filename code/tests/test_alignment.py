import math

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as Fa

from hbasr import alignment as A


def test_collapse_and_greedy():
    assert A.ctc_collapse([1, 1, 0, 1, 2, 2]) == [1, 1, 2]
    assert A.ctc_collapse([0, 1, 1, 0]) == [1]
    lp = np.log(np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]]))
    assert A.ctc_greedy(lp) == [1, 2]


def test_ctc_hand_example():
    p = np.array([[0.2, 0.6, 0.2], [0.4, 0.3, 0.3], [0.1, 0.2, 0.7]])
    loss = A.ctc_forward(np.log(p), [1, 2])
    # aab .126 + abb .126 + -ab .042 + a-b .168 + ab- .018 = .48
    assert abs(math.exp(-loss) - 0.48) < 1e-12
    assert abs(math.exp(-A.ctc_forward(np.log(p), [1, 1])) - 0.6 * 0.4 * 0.2) < 1e-12


def test_ctc_vs_torch_random():
    rng = np.random.default_rng(1)
    for T, U in [(5, 2), (8, 3), (7, 1)]:
        lp = A.log_softmax(rng.standard_normal((T, 4)))
        lab = rng.integers(1, 4, U).tolist()
        bf, _ = A.ctc_brute_force(lp, lab) if T <= 7 else (None, None)
        fw = A.ctc_forward(lp, lab)
        t = F.ctc_loss(torch.tensor(lp)[:, None], torch.tensor([lab]), torch.tensor([T]), torch.tensor([U]),
                       reduction="sum").item()
        assert abs(fw - t) < 1e-6
        if bf is not None:
            assert abs(bf - fw) < 1e-9


def test_rnnt_hand_and_torchaudio():
    p = np.array([[[0.4, 0.6], [0.8, 0.2]], [[0.3, 0.7], [0.9, 0.1]]])
    loss = A.rnnt_forward(np.log(p), [1])
    assert abs(math.exp(-loss) - (0.6 * 0.8 * 0.9 + 0.4 * 0.7 * 0.9)) < 1e-12
    rng = np.random.default_rng(2)
    lp = A.log_softmax(rng.standard_normal((4, 3, 5)))
    lab = [2, 4]
    bf, paths = A.rnnt_brute_force(lp, lab)
    assert len(paths) == math.comb(4 + 2 - 1, 2)
    fw = A.rnnt_forward(lp, lab)
    t = Fa.rnnt_loss(torch.tensor(lp, dtype=torch.float32)[None], torch.tensor([lab], dtype=torch.int32),
                     torch.tensor([4], dtype=torch.int32), torch.tensor([2], dtype=torch.int32),
                     blank=0, reduction="sum").item()
    assert abs(bf - fw) < 1e-9 and abs(fw - t) < 1e-4
