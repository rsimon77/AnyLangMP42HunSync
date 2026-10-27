import numpy as np

from mp4translator.steps.voices import cluster_speakers, nearest_in_time, pick_reference


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def group(base, n, seed):
    rng = np.random.default_rng(seed)
    return [unit(np.asarray(base, dtype=float) + rng.normal(0, 0.02, len(base))) for _ in range(n)]


A = [1.0, 0.0, 0.0]
B = [0.0, 1.0, 0.0]
C = [0.0, 0.0, 1.0]


def test_cluster_two_groups():
    embs = np.stack(group(A, 4, 0) + group(B, 4, 1))
    labels = cluster_speakers(embs, [5.0] * 8, 0.35, 4.0)
    assert sorted(set(labels)) == [0, 1]
    assert len(set(labels[:4])) == 1 and len(set(labels[4:])) == 1
    assert labels[0] != labels[4]


def test_cluster_one_group():
    embs = np.stack(group(A, 5, 2))
    assert set(cluster_speakers(embs, [5.0] * 5, 0.35, 4.0)) == {0}


def test_cluster_empty_and_single():
    assert cluster_speakers(np.zeros((0, 3)), [], 0.35, 4.0) == []
    assert cluster_speakers(np.stack([unit(A)]), [1.0], 0.35, 4.0) == [0]


def test_cluster_small_merged_into_nearest():
    near_a = unit([1.0, 0.6, 0.0])  # closer to A than to B, but far enough to form its own cluster
    embs = np.stack(group(A, 3, 3) + group(B, 3, 4) + [near_a])
    durs = [5.0] * 6 + [0.5]
    labels = cluster_speakers(embs, durs, 0.1, 4.0)
    assert sorted(set(labels)) == [0, 1]
    assert labels[6] == labels[0]


def test_cluster_labels_ordered_by_duration():
    embs = np.stack(group(A, 2, 5) + group(B, 2, 6) + group(C, 2, 7))
    durs = [5.0, 5.0, 20.0, 20.0, 10.0, 10.0]
    labels = cluster_speakers(embs, durs, 0.35, 4.0)
    assert labels[2] == labels[3] == 0
    assert labels[4] == labels[5] == 1
    assert labels[0] == labels[1] == 2


def test_nearest_in_time():
    labelled = {0: 5, 4: 7}
    assert nearest_in_time(1, labelled) == 5
    assert nearest_in_time(3, labelled) == 7
    assert nearest_in_time(2, labelled) == 5  # tie -> earlier
    assert nearest_in_time(10, labelled) == 7


def test_pick_reference_prefers_similar_and_long():
    cands = [(0.0, 2.0, 0.9), (10.0, 14.0, 0.9), (20.0, 24.0, 0.2)]
    assert pick_reference(cands, 4.0) == [(10.0, 14.0)]


def test_pick_reference_stops_at_target_and_sorted():
    cands = [(30.0, 34.0, 0.95), (0.0, 4.0, 0.9), (10.0, 14.0, 0.8), (20.0, 24.0, 0.7)]
    out = pick_reference(cands, 8.0)
    assert out == [(0.0, 4.0), (30.0, 34.0)]
    assert out == sorted(out)


def test_pick_reference_short_total_and_empty():
    assert pick_reference([(0.0, 1.0, 0.5)], 12.0) == [(0.0, 1.0)]
    assert pick_reference([], 12.0) == []
