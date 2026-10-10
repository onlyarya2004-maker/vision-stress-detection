"""
Tests for daisee_ensemble.shared_pairs() / start_background_load(): the single,
pre-warmed, process-wide copy of the CNN weights that every hosted session uses
(so a cold start is not a minute of blank page for the first visitor, and two
sessions never load two copies of 50 MB of weights).
"""
import threading
import time

import numpy as np
import pytest

import daisee_ensemble


@pytest.fixture
def fresh(monkeypatch):
    """Reset the module's singleton state and count build_pairs() calls."""
    monkeypatch.setattr(daisee_ensemble, '_SHARED', None)
    monkeypatch.setattr(daisee_ensemble, '_BG_THREAD', None)
    calls = {'build': 0, 'predict': 0}

    def fake_build(skip=False):
        calls['build'] += 1
        time.sleep(0.05)                                   # make races visible
        return [('infer', 160, None)]

    def fake_predict(face, pairs):
        calls['predict'] += 1
        return np.array([0.5, 0.5])

    monkeypatch.setattr(daisee_ensemble, 'build_pairs', fake_build)
    monkeypatch.setattr(daisee_ensemble, 'ensemble_predict', fake_predict)
    return calls


def test_shared_pairs_loads_once_and_warms_up_once(fresh):
    a = daisee_ensemble.shared_pairs()
    b = daisee_ensemble.shared_pairs()
    assert a is b
    assert fresh == {'build': 1, 'predict': 1}              # one load, one warm-up trace


def test_concurrent_sessions_still_trigger_only_one_load(fresh):
    results = []
    threads = [threading.Thread(target=lambda: results.append(daisee_ensemble.shared_pairs()))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert fresh['build'] == 1
    assert all(r is results[0] for r in results)


def test_background_load_starts_a_thread_once_and_a_later_caller_waits_for_it(fresh):
    daisee_ensemble.start_background_load()
    daisee_ensemble.start_background_load()                 # called on every script run
    daisee_ensemble.start_background_load()
    pairs = daisee_ensemble.shared_pairs()                  # blocks until the thread is done
    assert pairs and fresh['build'] == 1
    daisee_ensemble.start_background_load()                 # already loaded: a no-op
    assert fresh['build'] == 1


def test_a_failing_warm_up_does_not_break_loading(fresh, monkeypatch):
    def boom(face, pairs):
        raise RuntimeError("trace failed")
    monkeypatch.setattr(daisee_ensemble, 'ensemble_predict', boom)
    assert daisee_ensemble.shared_pairs()                   # still returns the loaded models
