import numpy as np

from kalshi_model.rng import STREAM_NAMES, make_streams


def test_same_seed_gives_same_draws_in_every_stream():
    first = make_streams(7)
    second = make_streams(7)
    for name in STREAM_NAMES:
        assert np.array_equal(first[name].random(5), second[name].random(5))


def test_streams_are_independent_of_each_other():
    untouched = make_streams(7)["world"].random(5)
    streams = make_streams(7)
    streams["network"].random(1000)
    streams["traits"].random(3)
    assert np.array_equal(streams["world"].random(5), untouched)


def test_streams_differ_by_name_and_seed():
    streams = make_streams(7)
    draws = {name: streams[name].random() for name in STREAM_NAMES}
    assert len(set(draws.values())) == len(STREAM_NAMES)
    assert make_streams(8)["world"].random() != make_streams(7)["world"].random()


def test_stream_names_are_unique():
    assert len(set(STREAM_NAMES)) == len(STREAM_NAMES)
