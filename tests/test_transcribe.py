from mp4translator.steps.transcribe import Word, group_words


def W(text, start, end):
    return Word(text, start, end)


def test_empty_and_whitespace_only():
    assert group_words([]) == []
    assert group_words([W(" ", 0, 0.1), W("  ", 0.1, 0.2)]) == []


def test_sentence_end_splits_when_long_enough():
    words = [W(" Hello", 0, 0.6), W(" world.", 0.6, 1.2), W(" Next", 1.3, 1.6), W(" one.", 1.6, 2.5)]
    assert group_words(words) == [(0, 1.2, "Hello world."), (1.3, 2.5, "Next one.")]


def test_sentence_end_respects_min_dur():
    words = [W(" Hi.", 0, 0.4), W(" There", 0.5, 0.9), W(" you.", 0.9, 1.6)]
    assert group_words(words) == [(0, 1.6, "Hi. There you.")]


def test_min_dur_is_configurable():
    words = [W(" Hi.", 0, 0.4), W(" There.", 0.5, 0.9)]
    assert group_words(words, min_dur=0.3) == [(0, 0.4, "Hi."), (0.5, 0.9, "There.")]


def test_question_and_exclamation_end_sentences():
    words = [W(" Why?", 0, 1.2), W(" Wow!", 1.3, 2.5)]
    assert group_words(words) == [(0, 1.2, "Why?"), (1.3, 2.5, "Wow!")]


def test_pause_splits():
    words = [W(" one", 0, 0.5), W(" two", 0.5, 1.0), W(" three", 2.0, 2.5)]
    assert group_words(words) == [(0, 1.0, "one two"), (2.0, 2.5, "three")]


def test_short_pause_does_not_split():
    words = [W(" one", 0, 0.5), W(" two", 1.0, 1.5)]
    assert group_words(words) == [(0, 1.5, "one two")]


def test_max_dur_splits_at_last_clause_boundary():
    words = [
        W(" a", 0, 1),
        W(" b,", 1, 2),
        W(" c", 2, 3),
        W(" d,", 3, 4),
        W(" e", 4, 5),
        W(" f", 5, 6),
    ]
    assert group_words(words, max_dur=5.5) == [(0, 4, "a b, c d,"), (4, 6, "e f")]


def test_max_dur_plain_split_without_clause_boundary():
    words = [W(f" w{i}", i, i + 1) for i in range(6)]
    assert group_words(words, max_dur=4.5) == [(0, 4, "w0 w1 w2 w3"), (4, 6, "w4 w5")]


def test_clause_boundary_on_first_word_is_not_used():
    words = [W(" a,", 0, 1), W(" b", 1, 2), W(" c", 2, 3), W(" d", 3, 4)]
    assert group_words(words, max_dur=3.5) == [(0, 3, "a, b c"), (3, 4, "d")]


def test_trailing_words_flushed():
    assert group_words([W(" tail", 5, 5.5)]) == [(5, 5.5, "tail")]
