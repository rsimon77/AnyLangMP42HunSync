from mp4translator.text import count_hu_syllables, format_srt_time, to_srt


def test_syllables_basic():
    assert count_hu_syllables("") == 0
    assert count_hu_syllables("szia") == 2
    assert count_hu_syllables("bcd") == 0


def test_syllables_accented_vowels_and_case():
    assert count_hu_syllables("áéíóöőúüű") == 9
    assert count_hu_syllables("ÁÉÍÓÖŐÚÜŰ") == 9
    assert count_hu_syllables("AEIOU") == 5


def test_syllables_digits():
    assert count_hu_syllables("2025") == 8
    assert count_hu_syllables("5 és 10") == 2 + 1 + 4


def test_format_srt_time():
    assert format_srt_time(0) == "00:00:00,000"
    assert format_srt_time(1.5) == "00:00:01,500"
    assert format_srt_time(3661.007) == "01:01:01,007"
    assert format_srt_time(59.9996) == "00:01:00,000"


def test_format_srt_time_negative_clamped():
    assert format_srt_time(-3) == "00:00:00,000"


def test_to_srt():
    out = to_srt([(0, 1.5, " Szia "), (2, 3.25, "Világ")])
    assert out == "1\n00:00:00,000 --> 00:00:01,500\nSzia\n\n2\n00:00:02,000 --> 00:00:03,250\nVilág\n"


def test_to_srt_empty():
    assert to_srt([]) == ""
