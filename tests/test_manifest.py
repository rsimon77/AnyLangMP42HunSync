from mp4translator.manifest import Manifest, Segment


def make():
    return Manifest(
        source="in.mp4",
        duration=10.0,
        summary="sum",
        steps_done=["transcribe"],
        segments=[
            Segment(id=0, start=0.0, end=2.0, text_en="Hello."),
            Segment(id=1, start=3.0, end=5.0, text_en="World.", text_hu="Világ.", speed_factor=1.2),
        ],
    )


def test_next_start():
    m = make()
    assert m.next_start(0) == 3.0


def test_next_start_last_is_duration():
    assert make().next_start(1) == 10.0


def test_save_load_roundtrip(tmp_path):
    m = make()
    path = tmp_path / "manifest.json"
    m.save(path)
    assert Manifest.load(path) == m
    assert not path.with_suffix(".tmp").exists()


def test_save_overwrites(tmp_path):
    path = tmp_path / "manifest.json"
    make().save(path)
    m2 = Manifest(source="other.mp4")
    m2.save(path)
    assert Manifest.load(path) == m2


def test_save_preserves_unicode(tmp_path):
    path = tmp_path / "m.json"
    make().save(path)
    assert Manifest.load(path).segments[1].text_hu == "Világ."
