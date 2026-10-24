import pytest

from knowledge_base.preprocess import chunk_text, clean_text, html_title, html_to_text, is_table_debris

PAGE = """<html><head><title>Irrigation Management :: Drip</title>
<style>.x{color:red}</style><script>alert('x')</script></head><body>
<div>Home | About Us | Contact | Disclaimer</div>
<h2>Drip Irrigation</h2><p>Drip irrigation saves water.</p>
<table><tr><td>Rice</td><td>Initial tillering</td></tr></table>
<p>&copy; All Rights Reserved. TNAU-2016.</p></body></html>"""


def test_html_to_text_removes_navigation_scripts_and_footer():
    text = html_to_text(PAGE)
    assert "Drip irrigation saves water." in text
    assert "Rice Initial tillering" in text  # table cells joined on one line
    assert "alert" not in text and "color:red" not in text
    assert "About Us" not in text
    assert "Rights Reserved" not in text


def test_html_title():
    assert html_title(PAGE) == "Irrigation Management :: Drip"
    assert html_title("<p>no title</p>") is None


def test_clean_text_normalises_tamil_and_removes_invisible_chars():
    decomposed = "கொ"  # already NFC; build the decomposed form explicitly
    nfd = "கொ"  # க + ெ + ா  -> NFC composes to கொ
    cleaned = clean_text(f"  {nfd}​  நெல்\n\n\n\nசாகுபடி ")
    assert decomposed in cleaned
    assert "​" not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.endswith("சாகுபடி")


def test_clean_text_repairs_hyphenated_line_breaks():
    assert clean_text("fertil-\nizer") == "fertilizer"


def test_chunk_text_respects_size_and_overlap():
    sentences = [f"Sentence number {i} talks about irrigation water management." for i in range(40)]
    chunks = chunk_text(" ".join(sentences), chunk_size=200, overlap=50)
    assert len(chunks) > 5
    assert all(len(c) <= 200 for c in chunks)
    # consecutive chunks share text (overlap)
    for a, b in zip(chunks, chunks[1:]):
        assert any(word in b[:60] for word in a[-50:].split()[-3:])


def test_chunk_text_hard_splits_long_sentence():
    chunks = chunk_text("word " * 300, chunk_size=100, overlap=10)
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) >= 12


def test_chunk_text_tamil():
    text = "நெல் பயிருக்கு நீர் மேலாண்மை மிகவும் முக்கியம். " * 20
    chunks = chunk_text(text, chunk_size=150, overlap=30)
    assert chunks and all(len(c) <= 150 for c in chunks)
    assert all("நெல்" in c or "நீர்" in c for c in chunks)


def test_chunk_text_empty_and_invalid_arguments():
    assert chunk_text("   ") == []
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=10)


def test_table_debris_detected_in_both_languages():
    assert is_table_debris("3-6 44 62 1.20 75,000 பூசணி வகைகள் 1-2 3-6 15 54 1.20 75,000")
    assert is_table_debris("60 - 130 98 45 5.0 34,000 Banana 8 –12 30 - 40 52 45 2.0 56,000")
    assert is_table_debris("   ")


def test_prose_and_dosage_tables_are_not_debris():
    assert not is_table_debris("குலைநோய் அறிகுறிகள்: இலைகளின் மேல் கண் வடிவ புள்ளிகள் காணப்படும்.")
    assert not is_table_debris("Acephate 75 % SP 400 g/ac Carbofuran 3% CG 10 kg/ac Fipronil 80%WG 20-25 g/ac")
