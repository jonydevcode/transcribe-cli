# SPDX-License-Identifier: Apache-2.0
from transcribe_cli.stitch import join_chunks, stitch_chunks, text_tokens


def test_join_chunks_spaces_latin_but_not_han() -> None:
    assert join_chunks(["你好", "世界", "Hello"]) == "你好世界 Hello"
    assert join_chunks(["", "  ", "a"]) == "a"


def test_tokens_split_han_from_latin() -> None:
    assert text_tokens("Hello你好世界") == [("hello", 5), ("你", 6), ("好", 7), ("世", 8), ("界", 9)]


def test_exact_overlap_is_removed() -> None:
    assert stitch_chunks([
        "Please review the guest room door.",
        "the guest room door. Please find the exits.",
    ]) == ("Please review the guest room door. Please find the exits.", 0)


def test_unmatched_words_are_preserved_and_counted() -> None:
    assert stitch_chunks([
        "Please open the door.",
        "Do not open the door. There is smoke outside.",
    ]) == ("Please open the door. Do not open the door. There is smoke outside.", 1)
    assert stitch_chunks([
        "They tell guests to open the big red heavy door.",
        "Do not open the big red heavy door. It is dangerous.",
    ]) == ("They tell guests to open the big red heavy door. "
           "Do not open the big red heavy door. It is dangerous.", 1)


def test_fuzzy_leading_mismatch_still_aligns() -> None:
    assert stitch_chunks([
        "If the door is warm or impassable, please wet towels at the",
        "Impossible please web towels at the base of the door.",
    ]) == ("If the door is warm or impassable, please wet towels at the base of the door.", 0)


def test_han_overlap_needs_six_characters() -> None:
    assert stitch_chunks(["天气很好我们出去散步", "我们出去散步然后吃饭"]) == ("天气很好我们出去散步然后吃饭", 0)


def test_repeated_single_word_is_not_an_overlap() -> None:
    assert stitch_chunks(["Yes yes yes.", "Yes yes yes, please."]) == ("Yes yes yes. Yes yes yes, please.", 1)
