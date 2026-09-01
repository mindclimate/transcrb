"""What a recording and its transcript end up being called.

The name is the only handle a person has on a meeting once it is over: the wav,
the transcript folder and the files inside it all carry it. It therefore has to
survive being a filename on a Mac, being read back a month later, and being
typed in Hebrew.
"""
import time

import pytest

from engine.naming import compose, rename, slugify, stamp


def _at(text: str) -> float:
    """Seconds since the epoch for a local 'YYYY-MM-DD HH:MM' string."""
    return time.mktime(time.strptime(text, "%Y-%m-%d %H:%M"))


# ---- the stamp ---------------------------------------------------------------

def test_the_stamp_is_the_day_and_the_time_of_day():
    assert stamp(_at("2026-09-02 19:12")) == "02.09.2026-1912"

def test_single_digit_days_hours_and_minutes_are_padded():
    # Unpadded, the folder list stops lining up and the eye stops finding things.
    assert stamp(_at("2026-01-05 09:05")) == "05.01.2026-0905"

def test_midnight_is_a_time_of_day_like_any_other():
    assert stamp(_at("2026-09-02 00:00")) == "02.09.2026-0000"

def test_the_stamp_defaults_to_now():
    assert stamp() == stamp(time.time())


# ---- the user's part ---------------------------------------------------------

def test_a_name_keeps_its_spaces():
    # The name is read far more often than it is typed. Hyphens made every
    # transcript folder look like a slug of itself.
    assert slugify("Weekly with Vladi") == "Weekly with Vladi"

def test_runs_of_whitespace_collapse_to_one_space():
    assert slugify("  RPO   /  Bohdan + Daniil  ") == "RPO Bohdan Daniil"

def test_a_name_cannot_escape_the_output_directory():
    assert slugify("../../etc/passwd") == "etcpasswd"
    assert slugify("/") == ""

def test_a_name_cannot_contain_a_colon():
    # Finder shows a colon as a slash and some tools refuse the path outright.
    assert ":" not in slugify("14:11 standup")

def test_a_name_cannot_contain_a_dot():
    # The stamp uses dots. A name that also has them makes the two unreadable
    # where they meet, and a leading dot hides the folder.
    assert slugify("v1.2 review") == "v12 review"

def test_an_empty_or_symbol_only_name_is_unusable():
    assert slugify("") == ""
    assert slugify("!!!") == ""
    assert slugify(None) == ""

def test_a_very_long_name_is_cut_rather_than_refused():
    assert len(slugify("x" * 200)) == 60

def test_a_hebrew_name_survives_being_a_filename():
    # Half this project's meetings are in Hebrew and the name is the only place
    # a person recognises them by.
    assert slugify("פגישה עם בוהדן") == "פגישה עם בוהדן"


# ---- the two together --------------------------------------------------------

def test_an_unnamed_recording_is_called_by_its_day_and_time():
    assert compose("", _at("2026-09-02 19:12")) == "02.09.2026-1912"

def test_a_named_recording_keeps_the_stamp_in_front_of_the_name():
    # In front, so the folder list is still in the order the meetings happened.
    assert compose("Weekly with Vladi", _at("2026-09-02 19:12")) == \
        "02.09.2026-1912 Weekly with Vladi"

def test_a_name_that_slugifies_to_nothing_leaves_the_stamp_alone():
    assert compose("!!!", _at("2026-09-02 19:12")) == "02.09.2026-1912"
    assert compose(None, _at("2026-09-02 19:12")) == "02.09.2026-1912"


# ---- renaming something that already exists ----------------------------------

def test_renaming_keeps_the_moment_the_recording_was_made():
    # The stamp says when the meeting happened. Renaming it an hour later must
    # not move it to an hour later.
    assert rename("02.09.2026-1912 Weekly with Vladi", "RPO sync") == \
        "02.09.2026-1912 RPO sync"

def test_renaming_something_that_was_never_named_gains_a_name():
    assert rename("02.09.2026-1912", "RPO sync") == "02.09.2026-1912 RPO sync"

def test_clearing_the_name_leaves_the_stamp_behind():
    assert rename("02.09.2026-1912 Weekly with Vladi", "") == "02.09.2026-1912"

def test_an_uploaded_file_has_no_stamp_to_keep():
    # A dropped interview.m4a is called 'interview'. Renaming it should not
    # invent a time it was recorded at.
    assert rename("interview", "RPO sync") == "RPO sync"

def test_a_new_name_that_slugifies_to_nothing_is_refused():
    # Nothing to rename it to, and an unstamped folder would end up as ''.
    with pytest.raises(ValueError):
        rename("interview", "!!!")
