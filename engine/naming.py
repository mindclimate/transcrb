"""What a recording and its transcript are called.

The name a recording is given becomes its filename, and its filename becomes
the transcript's folder, so it has to survive being both. Kept to characters
that mean the same thing everywhere rather than escaped cleverly: a meeting
named with a slash is not worth a directory traversal.
"""
import re
import time

# Day first, then the time of day, because that is how the meetings are
# remembered. Dots separate the date so the name reads as a date rather than as
# a serial number; the time has none, so the eye finds the boundary.
STAMP_FORMAT = "%d.%m.%Y-%H%M"

# Long enough for a real meeting title, short enough that the whole thing still
# fits in a Finder column next to the stamp.
_SLUG_LIMIT = 60

def stamp(when: float | None = None) -> str:
    """The day and time of day a recording was made, as a filename would say it."""
    return time.strftime(STAMP_FORMAT, time.localtime(when))

def slugify(title: str | None) -> str:
    """The user's part of a name, made safe to be a filename. '' if unusable.

    Spaces are kept — the name is read far more often than it is typed. Dots
    and colons are not: the stamp uses dots, and a colon is displayed as a
    slash by Finder.
    """
    cleaned = re.sub(r"[^\w\s-]", "", (title or ""), flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
    return cleaned[:_SLUG_LIMIT].strip(" -")

def compose(title: str | None, when: float | None = None) -> str:
    """The full name: the stamp, then the user's name for it if there is one.

    The stamp goes in front so a folder listing is still in the order the
    meetings happened, whether or not anyone named them.
    """
    slug = slugify(title)
    return f"{stamp(when)} {slug}" if slug else stamp(when)

# A stamp at the front of an existing name, so renaming can keep it. Matched
# rather than parsed: the point is only to tell the stamp from the name.
_STAMP = re.compile(r"^(\d{2}\.\d{2}\.\d{4}-\d{4})(?: (.*))?$")

def rename(existing: str, title: str | None) -> str:
    """`existing` renamed to `title`, keeping when it was recorded.

    The stamp records the moment the meeting happened, so renaming a month
    later must not restamp it. Something without a stamp — an uploaded file,
    named after the file — has nothing to keep and becomes the name alone.

    Raises ValueError if that would leave nothing to call it.
    """
    slug = slugify(title)
    made = _STAMP.match(existing or "")
    if not made:
        if not slug:
            raise ValueError("a name is needed")
        return slug
    return f"{made.group(1)} {slug}" if slug else made.group(1)
