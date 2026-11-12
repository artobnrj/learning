"""Skill 2: get_current_time - the model has no clock, so give it one."""

from datetime import datetime
from datetime import timezone as dt_timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import skill


@skill(
    name="get_current_time",
    description=(
        "Get the current date and time. Pass an IANA timezone name such as "
        "'Asia/Tokyo' or 'America/New_York', or null for UTC."
    ),
    properties={
        "timezone": {
            "type": ["string", "null"],
            "description": "IANA timezone name, or null for UTC.",
        }
    },
)
def get_current_time(timezone: str | None = None) -> str:
    if not timezone:
        tz = dt_timezone.utc
    else:
        try:
            tz = ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(
                f"Unknown timezone {timezone!r}; use an IANA name like 'Europe/London'"
            ) from None
    now = datetime.now(tz)
    return f"{now.isoformat(timespec='seconds')} ({now.strftime('%A')}, {tz})"
