import datetime
import logging


class UTCFormatter(logging.Formatter):
    LOCAL_TZ = datetime.datetime.now().astimezone().tzinfo

    def formatTime(self, record, datefmt=None):
        utc = datetime.datetime.fromtimestamp(record.created, UTCFormatter.LOCAL_TZ)
        return utc.isoformat(timespec="milliseconds")
