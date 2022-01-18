from __future__ import print_function
import pytz
from datetime import datetime
def get_now(mode="sql"):
    now = datetime.now()

    if mode == 'sql':
        return now.replace(tzinfo=pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
    if mode == 'file':
        return now.replace(tzinfo=pytz.utc).strftime("%Y-%m-%d_%H-%M-%S")
