import base64

def topic_uid(tup) -> str:
    a, b = tup
    a = int(a)
    b = int(b) + 1  # Because unclustered is -1 and we can't convert negative to unsigned.
    combined = (a << 10) | b  # pack into 20 bits
    return base64.urlsafe_b64encode(combined.to_bytes(3, "big")).rstrip(b'=').decode()


def uid_to_ints(s: str):
    """Returns (layer, cluster_number)"""
    padded = s + '=' * (-len(s) % 4)
    combined = int.from_bytes(base64.urlsafe_b64decode(padded), "big")
    return combined >> 10, (combined & 0x3FF) - 1