from slowapi import Limiter
from slowapi.util import get_remote_address

# Initialize slowapi rate limiter using remote address key
limiter = Limiter(key_func=get_remote_address)
