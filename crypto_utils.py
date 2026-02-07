# crypto_utils.py
import os
from cryptography.fernet import Fernet

_KEY ='Y7-Dsht3fzSZ3b9RiuxpYgIqnPefA30nNB6s84iQCoA='

# _KEY will be set from app.py via os.environ.setdefault(...)
fernet = Fernet(_KEY.encode() if isinstance(_KEY, str) else _KEY)

def encrypt_str(s: str) -> str:
    """Encrypt a string and return a base64 token."""
    return fernet.encrypt(s.encode("utf-8")).decode("utf-8")

def decrypt_str(token: str) -> str:
    """Decrypt a base64 token back to original string."""
    return fernet.decrypt(token.encode("utf-8")).decode("utf-8")
