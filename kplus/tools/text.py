

__all__ = [
    "safepath"
]

####################
# Text Normalization
####################
def safepath(s: str) -> str:
    return "".join([c for c in s if c.isalpha() or c.isdigit() or c in ' _-']).strip()
