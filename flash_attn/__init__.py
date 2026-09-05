from pkgutil import extend_path

# look for every subdir with flash_attn base name such that fa2 and fa4 can be co-installed
__path__ = extend_path(__path__, __name__)

try:
    from importlib.metadata import version
    __version__ = version("flash_attn").split("+")[0]
except Exception:
    __version__ = "2.9.2.post2"

from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)
