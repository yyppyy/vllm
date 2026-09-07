"""Force hostname resolution through Google DNS (8.8.8.8) for hosts the
cluster resolver maps to unreachable CloudFront edges.

Usage:  python -m tools.net.dns8888  <module-or-script>  ...
or:     import tools.net.dns8888  (side-effect: patches socket.getaddrinfo)
"""
import socket
import subprocess

_cache = {}
_orig = socket.getaddrinfo


def _resolve8888(host):
    if host in _cache:
        return _cache[host]
    ips = []
    try:
        out = subprocess.run(
            ["dig", "+short", "+time=3", "+tries=2", "@8.8.8.8", host, "A"],
            capture_output=True, text=True, timeout=15).stdout
        ips = [l.strip() for l in out.splitlines()
               if l.strip() and l.strip()[0].isdigit() and l.count(".") == 3]
    except Exception:
        pass
    _cache[host] = ips
    return ips


def _patched(host, port, family=0, type=0, proto=0, flags=0):
    if isinstance(host, str) and (host.endswith("huggingface.co")
                                  or host.endswith("hf.co")):
        for ip in _resolve8888(host):
            try:
                res = _orig(ip, port, socket.AF_INET, type, proto, flags)
                if res:
                    return res
            except Exception:
                continue
    return _orig(host, port, family, type, proto, flags)


socket.getaddrinfo = _patched
