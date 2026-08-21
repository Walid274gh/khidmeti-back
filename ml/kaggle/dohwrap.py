#!/usr/bin/env python3
"""Contournement DNS cassé (Android/termux) : patch socket.getaddrinfo via
DNS-over-HTTPS sur IP directe (1.1.1.1 — aucun lookup requis, cert valide pour l'IP).
SNI/vérif TLS restent sur le vrai hostname (seul le connect TCP utilise l'IP).
Usage : python3 dohwrap.py <script.py> [args...]
"""
import json, socket, sys, urllib.request

_orig = socket.getaddrinfo
_cache = {}


def _doh(host):
    if host not in _cache:
        req = urllib.request.Request(
            f"https://1.1.1.1/dns-query?name={host}&type=A",
            headers={"accept": "application/dns-json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            ans = json.loads(r.read().decode()).get("Answer") or []
        ips = [a["data"] for a in ans if a.get("type") == 1]
        if not ips:
            raise socket.gaierror(f"DoH: no A record for {host}")
        _cache[host] = ips
    return _cache[host]


def _patched(host, port, family=0, type=0, proto=0, flags=0):
    try:
        return _orig(host, port, family, type, proto, flags)
    except socket.gaierror:
        return _orig(_doh(host)[0], port, family, type, proto, flags)


socket.getaddrinfo = _patched

if __name__ == "__main__":
    path = sys.argv[1]
    sys.argv = sys.argv[1:]
    exec(compile(open(path).read(), path, "exec"), {"__name__": "__main__", "__file__": path})
