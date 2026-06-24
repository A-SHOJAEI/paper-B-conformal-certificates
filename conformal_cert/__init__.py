"""Conformal recovery-deadline certificates — domain-agnostic core (Paper B).

The mechanism originated as the spacecraft RTA certificate in
``program/conformal_rta.py`` (WS2); this package factors the pure math out of
the spacecraft testbed so the same certificate runs on any Simplex-style
domain (see ``conformal_cert.domains``). The spacecraft module re-exports
from here, so its committed evidence stays reproducible.
"""
