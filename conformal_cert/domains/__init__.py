"""Domain adapters for the conformal recovery-deadline certificate (Paper B).

Each adapter is a self-contained Simplex-style testbed that produces the plain
recovery-time arrays the domain-agnostic ``conformal_cert.core`` consumes. The
spacecraft adapter lives in ``program/conformal_rta.py`` (the WS2 origin); the
testbeds here exist to show the certificate is a *mechanism*, not a spacecraft
trick — the same suppression pathology and the same conformal cure appear in an
unrelated plant.

- ``pendulum``: a torque-controlled inverted pendulum with an actuator
  effectiveness/sign fault (the 1-DOF analog of the spacecraft gain fault).
- ``proximity_ops``: a translational station-keeping regime (3-DOF double
  integrator) with thruster faults — the SECOND operating regime for Paper D
  ("assurance transfers, skill does not"). The same certificate, recalibrated on a
  structurally different plant; ``_basilisk_proxops_check`` audits the descope
  against Basilisk's 6-DOF translational dynamics.
"""
