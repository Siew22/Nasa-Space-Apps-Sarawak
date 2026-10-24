"""Quarantined scripts kept only for traceability. Do not use their outputs as results.

See src/legacy/README.md for why each script was retired. Importing anything from
this package stops the program unless FLAME_ALLOW_LEGACY=1 is set, so a legacy
script cannot silently overwrite the current reports.
"""

import os

if os.environ.get("FLAME_ALLOW_LEGACY") != "1":
    raise SystemExit(
        "src.legacy contains retired scripts whose outputs are not valid evidence "
        "(see src/legacy/README.md). Set FLAME_ALLOW_LEGACY=1 only to reproduce old artefacts."
    )
