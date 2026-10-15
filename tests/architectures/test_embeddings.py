import subprocess
import sys

SPHERICAL_HARMONICS_MODULE = (
    "otter.models.architectures.spherical_harmonics.spherical_harmonics_ylm"
)


def test_importing_embeddings_does_not_load_spherical_harmonics() -> None:
    # A fresh interpreter, since other tests may have loaded the module.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, otter.models.architectures.embeddings; "
            f"assert {SPHERICAL_HARMONICS_MODULE!r} not in sys.modules",
        ],
        check=True,
    )
