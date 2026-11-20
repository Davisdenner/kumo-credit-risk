#Fetch and verify the exact UCI archive used for the published results#
from hashlib import file_digest
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "uci.zip"
URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"


def verify(path):
    with path.open("rb") as source:
        if file_digest(source, "sha256").hexdigest() != SHA256:
            raise ValueError(f"Unexpected UCI archive checksum: {path}")


def prepare():
    if ARCHIVE.exists():
        verify(ARCHIVE)
        return ARCHIVE
    ARCHIVE.parent.mkdir(exist_ok=True)
    with NamedTemporaryFile(dir=ARCHIVE.parent, delete=False) as tmp:
        candidate = Path(tmp.name)
    try:
        with urlopen(URL, timeout=60) as response, candidate.open("wb") as dest:
            copyfileobj(response, dest)
        verify(candidate)
        candidate.replace(ARCHIVE)
    finally:
        candidate.unlink(missing_ok=True)
    return ARCHIVE
