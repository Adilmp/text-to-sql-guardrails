"""Spider benchmark loader.

Spider (Yu et al., 2018) is the standard cross-domain Text-to-SQL benchmark. Running against
it is what lets a claim like "X% execution accuracy" be compared with published numbers
instead of being an unfalsifiable statement about one hand-written suite.

Two design constraints shaped this module
------------------------------------------
**It must degrade honestly.** Spider's canonical distribution is a Google Drive link that
rate-limits and requires a confirmation token, and the machine this was built on had a very
slow connection. If the data cannot be fetched, this module raises a clear error and the
project reports synthetic-only results. It never substitutes an approximation and never lets
a caller quote a Spider number that was not measured.

**Only the dev split is used, and only its databases.** Spider ships one SQLite file per
database. The loader indexes them and exposes each as a normal :class:`Catalog`, so the
entire guardrail and evaluation stack runs against Spider unchanged — which is the real
value: it tests that the architecture is not secretly coupled to the demo schema.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from .. import schema as schema_pkg
from ..errors import MizanError
from ..logging import get_logger

logger = get_logger("db.spider")

#: Mirrors, tried in order.
#:
#: **Status as of 2026-09-21: none of these work unattended, and that is a data-availability
#: problem rather than a bug here.** The HuggingFace repo ``xlangai/spider`` publishes only
#: ``train``/``validation`` *parquet* files — question text and gold SQL, but **not** the
#: SQLite databases. Execution accuracy compares what two queries *return*, so without the
#: databases the benchmark cannot be run at all; question text alone would only support
#: string-comparison metrics, which this project deliberately rejects (see DECISIONS.md D9).
#:
#: The databases live only in the original Yale distribution, behind a Google Drive link
#: that rate-limits and requires an interactive confirmation token. That is a manual step.
#:
#: To install Spider by hand::
#:
#:     # download spider.zip from https://yale-lily.github.io/spider
#:     unzip spider.zip -d data/
#:     # expected result: data/spider/database/<db_id>/<db_id>.sqlite  +  data/spider/dev.json
#:
#: ``find_local`` picks it up automatically from there and nothing else needs changing.
SPIDER_MIRRORS: tuple[str, ...] = (
    "https://huggingface.co/datasets/xlangai/spider/resolve/main/spider.zip",
    "https://drive.usercontent.google.com/download?id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m&export=download&confirm=t",
)


class SpiderUnavailable(MizanError):
    """Spider data is not present and could not be fetched."""

    code = "spider_unavailable"


@dataclass(frozen=True)
class SpiderQuestion:
    db_id: str
    question: str
    gold_sql: str


class SpiderDataset:
    """A local Spider installation."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.database_dir = root / "database"
        if not self.database_dir.is_dir():
            raise SpiderUnavailable(
                "spider database directory not found", expected=str(self.database_dir)
            )

    @property
    def db_ids(self) -> list[str]:
        return sorted(p.name for p in self.database_dir.iterdir() if p.is_dir())

    def db_path(self, db_id: str) -> Path:
        path = self.database_dir / db_id / f"{db_id}.sqlite"
        if not path.exists():
            raise SpiderUnavailable(f"no sqlite file for db_id {db_id!r}", path=str(path))
        return path

    def catalog(self, db_id: str) -> schema_pkg.Catalog:
        """A Catalog for one Spider database.

        No glossary is applied — Spider is English-only, so the Arabic alias layer has
        nothing to contribute and pretending otherwise would be noise in the prompt.
        """
        return schema_pkg.Catalog.from_sqlite(self.db_path(db_id))

    def questions(self, split: str = "dev", limit: int | None = None) -> list[SpiderQuestion]:
        path = self.root / f"{split}.json"
        if not path.exists():
            raise SpiderUnavailable(f"missing split file {path.name}", path=str(path))
        records = json.loads(path.read_text(encoding="utf-8"))
        questions = [
            SpiderQuestion(
                db_id=record["db_id"],
                question=record["question"],
                gold_sql=record["query"],
            )
            for record in records
        ]
        return questions[:limit] if limit else questions


def find_local(data_dir: Path) -> SpiderDataset | None:
    """Return an already-extracted Spider installation, or ``None``."""
    for candidate in (data_dir / "spider", data_dir / "spider" / "spider"):
        if (candidate / "database").is_dir():
            logger.info("found local spider", extra={"root": str(candidate)})
            return SpiderDataset(candidate)
    return None


def download(data_dir: Path, *, timeout_s: float = 600.0) -> SpiderDataset:
    """Fetch and extract Spider. Raises :class:`SpiderUnavailable` on any failure.

    Deliberately not called automatically by the eval harness: a ~100 MB download is a
    decision the operator makes, not a side effect of running a test suite.
    """
    if existing := find_local(data_dir):
        return existing

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "spider.zip"
    errors: list[str] = []

    for url in SPIDER_MIRRORS:
        try:
            logger.info("downloading spider", extra={"url": url})
            with httpx.stream("GET", url, timeout=timeout_s, follow_redirects=True) as response:
                response.raise_for_status()
                with archive.open("wb") as handle:
                    for chunk in response.iter_bytes(chunk_size=1 << 20):
                        handle.write(chunk)
            break
        except (httpx.HTTPError, OSError) as exc:
            errors.append(f"{url}: {exc}")
            logger.warning("mirror failed", extra={"url": url, "error": str(exc)})
            archive.unlink(missing_ok=True)
    else:
        raise SpiderUnavailable(
            "every Spider mirror failed; download it manually and extract to "
            f"{data_dir / 'spider'}",
            attempts=errors,
        )

    try:
        with zipfile.ZipFile(archive) as zf:
            _extract_safely(zf, data_dir)
    except zipfile.BadZipFile as exc:
        archive.unlink(missing_ok=True)
        raise SpiderUnavailable(
            "downloaded file is not a valid zip (the mirror likely served an HTML error page)"
        ) from exc
    finally:
        archive.unlink(missing_ok=True)

    dataset = find_local(data_dir)
    if dataset is None:
        raise SpiderUnavailable("extraction produced no database directory")
    logger.info("spider ready", extra={"databases": len(dataset.db_ids)})
    return dataset


def _extract_safely(zf: zipfile.ZipFile, destination: Path) -> None:
    """Extract, refusing any member that would escape ``destination``.

    ``ZipFile.extractall`` will happily write to ``../../etc/`` if the archive says so —
    the Zip Slip vulnerability. Since this archive comes off the network, every resolved
    path is checked against the destination root before anything is written.
    """
    root = destination.resolve()
    for member in zf.namelist():
        target = (root / member).resolve()
        if not target.is_relative_to(root):
            raise SpiderUnavailable(
                "archive contains a path traversal entry; refusing to extract",
                member=member,
            )
    zf.extractall(root)
