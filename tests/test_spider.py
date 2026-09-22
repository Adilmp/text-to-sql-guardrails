"""Spider loader tests.

The Spider *data* is not available unattended (see the module docstring in
``mizan/db/spider.py``), so these tests cover what can be tested without it: graceful
degradation, and the archive-extraction safety check.

The Zip Slip test matters. `ZipFile.extractall` will follow `../` components straight out of
the destination directory, and this archive is fetched over the network from a third party.
A path-traversal entry writing to `~/.ssh/authorized_keys` is the textbook exploitation, so
the guard gets a test rather than a comment.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from mizan.db.spider import SpiderDataset, SpiderUnavailable, _extract_safely, find_local


class TestGracefulDegradation:
    def test_find_local_returns_none_when_absent(self, tmp_path: Path) -> None:
        assert find_local(tmp_path) is None

    def test_dataset_without_database_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SpiderUnavailable, match="database directory not found"):
            SpiderDataset(tmp_path)

    def test_missing_split_file_raises(self, tmp_path: Path) -> None:
        (tmp_path / "database").mkdir()
        dataset = SpiderDataset(tmp_path)
        with pytest.raises(SpiderUnavailable, match="missing split file"):
            dataset.questions("dev")

    def test_missing_db_file_raises(self, tmp_path: Path) -> None:
        (tmp_path / "database" / "concert_singer").mkdir(parents=True)
        dataset = SpiderDataset(tmp_path)
        with pytest.raises(SpiderUnavailable, match="no sqlite file"):
            dataset.db_path("concert_singer")


class TestExtractionSafety:
    def test_path_traversal_entry_is_refused(self, tmp_path: Path) -> None:
        """Zip Slip: an archive member that escapes the destination must not be written."""
        archive = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("../../escaped.txt", "pwned")

        destination = tmp_path / "dest"
        destination.mkdir()
        with zipfile.ZipFile(archive) as zf, pytest.raises(
            SpiderUnavailable, match="path traversal"
        ):
            _extract_safely(zf, destination)

        assert not (tmp_path.parent / "escaped.txt").exists()
        assert not (tmp_path / "escaped.txt").exists()

    def test_absolute_path_entry_is_refused(self, tmp_path: Path) -> None:
        """An archive member with an absolute path must be refused, not written.

        This relies on a pathlib behaviour that is itself a trap worth knowing:
        ``Path("/some/root") / "/tmp/evil.txt"`` evaluates to ``/tmp/evil.txt`` — joining
        with an absolute right-hand side *discards* the left side entirely. So the resolved
        target escapes the destination root, `is_relative_to` returns False, and the guard
        fires. Code that builds paths by concatenating strings instead would have silently
        produced ``/some/root/tmp/evil.txt`` and written the file.
        """
        archive = tmp_path / "abs.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("/tmp/mizan-should-not-exist.txt", "nope")

        destination = tmp_path / "dest"
        destination.mkdir()
        with zipfile.ZipFile(archive) as zf, pytest.raises(SpiderUnavailable, match="traversal"):
            _extract_safely(zf, destination)

        assert not Path("/tmp/mizan-should-not-exist.txt").exists()

    def test_normal_archive_extracts(self, tmp_path: Path) -> None:
        archive = tmp_path / "good.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("spider/dev.json", "[]")

        destination = tmp_path / "dest"
        destination.mkdir()
        with zipfile.ZipFile(archive) as zf:
            _extract_safely(zf, destination)
        assert (destination / "spider" / "dev.json").read_text() == "[]"
