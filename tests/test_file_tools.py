"""Tests for tools/file_tools.py."""

from pathlib import Path

from miniclaw.tools.file_tools import ListFiles, ReadFile, WriteFile, MAX_READ_CHARS


# ============================================================
# ListFiles
# ============================================================


class TestListFiles:
    def test_list_directory(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world")
        (tmp_path / "subdir").mkdir()

        result = ListFiles().run(path=str(tmp_path))
        assert "entries" in result
        names = [e["name"] for e in result["entries"]]
        assert "a.txt" in names
        assert "b.txt" in names
        assert "subdir" in names

    def test_entry_types(self, tmp_path: Path):
        (tmp_path / "file.txt").write_text("x")
        (tmp_path / "dir").mkdir()

        result = ListFiles().run(path=str(tmp_path))
        by_name = {e["name"]: e["type"] for e in result["entries"]}
        assert by_name["file.txt"] == "file"
        assert by_name["dir"] == "directory"

    def test_sorted_output(self, tmp_path: Path):
        (tmp_path / "z.txt").write_text("z")
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "m.txt").write_text("m")

        result = ListFiles().run(path=str(tmp_path))
        names = [e["name"] for e in result["entries"]]
        assert names == sorted(names)

    def test_empty_directory(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = ListFiles().run(path=str(empty))
        assert result["entries"] == []

    def test_nonexistent_path(self):
        result = ListFiles().run(path="/no/such/path/exists")
        assert "error" in result
        assert "does not exist" in result["error"]

    def test_path_is_file(self, tmp_path: Path):
        f = tmp_path / "file.txt"
        f.write_text("hi")
        result = ListFiles().run(path=str(f))
        assert "error" in result
        assert "Not a directory" in result["error"]

    def test_workspace_boundary_blocks_outside_path(self, tmp_path: Path):
        root = tmp_path / "workspace"
        outside = tmp_path / "outside"
        root.mkdir()
        outside.mkdir()

        result = ListFiles(workspace_root=root).run(path=str(outside))

        assert "error" in result
        assert "outside workspace" in result["error"]


# ============================================================
# ReadFile
# ============================================================


class TestReadFile:
    def test_read_text_file(self, tmp_path: Path):
        f = tmp_path / "hello.txt"
        f.write_text("Hello, World!")
        result = ReadFile().run(path=str(f))
        assert result["content"] == "Hello, World!"
        assert result["truncated"] is False
        assert result["chars"] == 13

    def test_read_utf8_content(self, tmp_path: Path):
        f = tmp_path / "utf8.txt"
        f.write_text("你好世界", encoding="utf-8")
        result = ReadFile().run(path=str(f))
        assert result["content"] == "你好世界"

    def test_truncation(self, tmp_path: Path):
        f = tmp_path / "big.txt"
        big_content = "x" * (MAX_READ_CHARS + 1000)
        f.write_text(big_content)
        result = ReadFile().run(path=str(f))
        assert result["truncated"] is True
        assert len(result["content"]) == MAX_READ_CHARS
        assert result["chars"] == MAX_READ_CHARS + 1000

    def test_exact_limit_not_truncated(self, tmp_path: Path):
        f = tmp_path / "exact.txt"
        f.write_text("a" * MAX_READ_CHARS)
        result = ReadFile().run(path=str(f))
        assert result["truncated"] is False
        assert result["chars"] == MAX_READ_CHARS

    def test_nonexistent_file(self):
        result = ReadFile().run(path="/no/such/file.txt")
        assert "error" in result
        assert "does not exist" in result["error"]

    def test_path_is_directory(self, tmp_path: Path):
        d = tmp_path / "dir"
        d.mkdir()
        result = ReadFile().run(path=str(d))
        assert "error" in result
        assert "Not a file" in result["error"]

    def test_non_utf8_file(self, tmp_path: Path):
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\x80\x81\x82\x83")
        result = ReadFile().run(path=str(f))
        assert "error" in result
        assert "UTF-8" in result["error"]

    def test_workspace_boundary_allows_inside_file(self, tmp_path: Path):
        root = tmp_path / "workspace"
        root.mkdir()
        f = root / "note.txt"
        f.write_text("inside", encoding="utf-8")

        result = ReadFile(workspace_root=root).run(path=str(f))

        assert result["content"] == "inside"


# ============================================================
# WriteFile
# ============================================================


class TestWriteFile:
    def test_write_creates_file(self, tmp_path: Path):
        f = tmp_path / "output.txt"
        result = WriteFile().run(path=str(f), content="Hello!")
        assert f.read_text() == "Hello!"
        assert result["chars_written"] == 6

    def test_write_creates_parent_dirs(self, tmp_path: Path):
        f = tmp_path / "a" / "b" / "c" / "deep.txt"
        WriteFile().run(path=str(f), content="deep")
        assert f.read_text() == "deep"
        assert f.parent.exists()

    def test_write_overwrites_existing(self, tmp_path: Path):
        f = tmp_path / "over.txt"
        f.write_text("old")
        WriteFile().run(path=str(f), content="new")
        assert f.read_text() == "new"

    def test_write_empty_content(self, tmp_path: Path):
        f = tmp_path / "empty.txt"
        result = WriteFile().run(path=str(f), content="")
        assert f.read_text() == ""
        assert result["chars_written"] == 0

    def test_write_unicode(self, tmp_path: Path):
        f = tmp_path / "uni.txt"
        content = "你好 🐾"
        result = WriteFile().run(path=str(f), content=content)
        assert f.read_text(encoding="utf-8") == content
        assert result["chars_written"] == len(content)

    def test_write_can_be_disabled(self, tmp_path: Path):
        f = tmp_path / "blocked.txt"

        result = WriteFile(allow_write=False).run(path=str(f), content="nope")

        assert "error" in result
        assert "disabled" in result["error"]
        assert not f.exists()

    def test_write_workspace_boundary_blocks_outside_path(self, tmp_path: Path):
        root = tmp_path / "workspace"
        outside = tmp_path / "outside.txt"
        root.mkdir()

        result = WriteFile(workspace_root=root).run(path=str(outside), content="nope")

        assert "error" in result
        assert "outside workspace" in result["error"]
        assert not outside.exists()
