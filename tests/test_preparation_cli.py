"""Preparation command integration tests, including fail-closed artifact handling."""
import json
import sys

import pytest

from scripts.prepare_passages import main


def test_preparation_cli_and_overwrite_protection(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.json"
    output = tmp_path / "passages.jsonl"
    corpus.write_text(json.dumps([{
        "title": "Synthetic", "body": "First sentence. Another supporting statement.",
        "url": "https://example.org/doc", "source": "Synthetic",
    }]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "prepare_passages", "--corpus", str(corpus), "--output", str(output),
        "--expected-articles", "1",
    ])
    main()
    report = json.loads(output.with_suffix(".report.json").read_text(encoding="utf-8"))
    assert report["input_articles"] == report["processed_articles"] == 1
    assert report["failures"] == []
    assert report["passage_count"] == len(output.read_text(encoding="utf-8").splitlines())
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        main()
    assert output.read_bytes() == original


def test_failed_document_never_saves_partial_dataset(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.json"
    output = tmp_path / "passages.jsonl"
    corpus.write_text(json.dumps([
        {"title": "Good", "body": "Valid content.", "url": "a", "source": "Synthetic"},
        {"title": "Bad", "body": "", "url": "b", "source": "Synthetic"},
    ]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "prepare_passages", "--corpus", str(corpus), "--output", str(output),
        "--expected-articles", "2",
    ])
    with pytest.raises(RuntimeError, match="no partial"):
        main()
    assert not output.exists()
    report = json.loads(output.with_suffix(".report.json").read_text(encoding="utf-8"))
    assert report["processed_articles"] == 1
    assert report["failures"][0]["doc_id"] == "doc_0001"
