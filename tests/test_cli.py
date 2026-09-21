"""Tests for the CLI entry point: argument parsing, --version, --json,
--no-color, and exit codes -- independent of whether torch is
installed."""
from __future__ import annotations

import json

import pytest

import torch_native_dropout_train_none_guard.core as core
from torch_native_dropout_train_none_guard.cli import main


def test_version_flag(capsys):
    code = main(["--version"])
    out = capsys.readouterr().out
    assert code == 0
    assert "torch-native-dropout-train-none-guard" in out


def test_json_output_is_valid_json_and_reports_status(capsys):
    torch = pytest.importorskip("torch")
    code = main(["--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert "torch_version" in report
    assert report["torch_version"] == torch.__version__
    assert code in (0, 1)


def test_text_output_no_color_has_no_ansi_escapes(capsys):
    pytest.importorskip("torch")
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "\x1b[" not in out


def test_torch_unavailable_json_output_reports_error_and_exit_2(capsys, monkeypatch):
    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis and guarding")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload == {"error": "torch is required for diagnosis and guarding"}
    assert code == 2


def test_torch_unavailable_text_output_shows_fail_headline_and_exit_2(capsys, monkeypatch):
    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis and guarding")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "[X] torch unavailable: torch is required for diagnosis and guarding" in out
    assert code == 2


def _fake_case(p, train_arg, diverges, guard_ok):
    eager_mask_all_true = False if (train_arg is None or train_arg is True) else True
    if diverges:
        compiled_mask_all_true = not eager_mask_all_true
    else:
        compiled_mask_all_true = eager_mask_all_true
    guarded_mask_all_true = eager_mask_all_true if guard_ok else compiled_mask_all_true
    return {
        "name": f"p={p}_train={train_arg}",
        "p": p,
        "train_arg": train_arg,
        "eager_mask_all_true": eager_mask_all_true,
        "compiled_mask_all_true": compiled_mask_all_true,
        "guarded_mask_all_true": guarded_mask_all_true,
        "diverges_unguarded": diverges,
        "guard_restores_agreement": guarded_mask_all_true == eager_mask_all_true,
    }


def _fake_report(none_diverges, guard_ok, explicit_diverges=False, explicit_guard_ok=True):
    cases = [
        _fake_case(0.5, None, none_diverges, guard_ok),
        _fake_case(0.5, True, explicit_diverges, explicit_guard_ok),
        _fake_case(0.5, False, explicit_diverges, explicit_guard_ok),
    ]
    none_cases = [c for c in cases if c["train_arg"] is None]
    explicit_cases = [c for c in cases if c["train_arg"] is not None]
    return {
        "torch_version": "0.0.0-fake",
        "issue_url": "https://github.com/pytorch/pytorch/issues/197846",
        "fix_pr_url": "https://github.com/pytorch/pytorch/pull/197854",
        "cases": cases,
        "any_none_divergence_reproduced": any(c["diverges_unguarded"] for c in none_cases),
        "guard_fully_restores_none_cases": all(c["guard_restores_agreement"] for c in none_cases),
        "explicit_cases_never_diverge_unguarded": all(not c["diverges_unguarded"] for c in explicit_cases),
        "explicit_cases_unaffected_by_guard": all(c["guard_restores_agreement"] for c in explicit_cases),
    }


def test_no_divergence_shows_info_message_not_fail(capsys, monkeypatch):
    monkeypatch.setattr(core, "diagnose", lambda **kwargs: _fake_report(False, True))
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "[i] no train=None divergence reproduced" in out
    assert code == 0


def test_guard_failed_label_shown_when_guard_ineffective(capsys, monkeypatch):
    monkeypatch.setattr(core, "diagnose", lambda **kwargs: _fake_report(True, False))
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "GUARD-FAILED" in out
    assert "[X] guard did NOT restore agreement for at least one train=None case" in out
    assert code == 1


def test_divergence_reproduced_shows_fail_headline_and_ok_guard(capsys, monkeypatch):
    monkeypatch.setattr(core, "diagnose", lambda **kwargs: _fake_report(True, True))
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "[X] train=None eager-vs-compiled divergence reproduced on this host" in out
    assert "[OK] guard restores eager agreement for every train=None case" in out
    assert "MPS-vs-CPU eager divergence" in out
    assert code == 0


def test_explicit_case_divergence_shows_fail(capsys, monkeypatch):
    monkeypatch.setattr(
        core,
        "diagnose",
        lambda **kwargs: _fake_report(True, True, explicit_diverges=True, explicit_guard_ok=True),
    )
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "[X] guard changed behavior for an explicit train=True/False case" in out
    assert code == 1
