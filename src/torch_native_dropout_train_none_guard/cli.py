"""Command-line interface: diagnose the torch.compile/eager and
MPS/CPU divergence for aten.native_dropout's undocumented train=None
argument, using the shared semantic-color design system.
"""
from __future__ import annotations

import argparse
import json
import sys

from .style import print_fields, resolve_style, section, status_headline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-native-dropout-train-none-guard",
        description=(
            "Diagnose whether the currently installed torch build's "
            "torch.compile (Inductor) disagrees with eager mode on "
            "torch.native_dropout(input, p, train=None) -- an "
            "undocumented argument value where CPU eager applies "
            "dropout but torch.compile silently skips it (pytorch/"
            "pytorch#197846). Also documents the separate MPS-vs-CPU "
            "eager divergence for the same argument (known limitation: "
            "not covered by CI, no MPS runner in GitHub Actions)."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-native-dropout-train-none-guard {__version__}")
        return 0

    from .core import TorchUnavailableError, diagnose

    try:
        report = diagnose()
    except TorchUnavailableError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            style = resolve_style(no_color_flag=args.no_color)
            print(status_headline(style, "fail", f"torch unavailable: {exc}"))
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        ok = (
            report["guard_fully_restores_none_cases"]
            and report["explicit_cases_never_diverge_unguarded"]
            and report["explicit_cases_unaffected_by_guard"]
        )
        return 0 if ok else 1

    style = resolve_style(no_color_flag=args.no_color)
    print_fields([("torch version", report["torch_version"])])
    print_fields([("upstream issue", report["issue_url"])])

    if report["any_none_divergence_reproduced"]:
        print(status_headline(style, "fail", "train=None eager-vs-compiled divergence reproduced on this host"))
    else:
        print(status_headline(style, "info", "no train=None divergence reproduced on this host's installed torch build"))

    if report["guard_fully_restores_none_cases"]:
        print(status_headline(style, "ok", "guard restores eager agreement for every train=None case"))
    else:
        print(status_headline(style, "fail", "guard did NOT restore agreement for at least one train=None case"))

    if report["explicit_cases_never_diverge_unguarded"] and report["explicit_cases_unaffected_by_guard"]:
        print(status_headline(style, "ok", "explicit train=True/False cases are unaffected (already agree, guard is a no-op there)"))
    else:
        print(status_headline(style, "fail", "guard changed behavior for an explicit train=True/False case -- unexpected"))

    print(status_headline(style, "warn", "MPS-vs-CPU eager divergence (same issue) is NOT covered by this tool's CI -- see README limitations"))

    section("cases (p, train -> eager/compiled/guarded mask_all_true?)")
    for c in report["cases"]:
        print_fields(
            [
                (
                    f"p={c['p']:.1f} train={str(c['train_arg']):5s}",
                    f"eager={str(c['eager_mask_all_true']):5s} "
                    f"compiled={str(c['compiled_mask_all_true']):5s} "
                    f"guarded={str(c['guarded_mask_all_true']):5s}  "
                    f"{'DIVERGED' if c['diverges_unguarded'] else 'ok':9s} "
                    f"{'guard-ok' if c['guard_restores_agreement'] else 'GUARD-FAILED'}",
                )
            ]
        )

    ok = (
        report["guard_fully_restores_none_cases"]
        and report["explicit_cases_never_diverge_unguarded"]
        and report["explicit_cases_unaffected_by_guard"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
