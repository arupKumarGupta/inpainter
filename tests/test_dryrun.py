"""Tests for the dry-run health checker."""
import os
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.dryrun import (
    CheckResult,
    check_output_dir_writable,
    check_python_version,
    check_uv_installed,
    run_all_checks,
)


class TestDryRunChecks(unittest.TestCase):
    def test_python_version_check(self):
        result = check_python_version()
        self.assertIn(result.status, ("PASS", "WARN", "FAIL"))
        self.assertEqual(result.name, "python_version")
        self.assertTrue(len(result.message) > 0)

    def test_uv_installed_check(self):
        result = check_uv_installed()
        self.assertIn(result.status, ("PASS", "WARN"))

    def test_output_dir_writable(self):
        result = check_output_dir_writable()
        self.assertEqual(result.status, "PASS")

    def test_run_all_checks_returns_results(self):
        results = run_all_checks()
        self.assertTrue(len(results) >= 8)
        for r in results:
            self.assertIsInstance(r, CheckResult)
            self.assertIn(r.status, ("PASS", "WARN", "FAIL"))


if __name__ == "__main__":
    unittest.main()
