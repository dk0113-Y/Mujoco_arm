from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from benchmark.seed_io import SeedFileError, load_seeds, parse_seed_text


class BenchmarkSeedIOTests(unittest.TestCase):
    def test_comments_blank_lines_inline_comments_and_order(self) -> None:
        text = """
        # Benchmark-0 smoke seeds
        42

        7  # a trailing comment is allowed
        0
        20260720 # pilot only
        """
        self.assertEqual(
            parse_seed_text(text, source="memory-seeds"),
            [42, 7, 0, 20260720],
        )

    def test_decimal_spelling_is_normalized_for_duplicate_detection(self) -> None:
        with self.assertRaises(SeedFileError) as caught:
            parse_seed_text("007\n7\n", source="pilot.txt")
        message = str(caught.exception)
        self.assertIn("pilot.txt:2", message)
        self.assertIn("duplicate seed 7", message)
        self.assertIn("line 1", message)

    def test_duplicate_with_comments_and_blank_lines_reports_physical_lines(self) -> None:
        text = "# first line\n\n42 # original\n7\n42 # duplicate\n"
        with self.assertRaises(SeedFileError) as caught:
            parse_seed_text(text, source="smoke_seeds.txt")
        message = str(caught.exception)
        self.assertIn("smoke_seeds.txt:5", message)
        self.assertIn("first declared on line 3", message)

    def test_non_decimal_or_negative_seed_is_rejected_with_line_context(self) -> None:
        invalid_values = (
            "-1",
            "+1",
            "1.0",
            "1 2",
            "0x10",
            "seed=42",
            "１２",
        )
        for invalid in invalid_values:
            with self.subTest(invalid=invalid):
                with self.assertRaises(SeedFileError) as caught:
                    parse_seed_text(f"5\n{invalid}\n", source="bad.txt")
                message = str(caught.exception)
                self.assertIn("bad.txt:2", message)
                self.assertIn("non-negative decimal integer", message)

    def test_empty_and_comment_only_inputs_are_rejected(self) -> None:
        for text in ("", "\n\r\n", "# comment\n  # another\n"):
            with self.subTest(text=repr(text)):
                with self.assertRaisesRegex(SeedFileError, "contains no seeds"):
                    parse_seed_text(text, source="empty.txt")

    def test_load_seeds_accepts_utf8_bom_and_preserves_file_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot_seeds.txt"
            path.write_text("\ufeff9\r\n# comment\r\n3\r\n11 # last\r\n", encoding="utf-8")
            loaded = load_seeds(path)
        self.assertEqual(loaded, [9, 3, 11])

    def test_load_seeds_rejects_missing_file_and_invalid_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.txt"
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                load_seeds(missing)

            invalid = Path(directory) / "invalid.txt"
            invalid.write_bytes(b"42\n\xff\n")
            with self.assertRaisesRegex(SeedFileError, "not valid UTF-8"):
                load_seeds(invalid)

    def test_large_non_negative_integer_is_preserved_exactly(self) -> None:
        seed = 2**128 + 12345
        self.assertEqual(parse_seed_text(f"{seed}\n"), [seed])


if __name__ == "__main__":
    unittest.main()
