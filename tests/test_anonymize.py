"""
Tests de la pseudonymisation des identifiants (RGPD).
Exécution : python -m unittest discover tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "producer"))
import anonymize  # noqa: E402

KEY = b"cle-de-test"


class TestPseudonymize(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(anonymize.pseudonymize("C1231006815", KEY),
                         anonymize.pseudonymize("C1231006815", KEY))

    def test_hides_original_and_keeps_account_type(self):
        client = anonymize.pseudonymize("C1231006815", KEY)
        merchant = anonymize.pseudonymize("M1979787155", KEY)
        self.assertNotIn("1231006815", client)
        self.assertTrue(client.startswith("C") and merchant.startswith("M"))
        self.assertEqual(len(client), 1 + anonymize.DIGEST_CHARS)

    def test_depends_on_key_and_account(self):
        base = anonymize.pseudonymize("C1231006815", KEY)
        self.assertNotEqual(base, anonymize.pseudonymize("C1231006815", b"autre-cle"))
        self.assertNotEqual(base, anonymize.pseudonymize("C1231006816", KEY))

    def test_missing_key_is_an_error(self):
        old = os.environ.pop("ANON_KEY", None)
        try:
            with self.assertRaises(RuntimeError):
                anonymize.get_key()
        finally:
            if old is not None:
                os.environ["ANON_KEY"] = old


if __name__ == "__main__":
    unittest.main()
