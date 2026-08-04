import unittest

from FYP2.services.auth_service import hash_password, verify_password


class PasswordTests(unittest.TestCase):
    def test_pbkdf2_password_round_trip(self):
        hashed = hash_password("correct horse battery staple")
        self.assertTrue(hashed.startswith("pbkdf2_sha256$"))
        self.assertTrue(verify_password("correct horse battery staple", hashed))
        self.assertFalse(verify_password("wrong password", hashed))

    def test_legacy_password_is_still_accepted_for_upgrade(self):
        self.assertTrue(verify_password("legacy", "legacy"))
        self.assertFalse(verify_password("wrong", "legacy"))


if __name__ == "__main__":
    unittest.main()
