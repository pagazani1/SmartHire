"""
tests/test_background_verification.py
--------------------------------------
Unit tests for src/background_verification.py

Run from the project root with:
    python -m unittest tests/test_background_verification.py -v

No third-party test framework is needed – only Python's built-in unittest.
"""

import unittest
from datetime import date

# We import from src because the project uses the src/ package convention.
from src.background_verification import (
    BackgroundVerifier,
    VALID_STATUSES,
    VALID_CATEGORIES,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fresh_verifier() -> BackgroundVerifier:
    """Return a new verifier with a sample candidate name."""
    return BackgroundVerifier(candidate_name="Test Candidate")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestAddPendingClaim(unittest.TestCase):
    """TC-01  Recording a pending claim."""

    def test_claim_is_stored_with_pending_status(self):
        v = _fresh_verifier()
        cid = v.add_claim(
            claim_text="BSc Computer Science, Example University",
            category="Degree / Education",
            status="Pending",
        )
        claim = v.get_claim(cid)
        self.assertEqual(claim["status"], "Pending")
        self.assertEqual(claim["claim_text"], "BSc Computer Science, Example University")
        self.assertEqual(claim["category"], "Degree / Education")

    def test_pending_claim_has_no_review_date(self):
        v = _fresh_verifier()
        cid = v.add_claim(claim_text="Python (5 years)", status="Pending")
        claim = v.get_claim(cid)
        self.assertIsNone(claim["review_date"],
            "A Pending claim should not have a review date set automatically.")

    def test_claim_id_is_auto_incremented(self):
        v = _fresh_verifier()
        id1 = v.add_claim(claim_text="Claim A")
        id2 = v.add_claim(claim_text="Claim B")
        id3 = v.add_claim(claim_text="Claim C")
        self.assertEqual([id1, id2, id3], [1, 2, 3])

    def test_evidence_is_empty_by_default_for_pending(self):
        v = _fresh_verifier()
        cid = v.add_claim(claim_text="AWS Certified Developer", status="Pending")
        claim = v.get_claim(cid)
        self.assertEqual(claim["evidence"], "")


class TestMarkVerifiedAfterEvidenceReview(unittest.TestCase):
    """TC-02  Marking a claim Verified only after a reviewer supplies evidence."""

    def test_claim_starts_pending_then_reviewer_verifies(self):
        v = _fresh_verifier()
        cid = v.add_claim(
            claim_text="AWS Certified Solutions Architect – Associate",
            category="Certification",
            status="Pending",
        )
        # Confirm it is Pending before any review
        self.assertEqual(v.get_claim(cid)["status"], "Pending")

        # Reviewer consults the certificate and updates the claim
        v.update_claim(
            claim_id=cid,
            status="Verified",
            evidence="Reviewed official AWS certificate PDF dated 2023-04-10.",
            reviewer_notes="Certificate ID confirmed on aws.training portal.",
            review_date=date(2026, 10, 27),
        )

        claim = v.get_claim(cid)
        self.assertEqual(claim["status"], "Verified")
        self.assertIn("AWS certificate PDF", claim["evidence"])   # text from evidence arg
        self.assertIn("aws.training", claim["reviewer_notes"])    # text from reviewer_notes arg
        self.assertEqual(claim["review_date"], date(2026, 10, 27))

    def test_verified_claim_has_nonempty_evidence(self):
        v = _fresh_verifier()
        cid = v.add_claim(claim_text="Django (3 years)", category="Skill")
        v.update_claim(
            claim_id=cid,
            status="Verified",
            evidence="Reference manager confirmed 3 years of Django usage.",
        )
        claim = v.get_claim(cid)
        self.assertNotEqual(claim["evidence"], "",
            "Verified claim should record what evidence was consulted.")

    def test_update_claim_sets_review_date_to_today_when_not_given(self):
        v = _fresh_verifier()
        cid = v.add_claim(claim_text="MSc Data Science, Tech University")
        v.update_claim(cid, status="Verified", evidence="Transcript verified.")
        claim = v.get_claim(cid)
        self.assertEqual(claim["review_date"], date.today())


class TestUnableToVerifyStatus(unittest.TestCase):
    """TC-03  Recording an Unable to Verify status."""

    def test_unable_to_verify_is_stored_correctly(self):
        v = _fresh_verifier()
        cid = v.add_claim(
            claim_text="Google Project Management Certificate (2022)",
            category="Certification",
            status="Pending",
        )
        v.update_claim(
            claim_id=cid,
            status="Unable to Verify",
            evidence="No response from the certifying body after two follow-up emails.",
            reviewer_notes="Candidate should supply a verification code or transcript.",
        )
        claim = v.get_claim(cid)
        self.assertEqual(claim["status"], "Unable to Verify")

    def test_unable_to_verify_does_not_mean_claim_is_false(self):
        """
        Unable to Verify must never be interpreted as proof of dishonesty.
        The status should simply reflect that verification was not possible
        at this time, not that the claim is untrue.
        """
        v = _fresh_verifier()
        cid = v.add_claim(claim_text="Software Intern at StartupXYZ (2023)")
        v.update_claim(
            claim_id=cid,
            status="Unable to Verify",
            evidence="Company closed, no response from former manager.",
        )
        claim = v.get_claim(cid)
        # The claim text is unchanged – we record inability to verify, not falsity
        self.assertIn("StartupXYZ", claim["claim_text"])
        self.assertEqual(claim["status"], "Unable to Verify")


class TestDiscrepancyFoundStatus(unittest.TestCase):
    """TC-04  Recording a Discrepancy Found status."""

    def test_discrepancy_found_is_stored_correctly(self):
        v = _fresh_verifier()
        cid = v.add_claim(
            claim_text="5 years of production Kubernetes experience",
            category="Skill",
        )
        v.update_claim(
            claim_id=cid,
            status="Discrepancy Found",
            evidence="Reference contact stated approximately 1 year, not 5.",
            reviewer_notes="Requires clarification; does not automatically indicate dishonesty.",
        )
        claim = v.get_claim(cid)
        self.assertEqual(claim["status"], "Discrepancy Found")
        self.assertIn("1 year", claim["evidence"])

    def test_discrepancy_is_not_automatic_rejection(self):
        """
        Discrepancy Found means conflicting information needs further review.
        This test documents that the status is purely informational and not
        a definitive finding of dishonesty.
        """
        v = _fresh_verifier()
        cid = v.add_claim(claim_text="Led a team of 20 engineers")
        v.update_claim(
            claim_id=cid,
            status="Discrepancy Found",
            evidence="LinkedIn shows team size of 8.",
            reviewer_notes="Further discussion with candidate recommended.",
        )
        claim = v.get_claim(cid)
        # Status is Discrepancy Found – it is NOT 'Rejected' or 'False'
        self.assertEqual(claim["status"], "Discrepancy Found")
        self.assertNotIn("Rejected", claim["status"])
        self.assertNotIn("False", claim["status"])


class TestVerificationReport(unittest.TestCase):
    """TC-05  Displaying a readable verification report."""

    def test_report_contains_candidate_name(self):
        v = BackgroundVerifier(candidate_name="Alice Smith")
        v.add_claim(claim_text="BSc Mathematics")
        report = v.get_report()
        self.assertIn("Alice Smith", report)

    def test_report_shows_all_four_statuses(self):
        v = _fresh_verifier()
        id1 = v.add_claim(claim_text="Claim A – pending")
        id2 = v.add_claim(claim_text="Claim B – verified")
        id3 = v.add_claim(claim_text="Claim C – unable to verify")
        id4 = v.add_claim(claim_text="Claim D – discrepancy")

        v.update_claim(id2, status="Verified",          evidence="Transcript seen.")
        v.update_claim(id3, status="Unable to Verify",  evidence="No reply from registrar.")
        v.update_claim(id4, status="Discrepancy Found", evidence="LinkedIn mismatch.")

        report = v.get_report()
        self.assertIn("Pending",            report)
        self.assertIn("Verified",           report)
        self.assertIn("Unable to Verify",   report)
        self.assertIn("Discrepancy Found",  report)

    def test_report_contains_all_claim_texts(self):
        v = _fresh_verifier()
        v.add_claim(claim_text="AWS Certified Developer")
        v.add_claim(claim_text="BSc Computer Engineering")
        report = v.get_report()
        self.assertIn("AWS Certified Developer",    report)
        self.assertIn("BSc Computer Engineering",   report)

    def test_report_includes_disclaimer_about_independence(self):
        """Report must mention that verification is independent of the ML fit category."""
        v = _fresh_verifier()
        report = v.get_report()
        self.assertIn("independent", report.lower())

    def test_empty_verifier_report_says_no_claims(self):
        v = BackgroundVerifier()
        report = v.get_report()
        self.assertIn("No claims recorded", report)

    def test_report_is_a_string(self):
        v = _fresh_verifier()
        v.add_claim(claim_text="Some claim")
        self.assertIsInstance(v.get_report(), str)


class TestInvalidStatusRejected(unittest.TestCase):
    """TC-06  Rejecting invalid status values."""

    def test_add_claim_rejects_invalid_status(self):
        v = _fresh_verifier()
        with self.assertRaises(ValueError) as ctx:
            v.add_claim(claim_text="Some skill", status="Approved")
        self.assertIn("Approved", str(ctx.exception))

    def test_update_claim_rejects_invalid_status(self):
        v = _fresh_verifier()
        cid = v.add_claim(claim_text="Some skill")
        with self.assertRaises(ValueError) as ctx:
            v.update_claim(cid, status="Rejected")
        self.assertIn("Rejected", str(ctx.exception))

    def test_add_claim_rejects_empty_string_status(self):
        v = _fresh_verifier()
        with self.assertRaises(ValueError):
            v.add_claim(claim_text="Some skill", status="")

    def test_all_valid_statuses_are_accepted(self):
        """Every entry in VALID_STATUSES must be accepted without error."""
        v = _fresh_verifier()
        for status in VALID_STATUSES:
            cid = v.add_claim(
                claim_text=f"Claim for status: {status}",
                status=status,
            )
            self.assertEqual(v.get_claim(cid)["status"], status)

    def test_add_claim_rejects_empty_claim_text(self):
        v = _fresh_verifier()
        with self.assertRaises(ValueError):
            v.add_claim(claim_text="   ")

    def test_update_claim_rejects_unknown_claim_id(self):
        v = _fresh_verifier()
        with self.assertRaises(KeyError):
            v.update_claim(claim_id=999, status="Verified", evidence="N/A")


class TestAdditionalBehaviour(unittest.TestCase):
    """Additional edge-case and correctness checks."""

    def test_all_claims_returns_correct_count(self):
        v = _fresh_verifier()
        for i in range(5):
            v.add_claim(claim_text=f"Claim {i}")
        self.assertEqual(len(v.all_claims()), 5)

    def test_get_claim_returns_a_copy_not_reference(self):
        """Mutating the returned dict must not change the internal record."""
        v = _fresh_verifier()
        cid = v.add_claim(claim_text="Python expert")
        copy = v.get_claim(cid)
        copy["status"] = "Verified"   # tamper with the copy
        self.assertEqual(v.get_claim(cid)["status"], "Pending",
            "get_claim() should return a copy, not a reference to the internal record.")

    def test_claim_text_is_stripped_of_whitespace(self):
        v = _fresh_verifier()
        cid = v.add_claim(claim_text="  Docker  ")
        self.assertEqual(v.get_claim(cid)["claim_text"], "Docker")

    def test_evidence_is_appended_not_overwritten_on_second_update(self):
        v = _fresh_verifier()
        cid = v.add_claim(claim_text="Machine Learning (3 years)")
        v.update_claim(cid, status="Pending",
                       evidence="First reference contacted.")
        v.update_claim(cid, status="Verified",
                       evidence="Second reference confirmed.")
        claim = v.get_claim(cid)
        self.assertIn("First reference",  claim["evidence"])
        self.assertIn("Second reference", claim["evidence"])

    def test_default_category_is_other(self):
        v = _fresh_verifier()
        cid = v.add_claim(claim_text="Some uncategorised claim")
        self.assertEqual(v.get_claim(cid)["category"], "Other")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
