"""
src/background_verification.py  –  SmartHire Candidate Background Verification
--------------------------------------------------------------------------------
A simple, in-memory module that lets a human reviewer record and track the
verification status of individual claims found in a candidate's resume (skills,
certifications, degrees, internships, work experience, etc.).

IMPORTANT SAFETY AND ACCURACY RULES
-------------------------------------
1.  A claim is NOT verified merely because it appears in a resume.
2.  Only a reviewer can mark a claim 'Verified' after consulting suitable
    supporting evidence (reference letter, transcript, certificate, etc.).
3.  Missing evidence does NOT automatically mean a claim is false.
4.  'Discrepancy Found' means conflicting information requires further review;
    it is NOT an automatic finding of dishonesty.
5.  This module does NOT collect sensitive identity documents or private
    personal information.
6.  Verification results are completely separate from the ML fit category.
    Never automatically reject a candidate based on verification status alone.

No new dependencies are needed – only Python's standard library is used.
"""

from datetime import date
from typing import Optional


# ---------------------------------------------------------------------------
# Allowed verification statuses (exact strings enforced by add_claim)
# ---------------------------------------------------------------------------

VALID_STATUSES = (
    "Pending",
    "Verified",
    "Unable to Verify",
    "Discrepancy Found",
)

# Allowed claim categories (used for display; not strictly validated so
# reviewers can add new categories without touching this file).
VALID_CATEGORIES = (
    "Skill",
    "Certification",
    "Degree / Education",
    "Internship",
    "Work Experience",
    "Other",
)


# ---------------------------------------------------------------------------
# BackgroundVerifier
# ---------------------------------------------------------------------------

class BackgroundVerifier:
    """
    Records and manages background-verification claims for one candidate.

    Each claim has:
        claim_id        – auto-assigned integer starting at 1
        claim_text      – what the candidate claims (e.g. "BSc Computer Science")
        category        – claim type (Skill, Certification, Degree / Education,
                          Internship, Work Experience, Other)
        status          – one of VALID_STATUSES (starts as "Pending")
        evidence        – reviewer's description of supporting material seen
        reviewer_notes  – free-text notes from the reviewer
        review_date     – date the status was last updated (or None if pending)
    """

    def __init__(self, candidate_name: str = ""):
        """
        Parameters
        ----------
        candidate_name : str
            Optional name label for the report header.  Do NOT store a full
            identity document or government ID here.
        """
        self.candidate_name: str = candidate_name
        self._claims: list[dict] = []   # internal list of claim records
        self._next_id: int = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_claim(
        self,
        claim_text: str,
        category: str = "Other",
        status: str = "Pending",
        evidence: str = "",
        reviewer_notes: str = "",
        review_date: Optional[date] = None,
    ) -> int:
        """
        Record a new claim and return its claim_id.

        Parameters
        ----------
        claim_text     : str  – the specific claim to verify
                                (e.g. "AWS Certified Solutions Architect")
        category       : str  – claim category (see VALID_CATEGORIES for
                                suggestions; any non-empty string is accepted)
        status         : str  – must be one of VALID_STATUSES
        evidence       : str  – description of supporting material the reviewer
                                consulted (leave blank for Pending claims)
        reviewer_notes : str  – free-text notes from the reviewer
        review_date    : date – date the status was last set
                                (defaults to today when status != 'Pending')

        Returns
        -------
        int – the claim_id assigned to the new claim

        Raises
        ------
        ValueError – if claim_text is empty or status is not a valid option
        """
        if not claim_text or not claim_text.strip():
            raise ValueError("claim_text must not be empty.")

        if status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. "
                f"Allowed values: {', '.join(VALID_STATUSES)}"
            )

        # Default review_date to today when the status is already known
        if review_date is None and status != "Pending":
            review_date = date.today()

        claim = {
            "claim_id":       self._next_id,
            "claim_text":     claim_text.strip(),
            "category":       category.strip() if category else "Other",
            "status":         status,
            "evidence":       evidence.strip(),
            "reviewer_notes": reviewer_notes.strip(),
            "review_date":    review_date,
        }

        self._claims.append(claim)
        self._next_id += 1
        return claim["claim_id"]

    def update_claim(
        self,
        claim_id: int,
        status: str,
        evidence: str = "",
        reviewer_notes: str = "",
        review_date: Optional[date] = None,
    ) -> None:
        """
        Update the status (and optionally evidence/notes) of an existing claim.

        Only a reviewer should call this after consulting supporting evidence.

        Parameters
        ----------
        claim_id       : int  – the id returned by add_claim
        status         : str  – new status (must be one of VALID_STATUSES)
        evidence       : str  – description of evidence consulted
        reviewer_notes : str  – any additional reviewer notes
        review_date    : date – date of update (defaults to today)

        Raises
        ------
        ValueError – if status is invalid
        KeyError   – if claim_id does not exist
        """
        if status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. "
                f"Allowed values: {', '.join(VALID_STATUSES)}"
            )

        claim = self._find_claim(claim_id)   # raises KeyError if not found

        claim["status"]         = status
        claim["review_date"]    = review_date if review_date else date.today()

        # Append new evidence/notes rather than silently overwriting
        if evidence.strip():
            existing = claim["evidence"]
            claim["evidence"] = (
                existing + "\n" + evidence.strip() if existing else evidence.strip()
            )
        if reviewer_notes.strip():
            existing = claim["reviewer_notes"]
            claim["reviewer_notes"] = (
                existing + "\n" + reviewer_notes.strip()
                if existing
                else reviewer_notes.strip()
            )

    def get_claim(self, claim_id: int) -> dict:
        """Return a copy of the claim record for the given claim_id."""
        return dict(self._find_claim(claim_id))

    def all_claims(self) -> list[dict]:
        """Return a list of copies of all claim records."""
        return [dict(c) for c in self._claims]

    def get_report(self) -> str:
        """
        Build and return a plain-text verification report for all claims.

        The report is human-readable and safe to print on any terminal.
        It clearly labels each claim's status and notes so a reviewer can
        see at a glance what is pending, verified, or flagged.
        """
        sep = "=" * 60
        lines: list[str] = [
            sep,
            "  SMARTHIRE BACKGROUND VERIFICATION REPORT",
        ]
        if self.candidate_name:
            lines.append(f"  Candidate : {self.candidate_name}")
        lines.append(f"  Total claims : {len(self._claims)}")
        lines.append(sep)

        if not self._claims:
            lines.append("  (No claims recorded yet.)")
        else:
            for claim in self._claims:
                lines.append("")
                lines.append(f"  Claim #{claim['claim_id']}")
                lines.append(f"    Statement  : {claim['claim_text']}")
                lines.append(f"    Category   : {claim['category']}")
                lines.append(f"    Status     : [{claim['status']}]")
                if claim["evidence"]:
                    # Indent multi-line evidence
                    ev_lines = claim["evidence"].splitlines()
                    lines.append(f"    Evidence   : {ev_lines[0]}")
                    for ev in ev_lines[1:]:
                        lines.append(f"                 {ev}")
                else:
                    lines.append("    Evidence   : (none recorded)")
                if claim["reviewer_notes"]:
                    note_lines = claim["reviewer_notes"].splitlines()
                    lines.append(f"    Notes      : {note_lines[0]}")
                    for n in note_lines[1:]:
                        lines.append(f"                 {n}")
                else:
                    lines.append("    Notes      : (none)")
                date_str = (
                    str(claim["review_date"]) if claim["review_date"] else "not yet reviewed"
                )
                lines.append(f"    Review date: {date_str}")
                lines.append("  " + "-" * 56)

        lines.append("")
        lines.append(
            "  NOTE: 'Verified' status is set only by a human reviewer after"
        )
        lines.append(
            "  consulting supporting evidence. Verification results are"
        )
        lines.append(
            "  independent of the ML resume-fit category."
        )
        lines.append(sep)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_claim(self, claim_id: int) -> dict:
        """Return the internal claim dict (not a copy) or raise KeyError."""
        for claim in self._claims:
            if claim["claim_id"] == claim_id:
                return claim
        raise KeyError(f"No claim found with claim_id={claim_id}.")


# ---------------------------------------------------------------------------
# Demo / sample data (run as __main__ to see output)
# ---------------------------------------------------------------------------

def _demo() -> None:
    """
    Demonstrate the module with realistic sample data.
    Shows all four verification statuses and the final report.
    """
    verifier = BackgroundVerifier(candidate_name="Jane Doe (sample)")

    # 1. Add a degree claim — starts Pending
    deg_id = verifier.add_claim(
        claim_text="BSc Computer Science, State University (2021)",
        category="Degree / Education",
        status="Pending",
    )

    # 2. Add a certification — will be Verified after evidence review
    cert_id = verifier.add_claim(
        claim_text="AWS Certified Solutions Architect – Associate (2023)",
        category="Certification",
        status="Pending",
    )

    # 3. Add a work-experience claim
    exp_id = verifier.add_claim(
        claim_text="Software Engineer Intern, TechCorp (Jun–Aug 2022)",
        category="Internship",
        status="Pending",
    )

    # 4. Add a skill claim with a discrepancy
    skill_id = verifier.add_claim(
        claim_text="5 years of production Kubernetes experience",
        category="Skill",
        status="Pending",
    )

    # --- Reviewer actions ---

    # Reviewer verified the certification after seeing the certificate PDF
    verifier.update_claim(
        claim_id=cert_id,
        status="Verified",
        evidence="Reviewed official AWS certificate PDF dated 2023-04-10. "
                 "Certificate ID verified on aws.training portal.",
        reviewer_notes="Valid and current. Expires 2026-04-10.",
        review_date=date(2026, 10, 27),
    )

    # Degree: could not get a response from the university registrar
    verifier.update_claim(
        claim_id=deg_id,
        status="Unable to Verify",
        evidence="Sent two emails to State University registrar; no reply within 10 days.",
        reviewer_notes="Candidate should be asked to provide an official transcript.",
        review_date=date(2026, 10, 27),
    )

    # Skill: reference indicated only 1 year of Kubernetes usage, not 5
    verifier.update_claim(
        claim_id=skill_id,
        status="Discrepancy Found",
        evidence="Reference contact (former manager) stated candidate used Kubernetes "
                 "for approximately 1 year, not 5.",
        reviewer_notes="Discrepancy between resume claim and reference account. "
                       "Requires further clarification with candidate before a conclusion "
                       "is drawn.",
        review_date=date(2026, 10, 27),
    )

    # Internship: still pending, no evidence yet
    # (exp_id claim left as Pending intentionally)

    print(_UNUSED := verifier.get_report())


if __name__ == "__main__":
    _demo()
