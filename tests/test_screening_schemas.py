import unittest
from uuid import uuid4

from pydantic import ValidationError

from backend.schemas import (
    PersistentScreeningRequest,
    PersistentScreeningResponse,
    ScreeningEvidenceRead,
    ScreeningRequirementResultRead,
    ScreeningRequirementSnapshot,
)


class TestPersistentScreeningSchemas(unittest.TestCase):
    def test_request_accepts_only_resume_and_job_version_ids(self):
        resume_id = uuid4()
        version_id = uuid4()

        request = PersistentScreeningRequest(
            resume_document_id=resume_id,
            job_description_version_id=version_id,
        )

        self.assertEqual(request.resume_document_id, resume_id)
        self.assertEqual(request.job_description_version_id, version_id)
        with self.assertRaises(ValidationError):
            PersistentScreeningRequest(
                resume_document_id=resume_id,
                job_description_version_id=version_id,
                final_label="Good Fit",
            )

    def test_request_rejects_missing_or_invalid_identifiers(self):
        with self.assertRaises(ValidationError):
            PersistentScreeningRequest(resume_document_id=uuid4())
        with self.assertRaises(ValidationError):
            PersistentScreeningRequest(
                resume_document_id="not-a-uuid",
                job_description_version_id=uuid4(),
            )

    def test_response_contains_results_and_explainable_evidence_not_keys(self):
        result = PersistentScreeningResponse(
            id=uuid4(),
            application_id=uuid4(),
            resume_document_id=uuid4(),
            job_description_version_id=uuid4(),
            run_number=1,
            final_label="Potential Fit",
            model_label="No Fit",
            model_probabilities={
                "Good Fit": 0.1,
                "Potential Fit": 0.2,
                "No Fit": 0.7,
            },
            decision_policy_version="required_skill_group_coverage_v2",
            model_artifact_version="production-model",
            decision_explanation="One of two required groups matched.",
            required_group_count=2,
            matched_required_group_count=1,
            missing_required_group_count=1,
            required_skill_coverage=0.5,
            required_skill_coverage_percentage=50.0,
            evidence=[
                ScreeningEvidenceRead(skill="Python", evidence_kind="resume_detected")
            ],
            requirement_results=[
                ScreeningRequirementResultRead(
                    requirement_group_id=uuid4(),
                    requirement_snapshot=ScreeningRequirementSnapshot(
                        canonical_identifier=str(uuid4()),
                        name="Python",
                        kind="required",
                        skills=["Python"],
                        source="structured",
                    ),
                    kind="required",
                    skills=["Python"],
                    matched=True,
                    matched_skill="Python",
                )
            ],
        )

        response_body = result.model_dump(mode="json")
        self.assertEqual(response_body["final_label"], "Potential Fit")
        self.assertEqual(response_body["model_label"], "No Fit")
        self.assertEqual(response_body["evidence"][0]["skill"], "Python")
        self.assertNotIn("idempotency_key", response_body)
        self.assertNotIn("storage_key", response_body)
        self.assertNotIn("storage_path", response_body)

        with self.assertRaises(ValidationError):
            PersistentScreeningResponse.model_validate(
                {
                    **response_body,
                    "storage_key": "must-not-be-returned",
                }
            )

    def test_requirement_snapshot_supports_structured_and_fallback_requirements(self):
        structured_id = uuid4()
        structured = ScreeningRequirementResultRead(
            requirement_group_id=structured_id,
            requirement_snapshot={
                "canonical_identifier": str(structured_id),
                "name": "Python or Java",
                "kind": "required",
                "skills": ["Python", "Java"],
                "source": "structured",
            },
            kind="required",
            skills=["Python", "Java"],
            matched=False,
            matched_skill=None,
        )
        fallback = ScreeningRequirementResultRead(
            requirement_group_id=None,
            requirement_snapshot={
                "canonical_identifier": "jd-required-0",
                "name": "Python or Java",
                "kind": "required",
                "skills": ["Python", "Java"],
                "source": "jd_text_fallback",
            },
            kind="required",
            skills=["Python", "Java"],
            matched=False,
            matched_skill=None,
        )

        self.assertEqual(structured.requirement_group_id, structured_id)
        self.assertEqual(structured.requirement_snapshot.source, "structured")
        self.assertIsNone(fallback.requirement_group_id)
        self.assertEqual(fallback.requirement_snapshot.source, "jd_text_fallback")

    def test_requirement_result_accepts_legacy_row_without_snapshot(self):
        historical = ScreeningRequirementResultRead(
            requirement_group_id=uuid4(),
            requirement_snapshot=None,
            kind="preferred",
            skills=["Docker"],
            matched=True,
            matched_skill="Docker",
        )

        self.assertIsNone(historical.requirement_snapshot)

    def test_fallback_snapshot_requires_identity_and_skill_options(self):
        with self.assertRaises(ValidationError):
            ScreeningRequirementSnapshot(
                kind="required",
                skills=["Python"],
                source="jd_text_fallback",
            )
        with self.assertRaises(ValidationError):
            ScreeningRequirementSnapshot(
                canonical_identifier="jd-required-0",
                kind="required",
                skills=[],
                source="jd_text_fallback",
            )
        with self.assertRaises(ValidationError):
            ScreeningRequirementResultRead(
                requirement_group_id=None,
                kind="required",
                skills=["Python"],
                matched=False,
                matched_skill=None,
            )


if __name__ == "__main__":
    unittest.main()
