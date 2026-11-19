import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AuthProvider } from "../context/AuthContext.jsx";
import CandidatesView from "./CandidatesView.jsx";
import JobsView from "./JobsView.jsx";
import { PersistentResultCard } from "./PersistentScreeningWorkspace.jsx";
import { mockApi, signInStorage } from "../test/helpers.js";

const wrap = (ui) => render(<AuthProvider>{ui}</AuthProvider>);

describe("JobsView", () => {
  it("blocks saving unsupported skills and suggests alternatives", async () => {
    signInStorage();
    const { calls } = mockApi({
      "GET /api/auth/me": { body: { id: "1", display_name: "T", role: "recruiter" } },
      "GET /api/jobs/?limit=100": { body: [] },
      "GET /api/skills/": { body: { skills: ["Python", "API Development", "PostgreSQL"] } },
    });
    wrap(<JobsView />);
    await userEvent.click(await screen.findByRole("button", { name: /new job/i }));
    await userEvent.type(screen.getByLabelText(/^title/i), "Backend");
    await userEvent.type(screen.getByLabelText(/job description/i), "Build things.");
    await userEvent.type(screen.getByLabelText(/required skill groups/i), "Python{enter}REST API");
    expect(await screen.findByText(/some skills are not supported/i)).toBeInTheDocument();
    expect(screen.getByText(/did you mean api development/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /create job/i }));
    expect(screen.getByText(/replace the unsupported skills/i)).toBeInTheDocument();
    expect(calls.some((c) => c.method === "POST" && c.url === "/api/jobs/")).toBe(false);
  });

  it("creates a job with alternatives parsed into groups", async () => {
    signInStorage();
    let posted = null;
    mockApi({
      "GET /api/auth/me": { body: { id: "1", display_name: "T", role: "recruiter" } },
      "GET /api/jobs/?limit=100": { body: [] },
      "GET /api/skills/": { body: { skills: ["Python", "Java", "PostgreSQL"] } },
      "POST /api/jobs/": ({ body }) => {
        posted = JSON.parse(body);
        return { status: 201, body: { id: "j1" } };
      },
    });
    wrap(<JobsView />);
    await userEvent.click(await screen.findByRole("button", { name: /new job/i }));
    await userEvent.type(screen.getByLabelText(/^title/i), "Backend Engineer");
    await userEvent.type(screen.getByLabelText(/job description/i), "Build APIs.");
    await userEvent.type(screen.getByLabelText(/required skill groups/i), "Python | Java{enter}PostgreSQL");
    await userEvent.click(screen.getByRole("button", { name: /create job/i }));

    await waitFor(() => expect(posted).not.toBeNull());
    expect(posted.required_skill_groups).toEqual([["Python", "Java"], ["PostgreSQL"]]);
    expect(posted.title).toBe("Backend Engineer");
    expect(posted).not.toHaveProperty("created_by_recruiter_id"); // the server decides the owner
  });

  it("warns that a job without required groups falls back to the model", async () => {
    signInStorage();
    mockApi({
      "GET /api/auth/me": { body: { id: "1", display_name: "T", role: "recruiter" } },
      "GET /api/jobs/?limit=100": { body: [] },
    });
    wrap(<JobsView />);
    await userEvent.click(await screen.findByRole("button", { name: /new job/i }));
    expect(screen.getByText(/no required skill groups/i)).toBeInTheDocument();
    expect(screen.getByText(/manual review/i)).toBeInTheDocument();
  });

  it("shows server validation errors instead of failing silently", async () => {
    signInStorage();
    mockApi({
      "GET /api/auth/me": { body: { id: "1", display_name: "T", role: "recruiter" } },
      "GET /api/jobs/?limit=100": { body: [] },
      "POST /api/jobs/": { status: 422, body: { detail: [{ loc: ["body", "title"], msg: "too long" }] } },
    });
    wrap(<JobsView />);
    await userEvent.click(await screen.findByRole("button", { name: /new job/i }));
    await userEvent.type(screen.getByLabelText(/^title/i), "X");
    await userEvent.type(screen.getByLabelText(/job description/i), "Y");
    await userEvent.click(screen.getByRole("button", { name: /create job/i }));
    expect(await screen.findByText(/title: too long/i)).toBeInTheDocument();
  });
});

const CANDIDATE = { id: "c1", display_name: "Asha Rao", email: "asha@example.test", phone: null, created_at: "2026-01-01T00:00:00Z", anonymized_at: null };
const detailRoutes = (consents, extra = {}) => ({
  "GET /api/auth/me": { body: { id: "1", display_name: "T", role: "recruiter" } },
  "GET /api/candidates/?limit=100": { body: [CANDIDATE] },
  "GET /api/jobs/?limit=100": { body: [{ id: "j1", title: "Backend Engineer", status: "open" }] },
  "GET /api/candidates/c1": { body: { ...CANDIDATE, applications: [{ id: "a1", job_id: "j1", status: "received", created_at: "2026-01-01T00:00:00Z" }] } },
  "GET /api/candidates/c1/consents/": { body: consents },
  "GET /api/candidates/c1/resumes/?limit=100": { body: [] },
  ...extra,
});
const ACTIVE_STORAGE = { id: "k1", candidate_id: "c1", purpose: "document_storage", status: "granted", captured_at: "2026-01-02T00:00:00Z", withdrawn_at: null, expires_at: null, notice_version: "v1", capture_method: "x", proof_reference: null };

async function openCandidate() {
  wrap(<CandidatesView onOpenScreening={vi.fn()} />);
  await userEvent.click(await screen.findByRole("button", { name: /asha rao/i }));
}

describe("CandidatesView", () => {
  it("blocks resume upload until an active storage consent exists", async () => {
    signInStorage();
    mockApi(detailRoutes([]));
    await openCandidate();
    expect(await screen.findByText(/no consent recorded/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^upload$/i })).toBeDisabled();
  });

  it("enables upload with an active consent and sends file plus consent id", async () => {
    signInStorage();
    let form = null;
    mockApi(detailRoutes([ACTIVE_STORAGE], {
      "POST /api/candidates/c1/resumes/": ({ body }) => {
        form = body;
        return { status: 201, body: { id: "r1" } };
      },
    }));
    await openCandidate();
    const input = await screen.findByLabelText(/pdf resume/i);
    const file = new File(["%PDF-1.4 test"], "cv.pdf", { type: "application/pdf" });
    await userEvent.upload(input, file);
    await userEvent.click(screen.getByRole("button", { name: /^upload$/i }));
    await waitFor(() => expect(form).not.toBeNull());
    expect(form.get("consent_record_id")).toBe("k1");
    expect(form.get("file").name).toBe("cv.pdf");
  });

  it("treats an expired consent as not usable", async () => {
    signInStorage();
    mockApi(detailRoutes([{ ...ACTIVE_STORAGE, expires_at: "2000-01-01T00:00:00Z" }]));
    await openCandidate();
    expect(await screen.findByText("expired")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^upload$/i })).toBeDisabled();
  });

  it("withdraws a consent only after confirmation", async () => {
    signInStorage();
    const { calls } = mockApi(detailRoutes([ACTIVE_STORAGE], {
      "POST /api/consents/k1/withdraw": { body: { ...ACTIVE_STORAGE, status: "withdrawn" } },
    }));
    const confirm = vi.spyOn(window, "confirm");
    await openCandidate();
    const button = await screen.findByRole("button", { name: /^withdraw$/i });

    confirm.mockReturnValueOnce(false);
    await userEvent.click(button);
    expect(calls.some((c) => c.url.includes("/withdraw"))).toBe(false);

    confirm.mockReturnValueOnce(true);
    await userEvent.click(button);
    await waitFor(() => expect(calls.some((c) => c.url === "/api/consents/k1/withdraw")).toBe(true));
  });

  it("erases personal data only after an explicit confirmation", async () => {
    signInStorage();
    const { calls } = mockApi(detailRoutes([], {
      "POST /api/candidates/c1/anonymize": { body: { candidate_id: "c1", anonymized_at: "2026-02-01T00:00:00Z", resumes_deleted: 0, consents_withdrawn: 0 } },
    }));
    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false);
    await openCandidate();
    const button = await screen.findByRole("button", { name: /erase this candidate/i });
    await userEvent.click(button);
    expect(calls.some((c) => c.url.endsWith("/anonymize"))).toBe(false);
    confirm.mockReturnValueOnce(true);
    await userEvent.click(button);
    await waitFor(() => expect(calls.some((c) => c.url === "/api/candidates/c1/anonymize")).toBe(true));
  });

  it("shows a forbidden error from the server", async () => {
    signInStorage();
    mockApi({
      ...detailRoutes([]),
      "GET /api/candidates/c1": { status: 403, body: { detail: "You do not have permission to access this candidate." } },
    });
    await openCandidate();
    expect(await screen.findByText(/do not have permission/i)).toBeInTheDocument();
  });
});

const RESULT = {
  run_number: 1,
  final_label: "Good Fit",
  model_label: "No Fit",
  model_probabilities: { "Good Fit": 0.1, "No Fit": 0.8, "Potential Fit": 0.1 },
  required_skill_coverage_percentage: 100,
  matched_required_group_count: 3,
  required_group_count: 3,
  decision_policy_version: "required_skill_group_coverage_v2",
  model_artifact_version: "abc",
  decision_explanation: "Candidate matched 3 out of 3 required skill groups.",
  evidence: [],
  requirement_results: [],
};

describe("PersistentResultCard", () => {
  it("renders the final policy label cleanly without disagreement banners", () => {
    render(<PersistentResultCard result={RESULT} />);
    expect(screen.getByText(/final: good fit/i)).toBeInTheDocument();
    expect(screen.queryByText(/text model disagrees/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/text-model reference:/i)).not.toBeInTheDocument();
  });

  it("flags manual review when there was no usable required-skill evidence", () => {
    render(<PersistentResultCard result={{ ...RESULT, required_skill_coverage_percentage: null }} />);
    expect(screen.getByText(/manual review needed/i)).toBeInTheDocument();
    expect(screen.queryByText(/disagrees/i)).not.toBeInTheDocument();
  });
});
