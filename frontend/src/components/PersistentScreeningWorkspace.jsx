import { useCallback, useEffect, useMemo, useState } from "react";
import Icon from "./Icon.jsx";

const FIT_BADGE_STYLES = {
  "Good Fit": "border-emerald-200 bg-emerald-50 text-emerald-800",
  "Potential Fit": "border-amber-200 bg-amber-50 text-amber-800",
  "No Fit": "border-rose-200 bg-rose-50 text-rose-800",
};

function generateIdempotencyKey() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `ui-${crypto.randomUUID()}`;
  }
  return `ui-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

function PersistentScreeningWorkspace() {
  const [candidates, setCandidates] = useState([]);
  const [jobsById, setJobsById] = useState({});
  const [selectedCandidateId, setSelectedCandidateId] = useState("");
  const [candidateDetail, setCandidateDetail] = useState(null);
  const [selectedApplicationId, setSelectedApplicationId] = useState("");
  const [resumes, setResumes] = useState([]);
  const [selectedResumeId, setSelectedResumeId] = useState("");
  const [jdVersionIdOverride, setJdVersionIdOverride] = useState("");

  const [idempotencyKey, setIdempotencyKey] = useState(() =>
    generateIdempotencyKey(),
  );
  const [isLoadingLists, setIsLoadingLists] = useState(true);
  const [isLoadingContext, setIsLoadingContext] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const [latestResult, setLatestResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState("");
  const [statusNotice, setStatusNotice] = useState("");

  // Load initial candidate and job directories
  const loadDirectories = useCallback(async () => {
    setIsLoadingLists(true);
    setError("");
    try {
      const [candidatesRes, jobsRes] = await Promise.all([
        fetch("/api/candidates/?limit=100"),
        fetch("/api/jobs/?limit=100"),
      ]);
      if (!candidatesRes.ok) {
        throw new Error(
          `Failed to load candidates (HTTP ${candidatesRes.status}).`,
        );
      }
      if (!jobsRes.ok) {
        throw new Error(`Failed to load jobs (HTTP ${jobsRes.status}).`);
      }
      const [candidatesData, jobsData] = await Promise.all([
        candidatesRes.json(),
        jobsRes.json(),
      ]);

      const nextJobsMap = {};
      for (const job of Array.isArray(jobsData) ? jobsData : []) {
        nextJobsMap[job.id] = job;
      }
      setJobsById(nextJobsMap);

      const candidateList = Array.isArray(candidatesData) ? candidatesData : [];
      setCandidates(candidateList);
      if (candidateList.length > 0) {
        setSelectedCandidateId((prev) =>
          candidateList.some((c) => c.id === prev) ? prev : candidateList[0].id,
        );
      }
    } catch (err) {
      setError(
        err instanceof TypeError
          ? "Could not connect to SmartHire API at http://127.0.0.1:8000."
          : err.message || "Unable to load workspace data.",
      );
    } finally {
      setIsLoadingLists(false);
    }
  }, []);

  useEffect(() => {
    loadDirectories();
  }, [loadDirectories]);

  // Load selected candidate's applications and private resumes
  useEffect(() => {
    if (!selectedCandidateId) {
      setCandidateDetail(null);
      setSelectedApplicationId("");
      setResumes([]);
      setSelectedResumeId("");
      return;
    }

    let cancelled = false;
    async function loadCandidateContext() {
      setIsLoadingContext(true);
      setError("");
      setLatestResult(null);
      setStatusNotice("");
      try {
        const [detailRes, resumesRes] = await Promise.all([
          fetch(`/api/candidates/${selectedCandidateId}`),
          fetch(`/api/candidates/${selectedCandidateId}/resumes/?limit=100`),
        ]);
        if (!detailRes.ok) {
          const detail = await readApiError(detailRes);
          throw new Error(
            detail || `Unable to load candidate profile (HTTP ${detailRes.status}).`,
          );
        }
        if (!resumesRes.ok) {
          const detail = await readApiError(resumesRes);
          throw new Error(
            detail || `Unable to load private resumes (HTTP ${resumesRes.status}).`,
          );
        }

        const [detailData, resumesData] = await Promise.all([
          detailRes.json(),
          resumesRes.json(),
        ]);
        if (cancelled) return;

        setCandidateDetail(detailData);
        const apps = Array.isArray(detailData.applications)
          ? detailData.applications
          : [];
        setSelectedApplicationId(apps.length > 0 ? apps[0].id : "");

        const resumeList = Array.isArray(resumesData) ? resumesData : [];
        setResumes(resumeList);
        const availableResume = resumeList.find(
          (doc) => doc.status === "available",
        );
        setSelectedResumeId(
          availableResume
            ? availableResume.id
            : resumeList.length > 0
              ? resumeList[0].id
              : "",
        );
        setJdVersionIdOverride("");
        setIdempotencyKey(generateIdempotencyKey());
      } catch (err) {
        if (!cancelled) {
          setError(err.message || "Failed to load candidate details.");
        }
      } finally {
        if (!cancelled) {
          setIsLoadingContext(false);
        }
      }
    }

    loadCandidateContext();
    return () => {
      cancelled = true;
    };
  }, [selectedCandidateId]);

  const activeApplication = useMemo(() => {
    const apps = candidateDetail?.applications ?? [];
    return apps.find((app) => app.id === selectedApplicationId) ?? null;
  }, [candidateDetail, selectedApplicationId]);

  const activeJob = useMemo(() => {
    if (!activeApplication) return null;
    return jobsById[activeApplication.job_id] ?? null;
  }, [activeApplication, jobsById]);

  const effectiveJdVersionId =
    jdVersionIdOverride.trim() || activeJob?.description_version_id || "";

  // Load application screening history
  const loadApplicationHistory = useCallback(async (applicationId) => {
    if (!applicationId) {
      setHistory([]);
      return;
    }
    try {
      const response = await fetch(
        `/api/applications/${applicationId}/screenings/?limit=50`,
      );
      if (!response.ok) {
        const detail = await readApiError(response);
        setError(
          detail ||
            `Unable to load screening history (HTTP ${response.status}).`,
        );
        return;
      }
      const data = await response.json();
      setHistory(Array.isArray(data) ? data : []);
    } catch {
      // Keep UI functional even if history fetch fails
    }
  }, []);

  useEffect(() => {
    loadApplicationHistory(selectedApplicationId);
  }, [selectedApplicationId, loadApplicationHistory]);

  function handleSelectionMutation(setter, value) {
    setter(value);
    setIdempotencyKey(generateIdempotencyKey());
    setStatusNotice("");
    setError("");
  }

  async function handleRunScreening(event) {
    event.preventDefault();
    if (!selectedApplicationId || !selectedResumeId || !effectiveJdVersionId) {
      setError(
        "Select an application, an available private resume, and a valid job description version.",
      );
      return;
    }

    const trimmedKey = idempotencyKey.trim();
    if (!trimmedKey) {
      setError("An Idempotency-Key is required to run persistent screening.");
      return;
    }

    setIsSubmitting(true);
    setError("");
    setStatusNotice("");

    try {
      const response = await fetch(
        `/api/applications/${selectedApplicationId}/screenings/`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Idempotency-Key": trimmedKey,
          },
          body: JSON.stringify({
            resume_document_id: selectedResumeId,
            job_description_version_id: effectiveJdVersionId,
          }),
        },
      );

      if (!response.ok) {
        const detail = await readApiError(response);
        throw new Error(formatScreeningHttpError(response.status, detail));
      }

      const result = await response.json();
      setLatestResult(result);
      setStatusNotice(
        `Persisted Run #${result.run_number} recorded under Idempotency-Key "${trimmedKey}". Re-submitting with the same key safely replays this exact run.`,
      );
      await loadApplicationHistory(selectedApplicationId);
    } catch (err) {
      setError(
        err instanceof TypeError
          ? "Could not reach the SmartHire backend at http://127.0.0.1:8000."
          : err.message || "Persistent screening failed.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.45fr)_minmax(340px,0.95fr)]">
        {/* Screening Configuration Form */}
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-[0_2px_12px_rgba(15,23,42,0.03)] sm:p-7">
          <div className="mb-6 flex items-start justify-between gap-4">
            <div className="flex items-start gap-4">
              <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-indigo-50 text-indigo-700">
                <Icon name="sparkles" className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-slate-900">
                  Run persistent application screening
                </h2>
                <p className="mt-1 text-sm leading-5 text-slate-500">
                  Evaluates a stored resume against a versioned job description
                  with consent validation, structured skill groups, and audit
                  persistence.
                </p>
              </div>
            </div>
            <button
              className="rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-600 transition hover:bg-slate-50"
              disabled={isLoadingLists}
              onClick={loadDirectories}
              type="button"
            >
              Refresh
            </button>
          </div>

          {isLoadingLists ? (
            <p className="py-6 text-sm text-slate-500">
              Loading candidates, applications, and job versions...
            </p>
          ) : candidates.length === 0 ? (
            <div className="rounded-xl border border-dashed border-slate-300 bg-slate-50 p-5 text-sm text-slate-600">
              No candidates found in the database yet. Create a job, candidate,
              active consents, and private resume via the API first.
            </div>
          ) : (
            <form className="space-y-5" onSubmit={handleRunScreening}>
              <div className="grid gap-4 sm:grid-cols-2">
                <div>
                  <label
                    className="block text-xs font-semibold uppercase tracking-wide text-slate-600"
                    htmlFor="candidate-select"
                  >
                    Candidate
                  </label>
                  <select
                    className="mt-1.5 w-full rounded-xl border border-slate-300 bg-white px-3.5 py-2.5 text-sm text-slate-900 focus:border-indigo-600 focus:outline-none"
                    id="candidate-select"
                    onChange={(e) => setSelectedCandidateId(e.target.value)}
                    value={selectedCandidateId}
                  >
                    {candidates.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.display_name}
                        {c.email ? ` (${c.email})` : ""}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label
                    className="block text-xs font-semibold uppercase tracking-wide text-slate-600"
                    htmlFor="application-select"
                  >
                    Application &amp; Role
                  </label>
                  <select
                    className="mt-1.5 w-full rounded-xl border border-slate-300 bg-white px-3.5 py-2.5 text-sm text-slate-900 focus:border-indigo-600 focus:outline-none"
                    disabled={
                      isLoadingContext ||
                      !(candidateDetail?.applications?.length > 0)
                    }
                    id="application-select"
                    onChange={(e) =>
                      handleSelectionMutation(
                        setSelectedApplicationId,
                        e.target.value,
                      )
                    }
                    value={selectedApplicationId}
                  >
                    {(candidateDetail?.applications ?? []).map((app) => {
                      const job = jobsById[app.job_id];
                      return (
                        <option key={app.id} value={app.id}>
                          {job ? job.title : `Job ${app.job_id.slice(0, 8)}`} —{" "}
                          {app.status}
                        </option>
                      );
                    })}
                  </select>
                </div>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <div>
                  <label
                    className="block text-xs font-semibold uppercase tracking-wide text-slate-600"
                    htmlFor="resume-select"
                  >
                    Private Resume Document
                  </label>
                  <select
                    className="mt-1.5 w-full rounded-xl border border-slate-300 bg-white px-3.5 py-2.5 text-sm text-slate-900 focus:border-indigo-600 focus:outline-none"
                    disabled={isLoadingContext || resumes.length === 0}
                    id="resume-select"
                    onChange={(e) =>
                      handleSelectionMutation(
                        setSelectedResumeId,
                        e.target.value,
                      )
                    }
                    value={selectedResumeId}
                  >
                    {resumes.length === 0 ? (
                      <option value="">No resumes uploaded</option>
                    ) : (
                      resumes.map((doc) => (
                        <option key={doc.id} value={doc.id}>
                          {doc.original_filename} [{doc.status}]
                        </option>
                      ))
                    )}
                  </select>
                </div>

                <div>
                  <label
                    className="block text-xs font-semibold uppercase tracking-wide text-slate-600"
                    htmlFor="jd-version-input"
                  >
                    JD Version ID{" "}
                    {activeJob?.description_version
                      ? `(Latest v${activeJob.description_version})`
                      : ""}
                  </label>
                  <input
                    className="mt-1.5 w-full rounded-xl border border-slate-300 bg-slate-50 px-3.5 py-2.5 font-mono text-xs text-slate-800 focus:border-indigo-600 focus:bg-white focus:outline-none"
                    id="jd-version-input"
                    onChange={(e) =>
                      handleSelectionMutation(
                        setJdVersionIdOverride,
                        e.target.value,
                      )
                    }
                    placeholder="Auto-filled from selected job"
                    type="text"
                    value={effectiveJdVersionId}
                  />
                </div>
              </div>

              {/* Idempotency Key Bar */}
              <div className="rounded-xl border border-slate-200 bg-slate-50/80 p-3.5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <label
                    className="text-xs font-semibold uppercase tracking-wide text-slate-600"
                    htmlFor="idempotency-key-input"
                  >
                    Idempotency-Key Header
                  </label>
                  <button
                    className="text-xs font-semibold text-indigo-700 hover:text-indigo-900"
                    onClick={() => {
                      setIdempotencyKey(generateIdempotencyKey());
                      setStatusNotice("");
                    }}
                    type="button"
                  >
                    Generate new key (new run)
                  </button>
                </div>
                <input
                  className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 font-mono text-xs text-slate-800 focus:border-indigo-600 focus:outline-none"
                  id="idempotency-key-input"
                  maxLength={255}
                  onChange={(e) => setIdempotencyKey(e.target.value)}
                  type="text"
                  value={idempotencyKey}
                />
                <p className="mt-1.5 text-[11px] leading-4 text-slate-500">
                  Re-submitting with the same key and inputs safely returns the
                  existing run; generate a new key to record a new sequential
                  run.
                </p>
              </div>

              <div className="flex flex-col gap-3 border-t border-slate-100 pt-5 sm:flex-row sm:items-center sm:justify-between">
                <p className="text-xs leading-5 text-slate-500">
                  Requires active <code className="font-mono">screening</code>{" "}
                  and <code className="font-mono">document_storage</code>{" "}
                  consents.
                </p>
                <button
                  className="inline-flex min-h-11 items-center justify-center gap-2 rounded-xl bg-indigo-700 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-800 disabled:cursor-not-allowed disabled:bg-slate-300"
                  disabled={
                    isSubmitting ||
                    isLoadingContext ||
                    !selectedApplicationId ||
                    !selectedResumeId ||
                    !effectiveJdVersionId
                  }
                  type="submit"
                >
                  {isSubmitting ? (
                    <>
                      <span
                        aria-hidden="true"
                        className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white"
                      />
                      Screening &amp; Persisting...
                    </>
                  ) : (
                    <>
                      <Icon name="sparkles" className="h-4 w-4" />
                      Run Persistent Screening
                      <Icon name="arrowRight" className="h-4 w-4" />
                    </>
                  )}
                </button>
              </div>

              {statusNotice && (
                <p className="rounded-lg border border-indigo-200 bg-indigo-50 px-4 py-3 text-xs leading-5 text-indigo-900">
                  {statusNotice}
                </p>
              )}
              {error && (
                <p
                  aria-live="polite"
                  className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm leading-5 text-rose-800"
                  role="alert"
                >
                  {error}
                </p>
              )}
            </form>
          )}
        </section>

        {/* Active Role Requirements Summary */}
        <aside className="space-y-6">
          <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-[0_2px_12px_rgba(15,23,42,0.03)]">
            <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
              Target Role Context
            </h3>
            {activeJob ? (
              <div className="mt-3 space-y-4">
                <div>
                  <p className="text-base font-bold text-slate-900">
                    {activeJob.title}
                  </p>
                  <p className="text-xs text-slate-500">
                    {[activeJob.department, activeJob.location]
                      .filter(Boolean)
                      .join(" · ") || "Active requisition"}{" "}
                    {activeJob.description_version
                      ? `• JD v${activeJob.description_version}`
                      : ""}
                  </p>
                </div>

                <div>
                  <p className="text-xs font-semibold text-slate-700">
                    Structured Required Skill Groups (
                    {activeJob.required_skill_groups?.length ?? 0})
                  </p>
                  {activeJob.required_skill_groups?.length > 0 ? (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {activeJob.required_skill_groups.map((grp, idx) => (
                        <span
                          className="rounded-md border border-indigo-100 bg-indigo-50/70 px-2.5 py-1 text-xs font-medium text-indigo-900"
                          key={`${idx}-${grp.join("|")}`}
                        >
                          {grp.join(" OR ")}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="mt-1 text-xs text-slate-500">
                      None defined — will fall back to JD text extraction.
                    </p>
                  )}
                </div>

                {activeJob.preferred_skill_groups?.length > 0 && (
                  <div>
                    <p className="text-xs font-semibold text-slate-700">
                      Preferred Skill Groups (
                      {activeJob.preferred_skill_groups.length})
                    </p>
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {activeJob.preferred_skill_groups.map((grp, idx) => (
                        <span
                          className="rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs text-slate-700"
                          key={`${idx}-${grp.join("|")}`}
                        >
                          {grp.join(" OR ")}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p className="mt-3 text-sm text-slate-500">
                Select a candidate application to inspect its job description
                version and requirement groups.
              </p>
            )}
          </section>
        </aside>
      </div>

      {/* Detailed Explainable Result Card */}
      {latestResult && <PersistentResultCard result={latestResult} />}

      {/* Application Screening History */}
      <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-[0_2px_12px_rgba(15,23,42,0.03)] sm:p-7">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-4">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Application Screening History
            </h2>
            <p className="mt-0.5 text-xs text-slate-500">
              Immutable audit log of persisted screening runs for this
              application (newest first).
            </p>
          </div>
          <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold tabular-nums text-slate-700">
            {history.length} {history.length === 1 ? "run" : "runs"}
          </span>
        </div>

        {history.length === 0 ? (
          <p className="py-6 text-sm text-slate-500">
            No persistent screening runs recorded for this application yet.
          </p>
        ) : (
          <div className="mt-4 overflow-x-auto">
            <table className="w-full border-collapse text-left text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-xs font-semibold uppercase tracking-wide text-slate-500">
                  <th className="py-2.5 pr-4">Timestamp</th>
                  <th className="px-4 py-2.5">Final Decision</th>
                  <th className="px-4 py-2.5">ML Prediction</th>
                  <th className="px-4 py-2.5">Required Coverage</th>
                  <th className="px-4 py-2.5">Matched / Missing Required</th>
                  <th className="pl-4 py-2.5">Policy</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {history.map((item) => {
                  const pct =
                    typeof item.required_skill_coverage === "number"
                      ? `${Number((item.required_skill_coverage * 100).toFixed(2))}%`
                      : "Fallback";
                  return (
                    <tr className="align-top" key={item.id}>
                      <td className="whitespace-nowrap py-3 pr-4 text-xs text-slate-600">
                        {new Date(item.created_at).toLocaleString()}
                      </td>
                      <td className="whitespace-nowrap px-4 py-3">
                        <span
                          className={`inline-flex rounded-full border px-2.5 py-0.5 text-xs font-semibold ${
                            FIT_BADGE_STYLES[item.final_label] ??
                            "border-slate-200 bg-slate-50 text-slate-700"
                          }`}
                        >
                          {item.final_label}
                        </span>
                      </td>
                      <td className="whitespace-nowrap px-4 py-3 text-xs font-medium text-slate-700">
                        {item.model_label}
                      </td>
                      <td className="whitespace-nowrap px-4 py-3 text-xs tabular-nums text-slate-700">
                        {item.matched_required_group_count}/
                        {item.required_group_count} ({pct})
                      </td>
                      <td className="px-4 py-3 text-xs text-slate-600">
                        {item.matched_required_skills.length > 0 && (
                          <div className="text-emerald-800">
                            Matched: {item.matched_required_skills.join(", ")}
                          </div>
                        )}
                        {item.missing_required_skills.length > 0 && (
                          <div className="text-amber-800">
                            Missing: {item.missing_required_skills.join(", ")}
                          </div>
                        )}
                        {item.matched_required_skills.length === 0 &&
                          item.missing_required_skills.length === 0 && (
                            <span className="text-slate-400">
                              No required skill items
                            </span>
                          )}
                      </td>
                      <td className="whitespace-nowrap pl-4 py-3 font-mono text-[11px] text-slate-500">
                        {item.decision_policy_version}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function PersistentResultCard({ result }) {
  const resumeDetected = (result.evidence ?? [])
    .filter((e) => e.evidence_kind === "resume_detected")
    .map((e) => e.skill);
  const matchedReq = (result.requirement_results ?? []).filter((r) => r.matched);
  const missingReq = (result.requirement_results ?? []).filter(
    (r) => !r.matched,
  );
  const modelProbs = Object.entries(result.model_probabilities ?? {});

  return (
    <section
      aria-live="polite"
      className="rounded-2xl border border-slate-200 bg-white p-5 shadow-[0_2px_12px_rgba(15,23,42,0.03)] sm:p-7"
    >
      <div className="flex flex-col justify-between gap-4 border-b border-slate-100 pb-5 sm:flex-row sm:items-center">
        <div>
          <p className="text-xs font-semibold tracking-wide text-indigo-700">
            PERSISTENT SCREENING RESULT &middot; RUN #{result.run_number}
          </p>
          <h2 className="mt-1 text-xl font-bold tracking-tight text-slate-950">
            Explainable Candidate Assessment
          </h2>
        </div>
        <div className="flex flex-wrap items-center gap-2.5">
          <span
            className={`inline-flex items-center gap-2 rounded-full border px-3.5 py-1.5 text-sm font-semibold ${
              FIT_BADGE_STYLES[result.final_label] ??
              "border-slate-200 bg-slate-50 text-slate-700"
            }`}
          >
            Final: {result.final_label}
          </span>
          <span className="inline-flex items-center rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-semibold text-slate-700">
            ML Model: {result.model_label}
          </span>
        </div>
      </div>

      {/* Top Metrics */}
      <div className="mt-6 grid gap-4 md:grid-cols-3">
        <div className="rounded-xl border border-slate-200 p-4">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
            Required Skill Group Coverage
          </p>
          <p className="mt-1.5 text-2xl font-bold tabular-nums text-slate-900">
            {result.required_skill_coverage_percentage !== null
              ? `${Number(result.required_skill_coverage_percentage.toFixed(2))}%`
              : "N/A"}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            {result.matched_required_group_count} of{" "}
            {result.required_group_count} required groups satisfied
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 p-4">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
            ML Model Probabilities (Separate from Policy)
          </p>
          <div className="mt-2 space-y-1">
            {modelProbs.map(([label, prob]) => (
              <div
                className="flex items-center justify-between text-xs"
                key={label}
              >
                <span className="font-medium text-slate-700">{label}</span>
                <span className="font-mono tabular-nums text-slate-900">
                  {(prob * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 p-4">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
            Audit Metadata
          </p>
          <p className="mt-1.5 font-mono text-xs text-slate-700">
            Policy: {result.decision_policy_version}
          </p>
          <p className="mt-1 truncate font-mono text-[11px] text-slate-500">
            Model: {result.model_artifact_version}
          </p>
        </div>
      </div>

      {/* Decision Explanation */}
      <div className="mt-5 rounded-xl border border-indigo-100 bg-indigo-50/50 p-4">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-indigo-900">
          Decision Policy Explanation
        </h3>
        <p className="mt-1.5 text-sm leading-6 text-slate-800">
          {result.decision_explanation || "No explanation provided."}
        </p>
      </div>

      {/* Requirement Group Evaluation */}
      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-200 p-4">
          <h3 className="text-sm font-semibold text-slate-900">
            Satisfied Requirement Groups ({matchedReq.length})
          </h3>
          {matchedReq.length > 0 ? (
            <ul className="mt-3 space-y-2">
              {matchedReq.map((req, idx) => (
                <li
                  className="flex items-center justify-between gap-2 rounded-lg border border-emerald-200 bg-emerald-50/70 px-3 py-2 text-xs text-emerald-950"
                  key={`matched-${idx}`}
                >
                  <span>
                    <strong className="uppercase">[{req.kind}]</strong>{" "}
                    {req.skills.join(" OR ")}
                    {req.matched_skill ? ` → matched ${req.matched_skill}` : ""}
                  </span>
                  {req.requirement_snapshot?.source && (
                    <span className="rounded bg-white/80 px-1.5 py-0.5 font-mono text-[10px] text-emerald-800">
                      {req.requirement_snapshot.source}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-2 text-xs text-slate-500">
              No requirement groups were satisfied.
            </p>
          )}
        </div>

        <div className="rounded-xl border border-slate-200 p-4">
          <h3 className="text-sm font-semibold text-slate-900">
            Missing Requirement Groups ({missingReq.length})
          </h3>
          {missingReq.length > 0 ? (
            <ul className="mt-3 space-y-2">
              {missingReq.map((req, idx) => (
                <li
                  className="flex items-center justify-between gap-2 rounded-lg border border-amber-200 bg-amber-50/70 px-3 py-2 text-xs text-amber-950"
                  key={`missing-${idx}`}
                >
                  <span>
                    <strong className="uppercase">[{req.kind}]</strong>{" "}
                    {req.skills.join(" OR ")}
                  </span>
                  {req.requirement_snapshot?.source && (
                    <span className="rounded bg-white/80 px-1.5 py-0.5 font-mono text-[10px] text-amber-800">
                      {req.requirement_snapshot.source}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-2 text-xs text-slate-500">
              No missing requirement groups.
            </p>
          )}
        </div>
      </div>

      {/* Skills Detected in Resume */}
      <div className="mt-5 rounded-xl border border-slate-200 p-4">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-600">
          Skills Detected in Resume ({resumeDetected.length})
        </h3>
        {resumeDetected.length > 0 ? (
          <div className="mt-2.5 flex flex-wrap gap-1.5">
            {resumeDetected.map((skill) => (
              <span
                className="rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-medium text-slate-800"
                key={skill}
              >
                {skill}
              </span>
            ))}
          </div>
        ) : (
          <p className="mt-1.5 text-xs text-slate-500">
            No canonical vocabulary skills detected in the resume.
          </p>
        )}
      </div>
    </section>
  );
}

async function readApiError(response) {
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string") return payload.detail;
    if (Array.isArray(payload.detail)) {
      return payload.detail
        .map((item) => item.msg)
        .filter(Boolean)
        .join(" ");
    }
  } catch {
    return "";
  }
  return "";
}

function formatScreeningHttpError(status, detail) {
  const suffix = detail ? ` ${detail}` : "";
  if (status === 403) {
    return `Consent or access check failed.${suffix}`;
  }
  if (status === 404) {
    return `Application, resume, or JD version not found.${suffix}`;
  }
  if (status === 409) {
    return `Idempotency conflict: this key was already used with different inputs.${suffix}`;
  }
  if (status === 400 || status === 422) {
    return `Screening validation error.${suffix}`;
  }
  return `Screening request failed (HTTP ${status}).${suffix}`;
}

export default PersistentScreeningWorkspace;