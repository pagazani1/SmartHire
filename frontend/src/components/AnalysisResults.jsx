import Icon from "./Icon.jsx";

const fitStyles = {
  "Good Fit": "border-emerald-200 bg-emerald-50 text-emerald-800",
  "Potential Fit": "border-amber-200 bg-amber-50 text-amber-800",
  "No Fit": "border-rose-200 bg-rose-50 text-rose-800",
};

function AnalysisResults({ result }) {
  const matchedSkills = stringList(result.matched_skills);
  const missingSkills = stringList(result.missing_skills);
  const limitations = stringList(result.extraction_limitations);
  const coverage = isRecord(result.coverage_metrics)
    ? result.coverage_metrics
    : {};
  const decisionEvidence = isRecord(result.decision_evidence)
    ? result.decision_evidence
    : {};
  const matchedRequiredGroups = firstNumber(
    decisionEvidence.matched_required_group_count,
    coverage.matched_required_group_count,
  );
  const requiredGroupCount = firstNumber(
    decisionEvidence.required_skill_group_count,
    coverage.required_skill_count,
  );
  const missingRequiredSkills = stringList(
    decisionEvidence.missing_required_skills ?? result.missing_required_skills,
  );
  const decisionExplanation =
    typeof result.decision_explanation === "string" &&
    result.decision_explanation.trim()
      ? result.decision_explanation
      : "Assessment explanation is not available for this response.";
  const finalClassification =
    typeof result.label === "string" && result.label.trim()
      ? result.label
      : "Fit category unavailable";

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-[0_2px_12px_rgba(15,23,42,0.03)] sm:p-7">
        <div className="flex flex-col justify-between gap-4 border-b border-slate-100 pb-5 sm:flex-row sm:items-center">
          <div>
            <p className="text-xs font-semibold tracking-wide text-indigo-700">
              ANALYSIS RESULTS
            </p>
            <h2 className="mt-1 text-xl font-bold tracking-tight text-slate-950">
              Candidate overview
            </h2>
          </div>
          <div
            className={`inline-flex w-fit items-center gap-2 rounded-full border px-3.5 py-2 text-sm font-semibold ${
              fitStyles[result.label] ??
              "border-slate-200 bg-slate-50 text-slate-700"
            }`}
          >
            <span className="h-2 w-2 rounded-full bg-current opacity-70" />
            {finalClassification}
          </div>
        </div>

        <div className="mt-6 rounded-xl border border-indigo-100 bg-indigo-50/50 p-4 sm:p-5">
          <h3 className="text-sm font-semibold text-slate-900">
            Assessment Summary
          </h3>
          <dl className="mt-4 grid gap-4 sm:grid-cols-2">
            <div>
              <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">
                Final classification
              </dt>
              <dd className="mt-1 text-sm font-semibold text-slate-900">
                {finalClassification}
              </dd>
            </div>
            <div>
              <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">
                Required skill groups matched
              </dt>
              <dd className="mt-1 text-sm font-semibold tabular-nums text-slate-900">
                {matchedRequiredGroups !== null && requiredGroupCount !== null
                  ? `${matchedRequiredGroups} of ${requiredGroupCount}`
                  : "Not available"}
              </dd>
            </div>
            <div className="sm:col-span-2">
              <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">
                Missing required skills
              </dt>
              <dd className="mt-2">
                {missingRequiredSkills.length > 0 ? (
                  <ul className="flex flex-wrap gap-2">
                    {missingRequiredSkills.map((skill) => (
                      <li
                        className="max-w-full break-words rounded-lg border border-amber-200 bg-amber-50 px-3 py-1.5 text-sm text-amber-900"
                        key={skill}
                      >
                        {skill}
                      </li>
                    ))}
                  </ul>
                ) : decisionEvidence.structured_evidence_available === true ||
                  (decisionEvidence.structured_evidence_available !== false &&
                    Array.isArray(result.missing_required_skills) &&
                    requiredGroupCount !== null) ? (
                  <p className="text-sm text-slate-600">
                    No missing required skills identified.
                  </p>
                ) : (
                  <p className="text-sm text-slate-500">
                    Required-skill evidence is not available.
                  </p>
                )}
              </dd>
            </div>
            <div className="sm:col-span-2">
              <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">
                Explanation
              </dt>
              <dd className="mt-1 text-sm leading-6 text-slate-700">
                {decisionExplanation}
              </dd>
            </div>
          </dl>
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-2">
        <SkillList
          title="Matched skills"
          skills={matchedSkills}
          emptyText="No matched skills were identified."
          tone="emerald"
        />
        <SkillList
          title="Missing skills"
          skills={missingSkills}
          emptyText="No missing skills were identified."
          tone="amber"
        />
      </div>

      <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-[0_2px_12px_rgba(15,23,42,0.03)] sm:p-7">
        <div className="mb-5">
          <h2 className="text-base font-semibold text-slate-900">
            Skill coverage
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            Coverage reflects detected skill groups in the job description.
          </p>
        </div>
        <div className="grid gap-4 md:grid-cols-3">
          <CoverageCard
            label="Required skills"
            value={coverage.required_skill_coverage}
            matched={coverage.matched_required_group_count}
            total={coverage.required_skill_count}
          />
          <CoverageCard
            label="Preferred skills"
            value={coverage.preferred_skill_coverage}
            matched={coverage.matched_preferred_group_count}
            total={coverage.preferred_skill_count}
          />
          <CoverageCard
            label="Overall skills"
            value={coverage.overall_skill_coverage}
            matched={coverage.matched_skill_count}
            total={
              typeof result.total_jd_skills === "number"
                ? result.total_jd_skills
                : undefined
            }
          />
        </div>
      </section>

      <section className="rounded-2xl border border-amber-200 bg-amber-50/70 p-5 sm:p-6">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-white text-amber-700 shadow-sm ring-1 ring-amber-200">
            <Icon name="shield" className="h-4 w-4" />
          </div>
          <div className="min-w-0">
            <h2 className="text-sm font-semibold text-amber-950">
              Interpretation and limitations
            </h2>
            <p className="mt-1 text-sm leading-5 text-amber-900/80">
              Skill matches are keyword detections, not verified evidence of
              competence. Use this analysis as decision support with human
              review.
            </p>
            {limitations.length > 0 ? (
              <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm leading-5 text-amber-950/80">
                {limitations.map((limitation) => (
                  <li key={limitation}>{limitation}</li>
                ))}
              </ul>
            ) : (
              <p className="mt-3 text-sm text-amber-950/70">
                No additional extraction limitations were provided.
              </p>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

function SkillList({ title, skills, emptyText, tone }) {
  const colors =
    tone === "emerald"
      ? "border-emerald-200 bg-emerald-50 text-emerald-800"
      : "border-amber-200 bg-amber-50 text-amber-900";

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-[0_2px_12px_rgba(15,23,42,0.03)] sm:p-6">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold text-slate-900">{title}</h2>
        <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-semibold tabular-nums text-slate-600">
          {skills.length}
        </span>
      </div>
      {skills.length > 0 ? (
        <ul className="mt-4 flex flex-wrap gap-2">
          {skills.map((skill) => (
            <li
              className={`max-w-full break-words rounded-lg border px-3 py-1.5 text-sm ${colors}`}
              key={skill}
            >
              {skill}
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-4 text-sm text-slate-500">{emptyText}</p>
      )}
    </section>
  );
}

function CoverageCard({ label, value, matched, total }) {
  const hasValue = typeof value === "number" && Number.isFinite(value);
  const percentage = hasValue
    ? Math.max(0, Math.min(100, value * 100))
    : null;
  const hasCounts = typeof matched === "number" && typeof total === "number";

  return (
    <div className="rounded-xl border border-slate-200 p-4">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-sm font-medium text-slate-700">{label}</h3>
        <span className="text-sm font-semibold tabular-nums text-slate-900">
          {percentage === null ? "Not available" : `${percentage.toFixed(0)}%`}
        </span>
      </div>
      {hasCounts && (
        <p className="mt-1 text-xs text-slate-500">
          {matched} of {total} identified
        </p>
      )}
      {percentage !== null && (
        <div
          aria-label={`${label} coverage`}
          aria-valuemax="100"
          aria-valuemin="0"
          aria-valuenow={percentage}
          className="mt-3 h-2 overflow-hidden rounded-full bg-slate-100"
          role="progressbar"
        >
          <div
            className="h-full rounded-full bg-indigo-600"
            style={{ width: `${percentage}%` }}
          />
        </div>
      )}
    </div>
  );
}

function stringList(value) {
  return Array.isArray(value)
    ? value.filter((item) => typeof item === "string" && item.trim())
    : [];
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function firstNumber(...values) {
  return values.find(
    (value) => typeof value === "number" && Number.isFinite(value),
  ) ?? null;
}

export default AnalysisResults;
