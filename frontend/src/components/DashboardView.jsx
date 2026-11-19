import { useEffect, useState } from "react";
import { describeError, useApi } from "../lib/api.js";
import { Alert, Badge, Card, SectionTitle, Spinner, PrimaryButton, SecondaryButton } from "./ui.jsx";

function Stat({ label, value, hint }) {
  return (
    <div className="rounded-xl border border-slate-200 p-4">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-1.5 text-3xl font-bold tabular-nums text-slate-900">{value}</p>
      {hint ? <p className="mt-1 text-xs text-slate-500">{hint}</p> : null}
    </div>
  );
}

function DashboardView({ onNavigate }) {
  const api = useApi();
  const [jobs, setJobs] = useState(null);
  const [candidates, setCandidates] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [jobList, candidateList] = await Promise.all([
          api.get("/api/jobs/?limit=100"),
          api.get("/api/candidates/?limit=100"),
        ]);
        if (cancelled) return;
        setJobs(Array.isArray(jobList) ? jobList : []);
        setCandidates(Array.isArray(candidateList) ? candidateList : []);
      } catch (err) {
        if (!cancelled) setError(describeError(err));
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [api]);

  if (error) return <Alert kind="error">{error}</Alert>;
  if (!jobs || !candidates) return <Spinner label="Loading your workspace" />;

  const openJobs = jobs.filter((job) => job.status === "open").length;
  const jobsWithoutRequirements = jobs.filter(
    (job) => (job.required_skill_groups ?? []).length === 0,
  );
  const activeCandidates = candidates.filter((c) => !c.anonymized_at);

  const steps = [
    {
      done: jobs.length > 0,
      title: "Create a job with required skill groups",
      detail: "Required groups drive the Good / Potential / No Fit decision.",
      action: ["Go to Jobs", "jobs"],
    },
    {
      done: candidates.length > 0,
      title: "Add a candidate to one of your jobs",
      detail: "Each candidate applies to a job you own.",
      action: ["Go to Candidates", "candidates"],
    },
    {
      done: false,
      title: "Record consent, upload a PDF resume, then screen",
      detail: "Screening needs storage and screening consent on file.",
      action: ["Open Screening", "persistent"],
    },
  ];

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-3">
        <Stat hint={`${openJobs} open`} label="Jobs" value={jobs.length} />
        <Stat
          hint={`${candidates.length - activeCandidates.length} erased`}
          label="Candidates"
          value={activeCandidates.length}
        />
        <Stat
          hint="Count of jobs"
          label="Needing requirements"
          value={jobsWithoutRequirements.length}
        />
      </div>

      {jobsWithoutRequirements.length > 0 ? (
        <Alert kind="warning" title="Some jobs have no required skill groups">
          Screenings for these jobs fall back to the text model&apos;s guess and are
          flagged for manual review:{" "}
          {jobsWithoutRequirements
            .slice(0, 3)
            .map((job) => job.title)
            .join(", ")}
          {jobsWithoutRequirements.length > 3 ? "…" : ""}
        </Alert>
      ) : null}

      <Card>
        <SectionTitle subtitle="The usual order of work in SmartHire." title="Getting started" />
        <ol className="space-y-3">
          {steps.map((step, index) => (
            <li
              className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-200 p-4"
              key={step.title}
            >
              <div className="flex items-start gap-3">
                <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-indigo-50 text-xs font-bold text-indigo-700">
                  {index + 1}
                </span>
                <div>
                  <p className="text-sm font-semibold text-slate-900">
                    {step.title} {step.done ? <Badge tone="green">Done</Badge> : null}
                  </p>
                  <p className="text-xs text-slate-500">{step.detail}</p>
                </div>
              </div>
              {index === 2 ? (
                <PrimaryButton onClick={() => onNavigate(step.action[1])}>{step.action[0]}</PrimaryButton>
              ) : (
                <SecondaryButton onClick={() => onNavigate(step.action[1])}>{step.action[0]}</SecondaryButton>
              )}
            </li>
          ))}
        </ol>
      </Card>

      <Alert kind="info" title="How to read a result">
        The final label comes from the share of required skill groups the resume matches
        (85% or more: Good Fit, 50% to under 85%: Potential Fit, under 50%: No Fit). The text
        model&apos;s prediction is shown only as a reference. Skill matches are keyword evidence,
        not proof of ability, so a person should always review the result.
      </Alert>
    </div>
  );
}

export default DashboardView;
