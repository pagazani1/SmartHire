import { useCallback, useEffect, useState } from "react";
import { findUnknownSkills, formatSkillGroups, parseSkillGroups, suggestSkills } from "../lib/skillGroups.js";
import { describeError, useApi } from "../lib/api.js";
import {
  Alert,
  Badge,
  Card,
  EmptyState,
  Field,
  PrimaryButton,
  SecondaryButton,
  SectionTitle,
  Spinner,
  formatDate,
  inputClass,
} from "./ui.jsx";

const STATUSES = ["draft", "open", "paused", "closed", "archived"];
const STATUS_TONE = { open: "green", draft: "slate", paused: "amber", closed: "red", archived: "slate" };

const EMPTY_FORM = {
  title: "",
  department: "",
  location: "",
  status: "open",
  description: "",
  required: "",
  preferred: "",
};

function toForm(job) {
  return {
    title: job.title ?? "",
    department: job.department ?? "",
    location: job.location ?? "",
    status: job.status ?? "open",
    description: job.description ?? "",
    required: formatSkillGroups(job.required_skill_groups),
    preferred: formatSkillGroups(job.preferred_skill_groups),
  };
}

function SupportedSkills({ vocabulary }) {
  const [filter, setFilter] = useState("");
  if (vocabulary.length === 0) return null;
  const shown = vocabulary.filter((skill) => skill.toLowerCase().includes(filter.trim().toLowerCase()));
  return (
    <details className="rounded-xl border border-slate-200 bg-slate-50/60 p-3 text-sm">
      <summary className="cursor-pointer font-medium text-slate-700">
        Supported skills ({vocabulary.length}). Only these names can be used in skill groups.
      </summary>
      <input
        aria-label="Filter supported skills"
        className={`${inputClass} mt-3`}
        onChange={(event) => setFilter(event.target.value)}
        placeholder="Filter, e.g. cloud"
        value={filter}
      />
      <ul className="mt-3 flex max-h-40 flex-wrap gap-1.5 overflow-y-auto">
        {shown.map((skill) => (
          <li className="rounded-full border border-slate-200 bg-white px-2.5 py-0.5 text-xs text-slate-700" key={skill}>
            {skill}
          </li>
        ))}
        {shown.length === 0 ? <li className="text-xs text-slate-500">No supported skill matches.</li> : null}
      </ul>
    </details>
  );
}

function UnknownSkillsAlert({ unknown, vocabulary }) {
  if (unknown.length === 0) return null;
  return (
    <Alert kind="error" title="Some skills are not supported">
      <ul className="mt-1 list-disc pl-5">
        {unknown.map((skill) => {
          const options = suggestSkills(skill, vocabulary);
          return (
            <li key={skill}>
              &ldquo;{skill}&rdquo;
              {options.length ? <> &mdash; did you mean {options.join(", ")}?</> : null}
            </li>
          );
        })}
      </ul>
    </Alert>
  );
}

function JobForm({ initial, isEdit, busy, onSubmit, onCancel, vocabulary }) {
  const [form, setForm] = useState(initial);
  const [localError, setLocalError] = useState("");
  const set = (key) => (event) => setForm((prev) => ({ ...prev, [key]: event.target.value }));
  const requiredGroups = parseSkillGroups(form.required);
  const preferredGroups = parseSkillGroups(form.preferred);
  const unknownSkills = findUnknownSkills([...requiredGroups, ...preferredGroups], vocabulary);

  function handleSubmit(event) {
    event.preventDefault();
    if (!form.title.trim()) return setLocalError("Give the job a title.");
    if (!form.description.trim()) return setLocalError("Add a job description.");
    if (unknownSkills.length > 0) {
      return setLocalError("Replace the unsupported skills listed below before saving.");
    }
    setLocalError("");
    onSubmit({
      title: form.title.trim(),
      department: form.department.trim() || null,
      location: form.location.trim() || null,
      status: form.status,
      description: form.description.trim(),
      required_skill_groups: requiredGroups,
      preferred_skill_groups: preferredGroups,
    });
  }

  return (
    <form className="space-y-4" onSubmit={handleSubmit}>
      {localError ? <Alert kind="error">{localError}</Alert> : null}
      <div className="grid gap-4 sm:grid-cols-2">
        <Field htmlFor="job-title" label="Title *">
          <input className={inputClass} id="job-title" onChange={set("title")} value={form.title} />
        </Field>
        <Field htmlFor="job-status" label="Status">
          <select className={inputClass} id="job-status" onChange={set("status")} value={form.status}>
            {STATUSES.map((status) => (
              <option key={status} value={status}>
                {status}
              </option>
            ))}
          </select>
        </Field>
        <Field htmlFor="job-dept" label="Department">
          <input className={inputClass} id="job-dept" onChange={set("department")} value={form.department} />
        </Field>
        <Field htmlFor="job-loc" label="Location">
          <input className={inputClass} id="job-loc" onChange={set("location")} value={form.location} />
        </Field>
      </div>
      <Field htmlFor="job-desc" label="Job description *">
        <textarea
          className={`${inputClass} min-h-32`}
          id="job-desc"
          onChange={set("description")}
          value={form.description}
        />
      </Field>
      <div className="grid gap-4 md:grid-cols-2">
        <Field
          hint="One group per line. Put alternatives on the same line: “Python | Java” is satisfied by either."
          htmlFor="job-required"
          label="Required skill groups"
        >
          <textarea
            className={`${inputClass} min-h-28 font-mono text-xs`}
            id="job-required"
            onChange={set("required")}
            placeholder={"Python | Java\nPostgreSQL\nDocker"}
            value={form.required}
          />
        </Field>
        <Field
          hint="Shown to the recruiter but never changes the final label."
          htmlFor="job-preferred"
          label="Preferred skill groups"
        >
          <textarea
            className={`${inputClass} min-h-28 font-mono text-xs`}
            id="job-preferred"
            onChange={set("preferred")}
            placeholder={"Kubernetes\nAWS | GCP"}
            value={form.preferred}
          />
        </Field>
      </div>
      <UnknownSkillsAlert unknown={unknownSkills} vocabulary={vocabulary} />
      <SupportedSkills vocabulary={vocabulary} />
      {requiredGroups.length === 0 ? (
        <Alert kind="warning" title="No required skill groups">
          Without them, the final label falls back to the text model&apos;s guess and every
          screening for this job is flagged for manual review.
        </Alert>
      ) : (
        <p className="text-xs text-slate-500">
          {requiredGroups.length} required group{requiredGroups.length === 1 ? "" : "s"} will be
          used for coverage.
        </p>
      )}
      {isEdit ? (
        <Alert kind="info">
          Changing the description or skill groups creates a new version. Earlier screenings keep
          the requirements they were run against.
        </Alert>
      ) : null}
      <div className="flex flex-wrap gap-3">
        <PrimaryButton busy={busy} type="submit">
          {isEdit ? "Save changes" : "Create job"}
        </PrimaryButton>
        <SecondaryButton onClick={onCancel}>Cancel</SecondaryButton>
      </div>
    </form>
  );
}

function JobsView() {
  const api = useApi();
  const [jobs, setJobs] = useState(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [mode, setMode] = useState(null); // null | "new" | job object being edited
  const [busy, setBusy] = useState(false);
  const [vocabulary, setVocabulary] = useState([]);

  const load = useCallback(async () => {
    try {
      const list = await api.get("/api/jobs/?limit=100");
      setJobs(Array.isArray(list) ? list : []);
      setError("");
    } catch (err) {
      setError(describeError(err));
    }
  }, [api]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    // If this fails the form still works; the server validates skill names anyway.
    api
      .get("/api/skills/")
      .then((data) => setVocabulary(Array.isArray(data?.skills) ? data.skills : []))
      .catch(() => setVocabulary([]));
  }, [api]);

  async function save(payload) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      if (mode && mode !== "new") {
        await api.put(`/api/jobs/${mode.id}`, payload);
        setNotice("Job updated.");
      } else {
        await api.post("/api/jobs/", payload);
        setNotice("Job created.");
      }
      setMode(null);
      await load();
    } catch (err) {
      setError(describeError(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      {error ? <Alert kind="error">{error}</Alert> : null}
      {notice ? <Alert kind="success">{notice}</Alert> : null}

      {mode ? (
        <Card>
          <SectionTitle title={mode === "new" ? "New job" : `Edit: ${mode.title}`} />
          <JobForm
            busy={busy}
            initial={mode === "new" ? EMPTY_FORM : toForm(mode)}
            isEdit={mode !== "new"}
            key={mode === "new" ? "new" : mode.id}
            onCancel={() => setMode(null)}
            onSubmit={save}
            vocabulary={vocabulary}
          />
        </Card>
      ) : null}

      <Card>
        <SectionTitle
          action={!mode ? <PrimaryButton onClick={() => setMode("new")}>New job</PrimaryButton> : null}
          subtitle="Jobs you own. Required skill groups decide the final classification."
          title="Jobs"
        />
        {jobs === null && !error ? <Spinner label="Loading jobs" /> : null}
        {jobs && jobs.length === 0 ? (
          <EmptyState title="No jobs yet">Create your first job to start screening candidates.</EmptyState>
        ) : null}
        {jobs && jobs.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-left text-sm">
              <thead className="text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="pb-2 font-semibold">Title</th>
                  <th className="pb-2 font-semibold">Status</th>
                  <th className="pb-2 font-semibold">Required groups</th>
                  <th className="pb-2 font-semibold">Version</th>
                  <th className="pb-2 font-semibold">Updated</th>
                  <th className="pb-2" />
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {jobs.map((job) => (
                  <tr key={job.id}>
                    <td className="py-3 pr-3 font-medium text-slate-900">
                      {job.title}
                      {job.department ? (
                        <span className="block text-xs font-normal text-slate-500">{job.department}</span>
                      ) : null}
                    </td>
                    <td className="py-3 pr-3">
                      <Badge tone={STATUS_TONE[job.status] ?? "slate"}>{job.status}</Badge>
                    </td>
                    <td className="py-3 pr-3 tabular-nums">
                      {(job.required_skill_groups ?? []).length === 0 ? (
                        <Badge tone="amber">none</Badge>
                      ) : (
                        job.required_skill_groups.length
                      )}
                    </td>
                    <td className="py-3 pr-3 tabular-nums text-slate-600">v{job.description_version ?? 1}</td>
                    <td className="py-3 pr-3 text-slate-600">{formatDate(job.updated_at)}</td>
                    <td className="py-3 text-right">
                      <SecondaryButton disabled={Boolean(mode)} onClick={() => setMode(job)}>
                        Edit
                      </SecondaryButton>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </Card>
    </div>
  );
}

export default JobsView;
