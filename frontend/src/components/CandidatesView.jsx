import { useCallback, useEffect, useMemo, useState } from "react";
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
  formatBytes,
  formatDate,
  inputClass,
} from "./ui.jsx";

const PURPOSES = [
  ["document_storage", "Store the resume document"],
  ["screening", "Screen the candidate against jobs"],
  ["contact", "Contact the candidate"],
  ["background_verification", "Background verification"],
];

function isConsentActive(consent) {
  if (consent.status !== "granted") return false;
  return !consent.expires_at || new Date(consent.expires_at) > new Date();
}

function consentTone(consent) {
  if (isConsentActive(consent)) return "green";
  return consent.status === "withdrawn" ? "red" : "amber";
}

function consentLabel(consent) {
  if (consent.status === "granted" && !isConsentActive(consent)) return "expired";
  return consent.status;
}

/* ---------------------------- add candidate ---------------------------- */

function AddCandidateForm({ jobs, onCreated, onCancel }) {
  const api = useApi();
  const usableJobs = jobs.filter((job) => job.status !== "archived");
  const [form, setForm] = useState({ job_id: usableJobs[0]?.id ?? "", display_name: "", email: "", phone: "" });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const set = (key) => (event) => setForm((prev) => ({ ...prev, [key]: event.target.value }));

  async function submit(event) {
    event.preventDefault();
    if (!form.job_id) return setError("Create a job first, then add candidates to it.");
    if (!form.display_name.trim()) return setError("Enter the candidate's name.");
    setBusy(true);
    setError("");
    try {
      const created = await api.post("/api/candidates/", {
        job_id: form.job_id,
        display_name: form.display_name.trim(),
        email: form.email.trim() || null,
        phone: form.phone.trim() || null,
      });
      onCreated(created.candidate.id);
    } catch (err) {
      setError(describeError(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="space-y-4" onSubmit={submit}>
      {error ? <Alert kind="error">{error}</Alert> : null}
      {usableJobs.length === 0 ? (
        <Alert kind="warning">You need a job before you can add a candidate. Open the Jobs page first.</Alert>
      ) : null}
      <div className="grid gap-4 sm:grid-cols-2">
        <Field htmlFor="cand-job" label="Applying for *">
          <select className={inputClass} id="cand-job" onChange={set("job_id")} value={form.job_id}>
            {usableJobs.map((job) => (
              <option key={job.id} value={job.id}>
                {job.title}
              </option>
            ))}
          </select>
        </Field>
        <Field htmlFor="cand-name" label="Full name *">
          <input className={inputClass} id="cand-name" onChange={set("display_name")} value={form.display_name} />
        </Field>
        <Field htmlFor="cand-email" label="Email">
          <input className={inputClass} id="cand-email" onChange={set("email")} type="email" value={form.email} />
        </Field>
        <Field htmlFor="cand-phone" label="Phone">
          <input className={inputClass} id="cand-phone" onChange={set("phone")} value={form.phone} />
        </Field>
      </div>
      <div className="flex gap-3">
        <PrimaryButton busy={busy} disabled={usableJobs.length === 0} type="submit">
          Add candidate
        </PrimaryButton>
        <SecondaryButton onClick={onCancel}>Cancel</SecondaryButton>
      </div>
    </form>
  );
}

/* ------------------------------- consent ------------------------------- */

function ConsentSection({ candidateId, consents, locked, onChanged, onError, onNotice }) {
  const api = useApi();
  const [showForm, setShowForm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [form, setForm] = useState({
    purpose: "document_storage",
    notice_version: "v1",
    capture_method: "Recorded by recruiter",
    proof_reference: "",
    expires: "",
  });
  const set = (key) => (event) => setForm((prev) => ({ ...prev, [key]: event.target.value }));

  async function grant(event) {
    event.preventDefault();
    setBusy(true);
    onError("");
    try {
      await api.post(`/api/candidates/${candidateId}/consents/`, {
        purpose: form.purpose,
        notice_version: form.notice_version.trim(),
        capture_method: form.capture_method.trim(),
        proof_reference: form.proof_reference.trim() || null,
        expires_at: form.expires ? new Date(`${form.expires}T23:59:59`).toISOString() : null,
      });
      onNotice("Consent recorded.");
      setShowForm(false);
      await onChanged();
    } catch (err) {
      onError(describeError(err));
    } finally {
      setBusy(false);
    }
  }

  async function withdraw(consent) {
    if (!window.confirm("Withdraw this consent? Resume access and screening that depend on it will stop immediately.")) return;
    onError("");
    try {
      await api.post(`/api/consents/${consent.id}/withdraw`);
      onNotice("Consent withdrawn.");
      await onChanged();
    } catch (err) {
      onError(describeError(err));
    }
  }

  return (
    <div>
      <SectionTitle
        action={
          !locked && !showForm ? (
            <SecondaryButton onClick={() => setShowForm(true)}>Record consent</SecondaryButton>
          ) : null
        }
        subtitle="Screening and resume storage each need an active consent on file."
        title="Consent"
      />
      {showForm ? (
        <form className="mb-5 space-y-4 rounded-xl border border-slate-200 bg-slate-50/60 p-4" onSubmit={grant}>
          <Alert kind="info">
            You are recording that this candidate gave consent. SmartHire stores the record and
            the time you saved it; keep the actual evidence (form, email) in your own records and
            reference it below.
          </Alert>
          <div className="grid gap-4 sm:grid-cols-2">
            <Field htmlFor="consent-purpose" label="Purpose">
              <select className={inputClass} id="consent-purpose" onChange={set("purpose")} value={form.purpose}>
                {PURPOSES.map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </Field>
            <Field htmlFor="consent-expires" label="Expires on (optional)">
              <input className={inputClass} id="consent-expires" onChange={set("expires")} type="date" value={form.expires} />
            </Field>
            <Field htmlFor="consent-notice" label="Privacy notice version">
              <input className={inputClass} id="consent-notice" onChange={set("notice_version")} value={form.notice_version} />
            </Field>
            <Field htmlFor="consent-method" label="How it was obtained">
              <input className={inputClass} id="consent-method" onChange={set("capture_method")} value={form.capture_method} />
            </Field>
          </div>
          <Field htmlFor="consent-proof" label="Evidence reference (optional)">
            <input
              className={inputClass}
              id="consent-proof"
              onChange={set("proof_reference")}
              placeholder="e.g. signed form #1042 or email date"
              value={form.proof_reference}
            />
          </Field>
          <div className="flex gap-3">
            <PrimaryButton busy={busy} type="submit">Save consent</PrimaryButton>
            <SecondaryButton onClick={() => setShowForm(false)}>Cancel</SecondaryButton>
          </div>
        </form>
      ) : null}
      {consents.length === 0 ? (
        <EmptyState title="No consent recorded">Record consent before uploading a resume or screening.</EmptyState>
      ) : (
        <ul className="divide-y divide-slate-100">
          {consents.map((consent) => (
            <li className="flex flex-wrap items-center justify-between gap-3 py-3" key={consent.id}>
              <div>
                <p className="text-sm font-medium text-slate-900">
                  {PURPOSES.find(([value]) => value === consent.purpose)?.[1] ?? consent.purpose}{" "}
                  <Badge tone={consentTone(consent)}>{consentLabel(consent)}</Badge>
                </p>
                <p className="text-xs text-slate-500">
                  Recorded {formatDate(consent.captured_at)} &middot; notice {consent.notice_version}
                  {consent.expires_at ? ` · expires ${formatDate(consent.expires_at)}` : ""}
                  {consent.withdrawn_at ? ` · withdrawn ${formatDate(consent.withdrawn_at)}` : ""}
                </p>
              </div>
              {consent.status === "granted" && !locked ? (
                <SecondaryButton danger onClick={() => withdraw(consent)}>Withdraw</SecondaryButton>
              ) : null}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/* ------------------------------- resumes ------------------------------- */

function ResumeSection({ candidateId, resumes, consents, locked, onChanged, onError, onNotice }) {
  const api = useApi();
  const storageConsents = consents.filter((c) => c.purpose === "document_storage" && isConsentActive(c));
  const [file, setFile] = useState(null);
  const [consentId, setConsentId] = useState("");
  const [busy, setBusy] = useState(false);
  const [inputKey, setInputKey] = useState(0);
  const effectiveConsent = storageConsents.some((c) => c.id === consentId) ? consentId : storageConsents[0]?.id ?? "";

  async function upload(event) {
    event.preventDefault();
    if (!file) return onError("Choose a PDF resume first.");
    if (file.size === 0) return onError("That file is empty.");
    if (file.size > 10 * 1024 * 1024) return onError("The file is larger than the 10 MB limit.");
    if (!effectiveConsent) return onError("Record an active document-storage consent first.");
    setBusy(true);
    onError("");
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("consent_record_id", effectiveConsent);
      await api.upload(`/api/candidates/${candidateId}/resumes/`, form);
      onNotice("Resume uploaded and stored privately.");
      setFile(null);
      setInputKey((key) => key + 1);
      await onChanged();
    } catch (err) {
      onError(describeError(err));
    } finally {
      setBusy(false);
    }
  }

  async function download(resume) {
    onError("");
    try {
      const blob = await api.download(`/api/resumes/${resume.id}/download`);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = resume.original_filename || "resume.pdf";
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (err) {
      onError(describeError(err));
    }
  }

  async function remove(resume) {
    if (!window.confirm(`Delete "${resume.original_filename}"? The file is removed permanently; screening history is kept.`)) return;
    onError("");
    try {
      await api.del(`/api/resumes/${resume.id}`);
      onNotice("Resume deleted.");
      await onChanged();
    } catch (err) {
      onError(describeError(err));
    }
  }

  return (
    <div>
      <SectionTitle subtitle="PDF only, up to 10 MB. Scanned images are not supported (no OCR)." title="Resumes" />
      {!locked ? (
        <form className="mb-5 flex flex-wrap items-end gap-3 rounded-xl border border-slate-200 bg-slate-50/60 p-4" onSubmit={upload}>
          <Field htmlFor="resume-file" label="PDF resume">
            <input
              accept="application/pdf,.pdf"
              className="block text-sm text-slate-700 file:mr-3 file:rounded-lg file:border-0 file:bg-indigo-50 file:px-3 file:py-2 file:text-sm file:font-semibold file:text-indigo-700"
              id="resume-file"
              key={inputKey}
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              type="file"
            />
          </Field>
          <Field htmlFor="resume-consent" label="Consent used">
            <select
              className={inputClass}
              disabled={storageConsents.length === 0}
              id="resume-consent"
              onChange={(event) => setConsentId(event.target.value)}
              value={effectiveConsent}
            >
              {storageConsents.length === 0 ? <option value="">No active consent</option> : null}
              {storageConsents.map((c) => (
                <option key={c.id} value={c.id}>
                  Recorded {formatDate(c.captured_at)}
                </option>
              ))}
            </select>
          </Field>
          <PrimaryButton busy={busy} disabled={storageConsents.length === 0} type="submit">
            Upload
          </PrimaryButton>
        </form>
      ) : null}
      {resumes.length === 0 ? (
        <EmptyState title="No resumes uploaded" />
      ) : (
        <ul className="divide-y divide-slate-100">
          {resumes.map((resume) => (
            <li className="flex flex-wrap items-center justify-between gap-3 py-3" key={resume.id}>
              <div>
                <p className="text-sm font-medium text-slate-900">
                  {resume.original_filename}{" "}
                  <Badge tone={resume.status === "available" ? "green" : "slate"}>{resume.status}</Badge>
                </p>
                <p className="text-xs text-slate-500">
                  {formatBytes(resume.byte_size)} &middot; uploaded {formatDate(resume.uploaded_at)}
                  {resume.retention_until ? ` · auto-delete ${formatDate(resume.retention_until)}` : ""}
                </p>
              </div>
              {resume.status === "available" ? (
                <div className="flex gap-2">
                  <SecondaryButton onClick={() => download(resume)}>Download</SecondaryButton>
                  <SecondaryButton danger onClick={() => remove(resume)}>Delete</SecondaryButton>
                </div>
              ) : null}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/* ------------------------------ detail pane ----------------------------- */

function CandidateDetail({ candidateId, jobsById, onOpenScreening, onListChanged }) {
  const api = useApi();
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [erasing, setErasing] = useState(false);

  const load = useCallback(async () => {
    try {
      const [candidate, consents, resumes] = await Promise.all([
        api.get(`/api/candidates/${candidateId}`),
        api.get(`/api/candidates/${candidateId}/consents/`),
        api.get(`/api/candidates/${candidateId}/resumes/?limit=100`),
      ]);
      setData({ candidate, consents, resumes });
    } catch (err) {
      setError(describeError(err));
    }
  }, [api, candidateId]);

  useEffect(() => {
    setData(null);
    setError("");
    setNotice("");
    load();
  }, [load]);

  async function erase() {
    const name = data.candidate.display_name;
    if (
      !window.confirm(
        `Erase all personal data for ${name}?\n\nThis permanently deletes their resume files, withdraws their consents and removes their name and contact details. It cannot be undone.`,
      )
    )
      return;
    setErasing(true);
    setError("");
    try {
      await api.post(`/api/candidates/${candidateId}/anonymize`);
      setNotice("Personal data erased.");
      await load();
      await onListChanged();
    } catch (err) {
      setError(describeError(err));
    } finally {
      setErasing(false);
    }
  }

  if (!data) return error ? <Alert kind="error">{error}</Alert> : <Spinner label="Loading candidate" />;
  const { candidate, consents, resumes } = data;
  const locked = Boolean(candidate.anonymized_at);
  const refresh = async () => {
    await load();
  };

  return (
    <div className="space-y-6">
      {error ? <Alert kind="error">{error}</Alert> : null}
      {notice ? <Alert kind="success">{notice}</Alert> : null}
      <Card>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">{candidate.display_name}</h2>
            <p className="text-sm text-slate-500">
              {candidate.email || "No email"} {candidate.phone ? `· ${candidate.phone}` : ""}
            </p>
          </div>
          {locked ? <Badge tone="slate">Personal data erased</Badge> : (
            <PrimaryButton onClick={() => onOpenScreening(candidate.id)}>Screen this candidate</PrimaryButton>
          )}
        </div>
        <div className="mt-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Applications</p>
          <ul className="mt-2 space-y-1.5">
            {candidate.applications.map((application) => (
              <li className="text-sm text-slate-700" key={application.id}>
                {jobsById[application.job_id]?.title ?? "Job"} <Badge>{application.status}</Badge>
              </li>
            ))}
          </ul>
        </div>
      </Card>
      <Card>
        <ConsentSection
          candidateId={candidateId}
          consents={consents}
          locked={locked}
          onChanged={refresh}
          onError={setError}
          onNotice={setNotice}
        />
      </Card>
      <Card>
        <ResumeSection
          candidateId={candidateId}
          consents={consents}
          locked={locked}
          onChanged={refresh}
          onError={setError}
          onNotice={setNotice}
          resumes={resumes}
        />
      </Card>
      {!locked ? (
        <Card className="border-rose-200">
          <SectionTitle
            subtitle="Use this when a candidate asks to be forgotten. Application and screening records stay (without identity) for statistics and audit."
            title="Erase personal data"
          />
          <SecondaryButton danger disabled={erasing} onClick={erase}>
            {erasing ? "Erasing…" : "Erase this candidate's data"}
          </SecondaryButton>
        </Card>
      ) : null}
    </div>
  );
}

/* ------------------------------ main view ------------------------------- */

function CandidatesView({ onOpenScreening }) {
  const api = useApi();
  const [candidates, setCandidates] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState("");

  const loadLists = useCallback(async () => {
    try {
      const [candidateList, jobList] = await Promise.all([
        api.get("/api/candidates/?limit=100"),
        api.get("/api/jobs/?limit=100"),
      ]);
      setCandidates(Array.isArray(candidateList) ? candidateList : []);
      setJobs(Array.isArray(jobList) ? jobList : []);
      setError("");
    } catch (err) {
      setError(describeError(err));
    }
  }, [api]);

  useEffect(() => {
    loadLists();
  }, [loadLists]);

  const jobsById = useMemo(() => Object.fromEntries(jobs.map((job) => [job.id, job])), [jobs]);

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(280px,0.8fr)_minmax(0,1.6fr)]">
      <div className="space-y-6">
        {error ? <Alert kind="error">{error}</Alert> : null}
        <Card>
          <SectionTitle
            action={!adding ? <PrimaryButton onClick={() => setAdding(true)}>Add</PrimaryButton> : null}
            title="Candidates"
          />
          {adding ? (
            <AddCandidateForm
              jobs={jobs}
              onCancel={() => setAdding(false)}
              onCreated={async (id) => {
                setAdding(false);
                await loadLists();
                setSelectedId(id);
              }}
            />
          ) : null}
          {candidates === null && !error ? <Spinner label="Loading candidates" /> : null}
          {candidates && candidates.length === 0 && !adding ? (
            <EmptyState title="No candidates yet">Add one to a job to begin.</EmptyState>
          ) : null}
          {candidates && candidates.length > 0 ? (
            <ul className="mt-2 divide-y divide-slate-100">
              {candidates.map((candidate) => (
                <li key={candidate.id}>
                  <button
                    aria-current={candidate.id === selectedId ? "true" : undefined}
                    className={`flex w-full items-center justify-between gap-3 rounded-lg px-2 py-3 text-left transition ${
                      candidate.id === selectedId ? "bg-indigo-50" : "hover:bg-slate-50"
                    }`}
                    onClick={() => setSelectedId(candidate.id)}
                    type="button"
                  >
                    <span>
                      <span className="block text-sm font-medium text-slate-900">{candidate.display_name}</span>
                      <span className="block text-xs text-slate-500">
                        {candidate.email || "No email"} · added {formatDate(candidate.created_at)}
                      </span>
                    </span>
                    {candidate.anonymized_at ? <Badge>erased</Badge> : null}
                  </button>
                </li>
              ))}
            </ul>
          ) : null}
        </Card>
      </div>
      <div>
        {selectedId ? (
          <CandidateDetail
            candidateId={selectedId}
            jobsById={jobsById}
            key={selectedId}
            onListChanged={loadLists}
            onOpenScreening={onOpenScreening}
          />
        ) : (
          <EmptyState title="Select a candidate">
            Manage their consent, resumes and applications here.
          </EmptyState>
        )}
      </div>
    </div>
  );
}

export default CandidatesView;
