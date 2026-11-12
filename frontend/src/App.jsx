import { useState } from "react";
import AnalysisResults from "./components/AnalysisResults.jsx";
import FileUpload from "./components/FileUpload.jsx";
import Icon from "./components/Icon.jsx";
import JobDescriptionField from "./components/JobDescriptionField.jsx";
import Sidebar from "./components/Sidebar.jsx";

const ANALYZE_ENDPOINT = "/api/analyze/";

function App() {
  const [file, setFile] = useState(null);
  const [jobDescription, setJobDescription] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState("");

  function handleFileChange(nextFile) {
    setFile(nextFile);
    setAnalysis(null);
    setError("");
  }

  function handleJobDescriptionChange(value) {
    setJobDescription(value);
    setAnalysis(null);
    setError("");
  }

  async function handleSubmit(event) {
    event.preventDefault();

    if (!file) {
      setError("Choose a PDF resume before starting the analysis.");
      return;
    }
    if (file.size === 0) {
      setError("The selected PDF is empty. Choose a non-empty PDF file.");
      return;
    }
    if (
      file.type !== "application/pdf" &&
      !file.name.toLowerCase().endsWith(".pdf")
    ) {
      setError("The resume must be a PDF file.");
      return;
    }
    if (!jobDescription.trim()) {
      setError("Enter a job description before starting the analysis.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("jd_text", jobDescription);

    setIsAnalyzing(true);
    setError("");
    setAnalysis(null);
    try {
      const response = await fetch(ANALYZE_ENDPOINT, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const detail = await readErrorDetail(response);
        throw new Error(getHttpErrorMessage(response.status, detail));
      }

      const result = await response.json();
      if (!result || typeof result !== "object" || Array.isArray(result)) {
        throw new Error("The analysis service returned an unexpected response.");
      }
      setAnalysis(result);
    } catch (requestError) {
      setError(
        requestError instanceof TypeError
          ? "Could not connect to SmartHire. Make sure the FastAPI backend is running at http://127.0.0.1:8000 and try again."
          : requestError.message || "Analysis failed. Please try again.",
      );
    } finally {
      setIsAnalyzing(false);
    }
  }

  return (
    <div className="min-h-screen bg-[#f7f8fa] text-slate-900">
      <Sidebar />

      <div className="min-h-screen pt-[68px] lg:pl-[260px] lg:pt-0">
        <header className="flex h-[76px] items-center justify-between border-b border-slate-200/80 bg-white/90 px-5 sm:px-8 lg:px-10">
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <span className="hidden sm:inline">Workspace</span>
            <Icon name="chevronRight" className="hidden h-4 w-4 sm:block" />
            <span className="font-semibold text-slate-800">New analysis</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="hidden text-sm text-slate-500 sm:block">
              Recruiter workspace
            </span>
            <div
              aria-label="Recruiter profile"
              className="flex h-9 w-9 items-center justify-center rounded-full bg-indigo-50 text-xs font-bold text-indigo-700 ring-1 ring-indigo-100"
            >
              HR
            </div>
          </div>
        </header>

        <main
          className="mx-auto max-w-[1440px] px-5 py-8 sm:px-8 sm:py-10 lg:px-10"
          id="workspace"
        >
          <section className="mb-9 flex flex-col justify-between gap-5 md:flex-row md:items-end">
            <div>
              <p className="mb-2 text-sm font-semibold tracking-wide text-indigo-700">
                RECRUITER WORKSPACE
              </p>
              <h1 className="text-3xl font-bold tracking-tight text-slate-950 sm:text-[2.1rem]">
                Resume analysis
              </h1>
              <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500 sm:text-base">
                Compare a candidate&apos;s resume with a role&apos;s requirements
                to get started.
              </p>
            </div>
            <div className="inline-flex w-fit items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-800">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
              Private workspace
            </div>
          </section>

          <div className="grid gap-6 xl:grid-cols-[minmax(0,1.55fr)_minmax(330px,0.9fr)]">
            <section
              className="rounded-2xl border border-slate-200 bg-white p-5 shadow-[0_2px_12px_rgba(15,23,42,0.03)] sm:p-7"
              id="new-analysis"
            >
              <div className="mb-7 flex items-start gap-4">
                <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-indigo-50 text-indigo-700">
                  <Icon name="sparkles" className="h-5 w-5" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">
                    Start a new analysis
                  </h2>
                  <p className="mt-1 text-sm leading-5 text-slate-500">
                    Add one resume and the job description you&apos;re hiring
                    for.
                  </p>
                </div>
              </div>

              <form className="space-y-7" onSubmit={handleSubmit}>
                <FileUpload file={file} onFileChange={handleFileChange} />
                <JobDescriptionField
                  value={jobDescription}
                  onChange={handleJobDescriptionChange}
                />

                <div className="flex flex-col gap-3 border-t border-slate-100 pt-6 sm:flex-row sm:items-center sm:justify-between">
                  <p className="text-xs leading-5 text-slate-500">
                    PDF text extraction and candidate analysis run through the
                    SmartHire API.
                  </p>
                  <button
                    className="inline-flex min-h-11 items-center justify-center gap-2 rounded-xl bg-indigo-700 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:shadow-none"
                    disabled={isAnalyzing || !file || !jobDescription.trim()}
                    type="submit"
                  >
                    {isAnalyzing ? (
                      <>
                        <span
                          aria-hidden="true"
                          className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white"
                        />
                        Analysing...
                      </>
                    ) : (
                      <>
                        <Icon name="sparkles" className="h-4 w-4" />
                        Analyse Resume
                        <Icon name="arrowRight" className="h-4 w-4" />
                      </>
                    )}
                  </button>
                </div>
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
            </section>

            <aside className="space-y-6">
              <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-[0_2px_12px_rgba(15,23,42,0.03)]">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-sky-50 text-sky-700">
                    <Icon name="shield" className="h-5 w-5" />
                  </div>
                  <div>
                    <h2 className="font-semibold text-slate-900">
                      Built for thoughtful hiring
                    </h2>
                  </div>
                </div>
                <p className="mt-4 text-sm leading-6 text-slate-500">
                  SmartHire is designed to help organize resume and role
                  requirements. Use analysis as one input to human review, not
                  as an automated hiring decision.
                </p>
              </section>

              {!analysis && (
                <section className="rounded-2xl border border-dashed border-slate-300 bg-slate-50/70 p-6">
                  <div className="flex items-start gap-3">
                    <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-white text-slate-500 shadow-sm ring-1 ring-slate-200">
                      <Icon name="chart" className="h-4 w-4" />
                    </div>
                    <div>
                      <h2 className="text-sm font-semibold text-slate-800">
                        Your results will appear here
                      </h2>
                      <p className="mt-1.5 text-sm leading-5 text-slate-500">
                        Fit assessment, skill matches, and coverage will appear
                        after you analyse a resume.
                      </p>
                    </div>
                  </div>
                </section>
              )}
            </aside>
          </div>

          {analysis && (
            <section aria-live="polite" className="mt-6" id="analysis-results">
              <AnalysisResults result={analysis} />
            </section>
          )}

          <footer className="mt-10 border-t border-slate-200/80 pt-5 text-xs leading-5 text-slate-400">
            SmartHire provides decision support. Resume keywords are not proof
            of a candidate&apos;s skills or qualifications.
          </footer>
        </main>
      </div>
    </div>
  );
}

async function readErrorDetail(response) {
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

function getHttpErrorMessage(status, detail) {
  const message = detail ? ` ${detail}` : "";
  if (status === 400) {
    return `The resume could not be processed. Check that it is a valid, text-based PDF.${message}`;
  }
  if (status === 422) {
    return `Some input is missing or invalid. Check the resume and job description.${message}`;
  }
  if (status >= 500) {
    return `SmartHire encountered a server error while analysing the resume. Please try again later.${message}`;
  }
  return `Analysis request failed (HTTP ${status}).${message}`;
}

export default App;
