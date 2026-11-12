import { useRef, useState } from "react";
import Icon from "./Icon.jsx";

function formatFileSize(size) {
  if (size < 1024 * 1024) return `${Math.max(1, Math.round(size / 1024))} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function FileUpload({ file, onFileChange }) {
  const inputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState("");

  function acceptFile(candidate) {
    if (!candidate) return;
    if (candidate.type !== "application/pdf" && !candidate.name.toLowerCase().endsWith(".pdf")) {
      setError("Please choose a PDF file.");
      return;
    }
    setError("");
    onFileChange(candidate);
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    acceptFile(event.dataTransfer.files?.[0]);
  }

  return (
    <div>
      <div className="mb-2.5 flex items-center justify-between gap-3">
        <label className="text-sm font-semibold text-slate-800" htmlFor="resume-file">
          Candidate resume
        </label>
        <span className="text-xs text-slate-400">PDF document</span>
      </div>

      {file ? (
        <div className="flex items-center gap-3 rounded-xl border border-indigo-200 bg-indigo-50/60 p-4">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-white text-indigo-700 shadow-sm">
            <Icon name="file" className="h-5 w-5" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-semibold text-slate-800">{file.name}</p>
            <p className="mt-0.5 text-xs text-slate-500">{formatFileSize(file.size)}</p>
          </div>
          <button
            aria-label="Remove selected resume"
            className="rounded-lg p-2 text-slate-400 transition hover:bg-white hover:text-slate-700 focus-visible:outline-2 focus-visible:outline-indigo-600"
            onClick={() => {
              onFileChange(null);
              setError("");
              if (inputRef.current) inputRef.current.value = "";
            }}
            type="button"
          >
            <Icon name="x" className="h-4 w-4" />
          </button>
        </div>
      ) : (
        <button
          className={`flex min-h-[178px] w-full flex-col items-center justify-center rounded-xl border border-dashed px-5 py-6 text-center transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 ${
            isDragging
              ? "border-indigo-500 bg-indigo-50"
              : "border-slate-300 bg-slate-50/70 hover:border-indigo-400 hover:bg-indigo-50/40"
          }`}
          onClick={() => inputRef.current?.click()}
          onDragEnter={(event) => {
            event.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={(event) => {
            event.preventDefault();
            setIsDragging(false);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDrop={handleDrop}
          type="button"
        >
          <span className="mb-3 flex h-11 w-11 items-center justify-center rounded-xl bg-white text-indigo-700 shadow-sm ring-1 ring-slate-200/80">
            <Icon name="upload" className="h-5 w-5" />
          </span>
          <span className="text-sm font-semibold text-slate-800">
            <span className="text-indigo-700">Click to upload</span> or drag and drop
          </span>
          <span className="mt-1.5 text-xs text-slate-500">
            PDF document only
          </span>
        </button>
      )}

      <input
        ref={inputRef}
        accept="application/pdf,.pdf"
        className="sr-only"
        id="resume-file"
        onChange={(event) => {
          acceptFile(event.target.files?.[0]);
          event.target.value = "";
        }}
        type="file"
      />
      {error && (
        <p className="mt-2 text-sm text-rose-700" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}

export default FileUpload;
