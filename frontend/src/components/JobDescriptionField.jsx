const MAX_LENGTH = 10000;

function JobDescriptionField({ value, onChange }) {
  return (
    <div>
      <div className="mb-2.5 flex items-center justify-between gap-3">
        <label
          className="text-sm font-semibold text-slate-800"
          htmlFor="job-description"
        >
          Job description
        </label>
        <span className="text-xs text-slate-400">Required</span>
      </div>
      <textarea
        className="min-h-[210px] w-full resize-y rounded-xl border border-slate-200 bg-white px-4 py-3.5 text-sm leading-6 text-slate-800 outline-none transition placeholder:text-slate-400 hover:border-slate-300 focus:border-indigo-500 focus:ring-4 focus:ring-indigo-100"
        id="job-description"
        maxLength={MAX_LENGTH}
        onChange={(event) => onChange(event.target.value)}
        placeholder="Paste the role overview, responsibilities, and required qualifications..."
        value={value}
      />
      <div className="mt-2 flex items-center justify-between gap-4">
        <p className="text-xs text-slate-400">
          Include the role&apos;s key responsibilities and skills.
        </p>
        <span aria-live="polite" className="shrink-0 text-xs tabular-nums text-slate-400">
          {value.length.toLocaleString()} / {MAX_LENGTH.toLocaleString()}
        </span>
      </div>
    </div>
  );
}

export default JobDescriptionField;
