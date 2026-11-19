// Small shared building blocks so every screen looks and behaves the same.

export const inputClass =
  "w-full rounded-xl border border-slate-200 bg-white px-3.5 py-2.5 text-sm text-slate-900 shadow-xs outline-none transition placeholder:text-slate-400 focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 disabled:bg-slate-50 disabled:text-slate-400";

export function Card({ children, className = "" }) {
  return (
    <section
      className={`rounded-2xl border border-slate-200 bg-white p-5 shadow-[0_2px_12px_rgba(15,23,42,0.03)] sm:p-6 ${className}`}
    >
      {children}
    </section>
  );
}

export function SectionTitle({ title, action, subtitle }) {
  return (
    <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
      <div>
        <h2 className="text-base font-semibold text-slate-900">{title}</h2>
        {subtitle ? (
          <p className="mt-0.5 text-xs leading-5 text-slate-500">{subtitle}</p>
        ) : null}
      </div>
      {action}
    </div>
  );
}

const ALERT_STYLES = {
  error: "border-rose-200 bg-rose-50 text-rose-800",
  warning: "border-amber-200 bg-amber-50 text-amber-900",
  success: "border-emerald-200 bg-emerald-50 text-emerald-800",
  info: "border-indigo-100 bg-indigo-50/60 text-indigo-900",
};

export function Alert({ kind = "info", children, title }) {
  return (
    <div
      className={`rounded-xl border px-4 py-3 text-sm leading-6 ${ALERT_STYLES[kind]}`}
      role={kind === "error" ? "alert" : "status"}
    >
      {title ? <p className="font-semibold">{title}</p> : null}
      <div>{children}</div>
    </div>
  );
}

export function Spinner({ label = "Loading" }) {
  return (
    <div className="flex items-center gap-2.5 py-6 text-sm text-slate-500" role="status">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-indigo-600" />
      {label}…
    </div>
  );
}

export function Field({ label, hint, children, htmlFor }) {
  return (
    <div>
      <label className="mb-1.5 block text-xs font-semibold text-slate-700" htmlFor={htmlFor}>
        {label}
      </label>
      {children}
      {hint ? <p className="mt-1 text-xs leading-5 text-slate-500">{hint}</p> : null}
    </div>
  );
}

export function PrimaryButton({ children, busy, disabled, ...props }) {
  return (
    <button
      className="inline-flex items-center justify-center gap-2 rounded-xl bg-indigo-700 px-4 py-2.5 text-sm font-semibold text-white shadow-xs transition hover:bg-indigo-800 disabled:bg-slate-300"
      disabled={disabled || busy}
      type="button"
      {...props}
    >
      {busy ? (
        <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/40 border-t-white" />
      ) : null}
      {children}
    </button>
  );
}

export function SecondaryButton({ children, danger = false, ...props }) {
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 rounded-xl border px-3.5 py-2 text-sm font-medium transition disabled:opacity-50 ${
        danger
          ? "border-rose-200 text-rose-700 hover:bg-rose-50"
          : "border-slate-200 text-slate-700 hover:bg-slate-50"
      }`}
      type="button"
      {...props}
    >
      {children}
    </button>
  );
}

const BADGE_STYLES = {
  green: "border-emerald-200 bg-emerald-50 text-emerald-800",
  amber: "border-amber-200 bg-amber-50 text-amber-800",
  red: "border-rose-200 bg-rose-50 text-rose-800",
  slate: "border-slate-200 bg-slate-50 text-slate-700",
  indigo: "border-indigo-200 bg-indigo-50 text-indigo-800",
};

export function Badge({ tone = "slate", children }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-[11px] font-semibold ${BADGE_STYLES[tone]}`}
    >
      {children}
    </span>
  );
}

export function EmptyState({ title, children }) {
  return (
    <div className="rounded-xl border border-dashed border-slate-300 px-5 py-8 text-center">
      <p className="text-sm font-semibold text-slate-700">{title}</p>
      {children ? <p className="mt-1 text-sm text-slate-500">{children}</p> : null}
    </div>
  );
}

export function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
