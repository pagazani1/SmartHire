import { useState } from "react";
import { useAuth } from "../context/AuthContext.jsx";
import Icon from "./Icon.jsx";

function getInitials(name) {
  if (!name) return "HR";
  const parts = name.trim().split(/\s+/);
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

const NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard", icon: "home" },
  { id: "jobs", label: "Jobs", icon: "file" },
  { id: "candidates", label: "Candidates", icon: "upload" },
  { id: "persistent", label: "Screening", icon: "shield" },
  { id: "quick", label: "Quick resume check", icon: "plus" },
];

function Sidebar({ activeView, onSelectView }) {
  const [isOpen, setIsOpen] = useState(false);
  const { recruiter, logout } = useAuth();

  function handleNavigate(view) {
    if (onSelectView) {
      onSelectView(view);
    }
    setIsOpen(false);
  }

  return (
    <>
      <div className="fixed left-0 top-0 z-30 flex h-[68px] w-full items-center justify-between border-b border-slate-200 bg-white px-5 lg:hidden">
        <Brand onSelect={() => handleNavigate("dashboard")} />
        <div className="flex items-center gap-2">
          {recruiter ? (
            <div
              className="flex h-8 w-8 items-center justify-center rounded-full bg-indigo-50 text-xs font-bold text-indigo-700 ring-1 ring-indigo-200"
              title={`${recruiter.display_name} (${recruiter.role})`}
            >
              {getInitials(recruiter.display_name)}
            </div>
          ) : null}
          <button
            aria-expanded={isOpen}
            aria-label={isOpen ? "Close navigation" : "Open navigation"}
            className="rounded-lg p-2 text-slate-600 hover:bg-slate-100"
            onClick={() => setIsOpen(!isOpen)}
            type="button"
          >
            <Icon name={isOpen ? "x" : "menu"} className="h-5 w-5" />
          </button>
        </div>
      </div>
      <aside
        className={`fixed inset-y-0 left-0 z-20 flex w-[260px] flex-col border-r border-slate-200 bg-white px-4 pb-5 pt-6 transition-transform lg:translate-x-0 ${
          isOpen ? "translate-x-0 pt-[84px]" : "-translate-x-full pt-[84px] lg:pt-6"
        }`}
      >
        <div className="hidden px-2 lg:block">
          <Brand onSelect={() => handleNavigate("dashboard")} />
        </div>

        <div className="mt-10 px-2 text-[10px] font-bold tracking-[0.16em] text-slate-400">
          WORKSPACE
        </div>
        <nav aria-label="Main navigation" className="mt-3 space-y-1">
          {NAV_ITEMS.map((item) => (
            <button
              aria-current={activeView === item.id ? "page" : undefined}
              className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm transition ${
                activeView === item.id
                  ? "bg-indigo-50 font-semibold text-indigo-800"
                  : "font-medium text-slate-500 hover:bg-slate-50 hover:text-slate-800"
              }`}
              key={item.id}
              onClick={() => handleNavigate(item.id)}
              type="button"
            >
              <Icon name={item.icon} className="h-[18px] w-[18px]" />
              {item.label}
            </button>
          ))}
        </nav>

        {/* Recruiter Authentication & Session Card */}
        <div className="mt-auto space-y-3">
          <div className="rounded-2xl border border-slate-200/80 bg-slate-50/70 p-3.5">
            {recruiter ? (
              <div>
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2.5 overflow-hidden">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-100 text-xs font-bold text-indigo-800 ring-1 ring-indigo-200">
                      {getInitials(recruiter.display_name)}
                    </div>
                    <div className="min-w-0">
                      <p className="truncate text-xs font-semibold text-slate-900">
                        {recruiter.display_name}
                      </p>
                      <p className="truncate text-[11px] text-slate-500">
                        {recruiter.email_normalized}
                      </p>
                    </div>
                  </div>
                  <span
                    className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${
                      recruiter.role === "admin"
                        ? "bg-purple-100 text-purple-800"
                        : "bg-sky-100 text-sky-800"
                    }`}
                  >
                    {recruiter.role}
                  </span>
                </div>
                <div className="mt-3 flex items-center justify-end border-t border-slate-200/60 pt-2.5">
                  <button
                    className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-rose-600 transition"
                    onClick={logout}
                    type="button"
                  >
                    <Icon name="logOut" className="h-3.5 w-3.5" />
                    Sign Out
                  </button>
                </div>
              </div>
            ) : null}
          </div>

          <div className="rounded-2xl bg-slate-50 p-3 ring-1 ring-slate-200/80">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-white text-indigo-700 shadow-xs ring-1 ring-slate-200/80">
              <Icon name="shield" className="h-3.5 w-3.5" />
            </div>
            <p className="mt-2 text-xs font-semibold text-slate-800">
              Human-led decisions
            </p>
            <p className="mt-0.5 text-[11px] leading-4 text-slate-500">
              Fair, contextual candidate review with full audit logging.
            </p>
          </div>
        </div>

        <div className="mt-3 border-t border-slate-100 px-2 pt-3 text-[11px] text-slate-400">
          SmartHire &middot; RBAC & Security Active
        </div>
      </aside>
      {isOpen && (
        <button
          aria-label="Close navigation menu"
          className="fixed inset-0 z-10 bg-slate-950/20 lg:hidden"
          onClick={() => setIsOpen(false)}
          type="button"
        />
      )}
    </>
  );
}

function Brand({ onSelect }) {
  return (
    <button
      className="inline-flex items-center gap-2.5 text-left"
      onClick={onSelect}
      type="button"
    >
      <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-700 text-white shadow-sm">
        <Icon name="sparkles" className="h-[18px] w-[18px]" />
      </span>
      <span className="text-[17px] font-bold tracking-tight text-slate-950">
        Smart<span className="text-indigo-700">Hire</span>
      </span>
    </button>
  );
}

export default Sidebar;
