import { useState } from "react";
import Icon from "./Icon.jsx";

function Sidebar() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <div className="fixed left-0 top-0 z-30 flex h-[68px] w-full items-center justify-between border-b border-slate-200 bg-white px-5 lg:hidden">
        <Brand />
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
      <aside
        className={`fixed inset-y-0 left-0 z-20 flex w-[260px] flex-col border-r border-slate-200 bg-white px-4 pb-5 pt-6 transition-transform lg:translate-x-0 ${
          isOpen ? "translate-x-0 pt-[84px]" : "-translate-x-full pt-[84px] lg:pt-6"
        }`}
      >
        <div className="hidden px-2 lg:block">
          <Brand />
        </div>

        <div className="mt-10 px-2 text-[10px] font-bold tracking-[0.16em] text-slate-400">
          WORKSPACE
        </div>
        <nav aria-label="Main navigation" className="mt-3 space-y-1">
          <a
            aria-current="page"
            className="flex items-center gap-3 rounded-xl bg-indigo-50 px-3 py-2.5 text-sm font-semibold text-indigo-800"
            href="#workspace"
            onClick={() => setIsOpen(false)}
          >
            <Icon name="home" className="h-[18px] w-[18px]" />
            Overview
          </a>
          <a
            className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium text-slate-500 transition hover:bg-slate-50 hover:text-slate-800"
            href="#new-analysis"
            onClick={() => setIsOpen(false)}
          >
            <Icon name="plus" className="h-[18px] w-[18px]" />
            New analysis
          </a>
        </nav>

        <div className="mt-auto rounded-2xl bg-slate-50 p-4 ring-1 ring-slate-200/80">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-white text-indigo-700 shadow-sm ring-1 ring-slate-200/80">
            <Icon name="shield" className="h-[18px] w-[18px]" />
          </div>
          <p className="mt-3 text-sm font-semibold text-slate-800">
            Human-led decisions
          </p>
          <p className="mt-1 text-xs leading-5 text-slate-500">
            Keep candidate review fair, contextual, and in human hands.
          </p>
        </div>
        <div className="mt-4 border-t border-slate-100 px-2 pt-4 text-xs text-slate-400">
          SmartHire · Recruiter tools
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

function Brand() {
  return (
    <a className="inline-flex items-center gap-2.5" href="#workspace">
      <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-700 text-white shadow-sm">
        <Icon name="sparkles" className="h-[18px] w-[18px]" />
      </span>
      <span className="text-[17px] font-bold tracking-tight text-slate-950">
        Smart<span className="text-indigo-700">Hire</span>
      </span>
    </a>
  );
}

export default Sidebar;
