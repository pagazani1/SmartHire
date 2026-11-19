/**
 * End-to-end check: the real React app (rendered in jsdom) driving the real forms against a
 * RUNNING backend and database. Nothing in the API layer is mocked.
 *
 * Needs: backend + (optionally the Vite proxy) running, a recruiter account, and a PDF.
 *   E2E_BASE_URL   e.g. http://127.0.0.1:5173 (through the Vite proxy) or http://127.0.0.1:8000
 *   E2E_EMAIL / E2E_PASSWORD   recruiter credentials
 *   E2E_PDF        path to a small text PDF mentioning Python, PostgreSQL and Docker
 * Run:  npm run test:e2e
 *
 * Limits: jsdom is not a real browser, so layout, CSS, focus rings and real file pickers
 * are not exercised.
 */
import { readFileSync } from "node:fs";
import { configure, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, describe, expect, it, vi } from "vitest";
import App from "../App.jsx";

// A real backend (password hashing, database, PDF parsing) is slower than the 1s default.
configure({ asyncUtilTimeout: 20000 });

const BASE = process.env.E2E_BASE_URL;
const EMAIL = process.env.E2E_EMAIL;
const PASSWORD = process.env.E2E_PASSWORD;
const PDF_PATH = process.env.E2E_PDF;
const run = BASE && EMAIL && PASSWORD && PDF_PATH ? describe : describe.skip;

const nodeFetch = globalThis.fetch;

// jsdom's FormData is not understood by Node's fetch, so build the multipart body by hand.
async function encodeMultipart(formData) {
  const boundary = `----smarthire${Math.random().toString(16).slice(2)}`;
  const chunks = [];
  for (const [name, value] of formData.entries()) {
    if (typeof value === "string") {
      chunks.push(Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="${name}"\r\n\r\n${value}\r\n`));
    } else {
      const bytes = Buffer.from(await value.arrayBuffer());
      chunks.push(
        Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="${name}"; filename="${value.name}"\r\nContent-Type: ${value.type || "application/octet-stream"}\r\n\r\n`),
        bytes,
        Buffer.from("\r\n"),
      );
    }
  }
  chunks.push(Buffer.from(`--${boundary}--\r\n`));
  return { body: Buffer.concat(chunks), contentType: `multipart/form-data; boundary=${boundary}` };
}

let corruptToken = false; // flips on to make the real server reject the session

async function realFetch(url, options = {}) {
  const target = String(url).startsWith("/") ? `${BASE}${url}` : url;
  // authFetch passes a Headers object; copy it properly (spreading it would drop every header).
  const headers = new Headers(options.headers || {});
  if (corruptToken && headers.has("Authorization")) headers.set("Authorization", "Bearer expired.invalid.token");
  const init = { ...options, headers };
  if (init.body && typeof init.body.entries === "function" && typeof init.body !== "string") {
    const { body, contentType } = await encodeMultipart(init.body);
    init.body = body;
    headers.set("Content-Type", contentType);
  }
  return nodeFetch(target, init);
}

const stamp = Date.now().toString(36);
const JOB_TITLE = `E2E Backend Engineer ${stamp}`;
const CANDIDATE = `E2E Candidate ${stamp}`;

async function chooseSidebar(name) {
  const nav = screen.getByRole("navigation", { name: /main navigation/i });
  await userEvent.click(within(nav).getByRole("button", { name: new RegExp(name, "i") }));
}

async function recordConsent(purposeValue) {
  await userEvent.click(await screen.findByRole("button", { name: /record consent/i }));
  await userEvent.selectOptions(screen.getByLabelText(/^purpose$/i), purposeValue);
  await userEvent.click(screen.getByRole("button", { name: /save consent/i }));
  await screen.findByText(/consent recorded/i);
}

run("SmartHire recruiter workflow against a live backend", () => {
  beforeAll(() => {
    vi.stubGlobal("fetch", realFetch);
  });

  it("covers login, job, candidate, consent, upload, screening, history, withdrawal, erasure and session expiry", async () => {
    window.confirm = vi.fn(() => true);
    render(<App />);

    // 1. wrong password is rejected with the server's message
    await userEvent.type(screen.getByLabelText("Email"), EMAIL);
    await userEvent.type(screen.getByLabelText("Password"), "definitely-wrong-password");
    await userEvent.click(screen.getByRole("button", { name: /^sign in$/i }));
    expect(await screen.findByText(/invalid email or password/i)).toBeInTheDocument();

    // 2. correct password reaches the dashboard
    await userEvent.clear(screen.getByLabelText("Password"));
    await userEvent.type(screen.getByLabelText("Password"), PASSWORD);
    await userEvent.click(screen.getByRole("button", { name: /^sign in$/i }));
    expect(await screen.findByText(/getting started/i)).toBeInTheDocument();

    // 2b. every dashboard shortcut goes where it says
    await userEvent.click(screen.getByRole("button", { name: /go to jobs/i }));
    expect(await screen.findByRole("heading", { name: /jobs & requirements/i })).toBeInTheDocument();
    await chooseSidebar("Dashboard");
    await userEvent.click(await screen.findByRole("button", { name: /go to candidates/i }));
    expect(await screen.findByRole("heading", { name: /candidates, consent & resumes/i })).toBeInTheDocument();
    await chooseSidebar("Dashboard");
    await userEvent.click(await screen.findByRole("button", { name: /open screening/i }));
    expect(await screen.findByRole("heading", { name: /application screening & history/i })).toBeInTheDocument();

    // 3. create a job (unsupported skill is stopped client-side first)
    await chooseSidebar("Jobs");
    await userEvent.click(await screen.findByRole("button", { name: /new job/i }));
    await userEvent.type(screen.getByLabelText(/^title/i), JOB_TITLE);
    await userEvent.type(screen.getByLabelText(/job description/i), "Build APIs with Python and PostgreSQL, deployed with Docker.");
    await userEvent.type(screen.getByLabelText(/required skill groups/i), "Python | Java{enter}PostgreSQL{enter}Docker{enter}REST");
    expect(await screen.findByText(/some skills are not supported/i)).toBeInTheDocument();
    const requiredBox = screen.getByLabelText(/required skill groups/i);
    await userEvent.clear(requiredBox);
    await userEvent.type(requiredBox, "Python | Java{enter}PostgreSQL{enter}Docker");
    await userEvent.click(screen.getByRole("button", { name: /create job/i }));
    expect(await screen.findByText(/job created/i)).toBeInTheDocument();
    expect(await screen.findByText(JOB_TITLE)).toBeInTheDocument();
    const row = screen.getByText(JOB_TITLE).closest("tr");
    expect(within(row).getByText("v1")).toBeInTheDocument();
    await userEvent.click(within(row).getByRole("button", { name: /^edit$/i }));
    await userEvent.type(await screen.findByLabelText(/job description/i), " Updated for e2e.");
    await userEvent.click(screen.getByRole("button", { name: /save changes/i }));
    expect(await screen.findByText(/job updated/i)).toBeInTheDocument();
    await waitFor(() => expect(within(screen.getByText(JOB_TITLE).closest("tr")).getByText("v2")).toBeInTheDocument());

    // 4. add a candidate to that job
    await chooseSidebar("Candidates");
    await userEvent.click(await screen.findByRole("button", { name: /^add$/i }));
    await waitFor(() => expect(screen.getByLabelText(/applying for/i).value).not.toBe(""));
    await userEvent.selectOptions(screen.getByLabelText(/applying for/i), screen.getByRole("option", { name: JOB_TITLE }));
    await userEvent.type(screen.getByLabelText(/full name/i), CANDIDATE);
    await userEvent.click(screen.getByRole("button", { name: /add candidate/i }));
    expect(await screen.findByRole("heading", { name: CANDIDATE })).toBeInTheDocument();

    // 5. upload is blocked until consent exists, then consents are captured
    expect(screen.getByRole("button", { name: /^upload$/i })).toBeDisabled();
    await recordConsent("document_storage");
    await recordConsent("screening");
    await waitFor(() => expect(screen.getAllByText("granted").length).toBe(2));

    // 6. upload the resume
    const pdf = readFileSync(PDF_PATH);
    await userEvent.upload(screen.getByLabelText(/pdf resume/i), new File([pdf], "resume.pdf", { type: "application/pdf" }));
    await waitFor(() => expect(screen.getByRole("button", { name: /^upload$/i })).toBeEnabled());
    await userEvent.click(screen.getByRole("button", { name: /^upload$/i }));
    expect(await screen.findByText(/resume uploaded and stored privately/i)).toBeInTheDocument();
    expect(await screen.findByText("resume.pdf")).toBeInTheDocument();

    // 6b. download the stored resume back
    const createObjectURL = vi.fn(() => "blob:smarthire-test");
    URL.createObjectURL = createObjectURL;
    URL.revokeObjectURL = vi.fn();
    await userEvent.click(screen.getByRole("button", { name: /^download$/i }));
    await waitFor(() => expect(createObjectURL).toHaveBeenCalled());
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    // 7. screen the candidate and read the explainable result
    await userEvent.click(screen.getByRole("button", { name: /screen this candidate/i }));
    expect(await screen.findByRole("button", { name: /run persistent screening/i })).toBeEnabled();
    await waitFor(() => expect(screen.getByLabelText(/^candidate$/i).selectedOptions[0].text).toContain(CANDIDATE));
    await userEvent.click(screen.getByRole("button", { name: /run persistent screening/i }));
    expect(await screen.findByText(/run #1 recorded/i)).toBeInTheDocument();
    expect((await screen.findAllByText("Good Fit")).length).toBeGreaterThan(0);
    expect(screen.getByText(/text-model reference/i, { selector: "span" })).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByText(/^1 run$/i).length).toBeGreaterThan(0));

    // 7b. refresh and "generate new key" really do something
    const keyBox = screen.getByLabelText(/idempotency-key header/i);
    const keyBefore = keyBox.value;
    await userEvent.click(screen.getByRole("button", { name: /generate new key/i }));
    expect(screen.getByLabelText(/idempotency-key header/i).value).not.toBe(keyBefore);
    await userEvent.click(screen.getByRole("button", { name: /^refresh$/i }));
    await waitFor(() => expect(screen.getByRole("button", { name: /run persistent screening/i })).toBeEnabled());

    // 8. a second run is recorded in history
    await userEvent.click(screen.getByRole("button", { name: /run persistent screening/i }));
    expect(await screen.findByText(/run #2 recorded/i)).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByText(/^2 runs$/i).length).toBeGreaterThan(0));

    // 9. withdraw screening consent -> screening is refused and the error is shown
    await chooseSidebar("Candidates");
    await userEvent.click(await screen.findByRole("button", { name: new RegExp(CANDIDATE, "i") }));
    const withdrawButtons = await screen.findAllByRole("button", { name: /^withdraw$/i });
    expect(withdrawButtons.length).toBe(2);
    await userEvent.click(withdrawButtons[0]);
    expect(await screen.findByText(/consent withdrawn/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /screen this candidate/i }));
    await userEvent.click(await screen.findByRole("button", { name: /run persistent screening/i }));
    const refusal = await screen.findByRole("alert");
    expect(refusal).toHaveTextContent(/consent/i);
    console.log("Screening after withdrawal shows:", refusal.textContent);
    await waitFor(() => expect(screen.queryByText(/run #3 recorded/i)).not.toBeInTheDocument());

    // 10. delete the resume, then erase the candidate
    await chooseSidebar("Candidates");
    await userEvent.click(await screen.findByRole("button", { name: new RegExp(CANDIDATE, "i") }));
    await userEvent.click(await screen.findByRole("button", { name: /^delete$/i }));
    expect(await screen.findByText(/resume deleted/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /erase this candidate/i }));
    expect(await screen.findByText(/personal data erased/i)).toBeInTheDocument();
    expect((await screen.findAllByText(/anonymized candidate/i)).length).toBeGreaterThan(0);
    // the detail pane reloads after the notice appears, so wait for the real name to disappear
    await waitFor(() => expect(screen.queryByRole("heading", { name: CANDIDATE })).not.toBeInTheDocument());
    await waitFor(() => expect(screen.queryByText(CANDIDATE)).not.toBeInTheDocument()); // list refreshed too
    expect(screen.queryByRole("button", { name: /erase this candidate/i })).not.toBeInTheDocument();

    // 10b. the stateless quick check works with a signed-in session
    await chooseSidebar("Quick resume check");
    await userEvent.upload(await screen.findByLabelText(/resume/i, { selector: "input[type=file]" }), new File([pdf], "quick.pdf", { type: "application/pdf" }));
    await userEvent.type(screen.getByLabelText(/job description/i), "Required skills: Python, PostgreSQL and Docker.");
    await userEvent.click(screen.getByRole("button", { name: /analyse resume/i }));
    expect((await screen.findAllByText(/good fit|potential fit|no fit/i)).length).toBeGreaterThan(0);

    // 11. an expired / invalid session sends the user back to sign-in
    corruptToken = true; // the server will now answer 401 to every request
    await chooseSidebar("Jobs");
    expect(await screen.findByRole("heading", { name: /recruiter sign in/i })).toBeInTheDocument();
    expect(sessionStorage.getItem("smart_hire_token")).toBeNull();
    corruptToken = false;
  });
});
