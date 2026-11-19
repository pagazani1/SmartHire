import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import App from "./App.jsx";
import { RECRUITER, mockApi, signInStorage } from "./test/helpers.js";

const lists = {
  "GET /api/jobs/?limit=100": { body: [] },
  "GET /api/candidates/?limit=100": { body: [] },
};

describe("sign-in gate", () => {
  it("shows only the login screen and calls no data endpoint when signed out", () => {
    const { calls } = mockApi({ ...lists });
    render(<App />);
    expect(screen.getByRole("heading", { name: /recruiter sign in/i })).toBeInTheDocument();
    expect(screen.queryByText(/getting started/i)).not.toBeInTheDocument();
    expect(calls).toHaveLength(0);
  });

  it("logs in, stores the session, and sends the token on every later request", async () => {
    const { calls } = mockApi({
      "POST /api/auth/login": { body: { access_token: "tok-abc", expires_at: "2099-01-01T00:00:00Z", recruiter: RECRUITER } },
      "GET /api/auth/me": { body: RECRUITER },
      ...lists,
    });
    render(<App />);
    await userEvent.type(screen.getByLabelText("Email"), "test@example.test");
    await userEvent.type(screen.getByLabelText("Password"), "correct-horse-battery");
    await userEvent.click(screen.getByRole("button", { name: /^sign in$/i }));

    expect(await screen.findByText(/getting started/i)).toBeInTheDocument();
    const dataCalls = calls.filter((c) => c.url.startsWith("/api/jobs") || c.url.startsWith("/api/candidates"));
    expect(dataCalls.length).toBeGreaterThan(0);
    for (const call of dataCalls) {
      expect(call.headers.get("Authorization")).toBe("Bearer tok-abc");
    }
    expect(sessionStorage.getItem("smart_hire_token")).toBe("tok-abc");
  });

  it("shows the server's message for wrong credentials and stays signed out", async () => {
    mockApi({ "POST /api/auth/login": { status: 401, body: { detail: "Invalid email or password." } } });
    render(<App />);
    await userEvent.type(screen.getByLabelText("Email"), "a@b.test");
    await userEvent.type(screen.getByLabelText("Password"), "wrong-password");
    await userEvent.click(screen.getByRole("button", { name: /^sign in$/i }));
    expect(await screen.findByText("Invalid email or password.")).toBeInTheDocument();
    expect(sessionStorage.getItem("smart_hire_token")).toBeNull();
  });

  it("explains what to do when the backend is down or the database is unavailable", async () => {
    mockApi({ "POST /api/auth/login": { status: 502, body: null } });
    render(<App />);
    await userEvent.type(screen.getByLabelText("Email"), "a@b.test");
    await userEvent.type(screen.getByLabelText("Password"), "some-password");
    await userEvent.click(screen.getByRole("button", { name: /^sign in$/i }));
    expect(await screen.findByText(/server is not responding/i)).toBeInTheDocument();
  });

  it("shows the lockout message after too many failed attempts", async () => {
    mockApi({ "POST /api/auth/login": { status: 429, body: { detail: "Too many failed sign-in attempts. Please try again later." } } });
    render(<App />);
    await userEvent.type(screen.getByLabelText("Email"), "a@b.test");
    await userEvent.type(screen.getByLabelText("Password"), "some-password");
    await userEvent.click(screen.getByRole("button", { name: /^sign in$/i }));
    expect(await screen.findByText(/too many failed sign-in attempts/i)).toBeInTheDocument();
  });

  it("asks for both fields before contacting the server", async () => {
    const { calls } = mockApi({});
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /^sign in$/i }));
    expect(await screen.findByText(/enter your recruiter email and password/i)).toBeInTheDocument();
    expect(calls).toHaveLength(0);
  });

  it("returns to the login screen when the server says the session expired (401)", async () => {
    signInStorage();
    mockApi({
      "GET /api/auth/me": { body: RECRUITER },
      "GET /api/jobs/?limit=100": { status: 401, body: { detail: "expired" } },
      "GET /api/candidates/?limit=100": { status: 401, body: { detail: "expired" } },
    });
    render(<App />);
    expect(await screen.findByRole("heading", { name: /recruiter sign in/i })).toBeInTheDocument();
    expect(sessionStorage.getItem("smart_hire_token")).toBeNull();
  });

  it("signs out from the header", async () => {
    signInStorage();
    mockApi({ "GET /api/auth/me": { body: RECRUITER }, ...lists });
    render(<App />);
    await screen.findByText(/getting started/i);
    await userEvent.click(screen.getAllByRole("button", { name: /sign out/i })[0]);
    await waitFor(() => expect(screen.getByRole("heading", { name: /recruiter sign in/i })).toBeInTheDocument());
  });
});
