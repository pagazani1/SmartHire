import { describe, expect, it, vi } from "vitest";
import { ApiError, buildClient, formatDetail, readErrorMessage } from "./api.js";

const json = (status, body) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

describe("formatDetail", () => {
  it("handles strings, validation lists and unknown shapes", () => {
    expect(formatDetail("Nope")).toBe("Nope");
    expect(formatDetail([{ loc: ["body", "title"], msg: "Field required" }])).toBe("title: Field required");
    expect(formatDetail(undefined)).toBeNull();
  });
});

describe("readErrorMessage", () => {
  it("prefers the server detail, then a friendly status message", async () => {
    expect(await readErrorMessage(json(403, { detail: "Not yours" }))).toBe("Not yours");
    expect(await readErrorMessage(new Response("<html>", { status: 413 }))).toMatch(/too large/i);
    expect(await readErrorMessage(new Response("x", { status: 500 }))).toMatch(/HTTP 500/);
  });
});

describe("api client", () => {
  it("sends JSON bodies through the authenticated fetch and parses the reply", async () => {
    const authFetch = vi.fn(async () => json(201, { id: "abc" }));
    const api = buildClient(authFetch);
    const result = await api.post("/api/jobs/", { title: "X" });
    expect(result).toEqual({ id: "abc" });
    const [url, options] = authFetch.mock.calls[0];
    expect(url).toBe("/api/jobs/");
    expect(options.method).toBe("POST");
    expect(options.headers["Content-Type"]).toBe("application/json");
    expect(options.body).toBe(JSON.stringify({ title: "X" }));
  });

  it("does not set a content type for file uploads so the browser adds the boundary", async () => {
    const authFetch = vi.fn(async () => json(201, {}));
    const form = new FormData();
    form.append("file", new Blob(["x"]), "r.pdf");
    await buildClient(authFetch).upload("/api/candidates/1/resumes/", form);
    const options = authFetch.mock.calls[0][1];
    expect(options.headers["Content-Type"]).toBeUndefined();
    expect(options.body).toBe(form);
  });

  it("turns HTTP errors into ApiError with the server message", async () => {
    const api = buildClient(async () => json(403, { detail: "You do not have permission to access this candidate." }));
    await expect(api.get("/api/candidates/1")).rejects.toMatchObject({
      name: "ApiError",
      status: 403,
      message: "You do not have permission to access this candidate.",
    });
  });

  it("turns network failures into a helpful message", async () => {
    const api = buildClient(async () => {
      throw new TypeError("Failed to fetch");
    });
    const error = await api.get("/api/jobs/").catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.message).toMatch(/backend is running/i);
  });

  it("returns null for 204 responses", async () => {
    const api = buildClient(async () => new Response(null, { status: 204 }));
    expect(await api.del("/api/resumes/1")).toBeNull();
  });
});
