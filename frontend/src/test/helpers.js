import { vi } from "vitest";

export const RECRUITER = {
  id: "11111111-1111-1111-1111-111111111111",
  display_name: "Test Recruiter",
  email_normalized: "test@example.test",
  role: "recruiter",
};

/** Install a fake fetch. `routes` maps "METHOD /path" to a handler or a {status, body} object. */
export function mockApi(routes) {
  const calls = [];
  const fetchMock = vi.fn(async (url, options = {}) => {
    const method = (options.method || "GET").toUpperCase();
    const key = `${method} ${url}`;
    const headers = new Headers(options.headers || {});
    calls.push({ key, url, method, headers, body: options.body });
    const route = routes[key];
    if (!route) {
      return new Response(JSON.stringify({ detail: `unmocked ${key}` }), { status: 599 });
    }
    const result = typeof route === "function" ? await route({ headers, body: options.body }) : route;
    const { status = 200, body = null } = result;
    return new Response(status === 204 ? null : JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls, fetchMock };
}

export function signInStorage() {
  sessionStorage.setItem("smart_hire_token", "tok-123");
  sessionStorage.setItem("smart_hire_recruiter", JSON.stringify(RECRUITER));
}
