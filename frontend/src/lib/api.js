import { useMemo } from "react";
import { useAuth } from "../context/AuthContext.jsx";

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

// FastAPI returns `detail` as a string, or as a list of {loc, msg} for validation errors.
export function formatDetail(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((item) => {
        if (!item || typeof item !== "object") return null;
        const field = Array.isArray(item.loc)
          ? item.loc.filter((part) => part !== "body").join(".")
          : "";
        return field ? `${field}: ${item.msg}` : item.msg;
      })
      .filter(Boolean);
    return parts.length ? parts.join("; ") : null;
  }
  return null;
}

const STATUS_MESSAGES = {
  401: "Your session has expired. Please sign in again.",
  403: "You do not have permission to do that.",
  404: "That item could not be found.",
  413: "The file is too large (limit 10 MB).",
  429: "Too many attempts. Please wait a few minutes and try again.",
};

export async function readErrorMessage(response) {
  let detail = null;
  try {
    const data = await response.json();
    detail = formatDetail(data?.detail);
  } catch {
    // body was not JSON
  }
  return (
    detail ||
    STATUS_MESSAGES[response.status] ||
    `The request failed (HTTP ${response.status}).`
  );
}

export function describeError(error) {
  if (error instanceof ApiError) return error.message;
  return error?.message || "Something went wrong. Please try again.";
}

function buildClient(authFetch) {
  async function request(method, path, { json, form, raw } = {}) {
    const options = { method, headers: {} };
    if (json !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(json);
    } else if (form) {
      options.body = form; // the browser sets the multipart boundary
    }

    let response;
    try {
      response = await authFetch(path, options);
    } catch {
      throw new ApiError(
        "Could not reach the SmartHire server. Make sure the backend is running on http://127.0.0.1:8000.",
      );
    }
    if (!response.ok) {
      throw new ApiError(await readErrorMessage(response), response.status);
    }
    if (raw) return response;
    if (response.status === 204) return null;
    try {
      return await response.json();
    } catch {
      return null;
    }
  }

  return {
    get: (path) => request("GET", path),
    post: (path, json) => request("POST", path, { json }),
    put: (path, json) => request("PUT", path, { json }),
    del: (path) => request("DELETE", path),
    upload: (path, form) => request("POST", path, { form }),
    download: async (path) => (await request("GET", path, { raw: true })).blob(),
  };
}

export function useApi() {
  const { authFetch } = useAuth();
  return useMemo(() => buildClient(authFetch), [authFetch]);
}

export { buildClient };
