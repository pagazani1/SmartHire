import { createContext, useCallback, useContext, useEffect, useState } from "react";

const AuthContext = createContext(null);

const TOKEN_STORAGE_KEY = "smart_hire_token";
const RECRUITER_STORAGE_KEY = "smart_hire_recruiter";

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => {
    try {
      return sessionStorage.getItem(TOKEN_STORAGE_KEY) || null;
    } catch {
      return null;
    }
  });

  const [recruiter, setRecruiter] = useState(() => {
    try {
      const stored = sessionStorage.getItem(RECRUITER_STORAGE_KEY);
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  });

  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);

  const openAuthModal = useCallback(() => setIsAuthModalOpen(true), []);
  const closeAuthModal = useCallback(() => setIsAuthModalOpen(false), []);

  const logout = useCallback(() => {
    setToken(null);
    setRecruiter(null);
    try {
      sessionStorage.removeItem(TOKEN_STORAGE_KEY);
      sessionStorage.removeItem(RECRUITER_STORAGE_KEY);
    } catch {
      // ignore
    }
  }, []);

  // Verify stored session on initial mount
  useEffect(() => {
    const activeToken = token || sessionStorage.getItem(TOKEN_STORAGE_KEY);
    if (!activeToken) return;

    let cancelled = false;
    async function verifySession() {
      try {
        const response = await fetch("/api/auth/me", {
          headers: {
            Authorization: `Bearer ${activeToken}`,
          },
        });
        if (response.ok) {
          const profile = await response.json();
          if (!cancelled) {
            setRecruiter(profile);
            sessionStorage.setItem(RECRUITER_STORAGE_KEY, JSON.stringify(profile));
          }
        } else if (response.status === 401) {
          if (!cancelled) {
            logout();
          }
        }
      } catch {
        // Keep existing session on network failure
      }
    }

    verifySession();
    return () => {
      cancelled = true;
    };
  }, [token, logout]);

  const login = useCallback(async (email, password) => {
    try {
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: email.trim(), password }),
      });

      if (!response.ok) {
        let errorDetail =
          response.status >= 500
            ? "The SmartHire server is not responding. Make sure the backend is running (see docs/LOCAL_SETUP.md)."
            : `Login failed (HTTP ${response.status}).`;
        try {
          const data = await response.json();
          if (typeof data.detail === "string") errorDetail = data.detail;
        } catch {
          // ignore
        }
        return { success: false, error: errorDetail };
      }

      const data = await response.json();
      const accessToken = data.access_token;
      const recruiterProfile = data.recruiter;

      setToken(accessToken);
      setRecruiter(recruiterProfile);
      sessionStorage.setItem(TOKEN_STORAGE_KEY, accessToken);
      sessionStorage.setItem(RECRUITER_STORAGE_KEY, JSON.stringify(recruiterProfile));
      setIsAuthModalOpen(false);

      return { success: true };
    } catch (err) {
      return {
        success: false,
        error:
          err instanceof TypeError
            ? "Unable to reach the SmartHire API server."
            : err.message || "Login failed.",
      };
    }
  }, []);

  const authFetch = useCallback(
    async (url, options = {}) => {
      const currentToken = token || sessionStorage.getItem(TOKEN_STORAGE_KEY);
      const headers = new Headers(options.headers || {});

      if (currentToken && !headers.has("Authorization")) {
        headers.set("Authorization", `Bearer ${currentToken}`);
      }

      const finalOptions = {
        ...options,
        headers,
      };

      const response = await fetch(url, finalOptions);

      if (response.status === 401) {
        // Token expired or invalid
        logout();
        setIsAuthModalOpen(true);
      }

      return response;
    },
    [token, logout],
  );

  return (
    <AuthContext.Provider
      value={{
        token,
        recruiter,
        isAuthModalOpen,
        openAuthModal,
        closeAuthModal,
        login,
        logout,
        authFetch,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
