import { useState } from "react";
import { useAuth } from "../context/AuthContext.jsx";
import { Alert, Field, PrimaryButton, inputClass } from "./ui.jsx";

function LoginScreen() {
  const { login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!email.trim() || !password) {
      setError("Enter your recruiter email and password.");
      return;
    }
    setIsSubmitting(true);
    setError("");
    const result = await login(email, password);
    setIsSubmitting(false);
    if (!result.success) setError(result.error);
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-[#f7f8fa] p-4">
      <form
        className="w-full max-w-md rounded-2xl border border-slate-200 bg-white p-7 shadow-[0_8px_30px_rgba(15,23,42,0.06)] sm:p-9"
        onSubmit={handleSubmit}
      >
        <p className="text-sm font-semibold tracking-wide text-indigo-700">SMARTHIRE</p>
        <h1 className="mt-1 text-2xl font-bold tracking-tight text-slate-950">
          Recruiter sign in
        </h1>
        <p className="mt-2 text-sm leading-6 text-slate-500">
          Explainable resume screening. Accounts are created by an administrator;
          there is no public registration.
        </p>

        <div className="mt-6 space-y-4">
          {error ? <Alert kind="error">{error}</Alert> : null}
          <Field htmlFor="login-email" label="Email">
            <input
              autoComplete="username"
              autoFocus
              className={inputClass}
              id="login-email"
              onChange={(event) => setEmail(event.target.value)}
              type="email"
              value={email}
            />
          </Field>
          <Field htmlFor="login-password" label="Password">
            <input
              autoComplete="current-password"
              className={inputClass}
              id="login-password"
              onChange={(event) => setPassword(event.target.value)}
              type="password"
              value={password}
            />
          </Field>
          <PrimaryButton busy={isSubmitting} className="w-full" type="submit">
            Sign in
          </PrimaryButton>
        </div>
      </form>
    </div>
  );
}

export default LoginScreen;
