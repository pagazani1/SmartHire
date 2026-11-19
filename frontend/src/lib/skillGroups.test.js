import { describe, expect, it } from "vitest";
import { findUnknownSkills, formatSkillGroups, parseSkillGroups, suggestSkills } from "./skillGroups.js";

describe("parseSkillGroups", () => {
  it("treats alternatives on one line as a single group", () => {
    expect(parseSkillGroups("Python | Java\nPostgreSQL")).toEqual([["Python", "Java"], ["PostgreSQL"]]);
  });

  it("accepts the word 'or' as a separator", () => {
    expect(parseSkillGroups("FastAPI or Flask")).toEqual([["FastAPI", "Flask"]]);
  });

  it("ignores blanks and removes duplicates within and across groups", () => {
    const groups = parseSkillGroups("\nPython | python\n\nJava | Python\nPython | Java\n  ");
    expect(groups).toEqual([["Python"], ["Java", "Python"]]);
  });

  it("returns an empty list for empty input", () => {
    expect(parseSkillGroups("")).toEqual([]);
    expect(parseSkillGroups(undefined)).toEqual([]);
  });

  it("round-trips through formatSkillGroups", () => {
    const groups = [["Python", "Java"], ["SQL"]];
    expect(parseSkillGroups(formatSkillGroups(groups))).toEqual(groups);
  });
});

describe("vocabulary helpers", () => {
  const vocabulary = ["Python", "API Development", "PostgreSQL", "Docker"];

  it("finds skills outside the vocabulary, ignoring case and duplicates", () => {
    const groups = [["python", "REST"], ["REST"], ["Docker"]];
    expect(findUnknownSkills(groups, vocabulary)).toEqual(["REST"]);
  });

  it("skips validation when the vocabulary could not be loaded", () => {
    expect(findUnknownSkills([["Anything"]], [])).toEqual([]);
  });

  it("suggests close matches", () => {
    expect(suggestSkills("api", vocabulary)).toContain("API Development");
    expect(suggestSkills("postgres", vocabulary)[0]).toBe("PostgreSQL");
    expect(suggestSkills("", vocabulary)).toEqual([]);
  });
});
