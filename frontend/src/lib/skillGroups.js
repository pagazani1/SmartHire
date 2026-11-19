// A requirement group is a list of alternatives: any ONE of them satisfies the group.
// In the text boxes, write one group per line and separate alternatives with "|" or " or ".
// Example line:  Python | Java   -> one group, satisfied by either skill.

const SEPARATOR = /\s*\|\s*|\s+or\s+/i;

export function parseSkillGroups(text) {
  const groups = [];
  const seenGroups = new Set();
  for (const line of String(text ?? "").split(/\r?\n/)) {
    const seenSkills = new Set();
    const skills = [];
    for (const part of line.split(SEPARATOR)) {
      const skill = part.trim().replace(/[;,]+$/, "").trim();
      const key = skill.toLowerCase();
      if (skill && !seenSkills.has(key)) {
        seenSkills.add(key);
        skills.push(skill);
      }
    }
    if (skills.length === 0) continue;
    const groupKey = [...seenSkills].sort().join("\u0000");
    if (seenGroups.has(groupKey)) continue;
    seenGroups.add(groupKey);
    groups.push(skills);
  }
  return groups;
}

export function formatSkillGroups(groups) {
  return (Array.isArray(groups) ? groups : [])
    .map((group) => (Array.isArray(group) ? group.join(" | ") : String(group)))
    .join("\n");
}

// Skills must come from the server's vocabulary. These helpers check a list of groups
// against it and propose close matches ("api" -> "API Development").
export function findUnknownSkills(groups, vocabulary) {
  const known = new Set((vocabulary ?? []).map((skill) => skill.toLowerCase()));
  if (known.size === 0) return [];
  const unknown = [];
  const seen = new Set();
  for (const group of groups) {
    for (const skill of group) {
      const key = skill.toLowerCase();
      if (!known.has(key) && !seen.has(key)) {
        seen.add(key);
        unknown.push(skill);
      }
    }
  }
  return unknown;
}

export function suggestSkills(name, vocabulary, limit = 3) {
  const needle = String(name ?? "").trim().toLowerCase();
  if (!needle) return [];
  const scored = [];
  for (const skill of vocabulary ?? []) {
    const hay = skill.toLowerCase();
    let score = 0;
    if (hay.startsWith(needle)) score = 3;
    else if (needle.startsWith(hay)) score = 2.5;
    else if (hay.includes(needle) || needle.includes(hay)) score = 2;
    else if (hay.split(/[\s/.-]+/).some((word) => word.length > 2 && needle.includes(word))) score = 1;
    if (score > 0) scored.push([score, skill]);
  }
  return scored
    .sort((a, b) => b[0] - a[0] || a[1].length - b[1].length)
    .slice(0, limit)
    .map(([, skill]) => skill);
}
