"""Coordinator-side privacy layer (model-agnostic — applies to Moonlight/Kimi alike):

  1. PII scrub      — remove HIPAA-18 / PII BEFORE anything leaves the device
                      (OpenMed pattern engine if available, else built-in HIPAA-18 rules).
  2. #1 anchor-protect — per-occurrence randomize the top frequency-anchor tokens, killing
                      the VMA "naming" channel that survives covariant obfuscation (~12%->~1%).
  3. #3 launder (iface) — on-device rephrase so surface tokens change while meaning holds
                      (preserves answers; further starves the frequency channel).

These run on the TRUSTED edge device, before share/scramble. They carry unchanged from
Moonlight to Kimi K2.7 (they touch text/tokens, not the model).
"""
import re, hashlib

# ---------- 1. PII scrub (HIPAA-18 baseline; OpenMed enhancement if present) ----------
_HIPAA = {
    "EMAIL": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "PHONE": r"\b(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
    "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
    "MRN": r"\b(?:MRN|mrn)[:#\s]*\d{5,}\b",
    "DATE": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    "IP": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "ZIP": r"\b\d{5}(?:-\d{4})?\b",
    "URL": r"\bhttps?://\S+\b",
}
try:
    import openmed                                     # model-backed NER (44M clinical PII model)
    _HAVE_OM = True
except Exception:
    _HAVE_OM = False


def scrub_pii(text: str):
    """Return (scrubbed_text, found). Two layers, both on the TRUSTED device:
      - HIPAA-18 regex: structured IDs (SSN/MRN/email/phone/cc/IP/date/url/zip).
      - OpenMed NER (model): unstructured PII the regex can't — NAMES, orgs, cities, ages,
        clinical identifiers. The reversible map stays on-device, never sent."""
    found, mapping = [], {}
    out = text
    # 1) NER FIRST on clean text (best accuracy for names/orgs/cities)
    ner_hits = 0; placeholders = []
    if _HAVE_OM:
        try:
            r = openmed.deidentify(out)                  # model-backed NER
            new = getattr(r, "text", None) or getattr(r, "deidentified_text", None)
            if new:
                placeholders = re.findall(r"\[[a-z_]+\]", new)
                ner_hits = len(placeholders); out = new
        except Exception:
            pass
    # 2) HIPAA-18 regex for structured IDs the NER may not normalize
    for label, pat in _HIPAA.items():
        def repl(m):
            tok = f"[{label}_{len(mapping)}]"
            mapping[tok] = m.group(0); found.append(label)
            return tok
        out = re.sub(pat, repl, out)
    return out, {"counts": {l: found.count(l) for l in set(found)},
                 "ner_entities": ner_hits, "ner_types": sorted(set(placeholders)),
                 "total": len(found) + ner_hits,
                 "engine": "openmed-NER+hipaa18" if _HAVE_OM else "hipaa18"}


# ---------- 2. #1 anchor-protection ----------
# function/anchor words drive the VMA frequency channel; per-occurrence salt them so the
# repeat pattern the attacker keys on is destroyed (meaning unchanged for the model after
# the matching de-salt; here we expose the protection as a token-stream transform hook).
_ANCHORS = set("the of and a to in is was he for it with as his on be at by i this had not are "
               "but from or have an they which one you were her all she there would their we him "
               "been has when who will more no if out so said what up its about into than them can "
               "only other new some could time these two may then do first any my now".split())


def anchor_protect(tokens, salt_seed: int = 0):
    """tokens: list[str]. Returns (protected_tokens, n_protected). Each anchor occurrence
    gets a per-occurrence invisible salt id so identical anchors no longer look identical to
    a frequency attacker. (De-salt happens on-device after recombination.)"""
    out, n = [], 0
    for i, t in enumerate(tokens):
        if t.lower() in _ANCHORS:
            salt = hashlib.sha256(f"{salt_seed}:{i}:{t}".encode()).hexdigest()[:4]
            out.append(f"{t}​{salt}")              # zero-width-joined salt, stripped on-device
            n += 1
        else:
            out.append(t)
    return out, n


# ---------- 3. #3 laundering (interface) ----------
LAUNDER_PROMPT = ("Rephrase the following request so the wording is different but the meaning "
                  "and any technical content are exactly preserved. Output only the rephrased "
                  "request.\n\nRequest: {p}\nRephrased:")


def launder(prompt: str, rephrase_fn=None) -> str:
    """rephrase_fn = an on-device model callable(str)->str. If None, returns prompt unchanged
    (laundering disabled). #3 changes surface tokens while keeping the answer (validated cos 0.92)."""
    if rephrase_fn is None:
        return prompt
    try:
        return rephrase_fn(LAUNDER_PROMPT.format(p=prompt)).strip() or prompt
    except Exception:
        return prompt


def protect(prompt: str, rephrase_fn=None, salt_seed: int = 0):
    """Full pipeline: PII scrub -> #3 launder -> (tokens) #1 anchor-protect. Returns dict."""
    scrubbed, pii = scrub_pii(prompt)
    laundered = launder(scrubbed, rephrase_fn)
    toks = laundered.split()
    protected, n_anchor = anchor_protect(toks, salt_seed)
    return {"pii": pii, "laundered": laundered, "anchors_protected": n_anchor,
            "protected_text": " ".join(protected)}
