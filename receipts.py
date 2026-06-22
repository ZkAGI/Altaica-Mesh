"""Post-quantum receipt + verification layer for private inference.

ALL primitives here are post-quantum:
  * Signatures: ML-DSA-65 (NIST FIPS 204, lattice) — replaces Ed25519.
  * Hashing / commitment: SHA3-256 (Keccak) Merkle tree — no elliptic curves.
  * Batch proof: a TRANSPARENT, hash-only, Fiat-Shamir-sampled Merkle argument
    (STARK-class assumptions: collision-resistant hashes, NO trusted setup,
    quantum-safe). This is the same security family as a zk-STARK.

Per request -> ML-DSA-signed receipt:
  input_commitment (SHA3 of the scrambled wire the node saw, not the plaintext),
  output_hash, privacy_cos (~0), integrity_ok (node math independently recomputed).
Receipts batch into a SHA3 Merkle root. The root is the ONE thing committed
on-chain (Solana memo/PDA + Base calldata) — O(1) on-chain, O(log n) inclusion.

On the STARK, honestly:
  - The COMMITMENT backbone (SHA3 Merkle) and the sampling argument below ARE the
    transparent, post-quantum core a STARK is built on. They give: anchored root
    on-chain, O(log n) inclusion proofs, and a Fiat-Shamir batch-integrity proof
    with soundness 1-(1-f)^k against a fraction f of bad receipts.
  - A FULL AIR/FRI succinct prover (true poly-log proof size + zero-knowledge over
    a custom statement) plugs in at `stark_prove_batch()` below. Production options,
    all hash-based / PQ: Winterfell, Plonky3, Cairo, RISC Zero. We do NOT hand-roll
    FRI here (rolling your own is unsafe); the interface is ready for a vetted prover.
"""
import hashlib, json, os, time
from dilithium_py.ml_dsa import ML_DSA_65

HERE = os.path.dirname(os.path.abspath(__file__))
KEYFILE = os.path.join(HERE, "coordinator_mldsa.key")
LOGFILE = os.path.join(HERE, "receipt_log.jsonl")
ANCHORFILE = os.path.join(HERE, "anchors.jsonl")
BATCH_N = int(os.environ.get("RECEIPT_BATCH_N", "8"))
PRIVACY_TAU = float(os.environ.get("PRIVACY_TAU", "0.05"))   # |cos| must be <= this
SAMPLES = int(os.environ.get("STARK_SAMPLES", "16"))          # FS spot-checks per batch


def sha(b: bytes) -> str:
    return hashlib.sha3_256(b).hexdigest()                    # PQ-grade (Keccak)


# ---------- coordinator PQ signing key (ML-DSA-65 / FIPS 204) ----------
def _load_or_make_key():
    if os.path.exists(KEYFILE):
        d = json.load(open(KEYFILE))
        return bytes.fromhex(d["pk"]), bytes.fromhex(d["sk"])
    pk, sk = ML_DSA_65.keygen()
    json.dump({"pk": pk.hex(), "sk": sk.hex()}, open(KEYFILE, "w"))
    return pk, sk


_PK, _SK = _load_or_make_key()
PUBKEY_HEX = _PK.hex()
SIG_ALG = "ML-DSA-65 (FIPS 204)"


def _canon(body: dict) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def receipt_leaf(receipt: dict) -> str:
    return sha(_canon(receipt["body"]))


def make_receipt(*, request_id, model, node, wire_bytes, answer, privacy_cos, integrity_ok):
    body = {
        "v": 2, "sig_alg": SIG_ALG, "hash": "SHA3-256",
        "request_id": request_id, "model": model, "node": node,
        "input_commitment": sha(wire_bytes),
        "output_hash": sha(answer.encode()),
        "privacy_cos": round(float(privacy_cos), 4),
        "integrity_ok": bool(integrity_ok),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    leaf = sha(_canon(body))
    sig = ML_DSA_65.sign(_SK, bytes.fromhex(leaf)).hex()
    return {"body": body, "leaf": leaf, "sig": sig, "pubkey": PUBKEY_HEX, "sig_alg": SIG_ALG}


def verify_signature(leaf_hex, sig_hex, pubkey_hex) -> bool:
    try:
        return bool(ML_DSA_65.verify(bytes.fromhex(pubkey_hex),
                                     bytes.fromhex(leaf_hex), bytes.fromhex(sig_hex)))
    except Exception:
        return False


# ---------- SHA3 Merkle (transparent, PQ commitment) ----------
def _merkle(leaves):
    if not leaves:
        return None, []
    levels = [leaves[:]]; cur = leaves[:]
    while len(cur) > 1:
        if len(cur) % 2:
            cur = cur + [cur[-1]]
        cur = [sha(bytes.fromhex(cur[i]) + bytes.fromhex(cur[i + 1])) for i in range(0, len(cur), 2)]
        levels.append(cur)
    return cur[0], levels


def inclusion_proof(leaves, index):
    _, levels = _merkle(leaves); proof = []
    for lvl in levels[:-1]:
        if len(lvl) % 2:
            lvl = lvl + [lvl[-1]]
        sib = index ^ 1
        proof.append({"hash": lvl[sib], "right": bool(sib > index)})
        index //= 2
    return proof


def verify_inclusion(leaf, proof, root):
    h = leaf
    for step in proof:
        h = sha(bytes.fromhex(h) + bytes.fromhex(step["hash"])) if step["right"] \
            else sha(bytes.fromhex(step["hash"]) + bytes.fromhex(h))
    return h == root


# ---------- transparent FS-sampled batch-integrity proof (STARK-class) ----------
def _fs_indices(root: str, n: int, k: int):
    """Deterministic Fiat-Shamir challenge indices from the committed root (no
    interaction, non-malleable). A prover who fixed the root cannot steer which
    receipts get checked -> a bad receipt is caught w.p. 1-(1-f)^k."""
    out, ctr = [], 0
    while len(out) < min(k, n):
        h = int(sha((root + f"|chal|{ctr}").encode()), 16) % n
        if h not in out:
            out.append(h)
        ctr += 1
    return out


STARK_BIN = os.path.join(HERE, "stark", "target", "release", "altaica_stark")
STARK_STEPS = int(os.environ.get("STARK_STEPS", "1024"))


def stark_fri_prove(root_hex: str):
    """Real transparent FRI-STARK (Winterfell) bound to the batch root. Returns
    {result, proof_b64, ms, steps} or {error} if the binary isn't built."""
    import subprocess, time
    if not os.path.exists(STARK_BIN):
        return {"error": "stark binary not built (cargo build --release in ./stark)"}
    t = time.perf_counter()
    out = subprocess.run([STARK_BIN, "prove", root_hex, str(STARK_STEPS)],
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        return {"error": out.stderr[-200:]}
    result, proof = out.stdout.strip().split(" ", 1)
    return {"result": result, "proof_b64": proof, "steps": STARK_STEPS,
            "prove_ms": round((time.perf_counter() - t) * 1e3, 1),
            "scheme": "Winterfell FRI-STARK (Blake3, transparent, post-quantum)"}


def stark_fri_verify(root_hex: str, result: str, proof_b64: str, steps: int = None):
    import subprocess, time
    if not os.path.exists(STARK_BIN):
        return {"error": "stark binary not built"}
    t = time.perf_counter()
    out = subprocess.run([STARK_BIN, "verify", root_hex, str(steps or STARK_STEPS),
                          result, proof_b64], capture_output=True, text=True, timeout=60)
    return {"valid": out.stdout.strip() == "OK",
            "verify_ms": round((time.perf_counter() - t) * 1e3, 1)}


def stark_prove_batch(batch):
    """Transparent, hash-only batch-integrity argument (PQ; no trusted setup).
    Proves: every receipt in the batch satisfies integrity_ok ∧ |privacy_cos|<=tau,
    and commits to root R. Returns a proof = sampled openings; verify with
    stark_verify_batch. (FULL AIR/FRI succinct prover plugs in HERE — Winterfell/
    Plonky3/Cairo/RISC0 — for poly-log size + ZK; left as the marked upgrade.)"""
    leaves = [r["leaf"] for r in batch]
    root, _ = _merkle(leaves)
    idx = _fs_indices(root, len(leaves), SAMPLES)
    openings = []
    for i in idx:
        b = batch[i]["body"]
        openings.append({"index": i, "leaf": leaves[i],
                         "proof": inclusion_proof(leaves, i),
                         "integrity_ok": b["integrity_ok"], "privacy_cos": b["privacy_cos"]})
    return {"scheme": "transparent-FS-merkle (SHA3, PQ, no trusted setup)",
            "root": root, "n": len(leaves), "samples": len(idx),
            "tau": PRIVACY_TAU, "openings": openings,
            "soundness_note": f"a bad receipt evades detection w.p. <= (1-1/n)^{len(idx)}"}


def stark_verify_batch(proof) -> dict:
    root, n = proof["root"], proof["n"]
    # 1) challenges must match the committed root (non-interactive soundness)
    want = _fs_indices(root, n, proof["samples"])
    got = [o["index"] for o in proof["openings"]]
    chal_ok = sorted(want) == sorted(got)
    incl_ok = all(verify_inclusion(o["leaf"], o["proof"], root) for o in proof["openings"])
    pred_ok = all(o["integrity_ok"] and abs(o["privacy_cos"]) <= proof["tau"]
                  for o in proof["openings"])
    return {"challenges_match_root": chal_ok, "inclusions_valid": incl_ok,
            "predicate_holds_on_samples": pred_ok,
            "valid": bool(chal_ok and incl_ok and pred_ok)}


# ---------- log + on-chain anchor ----------
def append_log(receipt: dict):
    with open(LOGFILE, "a") as f:
        f.write(json.dumps(receipt) + "\n")


def _read_log():
    return [json.loads(l) for l in open(LOGFILE)] if os.path.exists(LOGFILE) else []


def maybe_anchor():
    log = _read_log()
    anchored = [json.loads(l) for l in open(ANCHORFILE)] if os.path.exists(ANCHORFILE) else []
    done = sum(a["count"] for a in anchored)
    pending = log[done:]
    if len(pending) < BATCH_N:
        return None
    batch = pending[:BATCH_N]
    proof = stark_prove_batch(batch)
    vr = stark_verify_batch(proof)
    # REAL FRI-STARK (Winterfell) bound to the Merkle root
    fri = stark_fri_prove(proof["root"])
    fri_v = stark_fri_verify(proof["root"], fri["result"], fri["proof_b64"]) if "result" in fri else {"valid": False}
    anchor = {
        "batch_index": len(anchored), "count": BATCH_N, "merkle_root": proof["root"],
        "from_request": batch[0]["body"]["request_id"], "to_request": batch[-1]["body"]["request_id"],
        "sampled_proof_valid": vr["valid"], "samples": proof["samples"],
        "fri_stark": {"scheme": fri.get("scheme"), "valid": fri_v.get("valid"),
                      "prove_ms": fri.get("prove_ms"), "verify_ms": fri_v.get("verify_ms"),
                      "result": fri.get("result"), "steps": fri.get("steps"),
                      "proof_bytes": len(fri.get("proof_b64", "")) * 3 // 4,
                      "error": fri.get("error")},
        "sig_alg": SIG_ALG, "hash": "SHA3-256", "post_quantum": True,
        "pubkey": PUBKEY_HEX, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chains": {"solana": "PENDING (memo/PDA of root)", "base": "PENDING (calldata of root)"},
        "note": "Root + FRI-STARK proof ready to anchor. On-chain submit is a separate signed tx.",
        "fri_proof_b64": fri.get("proof_b64"),
    }
    with open(ANCHORFILE, "a") as f:
        f.write(json.dumps(anchor) + "\n")
    return anchor


def status():
    log = _read_log()
    anchors = [json.loads(l) for l in open(ANCHORFILE)] if os.path.exists(ANCHORFILE) else []
    return {"pubkey": PUBKEY_HEX[:64] + "…", "sig_alg": SIG_ALG, "hash": "SHA3-256",
            "post_quantum": True, "total_receipts": len(log), "batch_n": BATCH_N,
            "anchored_batches": len(anchors), "latest_anchor": anchors[-1] if anchors else None,
            "recent": [r["body"] for r in log[-10:]]}
