from __future__ import annotations

import argparse
import csv
import ipaddress
import json
import os
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from anthropic import APIConnectionError, Anthropic
from bs4 import BeautifulSoup
from dotenv import load_dotenv


# ============================================================
# Configuration
# ============================================================

load_dotenv()

JUSTWOKER_API_KEY = os.getenv("JUSTWOKER_API_KEY")
JUSTWOKER_BASE_URL = os.getenv(
    "JUSTWOKER_BASE_URL",
    "https://api.justwoker.icu",
).rstrip("/")
JUSTWOKER_MODEL = os.getenv(
    "JUSTWOKER_MODEL",
    "claude-opus-4-8",
)
SEARXNG_URL = os.getenv(
    "SEARXNG_URL",
    "http://localhost:8080",
).rstrip("/")

_client: Anthropic | None = None

HTTP = requests.Session()
HTTP.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "FeatureScout/1.0"
        )
    }
)


# ============================================================
# Models
# ============================================================

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    engine: str = ""
    query: str = ""


@dataclass
class FeatureCandidate:
    feature_name: str
    family: str
    definition: str
    formula: str
    required_columns: list[str]
    time_window: str
    intuition: str
    evidence_type: str
    source_urls: list[str]
    leakage_risk: str
    computational_cost: str
    implementation_difficulty: str
    expected_signal: str
    notes: str = ""


# ============================================================
# JustWoker helper
# ============================================================

def get_client() -> Anthropic:
    """Create the API client lazily so imports and --help work without secrets."""
    global _client

    if _client is None:
        if not JUSTWOKER_API_KEY:
            raise RuntimeError(
                "JUSTWOKER_API_KEY is missing. Add it to .env or export it "
                "before starting a research run."
            )

        _client = Anthropic(
            api_key=JUSTWOKER_API_KEY,
            base_url=JUSTWOKER_BASE_URL,
            timeout=180.0,
            max_retries=2,
        )

    return _client

def ask_llm(
    prompt: str,
    *,
    system: str = "",
    max_tokens: int = 5000,

) -> str:
    kwargs: dict[str, Any] = {
        "model": JUSTWOKER_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }

    if system:
        kwargs["system"] = system

    try:
        response = get_client().messages.create(**kwargs)
    except APIConnectionError as exc:
        raise RuntimeError(
            "Could not connect to the configured LLM endpoint "
            f"{JUSTWOKER_BASE_URL!r}. Check DNS/network access and "
            "JUSTWOKER_BASE_URL."
        ) from exc

    text = "\n".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ).strip()

    if not text:
        returned_types = [
            getattr(block, "type", type(block).__name__)
            for block in response.content
        ]
        raise RuntimeError(
            "No final text returned. "
            f"Content block types: {returned_types}"
        )

    return text


def parse_json(text: str) -> Any:
    text = text.strip()

    # Remove markdown fences when present.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # First try exact parse.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Then recover the largest JSON array/object.
    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start >= 0 and array_end > array_start:
        try:
            return json.loads(text[array_start : array_end + 1])
        except json.JSONDecodeError:
            pass

    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start >= 0 and obj_end > obj_start:
        return json.loads(text[obj_start : obj_end + 1])

    raise ValueError("Could not parse JSON from model response")


# ============================================================
# SearXNG
# ============================================================

def searx_search(
    query: str,
    *,
    page: int = 1,
    max_results: int = 12,
) -> list[SearchResult]:
    response = HTTP.get(
        f"{SEARXNG_URL}/search",
        params={
            "q": query,
            "format": "json",
            "language": "en",
            "pageno": page,
            "safesearch": 0,
        },
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    results: list[SearchResult] = []

    for item in data.get("results", [])[:max_results]:
        url = item.get("url") or ""
        if not url:
            continue

        results.append(
            SearchResult(
                title=item.get("title") or "",
                url=url,
                snippet=item.get("content") or "",
                engine=item.get("engine") or "",
                query=query,
            )
        )

    return results


# ============================================================
# Safe public URL reader
# ============================================================

def _host_is_public(hostname: str) -> bool:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
        ):
            return False
    return True


def fetch_url(url: str, max_chars: int = 18000) -> str:
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""

    if not _host_is_public(parsed.hostname):
        return ""

    # Convert normal GitHub file pages to raw text when possible.
    if parsed.hostname == "github.com" and "/blob/" in parsed.path:
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 5:
            owner, repo = parts[0], parts[1]
            branch = parts[3]
            file_path = "/".join(parts[4:])
            url = (
                "https://raw.githubusercontent.com/"
                f"{owner}/{repo}/{branch}/{file_path}"
            )

    try:
        # Validate every redirect target to prevent redirects to private hosts.
        for _ in range(6):
            parsed = urlparse(url)
            if not parsed.hostname or not _host_is_public(parsed.hostname):
                return ""

            r = HTTP.get(url, timeout=25, allow_redirects=False)
            if not r.is_redirect:
                r.raise_for_status()
                break

            location = r.headers.get("location")
            if not location:
                return ""
            url = urljoin(url, location)
        else:
            return ""
    except requests.RequestException:
        return ""

    ctype = (r.headers.get("content-type") or "").lower()

    if "application/pdf" in ctype:
        return ""

    if (
        "text/plain" in ctype
        or "application/json" in ctype
        or "text/markdown" in ctype
    ):
        return r.text[:max_chars]

    if "html" not in ctype and not ctype.startswith("text/"):
        return ""

    soup = BeautifulSoup(r.text, "html.parser")

    for tag in soup(
        ["script", "style", "noscript", "svg", "nav", "footer", "header"]
    ):
        tag.decompose()

    text = "\n".join(
        line.strip()
        for line in soup.get_text("\n").splitlines()
        if line.strip()
    )

    return text[:max_chars]


# ============================================================
# Research planning
# ============================================================

QUERY_SYSTEM = """
You are a senior ML feature-research planner.

Your only job is to produce diverse web-search queries that can uncover
feature-engineering ideas for the supplied prediction problem.

Search beyond the exact problem wording. Include adjacent tasks, industries,
competition solutions, GitHub implementations, Kaggle notebooks, papers,
survival analysis, sequence modelling, retention, repeat-purchase prediction,
CLV, recommender systems, behavioral modelling, and anomaly/change detection.

Do not return explanations. Return JSON only.
"""


def generate_queries(
    problem: str,
    existing_features: list[FeatureCandidate],
    round_no: int,
    n_queries: int = 14,
) -> list[str]:
    existing_names = [f.feature_name for f in existing_features[-120:]]

    prompt = f"""
RESEARCH ROUND: {round_no}

PROBLEM:
{problem}

FEATURES ALREADY FOUND:
{json.dumps(existing_names, ensure_ascii=False)}

Generate {n_queries} NEW search queries.

Requirements:
- do not simply repeat previous feature names
- cover different feature families
- include Kaggle/competition/GitHub/paper queries
- include adjacent prediction problems
- include advanced concepts such as entropy, concentration, transitions,
  change points, personal baselines, volatility, acceleration, sequence
  irregularity, lifecycle and cohort-relative behavior
- query strings should be useful directly in a web search engine

Return exactly:
[
  "query 1",
  "query 2"
]
"""

    raw = ask_llm(
        prompt,
        system=QUERY_SYSTEM,
        max_tokens=1800,
    )
    data = parse_json(raw)

    queries = []
    for q in data:
        if isinstance(q, str) and q.strip():
            queries.append(q.strip())

    # Deterministic source-oriented additions each round.
    seed = [
        "ecommerce churn feature engineering Kaggle competition",
        "repeat purchase prediction feature engineering GitHub",
        "customer retention feature engineering winning solution",
        "customer behavior entropy features churn",
        "purchase sequence features churn prediction",
        "survival analysis customer churn covariates ecommerce",
        "customer behavioral drift features retention",
        "cart abandonment trend features churn",
    ]

    if round_no == 1:
        queries.extend(seed)

    # Preserve order, remove duplicates.
    seen = set()
    out = []
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            out.append(q)

    return out[: max(n_queries, 12)]


# ============================================================
# Search + reading
# ============================================================

def collect_search_results(
    queries: list[str],
    pages: int = 2,
    per_query: int = 10,
) -> list[SearchResult]:
    jobs = []

    with ThreadPoolExecutor(max_workers=8) as ex:
        for query in queries:
            for page in range(1, pages + 1):
                jobs.append(
                    ex.submit(
                        searx_search,
                        query,
                        page=page,
                        max_results=per_query,
                    )
                )

        results: list[SearchResult] = []
        errors: list[str] = []

        for job in as_completed(jobs):
            try:
                results.extend(job.result())
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")

    if errors and not results:
        raise RuntimeError(
            "All SearXNG searches failed. Check SEARXNG_URL and confirm "
            "JSON output is enabled. First error: " + errors[0]
        )
    if errors:
        print(f"Warning: {len(errors)} SearXNG requests failed.")

    # Deduplicate by URL.
    unique: dict[str, SearchResult] = {}
    for item in results:
        unique.setdefault(item.url, item)

    return list(unique.values())


def rank_sources(
    results: list[SearchResult],
    limit: int = 35,
) -> list[SearchResult]:
    preferred = (
        "kaggle.com",
        "github.com",
        "arxiv.org",
        "medium.com",
        "towardsdatascience.com",
        "paperswithcode.com",
    )

    def score(r: SearchResult) -> tuple[int, int]:
        url = r.url.lower()
        source_score = 1 if any(d in url for d in preferred) else 0
        snippet_score = min(len(r.snippet), 500)
        return source_score, snippet_score

    return sorted(results, key=score, reverse=True)[:limit]


def read_sources(
    results: list[SearchResult],
    limit: int = 25,
) -> list[dict[str, str]]:
    selected = rank_sources(results, limit=limit)
    source_docs: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=6) as ex:
        future_to_result = {
            ex.submit(fetch_url, r.url): r
            for r in selected
        }

        for future in as_completed(future_to_result):
            r = future_to_result[future]
            try:
                body = future.result()
            except Exception:
                body = ""

            # Keep snippet even if page fetching is blocked.
            text = body if body else r.snippet

            if not text:
                continue

            source_docs.append(
                {
                    "title": r.title,
                    "url": r.url,
                    "query": r.query,
                    "text": text,
                    "evidence_level": (
                        "FULL_PAGE" if body else "SNIPPET_ONLY"
                    ),
                }
            )

    return source_docs


# ============================================================
# Feature extraction
# ============================================================

EXTRACT_SYSTEM = """
You are a senior data scientist extracting feature-engineering ideas from
retrieved technical evidence.

Rules:
1. Feature definitions must be implementable.
2. Never use information after the prediction timestamp.
3. Distinguish:
   FOUND   = directly supported by retrieved evidence.
   ADAPTED = a sourced technique adapted to this problem.
   NOVEL   = your own combination/extension inspired by evidence.
4. Do not call something FOUND if the evidence does not show it.
5. Avoid trivial renamings and semantic duplicates.
6. Prefer behavioral, temporal, sequence, trend, volatility, ratio,
   interaction, entropy, diversity, lifecycle and baseline-deviation features.
7. Do not return model changes; return features.
8. Return JSON only.
"""


def _source_packet(
    docs: list[dict[str, str]],
    max_docs: int = 12,
    max_chars_per_doc: int = 6500,
) -> str:
    chunks = []

    for i, d in enumerate(docs[:max_docs], 1):
        chunks.append(
            f"""
SOURCE {i}
TITLE: {d['title']}
URL: {d['url']}
EVIDENCE: {d['evidence_level']}
QUERY: {d['query']}
TEXT:
{d['text'][:max_chars_per_doc]}
"""
        )

    return "\n".join(chunks)


def extract_features_from_sources(
    problem: str,
    docs: list[dict[str, str]],
    existing_features: list[FeatureCandidate],
) -> list[FeatureCandidate]:
    if not docs:
        return []

    existing_names = [f.feature_name for f in existing_features[-150:]]
    packet = _source_packet(docs)

    prompt = f"""
PROBLEM:
{problem}

EXISTING FEATURE NAMES:
{json.dumps(existing_names, ensure_ascii=False)}

RETRIEVED SOURCES:
{packet}

Extract genuinely useful feature candidates.

Return a JSON array. Every object MUST use exactly these keys:

{{
  "feature_name": "snake_case_name",
  "family": "temporal|behavioral|sequence|pricing|diversity|etc",
  "definition": "precise implementable definition",
  "formula": "formula or pseudocode",
  "required_columns": ["col1", "col2"],
  "time_window": "for example 30d, 90d, lifetime-to-date",
  "intuition": "why it may predict the target",
  "evidence_type": "FOUND|ADAPTED|NOVEL|SNIPPET_ONLY",
  "source_urls": ["https://..."],
  "leakage_risk": "LOW|MEDIUM|HIGH plus one short reason",
  "computational_cost": "LOW|MEDIUM|HIGH",
  "implementation_difficulty": "LOW|MEDIUM|HIGH",
  "expected_signal": "short explanation",
  "notes": "optional"
}}

Only include source URLs that occur in the supplied evidence.
Do not repeat obvious basic RFM unless the feature is a meaningful extension.
"""

    raw = ask_llm(
        prompt,
        system=EXTRACT_SYSTEM,
        max_tokens=6500,
    )
    data = parse_json(raw)

    out: list[FeatureCandidate] = []

    allowed_urls = {d["url"] for d in docs}

    for item in data:
        if not isinstance(item, dict):
            continue

        urls = [
            u for u in item.get("source_urls", [])
            if u in allowed_urls
        ]

        evidence_type = str(
            item.get("evidence_type", "NOVEL")
        ).upper()

        # A sourced label without a retained source is not allowed.
        if evidence_type in {"FOUND", "ADAPTED", "SNIPPET_ONLY"} and not urls:
            evidence_type = "NOVEL"

        try:
            out.append(
                FeatureCandidate(
                    feature_name=str(item["feature_name"]).strip(),
                    family=str(item.get("family", "other")).strip(),
                    definition=str(item.get("definition", "")).strip(),
                    formula=str(item.get("formula", "")).strip(),
                    required_columns=[
                        str(x) for x in item.get("required_columns", [])
                    ],
                    time_window=str(item.get("time_window", "")).strip(),
                    intuition=str(item.get("intuition", "")).strip(),
                    evidence_type=evidence_type,
                    source_urls=urls,
                    leakage_risk=str(
                        item.get("leakage_risk", "UNKNOWN")
                    ).strip(),
                    computational_cost=str(
                        item.get("computational_cost", "UNKNOWN")
                    ).strip(),
                    implementation_difficulty=str(
                        item.get("implementation_difficulty", "UNKNOWN")
                    ).strip(),
                    expected_signal=str(
                        item.get("expected_signal", "")
                    ).strip(),
                    notes=str(item.get("notes", "")).strip(),
                )
            )
        except KeyError:
            continue

    return out


# ============================================================
# Deduplication + gap analysis
# ============================================================

def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def cheap_deduplicate(
    features: list[FeatureCandidate],
) -> list[FeatureCandidate]:
    seen: set[str] = set()
    out: list[FeatureCandidate] = []

    for feature in features:
        key = normalize_name(feature.feature_name)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(feature)

    return out


CRITIC_SYSTEM = """
You are a strict feature-engineering research critic.
Find semantic duplicates, target leakage, unsupported claims, weak definitions,
and under-explored feature families. Return JSON only.
"""


def critique_feature_bank(
    problem: str,
    features: list[FeatureCandidate],
) -> dict[str, Any]:
    compact = [
        {
            "feature_name": f.feature_name,
            "family": f.family,
            "definition": f.definition,
            "evidence_type": f.evidence_type,
            "leakage_risk": f.leakage_risk,
        }
        for f in features[-120:]
    ]

    prompt = f"""
PROBLEM:
{problem}

CURRENT FEATURE BANK:
{json.dumps(compact, ensure_ascii=False)}

Return:
{{
  "duplicate_groups": [
    ["feature_a", "feature_b"]
  ],
  "high_leakage_features": [
    "feature_name"
  ],
  "weak_features": [
    "feature_name"
  ],
  "underexplored_families": [
    "family / direction"
  ],
  "next_search_queries": [
    "query"
  ]
}}

Be strict. The next_search_queries must target genuinely missing directions.
"""

    raw = ask_llm(
        prompt,
        system=CRITIC_SYSTEM,
        max_tokens=3500,
    )
    return parse_json(raw)


def remove_semantic_duplicates(
    features: list[FeatureCandidate],
    duplicate_groups: list[list[str]],
) -> list[FeatureCandidate]:
    drop: set[str] = set()

    by_name = {
        normalize_name(f.feature_name): f
        for f in features
    }

    for group in duplicate_groups:
        normalized = [
            normalize_name(x)
            for x in group
            if normalize_name(x) in by_name
        ]

        # Keep the first one, drop the rest.
        for name in normalized[1:]:
            drop.add(name)

    return [
        f for f in features
        if normalize_name(f.feature_name) not in drop
    ]


# ============================================================
# Persistence / export
# ============================================================

def save_feature_bank(
    features: list[FeatureCandidate],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "feature_bank.json"
    csv_path = out_dir / "feature_bank.csv"

    json_path.write_text(
        json.dumps(
            [asdict(f) for f in features],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = list(asdict(features[0]).keys()) if features else [
            "feature_name",
            "family",
            "definition",
            "formula",
            "required_columns",
            "time_window",
            "intuition",
            "evidence_type",
            "source_urls",
            "leakage_risk",
            "computational_cost",
            "implementation_difficulty",
            "expected_signal",
            "notes",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for feature in features:
            row = asdict(feature)
            row["required_columns"] = json.dumps(
                row["required_columns"],
                ensure_ascii=False,
            )
            row["source_urls"] = json.dumps(
                row["source_urls"],
                ensure_ascii=False,
            )
            writer.writerow(row)


# ============================================================
# Deep research loop
# ============================================================

def deep_feature_research(
    problem: str,
    *,
    max_rounds: int = 6,
    target_features: int = 100,
    min_new_features: int = 5,
    stop_after_low_novelty_rounds: int = 2,
    output_dir: str = "feature_research_output",
) -> list[FeatureCandidate]:

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_bank: list[FeatureCandidate] = []
    low_novelty_rounds = 0
    critic_queries: list[str] = []

    for round_no in range(1, max_rounds + 1):

        print(f"\n{'=' * 70}")
        print(f"RESEARCH ROUND {round_no}")
        print(f"Current feature count: {len(feature_bank)}")
        print(f"{'=' * 70}")

        queries = generate_queries(
            problem,
            feature_bank,
            round_no,
        )

        # Add gap-directed queries from the previous critic.
        queries = list(
            dict.fromkeys(
                queries + critic_queries
            )
        )[:20]

        print(f"Search queries: {len(queries)}")

        results = collect_search_results(
            queries,
            pages=2 if round_no <= 3 else 3,
            per_query=10,
        )

        print(f"Unique search results: {len(results)}")

        docs = read_sources(
            results,
            limit=30,
        )

        print(f"Readable evidence sources: {len(docs)}")

        # Extract in batches to avoid massive prompts.
        new_features: list[FeatureCandidate] = []

        batch_size = 10

        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]

            try:
                extracted = extract_features_from_sources(
                    problem,
                    batch,
                    feature_bank + new_features,
                )
                new_features.extend(extracted)
            except Exception as exc:
                print(
                    f"Extraction batch failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        before = len(feature_bank)

        feature_bank = cheap_deduplicate(
            feature_bank + new_features
        )

        raw_added = len(feature_bank) - before

        print(f"New unique-name features: {raw_added}")

        critic_queries = []

        if feature_bank:
            try:
                critique = critique_feature_bank(
                    problem,
                    feature_bank,
                )

                feature_bank = remove_semantic_duplicates(
                    feature_bank,
                    critique.get(
                        "duplicate_groups",
                        [],
                    ),
                )

                critic_queries = [
                    str(q)
                    for q in critique.get(
                        "next_search_queries",
                        [],
                    )
                    if str(q).strip()
                ][:10]

                (out_dir / f"critique_round_{round_no}.json").write_text(
                    json.dumps(
                        critique,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

            except Exception as exc:
                print(
                    f"Critic failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        after_critic = len(feature_bank)

        print(
            f"Feature bank after critic: "
            f"{after_critic}"
        )

        save_feature_bank(
            feature_bank,
            out_dir,
        )

        # Save search evidence for auditability.
        (out_dir / f"sources_round_{round_no}.json").write_text(
            json.dumps(
                docs,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        genuinely_added = max(
            0,
            after_critic - before,
        )

        if genuinely_added < min_new_features:
            low_novelty_rounds += 1
        else:
            low_novelty_rounds = 0

        print(
            f"Genuinely added this round: "
            f"{genuinely_added}"
        )

        if (
            len(feature_bank) >= target_features
            and low_novelty_rounds >= 1
        ):
            print(
                "\nStopping: target reached and novelty is slowing."
            )
            break

        if (
            low_novelty_rounds
            >= stop_after_low_novelty_rounds
        ):
            print(
                "\nStopping: novelty stayed low for "
                f"{low_novelty_rounds} consecutive rounds."
            )
            break

    print(
        f"\nFINAL FEATURE COUNT: "
        f"{len(feature_bank)}"
    )

    return feature_bank


# ============================================================
# Example
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research evidence-backed feature candidates with an "
            "Anthropic-compatible model and SearXNG."
        )
    )
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--target-features", type=int, default=100)
    parser.add_argument("--min-new-features", type=int, default=5)
    parser.add_argument("--stop-after-low-novelty-rounds", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        default="feature_research_output",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate local configuration without making network/API calls.",
    )
    args = parser.parse_args()

    for name in (
        "max_rounds",
        "target_features",
        "min_new_features",
        "stop_after_low_novelty_rounds",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")

    return args


def check_config() -> None:
    if not JUSTWOKER_API_KEY:
        raise RuntimeError("JUSTWOKER_API_KEY is missing")
    if urlparse(JUSTWOKER_BASE_URL).scheme not in {"http", "https"}:
        raise RuntimeError("JUSTWOKER_BASE_URL must be an HTTP(S) URL")
    if urlparse(SEARXNG_URL).scheme not in {"http", "https"}:
        raise RuntimeError("SEARXNG_URL must be an HTTP(S) URL")

    print("Configuration is valid.")
    print(f"Model: {JUSTWOKER_MODEL}")
    print(f"LLM endpoint: {JUSTWOKER_BASE_URL}")
    print(f"SearXNG endpoint: {SEARXNG_URL}")


if __name__ == "__main__":
    args = parse_args()

    if args.check_config:
        check_config()
        raise SystemExit(0)

    PROBLEM = """
Predict ecommerce customer churn.

ENTITY
customer_id

TARGET
A customer churns if they make no purchase in the 60 days
after the prediction timestamp.

LEAKAGE BOUNDARY
Every feature must use only information available on or before
the prediction timestamp.

AVAILABLE TRANSACTION DATA
- customer_id
- order_id
- order_timestamp
- product_id
- category_id
- quantity
- unit_price
- discount
- payment_method

AVAILABLE SESSION DATA
- customer_id
- session_id
- session_timestamp
- page_views
- product_views
- category_views
- cart_additions
- checkout_started
- purchase_completed
- device_type
- traffic_source

RESEARCH GOAL
Discover a large, diverse, evidence-backed feature space.

Do not stop at basic RFM.

Search across:
- Kaggle
- GitHub
- competition writeups
- papers
- technical blogs
- ecommerce retention
- repeat purchase prediction
- CLV
- survival analysis
- recommender systems
- SaaS / telecom / gaming churn
- behavioral modelling

Prioritize:
- temporal change
- inter-purchase intervals
- acceleration/deceleration
- volatility/stability
- sequence behavior
- transition behavior
- cart/funnel deterioration
- product/category diversity
- entropy/concentration
- discount dependency
- price sensitivity
- customer-specific baselines
- lifecycle position
- behavioral drift
- cohort-relative behavior
- cross-window ratios
- interaction features
"""

    features = deep_feature_research(
        PROBLEM,
        max_rounds=args.max_rounds,
        target_features=args.target_features,
        min_new_features=args.min_new_features,
        stop_after_low_novelty_rounds=args.stop_after_low_novelty_rounds,
        output_dir=args.output_dir,
    )

    for feature in features[:20]:
        print(
            feature.feature_name,
            "|",
            feature.family,
            "|",
            feature.evidence_type,
        )
