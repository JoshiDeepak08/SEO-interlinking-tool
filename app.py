import os
import re
import math
import json
import warnings
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from bs4 import BeautifulSoup
from readability import Document
from langdetect import detect
import chardet

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import torch
from sentence_transformers import SentenceTransformer

import streamlit as st

# ================= STREAMLIT CONFIG =================

st.set_page_config(
    page_title="Internal Link Suggestion Tool",
    layout="wide",
)

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ================= UI: INPUTS =================

st.title("🔗 Internal Link Suggestion Tool")

col1, col2 = st.columns(2)
with col1:
    domain_input = st.text_input(
        "Domain (root URL)",
        value="https://en.wikipedia.org",
        help="Example: https://www.example.com",
    )
with col2:
    test_url_input = st.text_input(
        "Test URL (page to optimize)",
        value="https://en.wikipedia.org/wiki/History_of_Wikipedia",
        help="Must be inside the same domain.",
    )

adv = st.expander("Advanced settings (optional)")
with adv:
    lang_codes = st.text_input("Languages to keep (comma-separated)", "en")
    MIN_LEN = st.number_input("Minimum characters per page", 30, 10000, 80)
    MAX_PAGES = st.number_input("Max pages to crawl", 10, 2000, 60)
    MIN_SIM_BERT = st.slider("Min SBERT similarity", 0.0, 1.0, 0.25, 0.01)
    MIN_SIM_TFIDF = st.slider("Min TF-IDF similarity", 0.0, 1.0, 0.02, 0.01)
    TARGETS_PER_SOURCE = st.number_input("Max link suggestions", 3, 100, 20)
    TOP_KEYPHRASES = st.number_input("Keyphrases from Test URL", 10, 200, 60)
    ALPHA = st.slider("TF-IDF vs BERT weight (TF-IDF share)", 0.0, 1.0, 0.4, 0.05)
    ALLOW_SUBDOMAINS = st.checkbox("Allow subdomains", True)

LANGS = {c.strip() for c in lang_codes.split(",") if c.strip()}

run_button = st.button("🚀 Generate Suggestions")

# ================= CORE HELPERS =================


def make_session():
    UA = "SEO-InternalLinker/1.0 (+streamlit)"
    s = requests.Session()
    retries = Retry(
        total=4,
        backoff_factor=0.4,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s, UA


def smart_get(session, UA, url, timeout=15):
    try:
        r = session.get(
            url,
            headers={"User-Agent": UA},
            timeout=timeout,
            allow_redirects=True,
        )
        if r.status_code == 200:
            if r.encoding is None:
                enc = chardet.detect(r.content).get("encoding") or "utf-8"
                r.encoding = enc
            return r.text, r.url
        return None, f"HTTP {r.status_code}"
    except Exception as e:
        return None, f"ERR {e}"


def clean_url(u):
    if not u:
        return None
    u = u.split("#")[0].strip()
    if u.startswith("//"):
        u = "https:" + u
    return u


def same_site(u, root, allow_subdomains=True):
    uh, rh = urlparse(u).netloc, urlparse(root).netloc
    if not uh or not rh:
        return False
    if uh == rh:
        return True
    return allow_subdomains and uh.endswith("." + rh)


def parse_sitemap_xml(session, UA, xml_text, max_sitemaps=10, urls_per_sitemap=1000):
    soup = BeautifulSoup(xml_text, "xml")
    sitemap_tags = soup.find_all("sitemap")
    urls = []

    # sitemap index
    if sitemap_tags:
        for sm in sitemap_tags[:max_sitemaps]:
            loc = sm.find("loc")
            if not loc:
                continue
            xml, _ = smart_get(session, UA, loc.get_text(strip=True))
            if not xml:
                continue
            soup2 = BeautifulSoup(xml, "xml")
            for l in soup2.find_all("loc")[:urls_per_sitemap]:
                u = clean_url(l.get_text(strip=True))
                if u:
                    urls.append(u)
        return urls

    # simple sitemap
    for l in soup.find_all("loc")[:urls_per_sitemap]:
        u = clean_url(l.get_text(strip=True))
        if u:
            urls.append(u)
    return urls


def urls_from_sitemap(session, UA, domain):
    candidates = [
        urljoin(domain, "/sitemap.xml"),
        urljoin(domain, "/sitemap_index.xml"),
        urljoin(domain, "/sitemap-index.xml"),
    ]
    all_urls = []
    for sm in candidates:
        xml, _ = smart_get(session, UA, sm)
        if xml:
            all_urls.extend(parse_sitemap_xml(session, UA, xml))
    return list(dict.fromkeys(all_urls))


def crawl_internal_links(session, UA, seed_urls, root, max_pages, allow_subdomains):
    seen = set()
    queue = [clean_url(u) for u in seed_urls if clean_url(u)]
    out = []
    progress = st.progress(0.0, text="Crawling internal links...")

    # BFS crawl with safety limits
    while queue and len(out) < max_pages:
        url = queue.pop(0)
        if not url or url in seen or not same_site(url, root, allow_subdomains):
            continue
        seen.add(url)

        html, final = smart_get(session, UA, url)
        if not html:
            continue
        final = clean_url(final) or url
        if same_site(final, root, allow_subdomains) and final not in out:
            out.append(final)

        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            nu = clean_url(urljoin(final, a["href"]))
            if (
                nu
                and same_site(nu, root, allow_subdomains)
                and nu not in seen
                and len(out) + len(queue) < max_pages * 3
            ):
                queue.append(nu)

        progress.progress(min(len(out) / float(max_pages), 1.0))

    progress.empty()
    return out


def extract_main(html):
    doc = Document(html)
    title = (doc.short_title() or "").strip()
    content_html = doc.summary() or ""
    soup = BeautifulSoup(content_html, "lxml")

    h1 = soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else ""
    headings = " ".join(h.get_text(" ", strip=True) for h in soup.find_all(["h2", "h3"]))
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()

    soup_full = BeautifulSoup(html, "lxml")
    m = soup_full.find("meta", attrs={"name": re.compile("^description$", re.I)})
    meta_desc = m["content"].strip() if (m and m.has_attr("content")) else ""

    lang = "und"
    try:
        sample = (text[:1500] or title) if (text or title) else ""
        if sample:
            lang = detect(sample)
    except Exception:
        pass

    return title, h1, headings, meta_desc, text, lang


def extract_keyphrases(tfidf, source_text, topN):
    from collections import Counter

    toks = re.findall(r"\b[a-z0-9][a-z0-9\-]{1,}\b", source_text.lower())
    bad = {"click", "here", "read", "more", "login", "signup", "copyright"}

    vocab = set(tfidf.get_feature_names_out())
    freq = Counter([t for t in toks if t in vocab and t not in bad])

    bigrams = [" ".join(p) for p in zip(toks, toks[1:])]
    for bg in bigrams:
        ws = bg.split()
        if all(w in vocab for w in ws) and bg not in bad:
            freq[bg] += 1

    return [w for w, _ in freq.most_common(topN)]


# ================= PIPELINE FUNCTIONS =================


def build_index(domain, test_url, MIN_LEN, MAX_PAGES, LANGS, ALLOW_SUBDOMAINS):
    session, UA = make_session()

    st.write(f"**Domain:** {domain}")

    sitemap_urls = urls_from_sitemap(session, UA, domain)
    if sitemap_urls:
        st.write(f"Sitemap URLs found: **{len(sitemap_urls)}**")
    else:
        st.write("No/weak sitemap detected. Falling back to link crawl.")

    seed = list(sitemap_urls)
    if test_url not in seed:
        seed.append(test_url)
    if domain not in seed:
        seed.append(domain)

    urls = crawl_internal_links(
        session,
        UA,
        seed_urls=seed,
        root=domain,
        max_pages=MAX_PAGES,
        allow_subdomains=ALLOW_SUBDOMAINS,
    )

    urls = [u for u in (clean_url(x) for x in urls) if u]
    urls = [u for u in urls if same_site(u, domain, ALLOW_SUBDOMAINS)]
    urls = list(dict.fromkeys(urls))[:MAX_PAGES]

    st.write(f"Planned URLs after crawl & filter: **{len(urls)}**")

    if len(urls) < 2:
        st.error("Need at least 2 internal URLs. Increase MAX_PAGES or check the domain.")
        return None, None, None

    rows = []
    extract_bar = st.progress(0.0, text="Extracting main content...")
    for i, u in enumerate(urls):
        html, final = smart_get(session, UA, u)
        if not html:
            continue
        final = clean_url(final) or u
        if not same_site(final, domain, ALLOW_SUBDOMAINS):
            continue

        title, h1, heads, meta_desc, text, lang = extract_main(html)
        if LANGS and lang not in LANGS:
            continue
        if len(text) < MIN_LEN:
            continue

        doc_text = " ".join([title, h1, heads, meta_desc, text]).strip()
        rows.append(
            {
                "url": final,
                "title": title,
                "h1": h1,
                "headings": heads,
                "meta_description": meta_desc,
                "text": text,
                "doc": doc_text,
            }
        )
        if len(urls) > 0:
            extract_bar.progress(min((i + 1) / float(len(urls)), 1.0))
    extract_bar.empty()

    df = pd.DataFrame(rows).drop_duplicates(subset=["url"]).reset_index(drop=True)
    st.write(f"Kept pages after filters: **{len(df)}**")

    if len(df) < 2:
        st.error("Not enough usable pages (need at least 2). Try lowering MIN_LEN.")
        return None, None, None

    return df, session, UA


def compute_models(df, LANGS):
    st.write("Building TF-IDF + SBERT models...")

    n_docs = len(df)
    use_english_stop = "en" in LANGS

    # TF-IDF settings
    if n_docs < 5:
        min_df, max_df = 1, 1.0
    else:
        min_df, max_df = 2, 0.9

    def to_count(v, n):
        return math.floor(v * n) if isinstance(v, float) else int(v)

    if to_count(max_df, n_docs) < to_count(min_df, n_docs):
        min_df, max_df = 1, 1.0

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        stop_words="english" if use_english_stop else None,
    )

    corpus = df["doc"].astype(str).tolist()
    try:
        X_tfidf = tfidf.fit_transform(corpus)
    except ValueError:
        tfidf = CountVectorizer(
            ngram_range=(1, 2),
            stop_words="english" if use_english_stop else None,
        )
        X_tfidf = tfidf.fit_transform(corpus)

    # SBERT embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        _ = model.encode(["probe"], normalize_embeddings=True)
    except Exception:
        device = "cpu"
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    st.write(f"Embedding device: **{device}**")

    docs = df["doc"].tolist()
    bs = 64 if device == "cuda" else 16
    embs = []
    emb_bar = st.progress(0.0, text="Encoding pages with SBERT...")
    total = len(docs)

    for i in range(0, total, bs):
        chunk = model.encode(
            docs[i : i + bs],
            batch_size=bs,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embs.append(chunk)
        progress_val = min((i + bs) / float(total), 1.0)
        emb_bar.progress(progress_val)

    emb_bar.empty()
    X_emb = np.vstack(embs).astype("float32")
    return tfidf, X_tfidf, model, X_emb


def get_source_vectors(df, tfidf, model, X_tfidf, X_emb, test_url, session, UA, domain, allow_subdomains):
    url_to_idx = {u: i for i, u in enumerate(df["url"])}

    if test_url in url_to_idx:
        i = url_to_idx[test_url]
        return df.loc[i, "doc"], X_tfidf[i], X_emb[i].reshape(1, -1)

    html, _ = smart_get(session, UA, test_url)
    if not html:
        raise RuntimeError("Cannot fetch Test URL")
    title, h1, heads, meta_desc, text, _ = extract_main(html)
    full = " ".join([title, h1, heads, meta_desc, text]).strip()
    x_t = tfidf.transform([full])
    x_e = model.encode([full], normalize_embeddings=True)
    return full, x_t, x_e


def suggest_links(
    df,
    tfidf,
    X_tfidf,
    model,
    X_emb,
    domain,
    test_url,
    session,
    UA,
    MIN_SIM_BERT,
    MIN_SIM_TFIDF,
    TARGETS_PER_SOURCE,
    TOP_KEYPHRASES,
    ALPHA,
    allow_subdomains,
):
    url_to_idx = {u: i for i, u in enumerate(df["url"])}

    src_text, x_t, x_e = get_source_vectors(
        df, tfidf, model, X_tfidf, X_emb, test_url, session, UA, domain, allow_subdomains
    )

    keyphrases = extract_keyphrases(tfidf, src_text, TOP_KEYPHRASES)
    if not keyphrases:
        return []

    s_tfidf = cosine_similarity(x_t, X_tfidf)[0]
    s_bert = cosine_similarity(x_e, X_emb)[0]

    if test_url in url_to_idx:
        i = url_to_idx[test_url]
        s_tfidf[i] = -1
        s_bert[i] = -1

    score = ALPHA * s_tfidf + (1 - ALPHA) * s_bert

    keep = (s_bert >= MIN_SIM_BERT) & (s_tfidf >= MIN_SIM_TFIDF)
    score = np.where(keep, score, -1)

    titleh1 = (df["title"].fillna("") + " " + df["h1"].fillna("")).tolist()
    src_lower = src_text.lower()

    kp_embs = model.encode(keyphrases, normalize_embeddings=True)
    used_targets, used_phrases, out = set(), set(), []

    for j in np.argsort(-score):
        if len(out) >= TARGETS_PER_SOURCE or score[j] <= 0:
            break

        tgt_url = df["url"][j]
        if tgt_url in used_targets:
            continue

        tgt_title = titleh1[j] or df["title"][j]
        tgt_vec = model.encode([tgt_title], normalize_embeddings=True)[0]

        sims = kp_embs @ tgt_vec
        order = np.argsort(-sims)

        chosen = None
        for idx in order:
            phrase = keyphrases[idx]
            if phrase in src_lower and phrase not in used_phrases:
                chosen = phrase
                break

        if not chosen:
            continue

        out.append(
            {
                "anchor": chosen,
                "target_page": tgt_url,
                "score": round(float(score[j]), 3),
                "sim_tfidf": round(float(s_tfidf[j]), 3),
                "sim_bert": round(float(s_bert[j]), 3),
                "reason": f'Phrase "{chosen}" in source matches "{tgt_title[:80]}"',
            }
        )
        used_targets.add(tgt_url)
        used_phrases.add(chosen)

    return out


# ================= RUN ON BUTTON =================

if run_button:
    if not domain_input or not test_url_input:
        st.error("Please fill both Domain and Test URL.")
    elif not same_site(test_url_input, domain_input, ALLOW_SUBDOMAINS):
        st.error("Test URL must belong to the same domain.")
    else:
        try:
            with st.spinner("Running interlinking analysis..."):
                df, session, UA = build_index(
                    domain_input,
                    test_url_input,
                    MIN_LEN=MIN_LEN,
                    MAX_PAGES=int(MAX_PAGES),
                    LANGS=LANGS,
                    ALLOW_SUBDOMAINS=ALLOW_SUBDOMAINS,
                )

                if df is not None:
                    tfidf, X_tfidf, model, X_emb = compute_models(df, LANGS)

                    suggestions = suggest_links(
                        df,
                        tfidf,
                        X_tfidf,
                        model,
                        X_emb,
                        domain_input,
                        test_url_input,
                        session,
                        UA,
                        MIN_SIM_BERT=float(MIN_SIM_BERT),
                        MIN_SIM_TFIDF=float(MIN_SIM_TFIDF),
                        TARGETS_PER_SOURCE=int(TARGETS_PER_SOURCE),
                        TOP_KEYPHRASES=int(TOP_KEYPHRASES),
                        ALPHA=float(ALPHA),
                        allow_subdomains=ALLOW_SUBDOMAINS,
                    )

            if suggestions:
                out_df = pd.DataFrame(suggestions)
                st.subheader("Suggested Internal Links")
                st.dataframe(out_df, use_container_width=True)

                csv = out_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name="internal_link_suggestions.csv",
                    mime="text/csv",
                )
            else:
                st.warning(
                    "No suggestions found with current settings. "
                    "Try lowering similarity thresholds or increasing MAX_PAGES."
                )
        except Exception as e:
            st.error(f"Error: {e}")
