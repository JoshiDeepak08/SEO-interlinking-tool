#  **SEO Internal Link Suggestion Tool**

An automated **SEO internal linking assistant** that crawls a domain, extracts readable content, computes **TF-IDF + SBERT hybrid similarity**, and recommends high-quality internal links with suggested anchor texts. Runs as an interactive **Streamlit** app with progress feedback, adjustable thresholds, and CSV export.

**Live Demo:** [https://huggingface.co/spaces/joshi-deepak08/Seo-internal-linker](https://huggingface.co/spaces/joshi-deepak08/Seo-internal-linker)

<img width="1313" height="510" alt="image" src="https://github.com/user-attachments/assets/e05fabe7-ba82-4a69-ac9d-133aba9cab41" />

---

##  Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Folder Structure](#folder-structure)
4. [How to Run Locally](#how-to-run-locally)
5. [Architecture & Design Decisions](#architecture--design-decisions)
6. [Approach](#approach)
7. [Pipeline Design](#pipeline-design)
8. [Challenges & Trade-Offs](#challenges--trade-offs)

---

##  Overview

This project is an **Internal Link Suggestion Tool** for SEO teams, content writers, and technical SEOs.

Given:

* a **Domain (root URL)** – e.g. `https://www.example.com`
* a **Test URL** on that same domain – e.g. a blog post you want to optimize

…the tool:

1. Crawls the domain (using sitemap + internal links)
2. Extracts **main content** from each page using readability heuristics
3. Filters pages by language & content length
4. Builds **TF-IDF** and **SBERT** (SentenceTransformer MiniLM) representations
5. Computes a **hybrid similarity score** between the Test URL and all other pages
6. Extracts keyphrases from the Test URL and chooses **anchor → target** pairs
7. Outputs a ranked table of **internal link suggestions** with:

   * suggested anchor phrase
   * target URL
   * similarity scores (TF-IDF, SBERT, combined)
   * explanation / reason

All of this is wrapped in a **Streamlit dashboard** with interactive controls and CSV download.

---

##  Features

*  **Full-domain crawling** via sitemap + BFS internal link discovery
*  **Dual similarity engine**: TF-IDF (lexical) + SBERT (semantic)
*  **Readability-based content extraction** using `readability` + BeautifulSoup
*  **Language filtering** using `langdetect`
*  **Configurable thresholds & limits**:

  * Max pages to crawl
  * Minimum content length
  * Min TF-IDF & SBERT similarity
  * Number of suggestions per source page
*  **Keyphrase-based anchor selection** from the Test URL
*  **Explainable output**:

  * anchor text
  * target URL
  * similarity scores
  * reasoning text
* **CSV export** of internal link suggestions
* **Progress bars** for crawling, extraction, and embedding steps
* **Hugging Face Space deployment** (Dockerized environment)

---

## Folder Structure

```text
SEO-interlinking-tool/
│
├── app.py / main.py        # Streamlit app with full pipeline (UI + logic)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── (optionally) assets/    # Screenshots such as seo_interlinking.png
```

> In your Hugging Face Space, `app.py` is the Streamlit entrypoint.
> The code shown above is the main app logic.

---

## How to Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/JoshiDeepak08/SEO-interlinking-tool.git
cd SEO-interlinking-tool
```

### 2️⃣ Create & activate a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
# or
venv\Scripts\activate         # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit app

If your main file is `app.py`:

```bash
streamlit run app.py
```

(If the main file is named `main.py`, just change the command accordingly.)

The app will start at:

```text
http://localhost:8501
```

---

## Architecture & Design Decisions

Core components:

* **Streamlit** for UI → quick iteration, sliders, expanders, progress bars, CSV download
* **Requests + BeautifulSoup + Readability** for robust web crawling & content extraction
* **langdetect** to filter pages by language
* **sklearn TfidfVectorizer / CountVectorizer** for TF-IDF representations
* **SentenceTransformers / SBERT (`all-MiniLM-L6-v2`)** for semantic embeddings
* **Hybrid scoring function**: `score = α * TFIDF + (1 - α) * SBERT`
* **NumPy + pandas** for similarity computation and tabular output
* **Retrying HTTP Session** for resilient crawling (with backoff & error handling)

### Why TF-IDF + SBERT (Hybrid)?

* **TF-IDF** captures exact keyword matches and frequency
* **SBERT** captures semantic similarity beyond exact words
* Combining them via a tunable `ALPHA` parameter gives you:

  * control over “strict keyword” vs “semantic looseness”
  * better SEO relevance vs purely semantic matching

### Why Readability Extraction?

Instead of naive raw HTML scraping, content is cleaned to:

* reduce navigation/boilerplate noise
* focus on body text, headings, and meaningful content
* improve quality of embeddings and similarity scores

---

## Approach

1. **Input**: user provides:

   * Domain (root URL)
   * Test URL (page to optimize)
2. **Crawling strategy**:

   * Try **sitemap first** (`/sitemap.xml`, `/sitemap_index.xml`, etc.)
   * If weak or missing, fallback to **BFS crawling** via `<a>` tags
3. **Filtering**:

   * Keep only same-site URLs (optionally allow subdomains)
   * Filter by **language** (e.g., only `en`)
   * Filter by **minimum content length**
4. **Model building**:

   * Build TF-IDF / CountVectorizer on cleaned docs
   * Encode documents with SBERT (`all-MiniLM-L6-v2`)
5. **Source page representation**:

   * Use Test URL’s full extracted text
   * Compute its TF-IDF vector + SBERT embedding
6. **Keyphrase extraction**:

   * Extract token n-grams from Test URL that appear in TF-IDF vocab
   * Rank by frequency and filter out stopwords/boring phrases
7. **Similarity scoring + anchor selection**:

   * Compute TF-IDF cosine similarity + SBERT cosine similarity
   * Combine into hybrid score
   * Remove self page (Test URL) from candidates
   * Enforce similarity thresholds
   * Pick the best matching anchor phrase for each candidate target
8. **Output**:

   * Produce ranked list with anchors, target URLs, and explanation
   * Display in Streamlit data frame + allow CSV download

---

## Pipeline Design

### High-level flow

```mermaid
flowchart TD

A[User Inputs<br>Domain + Test URL + Settings] --> B[HTTP Session<br>+ Retry Logic]

B --> C[Crawl Domain<br>Sitemaps + Internal Links]
C --> D[Fetch HTML Pages]

D --> E[Readability Extraction<br>Title + H1 + Headings + Meta + Body Text]
E --> F[Language Filter + Length Filter]

F --> G[Document Corpus<br>(pandas DataFrame)]

G --> H[TF-IDF / Count Vectorizer]
G --> I[SBERT Embeddings]

H --> J[TF-IDF Matrix X_tfidf]
I --> K[Embedding Matrix X_emb]

J --> L[Similarity: TF-IDF cos]
K --> M[Similarity: SBERT cos]

L --> N[Hybrid Scoring using ALPHA]
M --> N

N --> O[Keyphrase Extraction from Test URL]
O --> P[Anchor Selection per Target Page]

P --> Q[Final Suggestions Table<br>(anchor, target, scores, reason)]

Q --> R[Streamlit UI: Table + CSV Download]

```

---

## Challenges & Trade-Offs

### 1. **Crawling vs Sitemap Quality**

* Some sites have perfect sitemaps → fast discovery
* Others need BFS crawling → slower & noisier
  ➡ Trade-off: **Hybrid approach** (try sitemaps first, fallback to crawl).

### 2. **Language & Content Filtering**

* Very aggressive filtering might discard useful pages
* Too loose filtering includes thin/irrelevant content
  ➡ Exposed **MIN_LEN** and **language whitelist** as UI settings.

### 3. **Similarity Threshold Tuning**

* High thresholds → only very strong matches; risk of no suggestions
* Low thresholds → many weak / irrelevant links
  ➡ User-controllable sliders for:

  * `MIN_SIM_TFIDF`
  * `MIN_SIM_BERT`
  * `ALPHA` (TF-IDF vs BERT weight)

### 4. **Model Performance vs Resource Usage**

* SBERT embeddings can be heavy on CPU
* GPU (if available) gives big speed boost
  ➡ Dynamic device detection (`cuda` if available, else `cpu`) and batch processing.

### 5. **Anchor Uniqueness**

* Same phrase should not be used for many URLs
  ➡ Tool tracks used anchors & targets to maintain diversity.

---

