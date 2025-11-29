# SEO Internal Link Suggestion Tool

A Streamlit-based utility to crawl a website, extract content from internal pages, compute TF-IDF + SBERT embeddings, and suggest **contextual internal links** (anchor text + target page) to improve site SEO and internal linking structure.  

👉 **Live demo:** https://huggingface.co/spaces/joshi-deepak08/Seo-internal-linker

---

## ⭐ Key Features

- Crawl site using sitemap.xml or internal link BFS crawling  
- Extract main content using `readability` + BeautifulSoup (title, headings, meta-description, main text)  
- Compute both TF-IDF and SBERT embeddings for content similarity  
- Suggest relevant internal link targets based on combined similarity score  
- Provide anchor-text suggestions based on top keyphrases from the source page  
- Configurable parameters: number of pages to crawl, similarity thresholds, subdomain allowance, keyphrase count, etc.  
- Export suggestions as CSV for manual review or bulk integration  

---

## 🚀 Try it online

You don’t need to clone or install anything — just visit the live deployment on :contentReference[oaicite:0]{index=0}:  
[https://huggingface.co/spaces/joshi-deepak08/Seo-internal-linker](https://huggingface.co/spaces/joshi-deepak08/Seo-internal-linker) — and start optimizing your website immediately.

---

## 📥 Setup & Local Use

If you prefer to run the tool locally or contribute:

### 1. Clone the repository

```bash
git clone https://github.com/your-username/seo-internal-linker.git
cd seo-internal-linker
