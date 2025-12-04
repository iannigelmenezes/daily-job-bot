import os
import sqlite3
import numpy as np
from datetime import datetime
from openai import OpenAI
from perplexityai import Perplexity

# -----------------------
# CONFIG
# -----------------------
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PERPLEXITY_API_KEY = os.environ['PERPLEXITY_API_KEY']
SENDGRID_API_KEY = os.environ['SENDGRID_API_KEY']
EMAIL_TO = "you@example.com"
EMAIL_FROM = "jobs-bot@example.com"

client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_perplexity = Perplexity(api_key=PERPLEXITY_API_KEY)

# -----------------------
# SQLITE DB SETUP
# -----------------------
conn = sqlite3.connect("jobs.db")
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    title TEXT,
    company TEXT,
    url TEXT,
    embedding BLOB,
    seen INTEGER DEFAULT 0
)
""")
conn.commit()

# -----------------------
# HELPER FUNCTIONS
# -----------------------
def search_jobs_perplexity(query, max_results=10):
    resp = client_perplexity.search.create(query=query, max_results=max_results)
    jobs = []
    for r in resp.results:
        jobs.append({
            "title": r.title,
            "url": r.url,
            "snippet": r.snippet,
            "company": "",  # optional, can try to parse from snippet
        })
    return jobs

def embed_text(text):
    resp = client_openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))

def is_similar_already(embedding, threshold=0.85):
    c.execute("SELECT embedding FROM jobs")
    for row in c.fetchall():
        stored_vec = np.frombuffer(row[0], dtype=np.float32)
        if cosine_similarity(stored_vec, embedding) >= threshold:
            return True
    return False

def store_job(job_id, title, company, url, embedding):
    c.execute(
        "INSERT OR IGNORE INTO jobs (job_id, title, company, url, embedding, seen) VALUES (?, ?, ?, ?, ?, 0)",
        (job_id, title, company, url, embedding.tobytes())
    )
    conn.commit()

def summarize_jobs(candidates):
    prompt = "Write a concise daily email listing these job openings. For each, give a 2-line summary. Output as HTML list.\nJobs:\n"
    for j in candidates:
        prompt += f"- {j['title']} | {j.get('company','')} | {j['url']}\n{j['snippet']}\n\n"
    resp = client_openai.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        max_output_tokens=600
    )
    return resp.output_text

def send_email(subject, html_body):
    import requests
    url = "https://api.sendgrid.com/v3/mail/send"
    headers = {
        "Authorization": f"Bearer {SENDGRID_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "personalizations": [{"to": [{"email": EMAIL_TO}], "subject": subject}],
        "from": {"email": EMAIL_FROM, "name": "Jobs Bot"},
        "content": [{"type": "text/html", "value": html_body}]
    }
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    print("Email sent successfully.")

# -----------------------
# MAIN LOOP
# -----------------------
def main():
    query = "Senior Treasury Manager Amsterdam remote"
    raw_jobs = search_jobs_perplexity(query)
    candidates = []
    for job in raw_jobs:
        text = f"{job['title']} {job.get('company','')} {job.get('snippet','')}"
        emb = embed_text(text)
        job_id = job['url']
        if not is_similar_already(emb):
            store_job(job_id, job['title'], job.get('company',''), job['url'], emb)
            candidates.append(job)
    if not candidates:
        print("No new jobs today.")
        return
    email_body = summarize_jobs(candidates)
    send_email("Your Daily Job Matches", email_body)

if __name__ == "__main__":
    main()