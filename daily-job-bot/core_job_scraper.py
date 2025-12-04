import os
import sqlite3
import numpy as np
from datetime import datetime
from openai import OpenAI
from perplexity import Perplexity
import base64
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# -----------------------
# CONFIG
# -----------------------
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PERPLEXITY_API_KEY = os.environ['PERPLEXITY_API_KEY']
EMAIL_TO = os.environ.get('EMAIL_TO', 'recipient@example.com')

# Gmail API OAuth2 configuration
SCOPES = ['https://www.googleapis.com/auth/gmail.send']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'

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

def get_gmail_service():
    """Get Gmail API service with OAuth2 authentication."""
    creds = None
    # Load existing token
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    
    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                raise FileNotFoundError(f"Please download OAuth2 credentials as '{CREDENTIALS_FILE}' from Google Cloud Console")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    
    return build('gmail', 'v1', credentials=creds)

def send_email(subject, html_body):
    """Send email using Gmail API with OAuth2."""
    try:
        service = get_gmail_service()
        
        # Create message
        message = MIMEText(html_body, 'html')
        message['To'] = EMAIL_TO
        message['Subject'] = subject
        
        # Encode message
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        
        # Send email
        send_message = service.users().messages().send(
            userId='me',
            body={'raw': raw_message}
        ).execute()
        
        print(f"Email sent successfully via Gmail API. Message ID: {send_message['id']}")
        
    except HttpError as error:
        print(f"Gmail API error occurred: {error}")
        raise
    except FileNotFoundError as error:
        print(f"OAuth2 setup error: {error}")
        print("\nTo set up OAuth2:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a new project or select existing one")
        print("3. Enable Gmail API")
        print("4. Create OAuth2 credentials (Desktop application)")
        print("5. Download credentials.json to this folder")
        raise

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