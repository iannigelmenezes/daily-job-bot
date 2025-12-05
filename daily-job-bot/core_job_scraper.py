import os
import sqlite3
import numpy as np
import requests
import json
from datetime import datetime
from openai import OpenAI
from mailjet_rest import Client

# -----------------------
# CONFIG
# -----------------------
PERPLEXITY_API_KEY = os.environ['PERPLEXITY_API_KEY']
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')  # Keep for embeddings
EMAIL_TO = os.environ.get('EMAIL_TO', 'recipient@example.com')

# Mailjet configuration
MAILJET_API_KEY = os.environ['MAILJET_API_KEY']
MAILJET_API_SECRET = os.environ['MAILJET_API_SECRET']
EMAIL_FROM = os.environ.get('EMAIL_FROM', 'noreply@yourdomain.com')
EMAIL_FROM_NAME = os.environ.get('EMAIL_FROM_NAME', 'Job Bot')

# Perplexity API configuration
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
perplexity_headers = {
    "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
    "Content-Type": "application/json"
}

# Keep OpenAI client for embeddings
client_openai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
mailjet = Client(auth=(MAILJET_API_KEY, MAILJET_API_SECRET), version='v3.1')

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
def load_prompt_from_file(file_path="prompt.txt"):
    """Load the job search prompt from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            prompt = file.read().strip()
        if not prompt:
            raise ValueError("Prompt file is empty")
        return prompt
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using default prompt.")
        return "Senior Treasury Manager Amsterdam remote"
    except Exception as e:
        print(f"Error reading prompt file: {e}. Using default prompt.")
        return "Senior Treasury Manager Amsterdam remote"

def search_jobs_perplexity(query, max_results=10):
    """Search for actual jobs using Perplexity AI's web search capabilities."""
    prompt = f"""
    Search the web for current job openings for: "{query}"
    
    Find real job postings from job boards like Indeed, LinkedIn, Glassdoor, company websites, and other job sites.
    Look for actual job listings that are currently posted and available.
    
    For each job found, extract:
    - Job title
    - Company name
    - Direct URL to the job posting
    - Brief description/snippet of requirements
    
    Return up to {max_results} actual job postings in this exact JSON format:
    {{
        "jobs": [
            {{
                "title": "Senior Treasury Manager",
                "company": "Example Corp", 
                "url": "https://www.indeed.com/viewjob?jk=abc123",
                "snippet": "Brief description of the role and key requirements from the actual job posting"
            }}
        ]
    }}
    
    Important: Only include real, current job postings with actual URLs. Do not generate fictional jobs.
    Focus specifically on: {query}
    """
    
    try:
        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a job search assistant with web access. Search for real, current job openings and return only actual job postings with real URLs. Always return valid JSON format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.3,
            "return_citations": True
        }
        
        response = requests.post(PERPLEXITY_API_URL, json=payload, headers=perplexity_headers)
        
        if response.status_code == 200:
            response_data = response.json()
            content = response_data['choices'][0]['message']['content']
            
            # Extract JSON from the response
            try:
                # Find JSON in the response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    result = json.loads(content)
                
                jobs = result.get('jobs', [])
                
                # Ensure we return the expected format
                formatted_jobs = []
                for job in jobs[:max_results]:
                    if job.get('url') and job.get('title'):  # Only include jobs with actual URLs
                        formatted_jobs.append({
                            "title": job.get('title', ''),
                            "url": job.get('url', ''),
                            "snippet": job.get('snippet', ''),
                            "company": job.get('company', '')
                        })
                
                return formatted_jobs
                
            except json.JSONDecodeError:
                print("Failed to parse JSON from Perplexity response")
                print(f"Response content: {content}")
                return []
        else:
            print(f"Perplexity API error: {response.status_code}")
            print(f"Response: {response.text}")
            return []
        
    except Exception as e:
        print(f"Error searching for jobs with Perplexity: {e}")
        return []

def embed_text(text):
    """Create embeddings for text using OpenAI or fallback to simple hash-based similarity."""
    if client_openai:
        try:
            resp = client_openai.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return np.array(resp.data[0].embedding, dtype=np.float32)
        except Exception as e:
            print(f"Error creating embedding with OpenAI: {e}")
            print("Falling back to simple text hash for similarity comparison")
    
    # Fallback: use a simple hash-based approach
    import hashlib
    text_hash = hashlib.md5(text.encode()).hexdigest()
    # Convert hash to a simple numerical representation
    return np.array([float(ord(c)) for c in text_hash[:32]], dtype=np.float32)

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
    """Summarize job listings using Perplexity AI, with OpenAI fallback."""
    prompt = "Write a concise daily email listing these job openings. For each, give a 2-line summary. Output as HTML list.\nJobs:\n"
    for j in candidates:
        prompt += f"- {j['title']} | {j.get('company','')} | {j['url']}\n{j['snippet']}\n\n"
    
    # Try Perplexity first
    try:
        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that creates professional job summary emails in HTML format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 600,
            "temperature": 0.3
        }
        
        response = requests.post(PERPLEXITY_API_URL, json=payload, headers=perplexity_headers)
        
        if response.status_code == 200:
            response_data = response.json()
            return response_data['choices'][0]['message']['content']
        else:
            print(f"Perplexity API error for summarization: {response.status_code}")
            raise Exception("Perplexity API failed")
            
    except Exception as e:
        print(f"Error using Perplexity for summarization: {e}")
        
        # Fallback to OpenAI if available
        if client_openai:
            try:
                print("Falling back to OpenAI for job summarization")
                response = client_openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that creates professional job summary emails in HTML format."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=600
                )
                return response.choices[0].message.content
            except Exception as openai_error:
                print(f"OpenAI fallback also failed: {openai_error}")
        
        # Last resort: simple HTML formatting
        html = "<ul>\n"
        for j in candidates:
            html += f"<li><strong>{j['title']}</strong> at {j.get('company', 'N/A')}<br>\n"
            html += f"<a href='{j['url']}'>{j['url']}</a><br>\n"
            html += f"{j.get('snippet', 'No description available')}</li>\n\n"
        html += "</ul>"
        return html

def send_email(subject, html_body):
    """Send email using Mailjet API."""
    try:
        data = {
            'Messages': [
                {
                    "From": {
                        "Email": EMAIL_FROM,
                        "Name": EMAIL_FROM_NAME
                    },
                    "To": [
                        {
                            "Email": EMAIL_TO
                        }
                    ],
                    "Subject": subject,
                    "HTMLPart": html_body
                }
            ]
        }
        
        result = mailjet.send.create(data=data)
        
        if result.status_code == 200:
            response_data = result.json()
            message_id = response_data['Messages'][0]['To'][0]['MessageID']
            print(f"Email sent successfully via Mailjet. Message ID: {message_id}")
        else:
            print(f"Failed to send email. Status code: {result.status_code}")
            print(f"Response: {result.json()}")
            raise Exception(f"Mailjet API error: {result.status_code}")
            
    except Exception as error:
        print(f"Error sending email via Mailjet: {error}")
        print("\nTo set up Mailjet:")
        print("1. Sign up at https://www.mailjet.com/")
        print("2. Go to Account Settings > API Keys")
        print("3. Set environment variables:")
        print("   - MAILJET_API_KEY: Your API Key")
        print("   - MAILJET_API_SECRET: Your Secret Key")
        print("   - EMAIL_FROM: Your verified sender email")
        print("   - EMAIL_FROM_NAME: Your sender name (optional)")
        raise

# -----------------------
# MAIN LOOP
# -----------------------
def main():
    query = load_prompt_from_file("daily-job-bot/prompt.txt")
    print(f"Searching for jobs with prompt: {query}")
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