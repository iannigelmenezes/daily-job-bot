import sqlite3
import numpy as np
from datetime import datetime

def connect_db():
    """Connect to the jobs database."""
    return sqlite3.connect("jobs.db")

def view_all_jobs():
    """Display all jobs in the database."""
    conn = connect_db()
    c = conn.cursor()
    
    c.execute("SELECT job_id, title, company, url, seen FROM jobs ORDER BY rowid")
    jobs = c.fetchall()
    
    if not jobs:
        print("📭 Database is empty - no jobs found.")
        conn.close()
        return
    
    print(f"📊 Found {len(jobs)} jobs in database:")
    print("=" * 100)
    print(f"{'#':<3} {'TITLE':<40} {'COMPANY':<20} {'SEEN':<6} {'URL':<30}")
    print("=" * 100)
    
    for i, (job_id, title, company, url, seen) in enumerate(jobs, 1):
        # Truncate long text for display
        title_display = title[:37] + "..." if len(title) > 40 else title
        company_display = company[:17] + "..." if len(company) > 20 else company
        url_display = url[:27] + "..." if len(url) > 30 else url
        seen_display = "✅ Yes" if seen else "❌ No"
        
        print(f"{i:<3} {title_display:<40} {company_display:<20} {seen_display:<6} {url_display:<30}")
    
    conn.close()

def view_job_details(job_number):
    """Display detailed information for a specific job."""
    conn = connect_db()
    c = conn.cursor()
    
    c.execute("SELECT job_id, title, company, url, embedding, seen FROM jobs ORDER BY rowid LIMIT 1 OFFSET ?", (job_number - 1,))
    job = c.fetchone()
    
    if not job:
        print(f"❌ Job #{job_number} not found.")
        conn.close()
        return
    
    job_id, title, company, url, embedding_blob, seen = job
    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
    
    print(f"\n📋 Job #{job_number} Details:")
    print("=" * 60)
    print(f"Title: {title}")
    print(f"Company: {company}")
    print(f"URL: {url}")
    print(f"Job ID: {job_id}")
    print(f"Seen: {'Yes' if seen else 'No'}")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"Embedding sample: [{', '.join(f'{x:.4f}' for x in embedding[:5])}...]")
    print("=" * 60)
    
    conn.close()

def count_jobs():
    """Show database statistics."""
    conn = connect_db()
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM jobs")
    total = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM jobs WHERE seen = 1")
    seen = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM jobs WHERE seen = 0")
    unseen = c.fetchone()[0]
    
    print(f"📊 Database Statistics:")
    print(f"   Total jobs: {total}")
    print(f"   Seen jobs: {seen}")
    print(f"   Unseen jobs: {unseen}")
    
    conn.close()

def clear_database():
    """Clear all jobs from the database (with confirmation)."""
    response = input("⚠️  Are you sure you want to delete ALL jobs? Type 'YES' to confirm: ")
    if response != 'YES':
        print("❌ Operation cancelled.")
        return
    
    conn = connect_db()
    c = conn.cursor()
    
    c.execute("DELETE FROM jobs")
    deleted_count = c.rowcount
    conn.commit()
    conn.close()
    
    print(f"🗑️  Deleted {deleted_count} jobs from database.")

def mark_job_seen(job_number):
    """Mark a specific job as seen."""
    conn = connect_db()
    c = conn.cursor()
    
    # Get the job
    c.execute("SELECT job_id, title FROM jobs ORDER BY rowid LIMIT 1 OFFSET ?", (job_number - 1,))
    job = c.fetchone()
    
    if not job:
        print(f"❌ Job #{job_number} not found.")
        conn.close()
        return
    
    # Update seen status
    c.execute("UPDATE jobs SET seen = 1 WHERE job_id = ?", (job[0],))
    conn.commit()
    
    print(f"✅ Marked job #{job_number} '{job[1]}' as seen.")
    conn.close()

def show_menu():
    """Display the main menu."""
    print("\n🔧 Database Management Menu:")
    print("1. View all jobs")
    print("2. View job details")
    print("3. Database statistics")
    print("4. Mark job as seen")
    print("5. Clear database")
    print("0. Exit")

def main():
    """Main interactive menu."""
    print("🗄️  Job Database Manager")
    print("=" * 30)
    
    while True:
        show_menu()
        choice = input("\nEnter your choice (0-5): ").strip()
        
        if choice == '0':
            print("👋 Goodbye!")
            break
        elif choice == '1':
            print("\n📋 All Jobs:")
            view_all_jobs()
        elif choice == '2':
            job_num = input("Enter job number: ").strip()
            try:
                view_job_details(int(job_num))
            except ValueError:
                print("❌ Please enter a valid number.")
        elif choice == '3':
            print()
            count_jobs()
        elif choice == '4':
            job_num = input("Enter job number to mark as seen: ").strip()
            try:
                mark_job_seen(int(job_num))
            except ValueError:
                print("❌ Please enter a valid number.")
        elif choice == '5':
            clear_database()
        else:
            print("❌ Invalid choice. Please enter 0-5.")

if __name__ == "__main__":
    main()