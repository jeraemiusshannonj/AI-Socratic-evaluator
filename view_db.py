import sqlite3

# Replace 'evaluator.db' with your actual database file name if it's different
conn = sqlite3.connect("evaluator.db")
cursor = conn.cursor()

# 1. Check what tables actually exist in the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print(f"\n--- Tables found in database: {tables} ---\n")

# 2. Try querying interactions if it exists
try:
  cursor.execute("SELECT * FROM interactions")
  rows = cursor.fetchall()
  print(f"--- Total Interactions Logged: {len(rows)} ---")
  for row in rows:
    print(row)
except sqlite3.OperationalError as e:
  print(f"❌ Error: {e}")

conn.close()