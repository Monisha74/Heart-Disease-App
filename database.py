import sqlite3

conn = sqlite3.connect("hospital.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS high_risk(
id INTEGER PRIMARY KEY AUTOINCREMENT,
name TEXT,
age INTEGER,
bp INTEGER,
chol INTEGER
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS low_risk(
id INTEGER PRIMARY KEY AUTOINCREMENT,
name TEXT,
age INTEGER,
bp INTEGER,
chol INTEGER
)
""")

conn.commit()
conn.close()