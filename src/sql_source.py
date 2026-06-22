
import os
import re
import sqlite3
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, LLM_MODEL

DB_PATH = "data/appsc.db"


def create_database():
    """Creates the SQLite database and fills it with APPSC data automatically."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS AppscSubjects (
            ID INTEGER PRIMARY KEY,
            ExamName TEXT,
            SubjectName TEXT,
            TotalMarks INTEGER,
            PaperType TEXT,
            ExamGroup TEXT
        )
    """)

    cursor.execute("SELECT COUNT(*) FROM AppscSubjects")
    if cursor.fetchone()[0] == 0:
        data = [
            (1, 'Group 1 Prelims', 'General Studies & Mental Ability', 150, 'Objective', 'Group 1'),
            (2, 'Group 1 Mains', 'History & Culture of India', 150, 'Descriptive', 'Group 1'),
            (3, 'Group 1 Mains', 'Economy & Development', 150, 'Descriptive', 'Group 1'),
            (4, 'Group 1 Mains', 'Indian Polity & Governance', 150, 'Descriptive', 'Group 1'),
            (5, 'Group 1 Mains', 'Science & Technology', 150, 'Descriptive', 'Group 1'),
            (6, 'Group 2 Prelims', 'General Studies', 150, 'Objective', 'Group 2'),
            (7, 'Group 2 Mains', 'General Science & Technology', 150, 'Descriptive', 'Group 2'),
            (8, 'Group 2 Mains', 'Andhra Pradesh History', 150, 'Descriptive', 'Group 2'),
            (9, 'SI Prelims', 'General Studies', 100, 'Objective', 'SI'),
            (10, 'SI Mains', 'Telugu & English', 100, 'Descriptive', 'SI'),
            (11, 'SI Mains', 'General Ability & Current Affairs', 100, 'Descriptive', 'SI'),
            (12, 'Group 1 Mains', 'Geography & Environment', 150, 'Descriptive', 'Group 1'),
            (13, 'Group 2 Mains', 'Indian Economy', 150, 'Descriptive', 'Group 2'),
            (14, 'SI Prelims', 'Reasoning & Mental Ability', 100, 'Objective', 'SI'),
            (15, 'Group 1 Mains', 'Ethics & Integrity', 150, 'Descriptive', 'Group 1'),
        ]
        cursor.executemany(
            "INSERT INTO AppscSubjects VALUES (?,?,?,?,?,?)", data
        )
        conn.commit()
        print(f"[sql_source] Database created at {DB_PATH} with {len(data)} rows")

    conn.close()


def get_schema() -> str:
    """Returns the table schema so the LLM knows what columns exist."""
    return """
    Table: AppscSubjects
    Columns:
    - ID (integer)
    - ExamName (text) e.g. 'Group 1 Prelims', 'Group 1 Mains', 'SI Prelims'
    - SubjectName (text) e.g. 'General Studies', 'History & Culture of India'
    - TotalMarks (integer) e.g. 100, 150
    - PaperType (text) e.g. 'Objective', 'Descriptive'
    - ExamGroup (text) e.g. 'Group 1', 'Group 2', 'SI'
    """


def clean_sql(sql: str) -> str:
    """Extracts only the SQL SELECT statement from LLM output."""
    match = re.search(r"(SELECT.*)", sql, re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(0)
    sql = re.sub(r"```sql|```", "", sql)
    if "Answer:" in sql:
        sql = sql.split("Answer:")[0].strip()
    sql = sql.strip().rstrip(";").strip()
    return sql


def run_query(sql: str):
    """Runs a SQL query on the SQLite database and returns results."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    conn.close()
    return columns, rows


def query_sql_source(question: str) -> str:
    """Full text-to-SQL flow using SQLite."""
    create_database()
    llm = ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, temperature=0)

    
    prompt = f"""You are a SQL expert. Generate a SQLite SELECT query to answer the question.
Schema:
{get_schema()}

Rules:
- Use SQLite syntax (no square brackets, use LIKE for text search)
- Return ONLY the SQL query, nothing else
- No explanation, no markdown, just the SQL

Question: {question}
SQL:"""

    sql_query = llm.invoke(prompt).content.strip()
    sql_query = clean_sql(sql_query)
    print(f"[sql_source] SQL: {sql_query}")

    try:
        columns, rows = run_query(sql_query)
        if not rows:
            return "NO_RESULTS"

        
        result_text = " | ".join(columns) + "\n"
        for row in rows:
            result_text += " | ".join(str(v) for v in row) + "\n"

        
        answer_prompt = (
            f"Question: {question}\n"
            f"Database result:\n{result_text}\n\n"
            f"Answer the question in simple plain English."
        )
        return llm.invoke(answer_prompt).content

    except Exception as e:
        print(f"[sql_source] Query failed: {e}")
        return "NO_RESULTS"


if __name__ == "__main__":
    create_database()
    print(query_sql_source("how many marks does Group 1 Prelims carry?"))
    print(query_sql_source("list all subjects for Group 2"))
    print(query_sql_source("marks for geography"))
