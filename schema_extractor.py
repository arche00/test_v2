import sqlite3

def get_table_schema(db_path, table_name):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query to get the schema of the specified table
    cursor.execute(f"PRAGMA table_info({table_name})")
    schema = cursor.fetchall()
    
    # Print the schema
    print(f"Schema for {table_name}:")
    for column in schema:
        print(column)
    
    # Close the connection
    conn.close()

# Path to the database
db_path = 'pattern_analysis_v2.db'

# Get schema for the tables
get_table_schema(db_path, 'pattern_analysis')
get_table_schema(db_path, 'session_prediction_results') 