import sqlite3

def check_raw_results():
    conn = sqlite3.connect('pattern_analysis_v2.db')
    cursor = conn.cursor()
    
    # Get all prediction results
    cursor.execute("SELECT prediction_results FROM session_prediction_results")
    results = cursor.fetchall()
    
    print("Raw Prediction Results:")
    print("-" * 50)
    
    total_length = 0
    for idx, result in enumerate(results, 1):
        if result[0]:
            print(f"Session {idx}: {result[0]}")
            total_length += len(result[0])
    
    print("-" * 50)
    print(f"Total number of sessions: {len(results)}")
    print(f"Total length of all results: {total_length}")
    
    conn.close()

if __name__ == "__main__":
    check_raw_results() 