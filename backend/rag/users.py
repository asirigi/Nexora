import sqlite3


def validate_user(username: str, password: str):
    username = username.lower()
    password = password.lower()
    conn = sqlite3.connect("users.db")
    print("Connected to the database successfully.")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = cursor.fetchone()
    print(user)
    conn.close()    
    return user
