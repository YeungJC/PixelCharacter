import sqlite3

import streamlit as st

class Database:
    def __init__(self, db_file="sqlite.db") -> None:
        self.db_file = db_file

    def get_connection(self):
        # Get connection and cursor to sqlite.db file
        try:
            conn = sqlite3.connect(self.db_file)
        except:
            # Attempt to provide error message to app admin
            # App will crash if it fails to connect and so we provide some logging
            print("Could not connect to database")
        cursor = conn.cursor()
        return conn, cursor

    def execute_sql(self, sql, parameters=None, get_id=False):
        # Execute sql query. Potentially return the id if we inserted a row
        lastrowid = None
        conn, cursor = self.get_connection()
        cursor.execute(sql, parameters)
        if get_id:
            lastrowid = cursor.lastrowid
        conn.commit()
        conn.close()
        return lastrowid

    def fetch_single(self, sql, parameters=None):
        # Fetch a single row from database for a provided query
        conn, cursor = self.get_connection()
        cursor.execute(sql, parameters)
        records = cursor.fetchone()
        conn.close()
        return records

    def fetch_all(self, sql, parameters=None):
        # Fetch all records from a database for a given query
        conn, cursor = self.get_connection()
        cursor.execute(sql, parameters)
        records = cursor.fetchall()
        conn.close()
        return records


class UserDatabase(Database):
    def __init__(self) -> None:
        super().__init__()

    def authenticate_login(self,username, hashed_password):
        # Check if user password pair exists
        query = "SELECT user_id FROM Users WHERE username = ? AND password = ?"
        row = self.fetch_single(query, (username, hashed_password))
        # If not records, write error message
        if row is None:
            st.write("Username / password not found")
            return None
        else:
            row = row[0]
            # Return user_id back to application as we use it for everything
        return row
    
    def insert_user(self, username, hashed_password):
        # Create user
        query = "INSERT INTO Users (username, password, date_created, date_updated) VALUES (?, ?, CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"
        user_id = self.execute_sql(query, (username, hashed_password), get_id=True)
        # Return user_id back to application as we use it for everything
        return user_id
    
    def insert_user_details(self,user_id, email):
        # Add user details (email)
        query = """
        INSERT INTO User_Details (user_id, email) 
        VALUES (?, ?)
        """
        self.execute_sql(query, (user_id, email))

    def not_unique_username(self, username):
        # Check if username already exists. Usernames must be unique
        query = """SELECT user_id FROM Users WHERE username = ?"""
        row = self.fetch_single(query, [username])

        if row is not None:
            return True
        else:
            return False
        
    def fetch_user_details(self, user_id):
        # Get username and email for a given user
        query = """
        SELECT u.username, ud.email
        FROM Users u 
        LEFT JOIN User_Details ud on u.user_id = ud.user_id
        WHERE u.user_id = ?;
        """
        row = self.fetch_single(query, [user_id])
        return row
    
    def update_password(self, hashed_password, user_id):
        # Update password for a user
        query = "UPDATE Users SET password = ? , date_updated = CURRENT_TIMESTAMP  WHERE user_id = ?"
        self.execute_sql(query, [hashed_password, user_id])
        return None
    
    def update_email(self, email,user_id):
        # Update email for a user
        query = "UPDATE User_Details SET email = ?  WHERE user_id = ?"
        self.execute_sql(query, [email, user_id])
        return None


class ImageDatabase(Database):
    def __init__(self) -> None:
        super().__init__()

    def get_image_pairs(self, img_id1, img_id2):
        # Get two images for a user
        latent_vectors = ", ".join([f"latent_vector_{i}" for i in range(40)])
        query = """
        SELECT {}
        FROM Image_Binary
        WHERE image_id in (?,?);
        """
        query = query.format(latent_vectors)
        records = self.fetch_all(query, [img_id1, img_id2])
        return records

    def get_image_details(self, UserId):
        # Get all images for user
        query = """
        SELECT ImD.user_id, ImD.image_id, ImD.name, ImB.raw  
        FROM Image_Details ImD 
        LEFT JOIN Image_Binary ImB on ImD.image_id = ImB.image_id
        WHERE ImD.user_id = ?;

        """
        records = self.fetch_all(query, [UserId])

        return records

    def create_image_details_record(self, user_id, image_name):
        # Create record in image details
        query = "INSERT INTO Image_details (user_id, name) VALUES (?, ?)"
        id = self.execute_sql(query, parameters=(user_id, image_name), get_id=True)
        return id

    def delete_image_id_from_image_details(self, img_id):
        # Delete record in image details
        query = "DELETE FROM Image_Details WHERE image_id = ?"
        self.execute_sql(query, [img_id])

    def delete_image_id_from_image_binary(self, img_id):
        # Delete record in image binary
        query = "DELETE FROM Image_Binary WHERE image_id = ?"
        self.execute_sql(query, [img_id])
    
    def insert_raw_image(self, latent_vectors, placeholders, args2):
        # Create record in image binary
        raw_query = "INSERT INTO Image_Binary (image_id, raw, {} ) VALUES (?,{}?)"
        raw_query = raw_query.format(latent_vectors, placeholders)
        self.execute_sql(raw_query, args2)



