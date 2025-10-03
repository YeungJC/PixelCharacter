import sqlite3
import streamlit as st
import math
from db import UserDatabase


class Users:
    def __init__(self, db_file="sqlite.db"):
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()

    def validate(self, username, password, email):
        # Validate username, password and email
        valid_username = self.validate_user(username)
        valid_password = self.validate_password(password)
        valid_email = self.validate_email(email)
        authorised = all([valid_username, valid_password, valid_email])
        return authorised

    def validate_user(self, username):
        # Check username is unique
        authorised = False
        if UserDatabase().not_unique_username(username):
            st.error("Username already taken")
        # Username cannot be empty
        elif len(username) == "":
            st.error("username cannot be empty")
        # Username cannot be greater than 20 characters
        elif len(username) > 20:
            st.error("username length must be less than 20")
        else:
            # Only return true is all validation has passed
            authorised = True
        return authorised

    def validate_password(self, password):
        authorised = False
        if len(password) < 8:
            # Password minimum length
            st.error("password length must be greater than 8")
        elif len(password) > 20:
            # password maximum length
            st.error("password length must be less than 20")
        elif not any(char.isdigit() for char in password):
            # password must contain number
            st.error("Password must have at least one number")

        elif not any(char.isupper() for char in password):
            # password must contain upper case
            st.error("Password must have at least one uppercase letter")
        elif not any(char.islower() for char in password):
            # password must contain lower
            st.error("Password must have at least one lowercase letter")
        else:
            # Only return true is all validation has passed
            authorised = True
        return authorised

    def validate_email(self, email):
        # Check @ is in middle
        authorised = False
        # Split the email into username and domain
            
        if "@" not in email or email.startswith("@") or email.endswith("@"):
            st.error("Please enter correct email")
        # Check if username and domain are not empty
        else:
            username, domain = email.split("@")
            if not username or not domain:
                st.error("Please enter correct email")
            # Check if there is a dot in the domain
            elif "." not in domain:
                st.error("Please enter correct email")
            else:
                # Only return true is all validation has passed
                authorised = True
        return authorised

    def login(self, username, password):
        user_id = None
        # hash password, don't store plain text
        hashed_password = self.hash_password(password)
        # Check if user,password exists
        row = UserDatabase().authenticate_login(username, hashed_password)
        if row is not None:
            user_id = row
        return user_id

    def signup(self, username, password, email):
        # validate username, password, email
        if not self.validate(username, password, email):
            st.error("Signup failed. Please check the validation criteria.")
            return False
        # hash password
        hashed_password = self.hash_password(password)
        # insert user password into db
        user_id = UserDatabase().insert_user(username, hashed_password)
        # add email to db
        UserDatabase().insert_user_details(user_id, email)
        return user_id

    def hash_password(self, password):
        # password hashing algorithm
        hashed_password = ""
        for x in password:
            ascii_character = ord(x)
            hashed_character = round(math.sqrt((ascii_character + 2) * 3))
            hashed_password += chr(hashed_character)

        return hashed_password
    
    def get_user_details(self,user_id):
        # Get user email for a user_id
        return UserDatabase().fetch_user_details(user_id)
    
    def update_email(self,email, user_id):
        # Update a users email if it meets criteria
        valid_email = self.validate_email(email)
        if valid_email:
            UserDatabase().update_email(email, user_id)
            st.success("Email updated")
        else:
            pass
    
    def update_password(self,password, user_id):
        # Update a users password if it meets criteria
        valid_password = self.validate_password(password)
        if valid_password:
            hashed_password = self.hash_password(password)
            UserDatabase().update_password(hashed_password, user_id)
            st.success("Email updated")
        else:
            pass

