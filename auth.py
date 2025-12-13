"""
User authentication for the application
Simple username/password authentication with session management
"""
import streamlit as st
import hashlib
from db_config import get_connection

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_users_table():
    """Create users table if not exists"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password_hash VARCHAR(64) NOT NULL,
                full_name VARCHAR(100),
                email VARCHAR(100),
                is_admin BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating users table: {e}")
        return False

def create_user(username: str, password: str, full_name: str = "", email: str = "", is_admin: bool = False) -> bool:
    """Create a new user"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        password_hash = hash_password(password)
        
        cursor.execute("""
            INSERT INTO users (username, password_hash, full_name, email, is_admin)
            VALUES (%s, %s, %s, %s, %s)
        """, (username, password_hash, full_name, email, is_admin))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating user: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False

def verify_user(username: str, password: str) -> dict | None:
    """Verify user credentials and return user info"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        password_hash = hash_password(password)
        
        cursor.execute("""
            SELECT id, username, full_name, email, is_admin
            FROM users
            WHERE username = %s AND password_hash = %s
        """, (username, password_hash))
        
        result = cursor.fetchone()
        
        if result:
            # Update last login
            cursor.execute("""
                UPDATE users
                SET last_login = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (result[0],))
            conn.commit()
            
            user_info = {
                'id': result[0],
                'username': result[1],
                'full_name': result[2],
                'email': result[3],
                'is_admin': result[4]
            }
            
            cursor.close()
            conn.close()
            return user_info
        
        cursor.close()
        conn.close()
        return None
        
    except Exception as e:
        print(f"Error verifying user: {e}")
        return None

def show_login_page():
    """Display login page"""
    st.title("🔐 Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if not username or not password:
                st.error("Please enter both username and password")
                return False
            
            user_info = verify_user(username, password)
            
            if user_info:
                st.session_state.logged_in = True
                st.session_state.user = user_info
                st.success(f"Welcome, {user_info['full_name'] or user_info['username']}!")
                st.rerun()
                return True
            else:
                st.error("Invalid username or password")
                return False
    
    return False

def show_logout_button():
    """Display logout button in sidebar"""
    if st.session_state.get('logged_in', False):
        user = st.session_state.get('user', {})
        st.sidebar.markdown("---")
        st.sidebar.write(f"👤 {user.get('full_name') or user.get('username')}")
        if user.get('is_admin'):
            st.sidebar.write("🔑 Administrator")
        
        # Show change password option
        show_change_password_dialog()
        
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()

def require_login():
    """Decorator/function to require login"""
    if not st.session_state.get('logged_in', False):
        show_login_page()
        st.stop()

def is_admin():
    """Check if current user is admin"""
    if not st.session_state.get('logged_in', False):
        return False
    user = st.session_state.get('user', {})
    return user.get('is_admin', False)

def change_password(username: str, old_password: str, new_password: str) -> bool:
    """Change user password"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Verify old password
        old_hash = hash_password(old_password)
        cursor.execute("""
            SELECT id FROM users
            WHERE username = %s AND password_hash = %s
        """, (username, old_hash))
        
        if not cursor.fetchone():
            cursor.close()
            conn.close()
            return False
        
        # Update to new password
        new_hash = hash_password(new_password)
        cursor.execute("""
            UPDATE users
            SET password_hash = %s
            WHERE username = %s
        """, (new_hash, username))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error changing password: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False

def show_change_password_dialog():
    """Display change password dialog"""
    if not st.session_state.get('logged_in', False):
        return
    
    user = st.session_state.get('user', {})
    
    st.sidebar.markdown("---")
    with st.sidebar.expander("🔑 Change Password"):
        with st.form("change_password_form"):
            old_password = st.text_input("Current Password", type="password", key="old_pwd")
            new_password = st.text_input("New Password", type="password", key="new_pwd")
            confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_pwd")
            submit = st.form_submit_button("Change Password")
            
            if submit:
                if not old_password or not new_password or not confirm_password:
                    st.error("Please fill in all fields")
                elif new_password != confirm_password:
                    st.error("New passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                elif old_password == new_password:
                    st.error("New password must be different from current password")
                else:
                    if change_password(user['username'], old_password, new_password):
                        st.success("Password changed successfully!")
                    else:
                        st.error("Current password is incorrect")

if __name__ == "__main__":
    print("Creating users table...")
    if create_users_table():
        print("✓ Users table created")
        
        # Create default admin user
        print("\nCreating default admin user...")
        if create_user("admin", "admin123", "Administrator", "", is_admin=True):
            print("✓ Default admin user created")
            print("  Username: admin")
            print("  Password: admin123")
            print("\n⚠ IMPORTANT: Please change this password after first login!")
        else:
            print("✗ Failed to create admin user (may already exist)")
    else:
        print("✗ Failed to create users table")
