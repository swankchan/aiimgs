"""
User management functions for administrators
"""
import streamlit as st
from db_config import get_connection
from auth import hash_password, is_admin

def get_all_users():
    """Get all users (admin only)"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, username, full_name, email, is_admin, created_at, last_login
            FROM users
            ORDER BY id
        """)
        
        users = []
        for row in cursor.fetchall():
            users.append({
                'id': row[0],
                'username': row[1],
                'full_name': row[2],
                'email': row[3],
                'is_admin': row[4],
                'created_at': row[5],
                'last_login': row[6]
            })
        
        cursor.close()
        conn.close()
        return users
        
    except Exception as e:
        st.error(f"Error fetching users: {e}")
        return []

def create_new_user(username: str, password: str, full_name: str = "", email: str = "", is_admin: bool = False):
    """Create new user (admin only)"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        password_hash = hash_password(password)
        
        cursor.execute("""
            INSERT INTO users (username, password_hash, full_name, email, is_admin)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (username, password_hash, full_name, email, is_admin))
        
        user_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        st.error(f"Error creating user: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False

def update_user(user_id: int, full_name: str = None, email: str = None, is_admin: bool = None):
    """Update user info (admin only)"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        updates = []
        params = []
        
        if full_name is not None:
            updates.append("full_name = %s")
            params.append(full_name)
        
        if email is not None:
            updates.append("email = %s")
            params.append(email)
        
        if is_admin is not None:
            updates.append("is_admin = %s")
            params.append(is_admin)
        
        if not updates:
            return True
        
        params.append(user_id)
        query = f"UPDATE users SET {', '.join(updates)} WHERE id = %s"
        
        cursor.execute(query, params)
        conn.commit()
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        st.error(f"Error updating user: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False

def delete_user(user_id: int):
    """Delete user (admin only)"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        st.error(f"Error deleting user: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False

def reset_user_password(user_id: int, new_password: str):
    """Reset user password (admin only)"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        password_hash = hash_password(new_password)
        cursor.execute("""
            UPDATE users
            SET password_hash = %s
            WHERE id = %s
        """, (password_hash, user_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        st.error(f"Error resetting password: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False

def show_user_management():
    """Display user management interface (admin only)"""
    if not is_admin():
        st.error("Access denied. Administrator privileges required.")
        return
    
    st.subheader("👥 User Management")
    
    # Create new user section
    with st.expander("➕ Create New User", expanded=False):
        with st.form("create_user_form"):
            new_username = st.text_input("Username*", key="new_user_username")
            new_password = st.text_input("Password*", type="password", key="new_user_password")
            new_full_name = st.text_input("Full Name", key="new_user_fullname")
            new_email = st.text_input("Email", key="new_user_email")
            new_is_admin = st.checkbox("Administrator", key="new_user_admin")
            
            if st.form_submit_button("Create User"):
                if not new_username or not new_password:
                    st.error("Username and password are required")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    if create_new_user(new_username, new_password, new_full_name, new_email, new_is_admin):
                        st.success(f"User '{new_username}' created successfully!")
                        st.rerun()
    
    st.divider()
    
    # List all users
    users = get_all_users()
    
    if not users:
        st.info("No users found")
        return
    
    st.markdown(f"**Total Users:** {len(users)}")
    
    for user in users:
        with st.expander(f"👤 {user['username']}" + (" 🔑" if user['is_admin'] else "")):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**ID:** {user['id']}")
                st.markdown(f"**Username:** {user['username']}")
                st.markdown(f"**Full Name:** {user['full_name'] or '_Not set_'}")
                st.markdown(f"**Email:** {user['email'] or '_Not set_'}")
                st.markdown(f"**Role:** {'Administrator' if user['is_admin'] else 'User'}")
                st.markdown(f"**Created:** {user['created_at']}")
                st.markdown(f"**Last Login:** {user['last_login'] or '_Never_'}")
            
            with col2:
                # Edit user
                with st.form(f"edit_user_{user['id']}"):
                    st.markdown("**Edit User**")
                    edit_full_name = st.text_input("Full Name", value=user['full_name'] or "", key=f"edit_fn_{user['id']}")
                    edit_email = st.text_input("Email", value=user['email'] or "", key=f"edit_email_{user['id']}")
                    edit_is_admin = st.checkbox("Administrator", value=user['is_admin'], key=f"edit_admin_{user['id']}")
                    
                    if st.form_submit_button("Update"):
                        if update_user(user['id'], edit_full_name, edit_email, edit_is_admin):
                            st.success("User updated!")
                            st.rerun()
                
                # Reset password
                with st.form(f"reset_pwd_{user['id']}"):
                    st.markdown("**Reset Password**")
                    new_pwd = st.text_input("New Password", type="password", key=f"reset_pwd_input_{user['id']}")
                    
                    if st.form_submit_button("Reset Password"):
                        if not new_pwd:
                            st.error("Password required")
                        elif len(new_pwd) < 6:
                            st.error("Password must be at least 6 characters")
                        else:
                            if reset_user_password(user['id'], new_pwd):
                                st.success("Password reset successfully!")
                
                # Delete user
                current_user = st.session_state.get('user', {})
                if user['id'] != current_user.get('id'):  # Can't delete yourself
                    st.markdown("**Delete User**")
                    if st.button("🗑️ Delete User", key=f"delete_{user['id']}", type="secondary"):
                        if delete_user(user['id']):
                            st.success(f"User '{user['username']}' deleted!")
                            st.rerun()
                else:
                    st.info("Cannot delete your own account")
