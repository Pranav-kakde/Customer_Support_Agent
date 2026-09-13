import sqlite3
import os
from datetime import datetime


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

MEMORY_DIR = "memory"

os.makedirs(
    MEMORY_DIR,
    exist_ok=True
)

DATABASE_PATH = os.path.join(
    MEMORY_DIR,
    "customer_memory.db"
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    return sqlite3.connect(
        DATABASE_PATH,
        check_same_thread=False
    )


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_database():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS customers (

            customer_id TEXT PRIMARY KEY,

            name TEXT,

            email TEXT,

            phone TEXT,

            preferences TEXT,

            notes TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()

    conn.close()


initialize_database()


# ============================================================
# CREATE / UPDATE CUSTOMER
# ============================================================

def save_customer(
    customer_id,
    name=None,
    email=None,
    phone=None,
    preferences=None,
    notes=None
):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO customers
        (
            customer_id,
            name,
            email,
            phone,
            preferences,
            notes
        )

        VALUES (?, ?, ?, ?, ?, ?)

        ON CONFLICT(customer_id)

        DO UPDATE SET

            name = COALESCE(excluded.name, customers.name),

            email = COALESCE(excluded.email, customers.email),

            phone = COALESCE(excluded.phone, customers.phone),

            preferences = COALESCE(
                excluded.preferences,
                customers.preferences
            ),

            notes = COALESCE(
                excluded.notes,
                customers.notes
            ),

            updated_at = CURRENT_TIMESTAMP
        """,
        (
            customer_id,
            name,
            email,
            phone,
            preferences,
            notes
        )
    )

    conn.commit()

    conn.close()


# ============================================================
# GET CUSTOMER
# ============================================================

def get_customer_memory(
    customer_id
):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            customer_id,
            name,
            email,
            phone,
            preferences,
            notes,
            created_at,
            updated_at

        FROM customers

        WHERE customer_id = ?
        """,
        (
            customer_id,
        )
    )

    result = cursor.fetchone()

    conn.close()

    if not result:

        return None

    return {

        "customer_id": result[0],

        "name": result[1],

        "email": result[2],

        "phone": result[3],

        "preferences": result[4],

        "notes": result[5],

        "created_at": result[6],

        "updated_at": result[7],
    }


# ============================================================
# UPDATE CUSTOMER PREFERENCE
# ============================================================

def update_preferences(
    customer_id,
    preferences
):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE customers

        SET
            preferences = ?,
            updated_at = CURRENT_TIMESTAMP

        WHERE customer_id = ?
        """,
        (
            preferences,
            customer_id
        )
    )

    conn.commit()

    conn.close()


# ============================================================
# ADD CUSTOMER NOTE
# ============================================================

def update_notes(
    customer_id,
    notes
):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE customers

        SET
            notes = ?,
            updated_at = CURRENT_TIMESTAMP

        WHERE customer_id = ?
        """,
        (
            notes,
            customer_id
        )
    )

    conn.commit()

    conn.close()
    
    
# Database Initilization    
def initialize_database():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS customers (

            customer_id TEXT PRIMARY KEY,

            name TEXT,

            email TEXT,

            phone TEXT,

            preferences TEXT,

            notes TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # --------------------------------------------------------
    # Demo customers
    # --------------------------------------------------------

    cursor.execute(
        """
        INSERT OR IGNORE INTO customers
        (
            customer_id,
            name,
            email,
            phone,
            preferences,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "C001",
            "Rahul Sharma",
            "rahul@example.com",
            "9876543210",
            "Prefers email communication",
            "Regular TechNova customer"
        )
    )

    cursor.execute(
        """
        INSERT OR IGNORE INTO customers
        (
            customer_id,
            name,
            email,
            phone,
            preferences,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "C002",
            "Priya Patil",
            "priya@example.com",
            "9876543211",
            "Prefers quick support responses",
            "TechNova Smart Watch customer"
        )
    )

    conn.commit()

    conn.close()    
    