"""
Database utilities for CRANE-X.
Connects to the same MySQL instance as the old CRANE project.
"""

import os
import sys
import mysql.connector
from mysql.connector import Error
try:
    from config import load_config as load_full_config
except ImportError:
    # When called without utils/ in sys.path (e.g. direct import)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import load_config as load_full_config


def load_config():
    """Load DB config from central config (config.yaml + .env + env vars)."""
    full = load_full_config()
    return full.get('db', {})


def get_connection(config=None):
    """Return a MySQL connection."""
    if config is None:
        config = load_config()
    try:
        conn = mysql.connector.connect(**config)
        return conn
    except Error as e:
        print(f"[DB] Connection error: {e}")
        return None
