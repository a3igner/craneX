"""
Central configuration loader for CRANE-X.

Loads all API endpoints and pipeline parameters from config.yaml,
and secrets (API keys, DB credentials) from environment variables / .env.

Usage:
    from utils.config import load_config
    cfg = load_config()
    eodhd_url = cfg['api']['eodhd']['base_url']
    deepseek_model = cfg['api']['deepseek']['model']
"""

import os
import sys
import yaml

# Project root — two levels up from utils/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cache so we only read files once
_config_cache = None


def _load_yaml():
    """Load config.yaml from project root, returning dict or empty dict."""
    path = os.path.join(PROJECT_ROOT, 'config.yaml')
    if not os.path.exists(path):
        print(f"[Config] WARNING: {path} not found — using defaults", file=sys.stderr)
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_env_file():
    """Parse .env file into a dict of key-value pairs."""
    env = {}
    for dotenv_path in [
        os.path.join(PROJECT_ROOT, '.env'),
        '/home/a3/.env',
        os.path.expanduser('~/.env'),
    ]:
        if os.path.exists(dotenv_path):
            with open(dotenv_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, v = line.split('=', 1)
                        env[k.strip()] = v.strip().strip('"').strip("'")
            break
    return env


def load_config():
    """
    Load full CRANE-X configuration.

    Priority (higher wins):
        1. Environment variables (os.environ)
        2. .env file
        3. config.yaml

    Returns a dict with keys:
        api:        nested dict with eodhd, deepseek, wsj_dylan, gauge endpoints
        pipeline:   timing / batch parameters
        wsj_tickers: list of instrument mappings
        assets:     list of asset symbols
        secrets:    merged API keys from env / .env (not from config.yaml)
        db:         MySQL connection parameters
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    yaml_cfg = _load_yaml()
    dotenv = _load_env_file()

    # --- Build result ---
    cfg = {}

    # API endpoints from yaml, with env fallback for individual keys
    cfg['api'] = yaml_cfg.get('api', {})

    # Pipeline parameters
    cfg['pipeline'] = yaml_cfg.get('pipeline', {})
    cfg['wsj_tickers'] = yaml_cfg.get('wsj_tickers', [])
    cfg['assets'] = yaml_cfg.get('assets', [])

    # --- Secrets: env var > .env > defaults ---
    def get_secret(key, default=None):
        return os.environ.get(key) or dotenv.get(key) or default

    cfg['secrets'] = {
        'eodhd_api_key': get_secret('EODHD_API_KEY'),
        'deepseek_api_key': get_secret('DEEPSEEK_API_KEY'),
        'wsj_entitlement_token': (
            os.environ.get('WSJ_ENTITLEMENT_TOKEN')
            or dotenv.get('WSJ_ENTITLEMENT_TOKEN')
            or cfg.get('api', {}).get('wsj_dylan', {}).get('entitlement_token')
        ),
        'wsj_ckey': (
            os.environ.get('WSJ_CKEY')
            or dotenv.get('WSJ_CKEY')
            or cfg.get('api', {}).get('wsj_dylan', {}).get('ckey')
        ),
    }

    # --- DB config ---
    cfg['db'] = {
        'host': get_secret('MYSQL_HOST', '127.0.0.1'),
        'port': int(get_secret('MYSQL_PORT', '3306')),
        'user': get_secret('MYSQL_USER', ''),
        'password': get_secret('MYSQL_PASSWORD', ''),
        'database': get_secret('MYSQL_DATABASE', ''),
    }

    _config_cache = cfg
    return cfg
