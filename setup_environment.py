#!/usr/bin/env python3
"""
Automated Environment Setup & Dependency Installer
for University AI Advisor (FYP).

Checks and configures:
1. Python environment & virtualenv
2. Project dependencies & EEG connectivity libraries (pylsl, muselsl)
3. Ollama service status & required LLM models
4. PostgreSQL credentials, database existence, and automatic backup restoration
5. Updates/creates FYP2/.env with active configuration
"""

import getpass
import os
import shutil
import subprocess
import sys
from pathlib import Path
import urllib.request
import json
import time

# Styling constants
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

PROJECT_ROOT = Path(__file__).resolve().parent
FYP2_DIR = PROJECT_ROOT / "FYP2"
ENV_FILE = FYP2_DIR / ".env"
ENV_EXAMPLE = FYP2_DIR / ".env.example"
REQ_FILE = PROJECT_ROOT / "requirements.txt"
DEFAULT_DUMP_FILE = PROJECT_ROOT / "database_backup.dump"
ALT_DUMP_PATH = Path("/Users/sharjeelahmed/fyprecovery")

REQUIRED_EEG_LIBS = ["pylsl", "muselsl"]
DEFAULT_MODELS = ["gemma2:2b", "mistral", "nomic-embed-text"]


def print_banner():
    print(f"\n{CYAN}{BOLD}======================================================{RESET}")
    print(f"{CYAN}{BOLD}    🎓 University AI Advisor - Automated Setup       {RESET}")
    print(f"{CYAN}{BOLD}======================================================{RESET}\n")


def log_step(title: str):
    print(f"\n{BLUE}{BOLD}▶ {title}{RESET}")


def log_success(msg: str):
    print(f"  {GREEN}✔ {msg}{RESET}")


def log_warn(msg: str):
    print(f"  {YELLOW}⚠ {msg}{RESET}")


def log_error(msg: str):
    print(f"  {RED}✖ {msg}{RESET}")


def log_info(msg: str):
    print(f"  {CYAN}ℹ {msg}{RESET}")


def find_executable(name: str) -> str | None:
    # First check standard PATH
    found = shutil.which(name)
    if found:
        return found

    # Check common macOS & Linux directories
    extra_paths = [
        Path("/Library/PostgreSQL/18/bin") / name,
        Path("/Library/PostgreSQL/17/bin") / name,
        Path("/Library/PostgreSQL/16/bin") / name,
        Path("/Library/PostgreSQL/15/bin") / name,
        Path("/usr/local/bin") / name,
        Path("/opt/homebrew/bin") / name,
        Path("/usr/bin") / name,
    ]
    for p in extra_paths:
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    return None


def get_python_and_pip():
    """Ensure we use the virtualenv python and pip if available."""
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    venv_pip = PROJECT_ROOT / ".venv" / "bin" / "pip"

    if venv_python.exists() and venv_pip.exists():
        return str(venv_python), str(venv_pip)

    # If running inside active venv
    if sys.prefix != sys.base_prefix:
        return sys.executable, shutil.which("pip") or f"{sys.executable} -m pip"

    return sys.executable, f"{sys.executable} -m pip"


def read_env_values() -> dict:
    values = {}
    target = ENV_FILE if ENV_FILE.exists() else ENV_EXAMPLE
    if target.exists():
        with open(target, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    # Check commented out defaults e.g. #DB_HOST=localhost
                    if "=" in line:
                        line = line.lstrip("#").strip()
                    else:
                        continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    values[k.strip()] = v.strip().strip("'\"")
    return values


def update_env_file(updates: dict):
    existing_lines = []
    keys_updated = set()

    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()

    new_lines = []
    for line in existing_lines:
        trimmed = line.strip()
        matched = False
        for k, v in updates.items():
            if trimmed.startswith(f"{k}=") or trimmed.startswith(f"#{k}="):
                new_lines.append(f"{k}={v}\n")
                keys_updated.add(k)
                matched = True
                break
        if not matched:
            new_lines.append(line)

    # Add any keys that weren't already in file
    for k, v in updates.items():
        if k not in keys_updated:
            new_lines.append(f"{k}={v}\n")

    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    log_success(f"Configuration saved to {ENV_FILE}")


# -------------------------------------------------------------
# STEP 1: Python Libraries & EEG Device Connectors
# -------------------------------------------------------------
def check_and_install_libraries():
    log_step("Checking Python Dependencies & EEG Device Libraries")
    python_bin, pip_bin = get_python_and_pip()
    log_info(f"Using Python: {python_bin}")

    # Check requirements.txt
    if REQ_FILE.exists():
        log_info(f"Verifying project packages from {REQ_FILE.name}...")
        try:
            cmd = [python_bin, "-m", "pip", "install", "-r", str(REQ_FILE)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                log_success("All core dependencies from requirements.txt are installed.")
            else:
                log_warn(f"pip encountered warnings:\n{res.stderr.strip()[:300]}")
        except Exception as e:
            log_error(f"Failed to run pip: {e}")
    else:
        log_warn("requirements.txt not found. Skipping batch install.")

    # Check EEG Libraries specifically
    for lib in REQUIRED_EEG_LIBS:
        try:
            res = subprocess.run([python_bin, "-c", f"import {lib}"], capture_output=True)
            if res.returncode == 0:
                log_success(f"EEG library '{lib}' is installed and importable.")
            else:
                log_info(f"Installing missing EEG library '{lib}'...")
                install_res = subprocess.run([python_bin, "-m", "pip", "install", lib], capture_output=True, text=True)
                if install_res.returncode == 0:
                    log_success(f"Successfully installed EEG library '{lib}'.")
                else:
                    log_error(f"Failed to install '{lib}': {install_res.stderr.strip()}")
        except Exception as e:
            log_error(f"Error checking EEG library '{lib}': {e}")


# -------------------------------------------------------------
# STEP 2: Ollama and LLM Models
# -------------------------------------------------------------
def check_and_setup_ollama():
    log_step("Checking Ollama System Installation & Models")
    ollama_bin = find_executable("ollama")

    if not ollama_bin:
        log_error("Ollama is not installed or not in PATH!")
        log_info("Please install Ollama: https://ollama.com/download or via 'brew install ollama'")
        return False

    log_success(f"Ollama binary found at: {ollama_bin}")

    # Check if Ollama service is running
    is_running = False
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                is_running = True
    except Exception:
        is_running = False

    if not is_running:
        log_info("Ollama service is not running. Attempting to start 'ollama serve' in background...")
        try:
            subprocess.Popen([ollama_bin, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Wait up to 8 seconds
            for _ in range(8):
                time.sleep(1)
                try:
                    with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as r:
                        if r.status == 200:
                            is_running = True
                            break
                except Exception:
                    continue
        except Exception as e:
            log_warn(f"Could not auto-start Ollama service: {e}")

    if is_running:
        log_success("Ollama service is up and running.")
    else:
        log_warn("Ollama service could not be reached. Please run 'ollama serve' in another terminal.")

    # Check models
    installed_models = set()
    try:
        res = subprocess.run([ollama_bin, "list"], capture_output=True, text=True)
        if res.returncode == 0:
            for line in res.stdout.splitlines()[1:]:
                parts = line.split()
                if parts:
                    installed_models.add(parts[0])
                    # Add base name without :latest if present
                    if ":latest" in parts[0]:
                        installed_models.add(parts[0].replace(":latest", ""))
    except Exception as e:
        log_warn(f"Could not query 'ollama list': {e}")

    # Models required by the advisor
    env_vals = read_env_values()
    eng_model = env_vals.get("OLLAMA_MODEL_ENGLISH", "gemma2:2b")
    urdu_model = env_vals.get("OLLAMA_MODEL_ROMAN_URDU", "mistral")
    models_to_check = list(set([eng_model, urdu_model] + DEFAULT_MODELS))

    for model in models_to_check:
        base_name = model.split(":")[0]
        has_model = any(m == model or m.startswith(f"{base_name}:") or m == base_name for m in installed_models)

        if has_model:
            log_success(f"Model '{model}' is present.")
        else:
            log_info(f"Model '{model}' is missing. Pulling model (this may take a few minutes)...")
            pull_res = subprocess.run([ollama_bin, "pull", model])
            if pull_res.returncode == 0:
                log_success(f"Successfully pulled model '{model}'.")
            else:
                log_error(f"Failed to pull model '{model}'. You can manually run: ollama pull {model}")

    return True


# -------------------------------------------------------------
# STEP 3: PostgreSQL & Database Setup / Restore
# -------------------------------------------------------------
def prompt_db_credentials() -> dict:
    env_vals = read_env_values()

    def_host = env_vals.get("DB_HOST", "localhost")
    def_port = env_vals.get("DB_PORT", "5432")
    def_user = env_vals.get("DB_USER", "postgres")
    def_name = env_vals.get("DB_NAME", "fyp_i")
    def_password = env_vals.get("DB_PASSWORD", "")

    print(f"\n{BOLD}Please confirm your PostgreSQL connection parameters:{RESET}")

    host = input(f"  • Database Host [{def_host}]: ").strip() or def_host
    port = input(f"  • Database Port [{def_port}]: ").strip() or def_port
    user = input(f"  • Database Username [{def_user}]: ").strip() or def_user

    if def_password:
        pwd_prompt = f"  • Database Password [stored in .env]: "
    else:
        pwd_prompt = f"  • Database Password: "

    pwd_input = getpass.getpass(pwd_prompt)
    password = pwd_input if pwd_input else def_password

    dbname = input(f"  • Database Name [{def_name}]: ").strip() or def_name

    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": dbname,
    }


def find_db_dump_file() -> Path | None:
    # Priority 1: Project root database_backup.dump
    if DEFAULT_DUMP_FILE.exists():
        return DEFAULT_DUMP_FILE
    # Priority 2: Alternative known path
    if ALT_DUMP_PATH.exists():
        return ALT_DUMP_PATH
    # Priority 3: Any .dump or .sql in project root
    for f in PROJECT_ROOT.glob("*.dump"):
        return f
    for f in PROJECT_ROOT.glob("*.sql"):
        return f
    return None


def restore_database(pg_restore_bin: str, creds: dict, dump_path: Path):
    log_info(f"Restoring database '{creds['database']}' from {dump_path}...")
    env = os.environ.copy()
    if creds.get("password"):
        env["PGPASSWORD"] = creds["password"]

    is_sql_script = str(dump_path).endswith(".sql")

    if is_sql_script:
        psql_bin = find_executable("psql")
        if not psql_bin:
            log_error("psql binary not found. Cannot execute plain SQL file.")
            return False
        cmd = [
            psql_bin,
            "-h", creds["host"],
            "-p", creds["port"],
            "-U", creds["user"],
            "-d", creds["database"],
            "-f", str(dump_path),
        ]
    else:
        cmd = [
            pg_restore_bin,
            "-h", creds["host"],
            "-p", creds["port"],
            "-U", creds["user"],
            "-d", creds["database"],
            "--no-owner",
            "--no-privileges",
            str(dump_path),
        ]

    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    # pg_restore often exits with code 1 for minor warnings (e.g. relation already exists)
    if res.returncode in (0, 1):
        log_success(f"Database restoration completed successfully!")
        return True
    else:
        log_error(f"Restore encountered an error:\n{res.stderr.strip()[:400]}")
        return False


def setup_database():
    log_step("Configuring PostgreSQL & Academic Database")

    # Locate PostgreSQL tools
    pg_restore_bin = find_executable("pg_restore")
    if not pg_restore_bin:
        log_warn("pg_restore not found in system PATH. Checking PostgreSQL 18/17 default locations...")
        for ver in [18, 17, 16, 15]:
            candidate = Path(f"/Library/PostgreSQL/{ver}/bin/pg_restore")
            if candidate.exists():
                pg_restore_bin = str(candidate)
                break

    if not pg_restore_bin:
        log_warn("Could not find 'pg_restore'. Database dump restore may require manual action.")
    else:
        log_success(f"PostgreSQL restore utility found: {pg_restore_bin}")

    creds = prompt_db_credentials()

    # Try connecting to PostgreSQL using psycopg2
    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    except ImportError:
        log_error("psycopg2 is not installed. Please run dependency installation first.")
        return False

    # 1. Connect to PostgreSQL server (connecting to maintenance db 'postgres')
    target_db = creds["database"]
    server_conn = None
    try:
        server_conn = psycopg2.connect(
            host=creds["host"],
            port=creds["port"],
            user=creds["user"],
            password=creds["password"],
            dbname="postgres",
            connect_timeout=5,
        )
        server_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        log_success("Connected to PostgreSQL server successfully.")
    except Exception as e:
        log_error(f"Could not connect to PostgreSQL server: {e}")
        log_info("Please verify your host, port, username, and password.")
        return False

    # 2. Check if the target database exists
    cursor = server_conn.cursor()
    cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (target_db,))
    db_exists = cursor.fetchone() is not None

    if not db_exists:
        log_info(f"Database '{target_db}' does not exist. Creating database now...")
        try:
            cursor.execute(f'CREATE DATABASE "{target_db}";')
            log_success(f"Database '{target_db}' created successfully.")
            db_exists = True
        except Exception as e:
            log_error(f"Failed to create database '{target_db}': {e}")
            cursor.close()
            server_conn.close()
            return False
    else:
        log_success(f"Database '{target_db}' already exists.")

    cursor.close()
    server_conn.close()

    # 3. Check table count in the target database
    table_count = 0
    try:
        db_conn = psycopg2.connect(
            host=creds["host"],
            port=creds["port"],
            user=creds["user"],
            password=creds["password"],
            dbname=target_db,
            connect_timeout=5,
        )
        with db_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';")
            table_count = cur.fetchone()[0]
        db_conn.close()
    except Exception as e:
        log_warn(f"Could not inspect tables in '{target_db}': {e}")

    # 4. If empty or requested, restore from backup dump
    dump_file = find_db_dump_file()
    if table_count == 0:
        log_info(f"Database '{target_db}' has 0 tables. Attempting automated data restoration...")
        if dump_file and dump_file.exists():
            log_info(f"Found backup file: {dump_file}")
            if pg_restore_bin:
                restore_database(pg_restore_bin, creds, dump_file)
            else:
                log_warn("Cannot restore automatically: 'pg_restore' is not found.")
        else:
            log_warn("No database backup file found in project. Please provide a backup dump.")
    else:
        log_success(f"Database '{target_db}' already contains {table_count} tables.")
        if dump_file:
            log_info(f"Backup file is available at: {dump_file}")

    # 5. Save credentials to FYP2/.env
    update_env_file({
        "DB_HOST": creds["host"],
        "DB_PORT": creds["port"],
        "DB_USER": creds["user"],
        "DB_PASSWORD": creds["password"],
        "DB_NAME": creds["database"],
    })
    return True


# -------------------------------------------------------------
# STEP 4: Knowledge Base / ChromaDB Check
# -------------------------------------------------------------
def check_knowledge_base():
    log_step("Checking Vector Knowledge Base")
    chroma_dir = FYP2_DIR / "chroma_db"
    rebuild_script = FYP2_DIR / "scripts" / "rebuild_knowledge_base.py"

    if chroma_dir.exists() and any(chroma_dir.iterdir()):
        log_success("Chroma vector database exists and contains indexed data.")
    else:
        log_info("Chroma vector database is not built yet.")
        if rebuild_script.exists():
            log_info("Building knowledge base from policy documents...")
            python_bin, _ = get_python_and_pip()
            res = subprocess.run([python_bin, str(rebuild_script)], capture_output=True, text=True)
            if res.returncode == 0:
                log_success("Vector knowledge base built successfully.")
            else:
                log_warn(f"Could not build knowledge base: {res.stderr.strip()[:200]}")


def main():
    print_banner()

    # Step 1: Python Libraries & EEG
    check_and_install_libraries()

    # Step 2: Ollama & Models
    check_and_setup_ollama()

    # Step 3: Database & Backup
    setup_database()

    # Step 4: Knowledge Base
    check_knowledge_base()

    # Finish Summary
    print(f"\n{GREEN}{BOLD}======================================================{RESET}")
    print(f"{GREEN}{BOLD}    🎉 Environment Setup & Verification Complete!     {RESET}")
    print(f"{GREEN}{BOLD}======================================================{RESET}")
    print(f"\nTo launch the FastAPI backend server, run:")
    print(f"  {CYAN}uvicorn FYP2.main:app --reload --port 8000{RESET}\n")


if __name__ == "__main__":
    main()
