from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
INSTALLER_DIR = SCRIPT_DIR.parent.parent
OUT_PATH = INSTALLER_DIR / "installer.pyz"
BUILD_DIR = INSTALLER_DIR / ".build_pyz"
APP_SRC = INSTALLER_DIR / "pyz_app"
APP_DEST = BUILD_DIR / "app" / "pyz_app"
REQUIREMENTS = APP_SRC /  "requirements.txt"

DISTRIBUTED_DEP_DB_DIR = INSTALLER_DIR / "dep_database.ignore"
CONSOLIDATED_DEP_DB_DIR = APP_SRC / "bundled_files" / "pip_dependency_database.json"
DEPENDENCY_OUT = APP_SRC / "bundled_files" / "pip_dependency_database.json"
PYPROJECT_LINK = APP_SRC / "bundled_files" / "pyproject.toml"
