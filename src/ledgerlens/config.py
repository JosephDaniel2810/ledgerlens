"""Runtime settings, read from environment variables (and a local .env file)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    sec_user_agent: str
    data_dir: Path

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


def get_settings() -> Settings:
    user_agent = os.getenv("SEC_USER_AGENT", "").strip()
    if not user_agent or "@" not in user_agent:
        raise RuntimeError(
            "SEC_USER_AGENT must be set to '<app> <name> <email>' (SEC policy). "
            "Copy .env.example to .env and fill it in."
        )
    return Settings(
        sec_user_agent=user_agent,
        data_dir=Path(os.getenv("LEDGERLENS_DATA_DIR", "data")),
    )
