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
    database_url: str
    embedding_model: str
    reranker_model: str

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def chunks_path(self) -> Path:
        return self.processed_dir / "chunks.jsonl"

    def require_sec_user_agent(self) -> str:
        if not self.sec_user_agent or "@" not in self.sec_user_agent:
            raise RuntimeError(
                "SEC_USER_AGENT must be set to '<app> <name> <email>' (SEC policy). "
                "Copy .env.example to .env and fill it in."
            )
        return self.sec_user_agent


def get_settings() -> Settings:
    return Settings(
        sec_user_agent=os.getenv("SEC_USER_AGENT", "").strip(),
        data_dir=Path(os.getenv("LEDGERLENS_DATA_DIR", "data")),
        database_url=os.getenv(
            "DATABASE_URL", "postgresql://ledgerlens:ledgerlens@localhost:5433/ledgerlens"
        ),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        reranker_model=os.getenv("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"),
    )
