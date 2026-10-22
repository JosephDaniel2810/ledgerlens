"""Thin client for SEC EDGAR: ticker lookup, filing index, and document download.

EDGAR rules we respect:
  * every request carries a descriptive User-Agent with a contact email
  * at most 10 requests per second (we stay well under that)
  * downloaded documents are cached on disk so reruns never re-hit SEC
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_FILE_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

MIN_REQUEST_INTERVAL = 0.15  # seconds; ~6 req/s, under SEC's 10 req/s limit


@dataclass(frozen=True)
class Filing:
    ticker: str
    cik: int
    company: str
    form: str
    accession: str
    filing_date: str
    report_date: str
    primary_document: str

    @property
    def fiscal_year(self) -> int:
        return int(self.report_date[:4])

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik=self.cik,
            accession=self.accession.replace("-", ""),
            document=self.primary_document,
        )

    def to_dict(self) -> dict:
        return asdict(self) | {"fiscal_year": self.fiscal_year, "url": self.url}


class EdgarClient:
    def __init__(self, user_agent: str, cache_dir: Path):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.cache_dir = cache_dir
        self._last_request = 0.0
        self._tickers: dict[str, dict] | None = None

    def _get(self, url: str) -> requests.Response:
        wait = MIN_REQUEST_INTERVAL - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(3):
            resp = self.session.get(url, timeout=30)
            self._last_request = time.monotonic()
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2**attempt)
                continue
            resp.raise_for_status()
            return resp
        resp.raise_for_status()
        return resp

    def lookup(self, ticker: str) -> dict:
        """Return {'cik_str', 'ticker', 'title'} for a ticker."""
        if self._tickers is None:
            data = self._get(TICKERS_URL).json()
            self._tickers = {row["ticker"].upper(): row for row in data.values()}
        try:
            return self._tickers[ticker.upper()]
        except KeyError:
            raise ValueError(f"Unknown ticker: {ticker}") from None

    def list_filings(self, ticker: str, forms: tuple[str, ...] = ("10-K",), limit: int = 3) -> list[Filing]:
        """Most recent filings of the given form types, newest first."""
        company = self.lookup(ticker)
        cik = int(company["cik_str"])
        submissions = self._get(SUBMISSIONS_URL.format(cik=cik)).json()

        # "recent" holds the latest ~1000 filings; frequent filers (e.g. banks)
        # need the older paginated index files to reach earlier annual reports.
        pages = [submissions["filings"]["recent"]]
        older = [f["name"] for f in submissions["filings"].get("files", [])]

        filings: list[Filing] = []
        while pages and len(filings) < limit:
            page = pages.pop(0)
            for i, form in enumerate(page["form"]):
                if form not in forms or not page["primaryDocument"][i]:
                    continue
                filings.append(
                    Filing(
                        ticker=ticker.upper(),
                        cik=cik,
                        company=company["title"],
                        form=form,
                        accession=page["accessionNumber"][i],
                        filing_date=page["filingDate"][i],
                        report_date=page["reportDate"][i] or page["filingDate"][i],
                        primary_document=page["primaryDocument"][i],
                    )
                )
                if len(filings) == limit:
                    break
            if not pages and older and len(filings) < limit:
                pages.append(self._get(SUBMISSIONS_FILE_URL.format(name=older.pop(0))).json())
        return filings

    def download(self, filing: Filing) -> Path:
        """Download the filing's primary HTML document (cached) and write its metadata."""
        folder = self.cache_dir / filing.ticker / filing.accession
        doc_path = folder / filing.primary_document
        if not doc_path.exists():
            folder.mkdir(parents=True, exist_ok=True)
            doc_path.write_bytes(self._get(filing.url).content)
            (folder / "metadata.json").write_text(json.dumps(filing.to_dict(), indent=2))
        return doc_path
