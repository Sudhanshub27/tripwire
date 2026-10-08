"""
================================================================================
NEWS & JOBS FRESHNESS SCRAPING PIPELINE (24-Hour Guarantee)
================================================================================

This module orchestrates data ingestion for News and Job boards with a strict
24-hour freshness guarantee, deduplication tracking, and date normalization.

Sources Covered:
  - News (5 Sources):
      1. Hacker News API (Official Firebase REST API)
      2. TechCrunch AI RSS Feed
      3. VentureBeat AI RSS Feed
      4. MIT Technology Review AI RSS Feed
      5. ArXiv AI Preprints Feed
  - Jobs (5 Sources):
      1. RemoteOK API (Official REST API)
      2. WeWorkRemotely RSS Feed
      3. Remotive API (Official REST API)
      4. Jobspresso RSS Feed
      5. Working Nomads API

Features:
  - Trafilatura full-text article extraction (stripping nav, ads, boilerplate).
  - Date normalization with relative date parsing & HTML/last-run heuristics.
  - 24-hour freshness filtering with audit logging.
  - Persistent SQLite deduplication tracking (cross-run avoidance).
  - Pydantic schema validation & JSONL export to data/processed/news.jsonl & jobs.jsonl.
================================================================================
"""

import argparse
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import feedparser
import structlog
import trafilatura
from config.settings import settings

from src.schemas.base import SourceMetadata
from src.schemas.job import Job
from src.schemas.news import News
from src.scrapers.base import AsyncScraper
from src.scrapers.date_parser import normalize_date, update_last_run_timestamp
from src.scrapers.dedup_tracker import DedupTracker
from src.scrapers.freshness_filter import FreshnessFilter

logger = structlog.get_logger(__name__)


class FreshnessPipelineScraper(AsyncScraper):
    """Pipeline scraper handling 5 News and 5 Job feeds with full-text parsing."""

    def __init__(self, max_concurrency: Optional[int] = None):
        super().__init__(max_concurrency=max_concurrency)
        self.dedup = DedupTracker()
        self.freshness_filter = FreshnessFilter(max_age_hours=24)

    async def fetch_full_text_trafilatura(self, url: str) -> Optional[str]:
        """Fetch article web page and extract clean full-text body using trafilatura."""
        if not url:
            return None
        try:
            html = await self.fetch(url)
            text = trafilatura.extract(html, include_links=False, include_images=False)
            return text
        except Exception as e:
            logger.debug("Trafilatura full text extraction failed", url=url, error=str(e))
            return None

    # --------------------------------------------------------------------------
    # NEWS SOURCE CRAWLERS (5 Sources)
    # --------------------------------------------------------------------------

    async def crawl_hacker_news(self, limit: int = 15) -> List[tuple[News, datetime]]:
        """Crawl top story items from Hacker News API."""
        url = "https://hacker-news.firebaseio.com/v0/topstories.json"
        logger.info("Crawling Hacker News API", url=url)
        results = []
        try:
            session = await self.get_session()
            async with session.get(url) as resp:
                if resp.status == 200:
                    story_ids = await resp.json()
                    for sid in story_ids[:limit]:
                        item_url = f"https://hacker-news.firebaseio.com/v0/item/{sid}.json"
                        async with session.get(item_url) as i_resp:
                            if i_resp.status == 200:
                                item = await i_resp.json()
                                if not item or item.get("type") != "story":
                                    continue
                                story_link = item.get("url") or f"https://news.ycombinator.com/item?id={sid}"

                                title = item.get("title", "HN Story")
                                inc_res = self.dedup.check_incremental(story_link, content=title)
                                if inc_res.action == "SKIP":
                                    continue

                                pub_timestamp = item.get("time")
                                pub_dt = (
                                    datetime.fromtimestamp(pub_timestamp, tz=timezone.utc)
                                    if pub_timestamp
                                    else normalize_date(None, record_id=f"hn_{sid}")
                                )

                                news_obj = News(
                                    title=title,
                                    summary=title,
                                    content=None,
                                    author=item.get("by"),
                                    published_at=pub_dt.isoformat(),
                                    categories_tags=["Hacker News", "Tech"],
                                    sentiment_score=0.75,
                                    source=SourceMetadata(name="Hacker News API", url=story_link),
                                )
                                self.dedup.save_record(story_link, content=title, source="Hacker News API")
                                results.append((news_obj, pub_dt))
        except Exception as e:
            logger.error("Hacker News API crawl failed", error=str(e))
        return results

    async def crawl_rss_news_source(
        self, source_name: str, feed_url: str, limit: int = 15
    ) -> List[tuple[News, datetime]]:
        """Generic RSS news feed crawler using feedparser and trafilatura full-text extraction."""
        logger.info("Crawling news RSS feed", source=source_name, feed_url=feed_url)
        results = []
        try:
            xml_text = await self.fetch(feed_url)
            feed = feedparser.parse(xml_text)

            for entry in feed.entries[:limit]:
                link = entry.get("link")
                if not link:
                    continue

                title = entry.get("title", "Untitled").strip()
                summary = entry.get("summary", title).strip()

                raw_pub_date = entry.get("published") or entry.get("updated")
                pub_dt = normalize_date(raw_pub_date, record_id=title[:20])

                # Extract full text using trafilatura
                full_text = await self.fetch_full_text_trafilatura(link)
                article_content = full_text or summary

                inc_res = self.dedup.check_incremental(link, content=article_content)
                if inc_res.action == "SKIP":
                    continue

                news_obj = News(
                    title=title,
                    summary=summary[:500],
                    content=article_content,
                    author=entry.get("author"),
                    published_at=pub_dt.isoformat(),
                    categories_tags=[source_name, "AI News"],
                    sentiment_score=0.8,
                    source=SourceMetadata(name=source_name, url=link),
                )
                self.dedup.save_record(link, content=article_content, source=source_name)
                results.append((news_obj, pub_dt))

        except Exception as e:
            logger.error("RSS News crawl failed", source=source_name, error=str(e))
        return results

    async def crawl_all_news(self, limit_per_source: int = 10) -> List[News]:
        """Crawl all 5 news sources concurrently."""
        news_sources = [
            ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/"),
            ("VentureBeat AI", "https://venturebeat.com/category/ai/feed/"),
            ("MIT Tech Review AI", "https://www.technologyreview.com/topic/artificial-intelligence/feed/"),
            ("ArXiv AI News", "https://export.arxiv.org/rss/cs.AI"),
        ]

        tasks = [self.crawl_hacker_news(limit=limit_per_source)]
        for name, feed_url in news_sources:
            tasks.append(self.crawl_rss_news_source(name, feed_url, limit=limit_per_source))

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        all_news_tuples: List[tuple[News, datetime]] = []

        for res in raw_results:
            if isinstance(res, list):
                all_news_tuples.extend(res)

        # Apply 24-hour freshness filter
        filtered_news = self.freshness_filter.filter_records(all_news_tuples)
        logger.info("News crawling completed", total_fresh_records=len(filtered_news))
        return filtered_news

    # --------------------------------------------------------------------------
    # JOB BOARD CRAWLERS (5 Sources)
    # --------------------------------------------------------------------------

    async def crawl_remoteok_jobs(self, limit: int = 15) -> List[tuple[Job, datetime]]:
        """Crawl AI/Developer job listings from RemoteOK API."""
        url = "https://remoteok.com/api"
        logger.info("Crawling RemoteOK Jobs API", url=url)
        results = []
        try:
            session = await self.get_session()
            headers = {"User-Agent": self.get_random_user_agent()}
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, list):
                        for job_item in data[1 : limit + 1]:  # Index 0 is legal disclaimer
                            if not isinstance(job_item, dict):
                                continue
                            job_url = job_item.get("url") or f"https://remoteok.com/remote-jobs/{job_item.get('id')}"
                            title = job_item.get("position", "Developer").strip()
                            company = job_item.get("company", "Tech Company").strip()
                            job_content = f"{title} {company} {job_item.get('description', '')}"

                            inc_res = self.dedup.check_incremental(job_url, content=job_content)
                            if inc_res.action == "SKIP":
                                continue

                            pub_date = job_item.get("date")
                            pub_dt = normalize_date(pub_date, record_id=f"{company}_{title}")

                            salary = ""
                            if job_item.get("salary_min") and job_item.get("salary_max"):
                                salary = f"${job_item.get('salary_min'):,} - ${job_item.get('salary_max'):,}"

                            job_obj = Job(
                                title=title,
                                company=company,
                                location=job_item.get("location") or "Remote",
                                job_type="Full-time",
                                salary_range=salary or None,
                                description=job_item.get("description"),
                                requirements=job_item.get("tags") or [],
                                posted_date=pub_dt.isoformat(),
                                apply_url=job_url,
                                source=SourceMetadata(name="RemoteOK API", url=job_url),
                            )
                            self.dedup.save_record(job_url, content=job_content, source="RemoteOK API")
                            results.append((job_obj, pub_dt))
        except Exception as e:
            logger.error("RemoteOK API crawl failed", error=str(e))
        return results

    async def crawl_remotive_jobs(self, limit: int = 15) -> List[tuple[Job, datetime]]:
        """Crawl developer jobs from Remotive API."""
        url = "https://remotive.com/api/remote-jobs?category=software-dev"
        logger.info("Crawling Remotive Jobs API", url=url)
        results = []
        try:
            session = await self.get_session()
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    jobs = data.get("jobs", [])
                    for job_item in jobs[:limit]:
                        job_url = job_item.get("url")
                        if not job_url:
                            continue

                        title = job_item.get("title", "Developer")
                        company = job_item.get("company_name", "Remote Company")
                        job_content = f"{title} {company} {job_item.get('description', '')}"

                        inc_res = self.dedup.check_incremental(job_url, content=job_content)
                        if inc_res.action == "SKIP":
                            continue

                        pub_date = job_item.get("publication_date")
                        pub_dt = normalize_date(pub_date, record_id=f"rem_{title}")

                        job_obj = Job(
                            title=title,
                            company=company,
                            location=job_item.get("candidate_required_location") or "Remote",
                            job_type=job_item.get("job_type") or "Full-time",
                            salary_range=job_item.get("salary") or None,
                            description=job_item.get("description"),
                            requirements=job_item.get("tags") or [],
                            posted_date=pub_dt.isoformat(),
                            apply_url=job_url,
                            source=SourceMetadata(name="Remotive API", url=job_url),
                        )
                        self.dedup.save_record(job_url, content=job_content, source="Remotive API")
                        results.append((job_obj, pub_dt))
        except Exception as e:
            logger.error("Remotive Jobs crawl failed", error=str(e))
        return results

    async def crawl_rss_job_board(
        self, source_name: str, feed_url: str, limit: int = 15
    ) -> List[tuple[Job, datetime]]:
        """Generic Job Board RSS feed crawler."""
        logger.info("Crawling Job Board RSS feed", source=source_name, url=feed_url)
        results = []
        try:
            xml_text = await self.fetch(feed_url)
            feed = feedparser.parse(xml_text)

            for entry in feed.entries[:limit]:
                link = entry.get("link")
                if not link:
                    continue

                raw_title = entry.get("title", "Software Engineer").strip()
                company = "Remote Tech"
                title = raw_title
                if " at " in raw_title:
                    parts = raw_title.split(" at ", 1)
                    title, company = parts[0].strip(), parts[1].strip()
                elif ":" in raw_title:
                    parts = raw_title.split(":", 1)
                    company, title = parts[0].strip(), parts[1].strip()

                job_content = f"{title} {company} {entry.get('summary', '')}"
                inc_res = self.dedup.check_incremental(link, content=job_content)
                if inc_res.action == "SKIP":
                    continue

                pub_date = entry.get("published") or entry.get("updated")
                pub_dt = normalize_date(pub_date, record_id=title[:20])

                job_obj = Job(
                    title=title,
                    company=company,
                    location="Remote",
                    job_type="Full-time",
                    salary_range=None,
                    description=entry.get("summary"),
                    requirements=["Software Engineering"],
                    posted_date=pub_dt.isoformat(),
                    apply_url=link,
                    source=SourceMetadata(name=source_name, url=link),
                )
                self.dedup.save_record(link, content=job_content, source=source_name)
                results.append((job_obj, pub_dt))


        except Exception as e:
            logger.error("RSS Job board crawl failed", source=source_name, error=str(e))
        return results

    async def crawl_all_jobs(self, limit_per_source: int = 10) -> List[Job]:
        """Crawl all 5 job boards concurrently."""
        rss_job_boards = [
            ("WeWorkRemotely", "https://weworkremotely.com/categories/remote-full-stack-programming-jobs.rss"),
            ("Jobspresso", "https://jobspresso.co/feed/"),
            ("Working Nomads", "https://www.workingnomads.com/jobs?category=development&rss=1"),
        ]

        tasks = [
            self.crawl_remoteok_jobs(limit=limit_per_source),
            self.crawl_remotive_jobs(limit=limit_per_source),
        ]
        for name, feed_url in rss_job_boards:
            tasks.append(self.crawl_rss_job_board(name, feed_url, limit=limit_per_source))

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        all_job_tuples: List[tuple[Job, datetime]] = []

        for res in raw_results:
            if isinstance(res, list):
                all_job_tuples.extend(res)

        # Apply 24-hour freshness filter
        filtered_jobs = self.freshness_filter.filter_records(all_job_tuples)
        logger.info("Job crawling completed", total_fresh_records=len(filtered_jobs))
        return filtered_jobs

    async def scrape(self) -> Dict[str, Any]:
        """Placeholder for AsyncScraper abstract method."""
        return {}


async def run_news_freshness_pipeline(limit_per_source: int = 10) -> Dict[str, Any]:
    """Execute complete 24-hour freshness scraping pipeline for News only."""
    settings.setup_directories()
    logger.info("Initiating News freshness scraping pipeline", limit_per_source=limit_per_source)
    scraper = FreshnessPipelineScraper()
    try:
        news_records = await scraper.crawl_all_news(limit_per_source=limit_per_source)
        news_file = settings.DATA_PROCESSED_DIR / "news.jsonl"
        with open(news_file, "w", encoding="utf-8") as f:
            for item in news_records:
                f.write(item.model_dump_json() + "\n")
        update_last_run_timestamp()
        summary = {"news_written": len(news_records), "news_file": str(news_file)}
        logger.info("News freshness pipeline completed successfully", **summary)
        return summary
    finally:
        await scraper.close()


async def run_jobs_freshness_pipeline(limit_per_source: int = 10) -> Dict[str, Any]:
    """Execute complete 24-hour freshness scraping pipeline for Jobs only."""
    settings.setup_directories()
    logger.info("Initiating Jobs freshness scraping pipeline", limit_per_source=limit_per_source)
    scraper = FreshnessPipelineScraper()
    try:
        job_records = await scraper.crawl_all_jobs(limit_per_source=limit_per_source)
        jobs_file = settings.DATA_PROCESSED_DIR / "jobs.jsonl"
        with open(jobs_file, "w", encoding="utf-8") as f:
            for item in job_records:
                f.write(item.model_dump_json() + "\n")
        update_last_run_timestamp()
        summary = {"jobs_written": len(job_records), "jobs_file": str(jobs_file)}
        logger.info("Jobs freshness pipeline completed successfully", **summary)
        return summary
    finally:
        await scraper.close()


async def run_freshness_pipeline(limit_per_source: int = 10) -> Dict[str, int]:
    """Execute complete 24-hour freshness scraping pipeline for News and Jobs."""
    settings.setup_directories()
    logger.info("Initiating News and Jobs freshness scraping pipeline", limit_per_source=limit_per_source)

    scraper = FreshnessPipelineScraper()
    try:
        news_task = scraper.crawl_all_news(limit_per_source=limit_per_source)
        jobs_task = scraper.crawl_all_jobs(limit_per_source=limit_per_source)

        news_records, job_records = await asyncio.gather(news_task, jobs_task)

        # Export News to data/processed/news.jsonl
        news_file = settings.DATA_PROCESSED_DIR / "news.jsonl"
        with open(news_file, "w", encoding="utf-8") as f:
            for item in news_records:
                f.write(item.model_dump_json() + "\n")

        # Export Jobs to data/processed/jobs.jsonl
        jobs_file = settings.DATA_PROCESSED_DIR / "jobs.jsonl"
        with open(jobs_file, "w", encoding="utf-8") as f:
            for item in job_records:
                f.write(item.model_dump_json() + "\n")

        # Record successful pipeline run execution timestamp
        update_last_run_timestamp()

        summary = {
            "news_written": len(news_records),
            "jobs_written": len(job_records),
            "news_file": str(news_file),
            "jobs_file": str(jobs_file),
        }
        logger.info("Freshness pipeline completed successfully", **summary)
        return summary

    finally:
        await scraper.close()


def main():
    parser = argparse.ArgumentParser(description="Run 24-hour Freshness Pipeline for News and Jobs.")
    parser.add_argument(
        "--limit-per-source",
        type=int,
        default=10,
        help="Max items to fetch per news/job source (default: 10)",
    )
    args = parser.parse_args()

    asyncio.run(run_freshness_pipeline(limit_per_source=args.limit_per_source))


if __name__ == "__main__":
    main()
