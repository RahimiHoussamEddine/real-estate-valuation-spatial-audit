"""Build aggregate analysis artifacts from the pinned UCI workbook."""

from __future__ import annotations

import json

from real_estate_audit.constants import RAW_DIR
from real_estate_audit.data import (
    data_quality_report,
    parse_archive,
    write_source_manifest,
)
from real_estate_audit.evaluation import build_report


def main() -> None:
    archive_path = RAW_DIR / "real-estate.zip"
    frame = parse_archive(archive_path)
    quality = data_quality_report(frame, archive_path)
    manifest = write_source_manifest(archive_path, frame)
    summary = build_report(frame, quality)
    print(
        json.dumps(
            {
                "rows": len(frame),
                "source_sha256": manifest["archive_sha256"],
                "report_metrics": len(summary["metrics"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
