import re
from pathlib import Path
import csv

INPUT_DATASET = Path("data/dataset.tsv")
OUTPUT_TSV = Path("simnibs_results.tsv")
OUTPUT_ROOT = Path("data/derivatives/simnibs_fem")

SITE_MAP = {
    "0001": "M1",
    "0002": "DLPFC",
    "0003": "SMA",
    "0004": "PPC_L",
    "0005": "PPC_R",
}

SITES = list(SITE_MAP.values())


def parse_fields_summary(path: Path) -> dict[str, tuple[str, str]]:
    results = {}
    for block in re.split(r'\n(?=sub-\S+_TMS_1-)', path.read_text()):
        sim_match = re.search(r'TMS_1-(\d{4})', block)
        e999_match = re.search(r'\|E\s+\|\s*([\d.e+\-]+)\s+V/m', block)
        foc50_match = re.search(r'\|E\s+\|[\d.e+\-]+\s+mm[^|]+\|\s*([\d.e+\-]+)\s+mm', block)
        if sim_match and e999_match and foc50_match:
            site = SITE_MAP.get(sim_match.group(1), sim_match.group(1))
            results[site] = (e999_match.group(1), foc50_match.group(1))
    return results


with open(INPUT_DATASET) as f:
    subjects = [row["subject_id"] for row in csv.DictReader(f, delimiter="\t")]

header = ["subject_id"] + [
    f"{site}_{col}"
    for site in SITES
    for col in ("E_999pct_Vm", "focality_50pct_mm3")
]

rows = []
for subject_id in subjects:
    site_data = {}
    for summary_file in sorted((OUTPUT_ROOT / subject_id).glob("**/fields_summary.txt")):
        site_data.update(parse_fields_summary(summary_file))

    row = [subject_id] + [v for site in SITES for v in site_data.get(site, ("", ""))]
    rows.append(row)

with open(OUTPUT_TSV, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(header)
    writer.writerows(rows)

print(f"Wrote {len(rows)} subjects to {OUTPUT_TSV}")
