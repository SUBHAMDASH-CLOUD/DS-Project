"""
download_dataset.py
-------------------
Downloads CSV files from the CIC IoT Dataset 2023.

The full dataset is hosted at:
    http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/

Usage:
    python data/download_dataset.py

Options:
    --sample     Download only a small sample file for quick testing
    --full       Attempt to download all CSV files (requires stable internet)
"""

import os
import sys
import argparse
import requests
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_URL = "http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/CSV/"
RAW_DIR  = Path(__file__).parent / "raw"

# Known CSV filenames in the dataset (partial list — check the dataset page for full list)
CSV_FILES = [
    "DDoS-ACK_Fragmentation.csv",
    "DDoS-UDP_Flood.csv",
    "DDoS-SlowLoris.csv",
    "DDoS-ICMP_Flood.csv",
    "DDoS-RSTFIN_Flood.csv",
    "DDoS-PSHACK_Flood.csv",
    "DDoS-HTTP_Flood.csv",
    "DDoS-UDP_Fragmentation.csv",
    "DDoS-TCP_Flood.csv",
    "DDoS-SYN_Flood.csv",
    "DDoS-SynonymousIP_Flood.csv",
    "DoS-TCP_Flood.csv",
    "DoS-HTTP_Flood.csv",
    "DoS-SYN_Flood.csv",
    "DoS-UDP_Flood.csv",
    "Recon-PingSweep.csv",
    "Recon-OSScan.csv",
    "Recon-VulScan.csv",
    "Recon-PortScan.csv",
    "Recon-HostDiscovery.csv",
    "SqlInjection.csv",
    "CommandInjection.csv",
    "Backdoor_Malware.csv",
    "Uploading_Attack.csv",
    "XSS.csv",
    "BrowserHijacking.csv",
    "DictionaryBruteForce.csv",
    "ArpSpoofing.csv",
    "DNS_Spoofing.csv",
    "Mirai-greip_flood.csv",
    "Mirai-greeth_flood.csv",
    "Mirai-udpplain.csv",
    "BenignTraffic.csv",
]


def download_file(url: str, dest: Path) -> bool:
    """Download a single file with progress indicator."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            total = int(response.headers.get("content-length", 0))
            dest.parent.mkdir(parents=True, exist_ok=True)
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  ↳ {dest.name}: {pct:5.1f}%", end="", flush=True)
            print(f"\r  ✓ {dest.name} ({downloaded/1024/1024:.1f} MB)       ")
            return True
        else:
            print(f"  ✗ {dest.name}: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"  ✗ {dest.name}: {e}")
        return False


def generate_sample_data():
    """
    Generate synthetic sample data that mirrors the real dataset structure.
    Useful for testing the pipeline without downloading the full dataset.
    """
    import pandas as pd
    import numpy as np

    print("\n📦 Generating synthetic sample data (mirrors real dataset structure)...")

    np.random.seed(42)
    n = 50_000  # 50k rows for a meaningful sample

    LABELS = [
        "DDoS-UDP_Flood", "DDoS-SYN_Flood", "DDoS-HTTP_Flood",
        "DoS-TCP_Flood", "DoS-SYN_Flood",
        "Recon-PortScan", "Recon-OSScan",
        "SqlInjection", "XSS", "CommandInjection",
        "DictionaryBruteForce",
        "ArpSpoofing", "DNS_Spoofing",
        "Mirai-greip_flood", "Mirai-udpplain",
        "BenignTraffic",
    ]

    label_col = np.random.choice(LABELS, size=n, p=[0.1]*10 + [0.05]*4 + [0.1]*2)

    data = {
        "flow_duration":     np.abs(np.random.exponential(5.77, n)),
        "Header_Length":     np.abs(np.random.exponential(76706, n)),
        "Protocol type":     np.random.choice([6, 17, 1, 0, 47], n),
        "Duration":          np.random.choice([0, 64, 128, 255], n),
        "Rate":              np.abs(np.random.exponential(9064, n)),
        "Srate":             np.abs(np.random.exponential(9064, n)),
        "Drate":             np.abs(np.random.exponential(0.000005, n)),
        "fin_flag_number":   np.random.randint(0, 2, n),
        "syn_flag_number":   np.random.randint(0, 2, n),
        "rst_flag_number":   np.random.randint(0, 2, n),
        "psh_flag_number":   np.random.randint(0, 2, n),
        "ack_flag_number":   np.random.randint(0, 2, n),
        "ece_flag_number":   np.random.randint(0, 2, n),
        "cwr_flag_number":   np.random.randint(0, 2, n),
        "ack_count":         np.abs(np.random.exponential(0.09, n)),
        "syn_count":         np.abs(np.random.exponential(0.33, n)),
        "fin_count":         np.abs(np.random.exponential(0.10, n)),
        "urg_count":         np.abs(np.random.exponential(6.24, n)),
        "rst_count":         np.abs(np.random.exponential(38.47, n)),
        "HTTP":              np.random.randint(0, 2, n),
        "HTTPS":             np.random.randint(0, 2, n),
        "DNS":               np.random.randint(0, 2, n),
        "Telnet":            np.random.randint(0, 2, n),
        "SMTP":              np.random.randint(0, 2, n),
        "SSH":               np.random.randint(0, 2, n),
        "IRC":               np.random.randint(0, 2, n),
        "TCP":               np.random.randint(0, 2, n),
        "UDP":               np.random.randint(0, 2, n),
        "DHCP":              np.random.randint(0, 2, n),
        "ARP":               np.random.randint(0, 2, n),
        "ICMP":              np.random.randint(0, 2, n),
        "IPv":               np.random.randint(0, 2, n),
        "LLC":               np.random.randint(0, 2, n),
        "Tot_sum":           np.abs(np.random.normal(1308, 2613, n)),
        "Min":               np.abs(np.random.normal(91, 140, n)),
        "Max":               np.abs(np.random.normal(182, 524, n)),
        "AVG":               np.abs(np.random.normal(125, 241, n)),
        "Std":               np.abs(np.random.normal(33, 160, n)),
        "Tot_size":          np.abs(np.random.normal(125, 242, n)),
        "IAT":               np.abs(np.random.normal(83182526, 17047352, n)),
        "Number":            np.abs(np.random.normal(9.5, 0.82, n)),
        "Magnitue":          np.abs(np.random.normal(13.12, 8.63, n)),
        "Radius":            np.abs(np.random.normal(47.09, 226.77, n)),
        "Covariance":        np.abs(np.random.normal(30724, 323711, n)),
        "Variance":          np.abs(np.random.normal(0.096, 0.233, n)),
        "Weight":            np.abs(np.random.normal(141.5, 21.1, n)),
        "label":             label_col,
    }

    # Inject some NaN and Inf to simulate real-world messiness
    df = pd.DataFrame(data)
    mask = np.random.random(df.shape) < 0.002
    df[df.select_dtypes(include="number").columns] = (
        df.select_dtypes(include="number").mask(
            pd.DataFrame(mask[:, :len(df.select_dtypes(include="number").columns)],
                         columns=df.select_dtypes(include="number").columns)
        )
    )
    # Add some inf values
    for col in ["Rate", "Srate", "Covariance"]:
        idx = np.random.choice(n, 20, replace=False)
        df.loc[idx, col] = np.inf

    output_path = Path(__file__).parent / "sample_data.csv"
    df.to_csv(output_path, index=False)
    print(f"  ✓ Sample data saved → {output_path}  ({n:,} rows × {df.shape[1]} cols)\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="CICIoT2023 Dataset Downloader")
    parser.add_argument("--sample", action="store_true",
                        help="Generate synthetic sample data only (no download)")
    parser.add_argument("--full", action="store_true",
                        help="Download ALL CSV files from the dataset server")
    parser.add_argument("--files", nargs="+", metavar="FILE",
                        help="Download specific CSV files by name")
    args = parser.parse_args()

    print("=" * 60)
    print("  CICIoT2023 Dataset Downloader")
    print("=" * 60)

    if args.sample or (not args.full and not args.files):
        generate_sample_data()
        print("ℹ️  To download real data, run:  python data/download_dataset.py --full")
        return

    files_to_download = args.files if args.files else CSV_FILES
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Saving to: {RAW_DIR}")
    print(f"📋 Files to download: {len(files_to_download)}\n")

    success, failed = 0, []
    for filename in files_to_download:
        url  = BASE_URL + filename
        dest = RAW_DIR / filename
        if dest.exists():
            print(f"  ⏭  {filename} already exists, skipping.")
            success += 1
            continue
        if download_file(url, dest):
            success += 1
        else:
            failed.append(filename)

    print(f"\n{'='*60}")
    print(f"✅ Downloaded: {success}/{len(files_to_download)}")
    if failed:
        print(f"❌ Failed ({len(failed)}): {', '.join(failed)}")
    print("="*60)


if __name__ == "__main__":
    main()
