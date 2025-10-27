
from __future__ import annotations
import argparse
from .runner import run_benchmark

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory of JSON samples')
    parser.add_argument('--models', type=str, required=True, help='Comma-separated model aliases')
    parser.add_argument('--modalities', type=str, default='all', help='Comma list from {url,html,image,all}')
    parser.add_argument('--max_samples', type=int, default=0, help='0 means all')
    parser.add_argument('--out_dir', type=str, default='./benchmark_out')
    args = parser.parse_args()

    model_aliases = [s.strip() for s in args.models.split(',') if s.strip()]
    modalities = [s.strip().lower() for s in args.modalities.split(',') if s.strip()]

    run_benchmark(args.data_dir, model_aliases, modalities, args.max_samples, args.out_dir)

if __name__ == "__main__":
    main()
