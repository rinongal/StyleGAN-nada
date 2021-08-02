from cleanfid import fid
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Compute FID between two folders')
    parser.add_argument('--gen_folder', help="Generated image folder.", required=True)
    parser.add_argument('--data_folder', help="Real data folder.", required=True)
    parser.add_argument('--use_kid', help="Use KID instead of FID", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    compute_func = fid.compute_kid if args.use_kid else fid.compute_fid

    score = compute_func(args.gen_folder, args.data_folder)
    score_str = "kid" if args.use_kid else "fid"

    print(f"Computed {score_str} score: {score}")

