import os
from smart_open import smart_open



def view_file(args):
    with smart_open(args.s3_file_uri, 'rb') as f:
        i = 0
        for line in f:
            if i > 50:
                continue

            print(f.readline().decode('utf-8'))
            i += 1


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("s3_file_uri", type=str, help="Path to the dataset root directory.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    view_file(args)


"""
python3 -u random_util_scripts/view_s3_file.py \
s3://kyle-sagemaker-training-outputs/susie_low_level/liberosplit2/gcdiffusion/default2/seed_0/20240507_010012/config.yaml

python3 -u random_util_scripts/view_s3_file.py \
s3://kyle-sagemaker-training-outputs/susie_low_level/liberosplit2/gcdiffusion/default/seed_0/20240301_015613/config.yaml

python3 -u random_util_scripts/view_s3_file.py \
s3://kyle-sagemaker-training-outputs/susie_low_level/liberosplit2/gcdiffusion/auggoaldiff/seed_0/20240228_024329/config.yaml
"""