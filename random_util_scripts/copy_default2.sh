#!/bin/bash

aws s3 cp --recursive s3://kyle-sagemaker-training-outputs/susie_low_level/calvin/gcdiffusion/default/seed_0/20240507_010213 s3://kyle-sagemaker-training-outputs/susie_low_level/calvin/gcdiffusion/default2/seed_0/20240507_010213
aws s3 cp --recursive s3://kyle-sagemaker-training-outputs/susie_low_level/calvin/gcdiffusion/default/seed_1/20240507_010213 s3://kyle-sagemaker-training-outputs/susie_low_level/calvin/gcdiffusion/default2/seed_1/20240507_010213
aws s3 cp --recursive s3://kyle-sagemaker-training-outputs/susie_low_level/calvin/gcdiffusion/default/seed_2/20240507_010213 s3://kyle-sagemaker-training-outputs/susie_low_level/calvin/gcdiffusion/default2/seed_2/20240507_010213
aws s3 cp --recursive s3://kyle-sagemaker-training-outputs/susie_low_level/calvin/gcdiffusion/default/seed_3/20240507_010213 s3://kyle-sagemaker-training-outputs/susie_low_level/calvin/gcdiffusion/default2/seed_3/20240507_010213


aws s3 cp --recursive s3://kyle-sagemaker-training-outputs/susie_low_level/liberosplit2/gcdiffusion/default/seed_0/20240507_010012 s3://kyle-sagemaker-training-outputs/susie_low_level/liberosplit2/gcdiffusion/default2/seed_0/20240507_010012
aws s3 cp --recursive s3://kyle-sagemaker-training-outputs/susie_low_level/liberosplit2/gcdiffusion/default/seed_1/20240507_010012 s3://kyle-sagemaker-training-outputs/susie_low_level/liberosplit2/gcdiffusion/default2/seed_1/20240507_010012
aws s3 cp --recursive s3://kyle-sagemaker-training-outputs/susie_low_level/liberosplit2/gcdiffusion/default/seed_2/20240507_010012 s3://kyle-sagemaker-training-outputs/susie_low_level/liberosplit2/gcdiffusion/default2/seed_2/20240507_010012
aws s3 cp --recursive s3://kyle-sagemaker-training-outputs/susie_low_level/liberosplit2/gcdiffusion/default/seed_3/20240507_010012 s3://kyle-sagemaker-training-outputs/susie_low_level/liberosplit2/gcdiffusion/default2/seed_3/20240507_010012

echo "Done."