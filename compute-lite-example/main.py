import pandas as pd
import os
from numerapi import NumerAPI
from lightgbm import LGBMRegressor

if not 'NUMERAI_PUBLIC_ID' in os.environ:
  raise Exception('missing NUMERAI_PUBLIC_ID')
if not 'NUMERAI_SECRET_KEY' in os.environ:
  raise Exception('missing NUMERAI_SECRET_KEY')
if not 'AWS_ACCESS_KEY_ID' in os.environ:
  raise Exception('missing AWS_ACCESS_KEY_ID')
if not 'AWS_SECRET_ACCESS_KEY' in os.environ:
  raise Exception('missing AWS_SECRET_ACCESS_KEY')

napi = NumerAPI()
napi.download_dataset("v4/train.parquet")
training_data = pd.read_parquet('v4/train.parquet')

model = LGBMRegressor()
model.fit(
   training_data[napi.feature_sets('small')],
   training_data['target']
)

model_id = 'dcbce83a-956a-43a1-9961-0fb9c5828ac9'
napi.deploy(model_id, model, napi.feature_sets('small'), 'requirements.txt')
