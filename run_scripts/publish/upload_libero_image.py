"""Upload the LIBERO image inference checkpoint without exposing credentials."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--token-file', type=Path, required=True)
p.add_argument('--repo-id', default='prehj/RLDX-1-IMG-LIBERO-60k')
a = p.parse_args()
state = json.loads((a.checkpoint / 'trainer_state.json').read_text())
assert state['global_step'] == 60000
index = json.loads((a.checkpoint / 'model.safetensors.index.json').read_text())
assert all((a.checkpoint / x).is_file() for x in set(index['weight_map'].values()))
api = HfApi(token=a.token_file.read_text().strip())
api.create_repo(a.repo_id, repo_type='model', private=True, exist_ok=True)
card = '''---
license: other
license_name: rlwrld-model-license-v1.0
license_link: LICENSE.md
base_model: RLWRLD/RLDX-1-PT-IMG
tags: [robotics, libero, rldx]
---
# RLDX-1 image LIBERO baseline (60k)

Fine-tuned from RLWRLD/RLDX-1-PT-IMG for 60,000 optimizer updates on
kimtaey/libero_gr00t_delta. Global batch 32, action horizon 16, 64 cognition
tokens, current-frame image inputs. Physics, memory, motion and RTC are off.
Top four LLM layers, cognition embeddings and action head follow the training
recipe in https://github.com/rakybond007/RLDX-1.

Evaluation uses replan_steps=5: execute the first five predicted actions, then
request a fresh chunk. Evaluation results are pending; no success rate is claimed.

This repository contains inference weights, processor/statistics, configuration
and training metadata. DeepSpeed optimizer states are retained in the original
training storage and are not included here. Use the RLDX-1 evaluation server
with this repository downloaded as a local model path.
'''
api.upload_file(repo_id=a.repo_id, path_or_fileobj=card.encode(), path_in_repo='README.md')
license_path = Path(__file__).resolve().parents[2] / 'models/RLDX-1-PT-IMG/LICENSE.md'
api.upload_file(repo_id=a.repo_id, path_or_fileobj=str(license_path), path_in_repo='LICENSE.md')
api.upload_folder(repo_id=a.repo_id, folder_path=str(a.checkpoint),
                  allow_patterns=['config.json', 'model*.safetensors', 'model.safetensors.index.json',
                                  'processor/**', 'experiment_cfg/**', 'trainer_state.json'],
                  commit_message='Upload LIBERO image baseline checkpoint at step 60000')
files = set(api.list_repo_files(a.repo_id))
required = set(index['weight_map'].values()) | {'config.json', 'model.safetensors.index.json', 'README.md', 'LICENSE.md'}
assert required <= files, required - files
print('UPLOAD_VERIFIED https://huggingface.co/' + a.repo_id, flush=True)
