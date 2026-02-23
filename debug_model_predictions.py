#!/usr/bin/env python3
"""
Debug script: run predictions with all candidate models for a given sequence length
Usage:
  python3 debug_model_predictions.py --video /path/to/video.mp4 --seq 40

This script loads Django settings, imports the model/dataset code from `ml_app.views`,
creates one validation_dataset sample for the video, finds all models in `models/`
whose filenames contain the numeric token equal to `seq`, and runs prediction with each.
It prints each model path, predicted class, confidence, and raw probabilities.

Run this from the project root.
"""

import os
import sys
import argparse
import glob
import torch

# Ensure project root is on path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Set Django settings module then setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project_settings.settings')
try:
    import django
    django.setup()
except Exception as e:
    print('Warning: django.setup() failed or not required:', e)

from ml_app.views import Model, validation_dataset, train_transforms, get_accurate_model, predict, sm
from django.conf import settings

parser = argparse.ArgumentParser()
parser.add_argument('--video', required=True, help='Path to video file')
parser.add_argument('--seq', type=int, required=True, help='Sequence length to test (e.g. 40)')
parser.add_argument('--device', default=None, help='Device to run on: cpu or cuda (optional)')
args = parser.parse_args()

# Resolve video path: accept absolute, project-root-relative, or ~ expansion
raw_video = args.video
video_path = os.path.expanduser(raw_video)
if not os.path.isabs(video_path):
    # try relative to project dir
    candidate = os.path.join(PROJECT_DIR, raw_video)
    if os.path.exists(candidate):
        video_path = candidate
    else:
        # fallback to abspath of given relative path
        video_path = os.path.abspath(raw_video)

print('Resolved video path:', video_path)
seq = args.seq

if args.device:
    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using device: {device}')

# Quick check: does opencv read frames from this path?
import cv2
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"OpenCV: Couldn't read video stream from file \"{video_path}\"")
    cap.release()
    sys.exit(1)

# Create dataset and sample
dataset = validation_dataset([video_path], sequence_length=seq, transform=train_transforms)
try:
    sample = dataset[0]
except Exception as e:
    print('Error creating dataset sample:', e)
    cap.release()
    sys.exit(1)

cap.release()

print('Sample shape (should be [1, seq, 3, 112, 112]):', getattr(sample, 'shape', None))

# Find candidate models that include the seq token
models_glob = glob.glob(os.path.join(settings.PROJECT_DIR, 'models', '*.pt'))
candidates = []
for p in models_glob:
    fname = os.path.basename(p)
    parts = fname.split('_')
    for token in parts:
        try:
            if int(token) == int(seq):
                candidates.append(p)
                break
        except Exception:
            continue

if not candidates:
    print('No candidate models found for sequence length', seq)
    print('Available models:', models_glob)
    sys.exit(1)

print('Found candidate models:')
for m in candidates:
    print(' -', m)

# Run prediction across candidates
for model_path in candidates:
    print('\n---')
    print('Model:', model_path)
    model = Model(2)
    model.to(device)
    try:
        print('Loading checkpoint...', model_path, flush=True)
        state = torch.load(model_path, map_location=device)
        print('Checkpoint loaded, type=', type(state), flush=True)

        # If state is a dict with 'model_state_dict' or similar, try to extract
        if isinstance(state, dict):
            # common keys: 'model_state_dict' or direct state_dict
            if 'model_state_dict' in state:
                state_dict = state['model_state_dict']
            elif 'state_dict' in state:
                state_dict = state['state_dict']
            else:
                # assume it's already a state_dict
                state_dict = state
        else:
            state_dict = state

        # Try loading state_dict; if keys are prefixed (e.g., 'module.'), strip prefixes and retry
        try:
            print('Attempting model.load_state_dict(...) with', len(state_dict) if hasattr(state_dict,'keys') else 'unknown', 'keys', flush=True)
            # print first few keys for diagnostics
            try:
                klist = list(state_dict.keys())[:10]
                print('state_dict keys sample:', klist, flush=True)
            except Exception:
                pass
            model.load_state_dict(state_dict)
        except Exception as load_exc:
            # attempt to normalize keys by removing common prefixes
            def strip_prefixes(sd, prefixes=('module.', 'model.')):
                new_sd = {}
                for k, v in sd.items():
                    new_k = k
                    for p in prefixes:
                        if new_k.startswith(p):
                            new_k = new_k[len(p):]
                    new_sd[new_k] = v
                return new_sd

            try:
                normalized = strip_prefixes(state_dict)
                print('Retrying with stripped prefixes, keys sample:', list(normalized.keys())[:10], flush=True)
                model.load_state_dict(normalized)
                print('Loaded state_dict after stripping common prefixes (module./model.).', flush=True)
            except Exception:
                # re-raise original exception to be printed by outer handler
                raise load_exc

        model.to(device)
        model.eval()
    except Exception as e:
        import traceback
        print('Failed to load model:', e)
        traceback.print_exc()
        continue

    # ensure sample on device
    inp = sample.to(device)
    with torch.no_grad():
        try:
            fmap, logits = model(inp)
            probs = sm(logits)
            _, pred = torch.max(probs, 1)
            pred_item = int(pred.item())
            confidence = float(probs[:, pred_item].item() * 100.0)
            print('Predicted class index:', pred_item, '(1==REAL, 0==FAKE)')
            print('Confidence:', confidence)
            print('Raw probs:', probs.detach().cpu().numpy())
        except Exception as e:
            import traceback
            print('Inference failed for model:', model_path)
            print('Error:', e)
            traceback.print_exc()
            continue

print('\nDone')
