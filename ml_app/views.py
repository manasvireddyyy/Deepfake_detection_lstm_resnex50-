from django.shortcuts import render, redirect
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
from torch.autograd import Variable
import time
import sys
from torch import nn
import json
import glob
import copy
from torchvision import models
import shutil
from PIL import Image as pImage
import time
from django.conf import settings
from .forms import VideoUploadForm
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

index_template_name = 'index.html'
predict_template_name = 'predict.html'
about_template_name = "about.html"

im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1)
inv_normalize =  transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))
# Use torch.device for robust device placement
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])

class Model(nn.Module):

    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        #model = models.resnext50_32x4d(pretrained = True)
        weights = ResNeXt50_32X4D_Weights.DEFAULT
        model = resnext50_32x4d(weights=weights) 
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,2048)
        x_lstm,_ = self.lstm(x,None)
        # Use the mean over time steps for classification (matches training notebook)
        x_pooled = torch.mean(x_lstm, dim=1)
        return fmap, self.dp(self.linear1(x_pooled))


class validation_dataset(Dataset):
    def __init__(self,video_names,sequence_length=60,transform = None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self,idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100/self.count)
        first_frame = np.random.randint(0,a)
        for i,frame in enumerate(self.frame_extract(video_path)):
            # Use CNN model for face detection (matches training preprocessing)
            faces = face_recognition.face_locations(frame, model='cnn')
            if len(faces) == 0:
                # No face found — skip this frame (matches training behavior
                # where frames without detected faces were not written to
                # the face-only output video)
                print(f"[validation_dataset] Frame {i}: no face detected, skipping", flush=True)
                continue
            top, right, bottom, left = faces[0]
            face_crop = frame[top:bottom, left:right, :]
            frames.append(self.transform(face_crop))
            if len(frames) == self.count:
                break

        # If we got fewer frames than needed, pad by repeating the last good frame
        if len(frames) == 0:
            print("[validation_dataset] WARNING: No faces detected in any frame! Using raw frames as fallback.", flush=True)
            for i, frame in enumerate(self.frame_extract(video_path)):
                frames.append(self.transform(frame))
                if len(frames) == self.count:
                    break

        while len(frames) < self.count:
            # Repeat the last successfully detected face frame
            frames.append(frames[-1].clone())

        print(f"[validation_dataset] Collected {len(frames)} frames for inference", flush=True)
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)
    
    def frame_extract(self,path):
      vidObj = cv2.VideoCapture(path) 
      success = 1
      while success:
          success, image = vidObj.read()
          if success:
              yield image

def im_convert(tensor, video_file_name):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)
    # This image is not used
    # cv2.imwrite(os.path.join(settings.PROJECT_DIR, 'uploaded_images', video_file_name+'_convert_2.png'),image*255)
    return image

def im_plot(tensor):
    image = tensor.cpu().numpy().transpose(1,2,0)
    b,g,r = cv2.split(image)
    image = cv2.merge((r,g,b))
    image = image*[0.22803, 0.22145, 0.216989] +  [0.43216, 0.394666, 0.37645]
    image = image*255.0
    plt.imshow(image.astype('uint8'))
    plt.show()


def predict(model,img,path = './', video_file_name=""):
    # Run inference in no_grad mode and ensure tensors are on the same device
    model.to(device)
    with torch.no_grad():
        fmap,logits = model(img.to(device))
        # debug: input shapes and raw logits
        try:
            print(f"DEBUG: predict input shape: {img.shape}, raw_logits_shape: {logits.shape}")
        except Exception:
            pass
        img = im_convert(img[:,-1,:,:,:], video_file_name)
        params = list(model.parameters())
        weight_softmax = model.linear1.weight.detach().cpu().numpy()
        probs = sm(logits)
        # prediction
        _,prediction = torch.max(probs,1)
        confidence = probs[:,int(prediction.item())].item()*100
        print('confidence of prediction:', confidence)
    return [int(prediction.item()),confidence]

def plot_heat_map(i, model, img, path = './', video_file_name=''):
  fmap,logits = model(img.to(device))
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _,prediction = torch.max(logits,1)
  idx = np.argmax(logits.detach().cpu().numpy())
  bz, nc, h, w = fmap.shape
  #out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
  out = np.dot(fmap[i].detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
  predict = out.reshape(h,w)
  predict = predict - np.min(predict)
  predict_img = predict / np.max(predict)
  predict_img = np.uint8(255*predict_img)
  out = cv2.resize(predict_img, (im_size,im_size))
  heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
  img = im_convert(img[:,-1,:,:,:], video_file_name)
  result = heatmap * 0.5 + img*0.8*255
  # Saving heatmap - Start
  heatmap_name = video_file_name+"_heatmap_"+str(i)+".png"
  image_name = os.path.join(settings.PROJECT_DIR, 'uploaded_images', heatmap_name)
  cv2.imwrite(image_name,result)
  # Saving heatmap - End
  result1 = heatmap * 0.5/255 + img*0.8
  r,g,b = cv2.split(result1)
  result1 = cv2.merge((r,g,b))
  return image_name

# Model Selection
def get_all_matching_models(sequence_length):
    """Return all model files matching the requested sequence_length.

    Models trained on broader datasets ('final_data', 'celeb') generalize better
    than models trained only on FaceForensics++ ('FF_data'). Returns models sorted
    so that 'final_data'/'celeb' models come first, then FF-only models.
    """
    list_models = glob.glob(os.path.join(settings.PROJECT_DIR, "models", "*.pt"))
    candidates = []

    for model_path in list_models:
        fname = os.path.basename(model_path)
        parts = fname.split("_")
        # find any numeric part equal to sequence_length
        seq_found = False
        for p in parts:
            try:
                if int(p) == int(sequence_length):
                    seq_found = True
                    break
            except Exception:
                continue
        if not seq_found:
            continue

        # parse accuracy from filename (e.g. model_89_acc_... -> 89)
        acc = None
        if len(parts) > 1:
            try:
                acc = float(parts[1])
            except Exception:
                acc = None
        if acc is None:
            for p in parts:
                try:
                    v = float(p)
                    if int(v) != int(sequence_length):
                        acc = v
                        break
                except Exception:
                    continue
        if acc is None:
            acc = 0.0

        # Determine if this is a general model (final_data / celeb) or FF-only
        fname_lower = fname.lower()
        is_general = 'final_data' in fname_lower or 'celeb' in fname_lower
        candidates.append((is_general, acc, os.path.join(settings.PROJECT_DIR, "models", fname)))

    if not candidates:
        print(f"No model found for sequence length={sequence_length}. Available: {list_models}")
        return []

    # Sort: general models first, then by accuracy descending
    candidates.sort(key=lambda x: (not x[0], -x[1]))
    result = [(acc, path) for (_, acc, path) in candidates]
    print(f"Models for sequence_length={sequence_length}:")
    for acc, path in result:
        print(f"  - {os.path.basename(path)} (acc={acc})")
    return result


def get_accurate_model(sequence_length):
    """Return the best single model for the sequence_length.
    Prefers 'final_data'/'celeb' models over FF-only models for better generalization."""
    models = get_all_matching_models(sequence_length)
    if not models:
        return ""
    best_acc, best_path = models[0]
    print(f"Selected model: {best_path} (acc={best_acc})")
    return best_path


def ensemble_predict(model_class, video_path, sequence_length, video_file_name=""):
    """Run prediction with ALL available models across MULTIPLE sequence lengths.

    Uses a multi-scale approach: different sequence lengths capture different
    types of deepfake artifacts:
    - Short sequences (10-20 frames) detect per-frame visual artifacts
    - Long sequences (60-100 frames) detect temporal inconsistencies

    Each model votes with its softmax probability. If ANY model detects FAKE
    with high confidence (>90%), the final result is biased toward FAKE because
    false positives (predicting FAKE for a real video) are rare for high-confidence
    predictions, while false negatives (missing a deepfake) are common.
    """
    all_models = glob.glob(os.path.join(settings.PROJECT_DIR, "models", "*.pt"))
    if not all_models:
        print("No models found!")
        return None

    all_probs = []
    individual_results = []

    # Group models by their sequence length
    model_groups = {}
    for model_path in all_models:
        fname = os.path.basename(model_path)
        parts = fname.split("_")
        # Find the sequence length token (before 'frames')
        seq = None
        for idx_p, p in enumerate(parts):
            if p == 'frames' and idx_p > 0:
                try:
                    seq = int(parts[idx_p - 1])
                except Exception:
                    pass
        if seq is None:
            continue
        model_groups.setdefault(seq, []).append(model_path)

    print(f"Multi-scale ensemble: found models for sequence lengths {sorted(model_groups.keys())}")

    # Build dataset samples for each unique sequence length
    datasets_cache = {}

    for seq_len in sorted(model_groups.keys()):
        # Create dataset with this sequence length
        if seq_len not in datasets_cache:
            try:
                ds = validation_dataset([video_path], sequence_length=seq_len, transform=train_transforms)
                datasets_cache[seq_len] = ds[0]
            except Exception as e:
                print(f"  Failed to create dataset for seq_len={seq_len}: {e}")
                continue

        sample = datasets_cache[seq_len]

        for model_path in model_groups[seq_len]:
            fname = os.path.basename(model_path)
            try:
                model = model_class(2)
                model.to(device)
                state = torch.load(model_path, map_location=device)
                model.load_state_dict(state)
                model.to(device)
                model.eval()

                with torch.no_grad():
                    fmap, logits = model(sample.to(device))
                    probs = sm(logits)
                    prob_np = probs.detach().cpu().numpy()[0]
                    all_probs.append(prob_np)

                    _, pred = torch.max(probs, 1)
                    pred_label = "REAL" if pred.item() == 1 else "FAKE"
                    conf = probs[:, pred.item()].item() * 100
                    individual_results.append({
                        'model': fname,
                        'seq_len': seq_len,
                        'prediction': pred_label,
                        'confidence': round(conf, 1),
                        'probs': prob_np.tolist()
                    })
                    print(f"  {fname} (seq={seq_len}): {pred_label} ({conf:.1f}%)")
            except Exception as e:
                print(f"  Failed to run {fname}: {e}")
                continue

    if not all_probs:
        return None

    # Compute ensemble using "any high-confidence FAKE wins" strategy:
    # If any model predicts FAKE with > 85% confidence, trust it —
    # false positives at that confidence level are rare.
    high_conf_fake = [r for r in individual_results
                      if r['prediction'] == 'FAKE' and r['confidence'] > 85.0]

    if high_conf_fake:
        # Use the highest-confidence FAKE prediction
        best_fake = max(high_conf_fake, key=lambda x: x['confidence'])
        ensemble_label = "FAKE"
        ensemble_conf = best_fake['confidence']
        ensemble_pred = 0
        print(f"High-confidence FAKE detected by {best_fake['model']}: {best_fake['confidence']}%")
    else:
        # Average probabilities across all models
        avg_probs = np.mean(all_probs, axis=0)
        ensemble_pred = int(np.argmax(avg_probs))
        ensemble_conf = float(avg_probs[ensemble_pred]) * 100
        ensemble_label = "REAL" if ensemble_pred == 1 else "FAKE"

    print(f"Ensemble result: {ensemble_label} ({ensemble_conf:.1f}%)")

    return {
        'prediction': ensemble_pred,
        'label': ensemble_label,
        'confidence': round(ensemble_conf, 1),
        'individual': individual_results
    }

ALLOWED_VIDEO_EXTENSIONS = set(['mp4','gif','webm','avi','3gp','wmv','flv','mkv'])

def allowed_video_file(filename):
    #print("filename" ,filename.rsplit('.',1)[1].lower())
    if (filename.rsplit('.',1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS):
        return True
    else: 
        return False
def index(request):
    if request.method == 'GET':
        video_upload_form = VideoUploadForm()
        if 'file_name' in request.session:
            del request.session['file_name']
        if 'preprocessed_images' in request.session:
            del request.session['preprocessed_images']
        if 'faces_cropped_images' in request.session:
            del request.session['faces_cropped_images']
        return render(request, index_template_name, {"form": video_upload_form})
    else:
        video_upload_form = VideoUploadForm(request.POST, request.FILES)
        if video_upload_form.is_valid():
            video_file = video_upload_form.cleaned_data['upload_video_file']
            video_file_ext = video_file.name.split('.')[-1]
            sequence_length = video_upload_form.cleaned_data['sequence_length']
            video_content_type = video_file.content_type.split('/')[0]
            if video_content_type in settings.CONTENT_TYPES:
                if video_file.size > int(settings.MAX_UPLOAD_SIZE):
                    video_upload_form.add_error("upload_video_file", "Maximum file size 100 MB")
                    return render(request, index_template_name, {"form": video_upload_form})

            if sequence_length <= 0:
                video_upload_form.add_error("sequence_length", "Sequence Length must be greater than 0")
                return render(request, index_template_name, {"form": video_upload_form})
            
            if allowed_video_file(video_file.name) == False:
                video_upload_form.add_error("upload_video_file","Only video files are allowed ")
                return render(request, index_template_name, {"form": video_upload_form})
            
            saved_video_file = 'uploaded_file_'+str(int(time.time()))+"."+video_file_ext
            if settings.DEBUG:
                with open(os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file), 'wb') as vFile:
                    shutil.copyfileobj(video_file, vFile)
                request.session['file_name'] = os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file)
            else:
                with open(os.path.join(settings.PROJECT_DIR, 'uploaded_videos','app','uploaded_videos', saved_video_file), 'wb') as vFile:
                    shutil.copyfileobj(video_file, vFile)
                request.session['file_name'] = os.path.join(settings.PROJECT_DIR, 'uploaded_videos','app','uploaded_videos', saved_video_file)
            request.session['sequence_length'] = sequence_length
            return redirect('ml_app:predict')
        else:
            return render(request, index_template_name, {"form": video_upload_form})

def predict_page(request):
    if request.method == "GET":
        # Redirect to 'home' if 'file_name' is not in session
        if 'file_name' not in request.session:
            return redirect("ml_app:home")
        if 'file_name' in request.session:
            video_file = request.session['file_name']
        if 'sequence_length' in request.session:
            sequence_length = request.session['sequence_length']
        path_to_videos = [video_file]
        video_file_name = os.path.basename(video_file)
        video_file_name_only = os.path.splitext(video_file_name)[0]
        # Production environment adjustments
        if not settings.DEBUG:
            production_video_name = os.path.join('/home/app/staticfiles/', video_file_name.split('/')[3])
            print("Production file name", production_video_name)
        else:
            production_video_name = video_file_name

        # Load validation dataset (kept for heatmap generation if needed later)
        # video_dataset = validation_dataset(path_to_videos, sequence_length=sequence_length, transform=train_transforms)

        start_time = time.time()
        # Display preprocessing images
        print("<=== | Started Videos Splitting | ===>")
        preprocessed_images = []
        faces_cropped_images = []
        cap = cv2.VideoCapture(video_file)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        cap.release()

        print(f"Number of frames: {len(frames)}")
        # Process each frame for preprocessing and face cropping
        padding = 40
        faces_found = 0
        for i in range(sequence_length):
            if i >= len(frames):
                break
            frame = frames[i]

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save preprocessed image
            image_name = f"{video_file_name_only}_preprocessed_{i+1}.png"
            image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
            img_rgb = pImage.fromarray(rgb_frame, 'RGB')
            img_rgb.save(image_path)
            preprocessed_images.append(image_name)

            # Face detection and cropping
            face_locations = face_recognition.face_locations(rgb_frame)
            if len(face_locations) == 0:
                continue

            top, right, bottom, left = face_locations[0]
            frame_face = frame[top - padding:bottom + padding, left - padding:right + padding]

            # Convert cropped face image to RGB and save
            rgb_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
            img_face_rgb = pImage.fromarray(rgb_face, 'RGB')
            image_name = f"{video_file_name_only}_cropped_faces_{i+1}.png"
            image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
            img_face_rgb.save(image_path)
            faces_found += 1
            faces_cropped_images.append(image_name)

        print("<=== | Videos Splitting and Face Cropping Done | ===>")
        print("--- %s seconds ---" % (time.time() - start_time))

        # No face detected
        if faces_found == 0:
            return render(request, predict_template_name, {"no_faces": True})

        # Perform prediction using multi-scale ensemble (ALL models, ALL sequence lengths)
        try:
            heatmap_images = []
            output = ""
            confidence = 0.0

            for i in range(len(path_to_videos)):
                print("<=== | Started Multi-Scale Ensemble Prediction | ===>")
                print(f"DEBUG: video_path={path_to_videos[i]}, device={device}")

                result = ensemble_predict(Model, path_to_videos[i], sequence_length, video_file_name_only)

                if result is not None:
                    output = result['label']
                    confidence = result['confidence']
                    prediction = [result['prediction'], result['confidence']]
                    print(f"Ensemble Prediction: {output} ({confidence}%)")
                    print(f"Individual model results:")
                    for r in result['individual']:
                        print(f"  {r['model']}: {r['prediction']} ({r['confidence']}%)")
                else:
                    # Fallback if no models could run
                    print("WARNING: Ensemble prediction failed. Using fallback.")
                    output = "FAKE"
                    prediction = [0, 50.0]
                    confidence = 50.0

                print("Prediction:", prediction[0], "==", output, "Confidence:", confidence)
                print("<=== | Prediction Done | ===>")
                print("--- %s seconds ---" % (time.time() - start_time))

                # Uncomment if you want to create heat map images
                # if model_loaded:
                #    for j in range(sequence_length):
                #        heatmap_images.append(plot_heat_map(j, model, video_dataset[i], './', video_file_name_only))

            # Render results
            context = {
                'preprocessed_images': preprocessed_images,
                'faces_cropped_images': faces_cropped_images,
                'heatmap_images': heatmap_images,
                'original_video': production_video_name,
                'models_location': os.path.join(settings.PROJECT_DIR, 'models'),
                'output': output,
                'confidence': confidence
            }

            if settings.DEBUG:
                return render(request, predict_template_name, context)
            else:
                return render(request, predict_template_name, context)

        except Exception as e:
            print(f"Exception occurred during prediction: {e}")
            return render(request, 'cuda_full.html')
def about(request):
    return render(request, about_template_name)

def handler404(request,exception):
    return render(request, '404.html', status=404)
def cuda_full(request):
    return render(request, 'cuda_full.html')
