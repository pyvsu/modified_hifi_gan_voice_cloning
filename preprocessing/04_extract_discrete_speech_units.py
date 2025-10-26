#!/usr/bin/env python3 
"""
Extract mHuBERT+KMeans (layer11, km1000) codes via asrp.

This produces integer unit sequences for each normalized audio file:
    unit_embeddings/<utt>.pt
""" 

import warnings 
warnings.filterwarnings("ignore", module="torchaudio")
warnings.filterwarnings("ignore", module="ffmpeg")
warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")
warnings.filterwarnings("ignore", message=".*deprecated.*") 
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*") 

import argparse 
from pathlib import Path 
from tqdm import tqdm 

import torch 
import torchaudio 
from transformers import HubertModel, Wav2Vec2FeatureExtractor 
from sklearn.cluster import MiniBatchKMeans 
import joblib 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


def parse_args(): 
    ap = argparse.ArgumentParser() 
    ap.add_argument("--in-root", type=Path, required=True, help="Folder of normalized wav files") 
    ap.add_argument("--out-root", type=Path, required=True, help="Folder to write .pt unit codes") 
    ap.add_argument("--km-file", type=str, required=True, help="Path to km.bin (kmeans)") 
    ap.add_argument("--layer", type=int, default=11, help="Hubert hidden layer to use") 
    ap.add_argument("--pattern", type=str, default="*.wav") 
    ap.add_argument("--overwrite", action="store_true") 
    return ap.parse_args() 


def ensure_parent(p): 
    p.parent.mkdir(parents=True, exist_ok=True) 


def load_km(path): 
    # joblib can load sklearn pickles safely 
    return joblib.load(path) 


def extract_hidden(model, feat_extractor, wav): 
    with torch.no_grad(): 
        # normalize waveform
        wav = wav.mean(dim=0, keepdim=True) 
        
        # force mono 
        feats = feat_extractor( 
            wav.squeeze().cpu().numpy(), 
            sampling_rate=16000, 
            return_tensors="pt", 
        ) 
        feats = feats.to(DEVICE) 
        
        outputs = model( 
            feats["input_values"], 
            output_hidden_states=True
        ) 
        
        return outputs.hidden_states 


def main(): 
    args = parse_args() 
    
    files = sorted(args.in_root.rglob(args.pattern)) 
    
    if not files: 
        raise SystemExit("No input wavs found.") 
    
    print("Loading mHuBERT (HubertModel)…") 
    model = HubertModel.from_pretrained("voidful/mhubert-base")
    model.to(DEVICE) 
    model.eval() 
    
    print("Loading KMeans quantizer…") 
    kmeans = load_km(args.km_file) 
    
    print("Loading Wav2Vec feature extractor…") 
    feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained("voidful/mhubert-base") 
    
    for wav_path in tqdm(files, desc="Units"): 
        out_path = args.out_root / wav_path.relative_to(args.in_root) 
        out_path = out_path.with_suffix(".pt") 
        
        if out_path.exists() and not args.overwrite: 
            continue 
        
        ensure_parent(out_path) 
        
        # load wav 
        wav, sr = torchaudio.load(wav_path) 
        
        # get hidden states 
        h = extract_hidden(model, feat_extractor, wav)[args.layer] # (1, T, D) 
        
        # flatten time dimension 
        feats = h.squeeze(0).cpu().numpy() 
        
        # assign cluster index to each frame 
        units = kmeans.predict(feats) 
        
        # save 
        torch.save(torch.tensor(units, dtype=torch.long), out_path) 
        print("DONE") 
        

if __name__ == "__main__":
    main()