import os, io, json, base64, librosa, numpy as np
from flask import Flask, render_template, request, send_file
from google import genai
from google.genai import types

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def agg(x):
    return {'mean': float(np.mean(x)), 'std': float(np.std(x)), 'min': float(np.min(x)), 'max': float(np.max(x))}

@app.route('/music', methods=['POST'])
def music():
    print('start')
    f = request.files['file']
    path = os.path.join('static', f.filename)
    f.save(path)
    y, sr = librosa.load(path, sr=None)
    tempo = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0])
    beat_int = np.diff(librosa.frames_to_time(librosa.beat.beat_track(y=y, sr=sr)[1], sr=sr))
    y_h, _ = librosa.effects.hpss(y)
    f0 = librosa.pyin(y_h, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr, hop_length=256)[0]
    valid_f0 = f0[~np.isnan(f0)]
    print('1')
    mfcc = librosa.feature.mfcc(y=y_h, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spec_con = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    ton = librosa.feature.tonnetz(y=y_h, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    print('2')
    feats = {'tempo_bpm': tempo,
             'beat_interval_mean': agg(beat_int)['mean'],
             'beat_interval_std': agg(beat_int)['std'],
             'f0_mean': agg(valid_f0)['mean'] if valid_f0.size else 0.0,
             'rms_mean': agg(rms)['mean'],
             'spectral_centroid_mean': agg(spec_cent)['mean'],
             'spectral_bandwidth_mean': agg(spec_bw)['mean'],
             'spectral_rolloff_mean': agg(spec_roll)['mean'],
             'spectral_contrast_mean': float(np.mean(spec_con)),
             'zcr_mean': agg(zcr)['mean'],
             'onset_mean': agg(onset)['mean'],
             'onset_std': agg(onset)['std']}
    print('3')
    for i in range(mfcc.shape[0]):
        feats[f'mfcc{i+1}_mean'] = agg(mfcc[i])['mean']
        feats[f'mfcc{i+1}_std'] = agg(mfcc[i])['std']

    pitch = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    feats['chroma_peak'] = pitch[int(np.argmax(np.mean(chroma, axis=1)))]
    for i, v in enumerate(np.mean(ton, axis=1), 1):
        feats[f'tonnetz{i}_mean'] = float(v)
    print('4')
    client = genai.Client(api_key='AIzaSyBC0yT0UerTu08pmIqM_GJMhgJa9ixizmw')
    mood_prompt = "다음 음악 특성에 기반해 이 음악의 분위기를 1-2문장 한국어로 설명해줘. " + json.dumps(feats, ensure_ascii=False)
    mood = client.models.generate_content(model='gemini-2.5-flash', contents=mood_prompt).candidates[0].content.parts[0].text.strip()

    img_prompt = mood + " | 자연환경 배경, 앨범 커버 스타일, vivid colors, 1024×1024"
    cfg = types.GenerateContentConfig(response_modalities=['TEXT', 'IMAGE'])
    img_resp = client.models.generate_content(model='gemini-2.0-flash-preview-image-generation', contents=img_prompt, config=cfg)
    raw_png = next(p.inline_data.data for p in img_resp.candidates[0].content.parts if p.inline_data)
    print('5')
    bio = io.BytesIO(raw_png)
    bio.seek(0)
    resp = send_file(bio, mimetype='image/png')
    return resp

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)