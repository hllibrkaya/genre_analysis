import pandas as pd
import numpy as np
import librosa
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import re

client_id = '7da03a37e45742b08663684e0f96e084'
client_secret = '9efa3aadcd4344ebb6834eb349b58cc7'

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def song_features(track):
    track_info = track["track"]
    track_id = track_info["id"]
    track_name = track_info['name']
    artist = [artist["name"] for artist in track["track"]["artists"]]
    track_name = track_name + "-" + artist[0]

    record_name = re.sub(r'[\\/:*?"<>|]', '_', track_name)  # removing invalid characters for mp3 files

    preview_url = track_info["preview_url"]

    if preview_url is not None:
        try:
            response = requests.get(preview_url)
            with open(f"songs/{record_name}.mp3", "wb") as f:
                f.write(response.content)
            y, sr = librosa.load(f"songs/{record_name}.mp3")
            audio_features = sp.audio_features([track_id])[0]
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            spectral_flatness = librosa.feature.spectral_flatness(y=y)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

            tempo_mean = np.mean(tempo)
            chroma_mean = np.mean(chroma)
            chroma_std = np.std(chroma)
            rms_mean = np.mean(rms)
            rms_std = np.std(rms)
            mel_spectrogram_mean = np.mean(mel_spectrogram)
            mel_spectrogram_std = np.std(mel_spectrogram)
            zero_crossing_rate_mean = np.mean(zero_crossing_rate)
            zero_crossing_rate_std = np.std(zero_crossing_rate)
            spectral_centroid_mean = np.mean(spectral_centroid)
            spectral_centroid_std = np.std(spectral_centroid)
            spectral_bandwidth_mean = np.mean(spectral_bandwidth)
            spectral_bandwidth_std = np.std(spectral_bandwidth)
            spectral_contrast_mean = np.mean(spectral_contrast)
            spectral_contrast_std = np.std(spectral_contrast)
            spectral_flatness_mean = np.mean(spectral_flatness)
            spectral_flatness_std = np.std(spectral_flatness)
            spectral_rolloff_mean = np.mean(spectral_rolloff)
            spectral_rolloff_std = np.std(spectral_rolloff)

            speechiness = audio_features['speechiness']
            duration = audio_features['duration_ms']
            valence = audio_features['valence']
            inst = audio_features['instrumentalness']
            key = audio_features['key']
            danceability = audio_features['danceability']
            energy = audio_features['energy']
            mode = audio_features['mode']
            acousticness = audio_features['acousticness']
            loudness = audio_features['loudness']

            return {
                'Track Name': track_name,
                'Tempo Mean': tempo_mean,
                'Chroma Mean': chroma_mean,
                'Chroma Std': chroma_std,
                'RMS Mean': rms_mean,
                'RMS Std': rms_std,
                'Mel Spectrogram Mean': mel_spectrogram_mean,
                'Mel Spectrogram Std': mel_spectrogram_std,
                'Zero Crossing Rate Mean': zero_crossing_rate_mean,
                'Zero Crossing Rate Std': zero_crossing_rate_std,
                'Spectral Centroid Mean': spectral_centroid_mean,
                'Spectral Centroid Std': spectral_centroid_std,
                'Spectral Bandwidth Mean': spectral_bandwidth_mean,
                'Spectral Bandwidth Std': spectral_bandwidth_std,
                'Spectral Contrast Mean': spectral_contrast_mean,
                'Spectral Contrast Std': spectral_contrast_std,
                'Spectral Flatness Mean': spectral_flatness_mean,
                'Spectral Flatness Std': spectral_flatness_std,
                'Spectral Rolloff Mean': spectral_rolloff_mean,
                'Spectral Rolloff Std': spectral_rolloff_std,
                "Speechiness": speechiness,
                "Duration": duration,
                "Valence": valence,
                "Instrumentalness": inst,
                "Key": key,
                "Danceability": danceability,
                "Energy": energy,
                "Loudness": loudness,
                "Mode": mode,
                "Acousticness": acousticness

            }
        except FileNotFoundError:
            return None


def collect(list):
    data = []
    for id in list:
        playlist = sp.playlist(id)
        tracks = playlist["tracks"]["items"]

        for track in tracks:
            features = song_features(track)
            if features:
                data.append({
                    **features,
                })
    df = pd.DataFrame(data)
    df.to_csv("tacks.csv", index=False)


artists_list = {"duman_id": "37i9dQZF1DZ06evO43NcNz", "manga_id": "37i9dQZF1DZ06evO4pjcGq",
                "mor_ve_ötesi_id": "37i9dQZF1DZ06evO37d71q", "yüz_kon_id": "37i9dQZF1DZ06evO4iwPYi",
                "gripin_id": "37i9dQZF1DZ06evO2kdV6i", "sebo_id": "37i9dQZF1DZ06evO4boMuq",
                "dktt_id": "37i9dQZF1DZ06evO0qnRN8", "müslüm_id": "37i9dQZF1DZ06evO2srcKu",
                "ferdi_id": "37i9dQZF1DZ06evO46IAAq", "orhan_id": "37i9dQZF1DZ06evO2XMuRB",
                "azer_id": "37i9dQZF1DZ06evO1wh6bn", "bergen_id": "37i9dQZF1DZ06evO0pNECV",
                "güllü_id": "37i9dQZF1DZ06evO2XTAGn", "cengiz_id": "37i9dQZF1DZ06evO4aMJc9",
                "ebru_id": "37i9dQZF1DZ06evO3hP8GV", "muratd_id": "37i9dQZF1DZ06evO3FyQ4o",
                "muratb_id": "37i9dQZF1DZ06evO2Y5YYd", "hadise_id": "37i9dQZF1DZ06evO0f9PCM",
                "gülşen_id": "37i9dQZF1DZ06evO0VwzCF", "hande_id": "37i9dQZF1DZ06evO00yqP5",
                "simge_id": "37i9dQZF1DZ06evO2Rwuis", "serdar_id": "37i9dQZF1DZ06evO4vmUE1",
                "aleyna_id": "37i9dQZF1DZ06evO2sapJA", "edis_id": "37i9dQZF1DZ06evO0SXlWG",
                "neşet_id": "37i9dQZF1DZ06evO0CZYS8", "musa_id": "37i9dQZF1DZ06evO1dBpgJ",
                "kıvırcık_id": "37i9dQZF1DZ06evO3antFX", "abdal_id": "37i9dQZF1DZ06evO3rmCwa",
                "oguz_id": "37i9dQZF1DZ06evO0c2O1g", "ahmet_id": "37i9dQZF1DZ06evO16XtP7",
                "selda_id": "37i9dQZF1DZ06evO1C2hTu", "mahzuni_id": "37i9dQZF1DZ06evO2xg7aQ",
                "semicenk_id": "37i9dQZF1DZ06evO0TOYhk", "tugkan_id": "37i9dQZF1DZ06evO1iny4U",
                "goksel_id": "37i9dQZF1DZ06evO2vCLCL", "sezen_id": "37i9dQZF1DZ06evO3zTpza",
                "kalben_id": "37i9dQZF1DZ06evO2BNFV6", "gece_yolc_id": "37i9dQZF1DZ06evO18YRrA",
                "kahraman_id": "37i9dQZF1DZ06evO1XGrC1", "pera_id": "37i9dQZF1DZ06evO0DNbwI",
                "soner_id": "37i9dQZF1DZ06evO2RUXEl", "model_id": "37i9dQZF1DZ06evO1agBCV",
                "emircan_id": "37i9dQZF1DZ06evO2UKnCF", "84_id": "37i9dQZF1DZ06evO4ziOJO",
                "oguz_yilmaz_id": "37i9dQZF1DZ06evO4ABqCQ", "namık_id": "37i9dQZF1DZ06evO0gradV",
                "ibocan_id": "37i9dQZF1DZ06evO3Yz8DC", "yasemin_id": "37i9dQZF1DZ06evO4Bv3Iu"}

artist_ids = artists_list.values()
collect(artist_ids)
