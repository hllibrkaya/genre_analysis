import pandas as pd
import numpy as np
import librosa
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import re
import time
from config import client_id, client_secret

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
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
            rms = np.mean(librosa.feature.rms(y=y))
            mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
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

            rows = {
                'Track Name': track_name,
                'Tempo': tempo,
                'Chroma Mean': chroma,
                'RMS Mean': rms,
                'Mel Spectrogram': mel_spectrogram,
                'Zero Crossing Rate': zero_crossing_rate,
                'Spectral Centroid': spectral_centroid,
                'Spectral Bandwidth': spectral_bandwidth,
                'Spectral Contrast': spectral_contrast,
                'Spectral Flatness': spectral_flatness,
                'Spectral Rolloff': spectral_rolloff,
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
            for i in range(mfcc.shape[0]):
                rows[f'mfcc_{i + 1}'] = np.mean(mfcc[i, :])

            return rows
        except FileNotFoundError:
            return None


def collect(list, name):
    data = []
    for id in list:
        playlist = sp.playlist(id)
        tracks = playlist["tracks"]["items"]

        for track in tracks:
            retry_count = 0
            while retry_count < 3:  # Maximum number of retry attempts
                try:
                    features = song_features(track)
                    if features:
                        data.append({
                            **features,
                        })
                    break  # Break out of the retry loop if successful
                except Exception as e:
                    print(f"Error processing track: {e}")
                    retry_count += 1
                    # backoff-retry strategy
                    wait_time = 2 ** retry_count
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
    df = pd.DataFrame(data)
    df.to_csv(name, index=False)


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
                "emircan_id": "37i9dQZF1DZ06evO2UKnCF", "84_id": "37i9dQZF1DZ06evO4ziOJO"
                }

artist_ids = artists_list.values()
collect(artist_ids, name="tracks.csv")
