from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import glob, os
import audio
from hparams import hparams
from os.path import exists
import librosa
import pypinyin as pinyin
from menuinst.knownfolders import PathNotFoundException


"""
The preprocess of aishell-1

"""


def __trans_dict(indir):
    """
    process wave file and transcript mapping.
    
    """
    print('the indir is {}'.format(indir))
    trans_path = ''.join([os.path.abspath(indir), '/transcript/aishell_transcript_v0.8.txt'])
    if not os.path.exists(trans_path):
        raise FileNotFoundError('the transcript path is not exitst :{}'.format(trans_path))
    print('the transripct path is {}'.format(trans_path))
    trans_dict = {}
    trans_texts = open(trans_path, mode='r', encoding='utf-8').readlines()
    for corpus in trans_texts:
        corpus = corpus.strip('\n').strip(' ')  # 去掉换行符和空格
        arr = corpus.split(' ')
        audio_id = arr[0]
        text = ''.join(arr[1:])
        pinyin_text = pinyin.lazy_pinyin(text, pinyin.Style.TONE3)
        trans = ' '.join(pinyin_text)
        trans_dict[audio_id] = trans
        print('len of transript dict {}'.format(len(trans_dict)))
    return trans_dict

def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    """
    Args:
        - in_dir: data_aishell dir.
        - out_dir: the preprocess result saved dir.
    """
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    trans_dict = __trans_dict(indir)
    index = 1
    wave_files = glob.glob(os.path.join(input_dir, '*', '*.wav'))
    print('the all wavefiles are {}'.format(len(wave_files)))
    for wav_file in wave_files:
        wav_path = wav_file
        audio_id = os.path.basename(wav_path).split('.')[0]
        text = trans_dict.get(audio_id)
        if text is None:
            print('audio id {} not in trans_dict!!!'.format(audio_id))
            continue
        else:
            print('audio id {}  in trans_dict!!!'.format(audio_id))
            speaker_id = int(os.path.basename(wav_path)[7:11])
            futures.append(executor.submit(partial(_process_utterance, out_dir, index, speaker_id, wav_path, text)))
            index += 1
    return [future.result() for future in tqdm(futures)]




def _process_utterance(out_dir, index, speaker_id, wav_path, text):
    sr = hparams.sample_rate

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    lab_path = wav_path.replace("wav48/", "lab/").replace(".wav", ".lab")

    # Trim silence 
    wav = audio.trim_silence(wav, hparams)
    
    if hparams.rescaling:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

    # Write the spectrograms to disk:
    spectrogram_filename = 'aishell1-spec-%05d.npy' % index
    mel_filename = 'aishell1-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example:
    return (spectrogram_filename, mel_filename, n_frames, text, speaker_id)
