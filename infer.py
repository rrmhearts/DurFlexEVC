import os
import json
import argparse
import torch
import torchaudio
from transformers import HubertModel

from models.evc.durflex import DurFlexEVC
from tasks.evc.evc_utils import VocoderInfer
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import set_hparams
from utils.audio.io import save_wav
from utils.audio import wav2spec

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config", type=str, default="./configs/exp/durflex_evc.yaml"
    )
    parser.add_argument("--src_wav", type=str, default="./sample/0011_000021.wav")
    parser.add_argument("--save_dir", type=str, default="./results")
    args = parser.parse_args()

    hparams = set_hparams(args.config)
    os.makedirs(args.save_dir, exist_ok=True)
    sample_rate = hparams["audio_sample_rate"]
    spk_dict = json.load(
        open(os.path.join(hparams["processed_data_dir"], "spk_map.json"))
    )
    emo_dict = {"Neutral": 0, "Angry": 1, "Happy": 2, "Sad": 3, "Surprise": 4}

    hubert = HubertModel.from_pretrained(
        hparams["hubert_model"], output_hidden_states=True
    ).cuda()
    
    model = DurFlexEVC(hparams["n_units"], hparams).cuda()
    print("work_dir:", hparams["work_dir"])
    load_ckpt(model, f"{hparams['work_dir']}/DurFlex", "model")
    vocoder = VocoderInfer(hparams)

    basename = os.path.basename(args.src_wav).replace(".wav", "")
    spk = basename.split("_")[0]
    item_id = int(basename.split("_")[1])
    if item_id < 351:
        label = "Neutral"
    elif item_id < 701:
        label = "Angry"
    elif item_id < 1051:
        label = "Happy"
    elif item_id < 1401:
        label = "Sad"
    else:
        label = "Surprise"

    y, sr = torchaudio.load(args.src_wav)
    x = hubert(y.cuda()).hidden_states[-1]
    spk_id = spk_dict[spk]
    src_emo = emo_dict[label]
    spk_id = torch.LongTensor([spk_id]).cuda()
    src_emo = torch.LongTensor([src_emo]).cuda()

    mel, mel_length = None, None
    if hparams["use_spk_encoder"]:
        wav2spec_dict = wav2spec(
                args.src_wav,
                fft_size=hparams["fft_size"],
                hop_size=hparams["hop_size"],
                win_length=hparams["win_size"],
                num_mels=hparams["audio_num_mel_bins"],
                fmin=hparams["fmin"],
                fmax=hparams["fmax"],
                sample_rate= sr, #hparams["audio_sample_rate"],
                loud_norm=hparams["loud_norm"],
                trim_long_sil=True,
            )
        mel = wav2spec_dict["mel"]
        # mel_length = torch.LongTensor([ mel.shape[0] ]) # there is only one file
        mels = torch.tensor([ mel ]).cuda() #collate_1d_or_2d([ mel ], 0.0)
        mel_lengths = torch.LongTensor([ mel.shape[0] ]).cuda()
    print("mel shape:", mel.shape)
    print("mel length:", mel_length)
    # y should look like,      y_lengths
    # torch.Size([1, 139, 80]) tensor([139], device='cuda:0')

    for k, v in emo_dict.items():

        tgt_emo = torch.LongTensor([v]).cuda()
        output = model(
            x, spk_id=spk_id, emotion_id=src_emo, tgt_emotion_id=tgt_emo, infer=True, y=mels, y_lengths=mel_lengths
        )
        mel_out = output["mel_out"]
        wav_out = vocoder.spec2wav(mel_out[0].cpu())
        save_wav(wav_out, f"{args.save_dir}/{basename}_{k}.wav", sample_rate)
        print(f"{basename} converted to {k}")
