import importlib
import torch
from utils.commons.hparams import hparams, set_hparams
import bigvgan

class VocoderInfer:
    def __init__(self, hparams):

        config_path = hparams["vocoder_config"]
        self.config = set_hparams(config_path, global_hparams=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # pkg = ".".join(hparams["vocoder_cls"].split(".")[:-1])
        # cls_name = hparams["vocoder_cls"].split(".")[-1]
        # vocoder = getattr(importlib.import_module(pkg), cls_name)
        # self.model = vocoder(config)
        self.model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_22khz_80band', use_cuda_kernel=False)

        # checkpoint_dict = torch.load(
        #     hparams["vocoder_ckpt"], map_location=self.device, weights_only=True
        # )
        # remove weight norm in the model and set to eval mode
        self.model.remove_weight_norm()
        self.model = self.model.eval().to(self.device)
        # self.model.load_state_dict(checkpoint_dict["generator"])
        # self.model.to(self.device)
        # self.model.eval()
        # # load wav file and compute mel spectrogram
        # wav_path = '/path/to/your/audio.wav'
        # wav, sr = librosa.load(wav_path, sr=model.h.sampling_rate, mono=True) # wav is np.ndarray with shape [T_time] and values in [-1, 1]
        # wav = torch.FloatTensor(wav).unsqueeze(0) # wav is FloatTensor with shape [B(1), T_time]

        # # compute mel spectrogram from the ground truth audio
        # mel = get_mel_spectrogram(wav, model.h).to(device) # mel is FloatTensor with shape [B(1), C_mel, T_frame]

    def spec2wav(self, mel, **kwargs):
        device = self.device
        with torch.no_grad():
            c = torch.FloatTensor(mel).unsqueeze(0).to(device)
            c = c.transpose(2, 1)
            y = self.model(c).view(-1)
        wav_out = y.cpu().numpy()
        return wav_out


def parse_dataset_configs():
    max_tokens = hparams["max_tokens"]
    max_sentences = hparams["max_sentences"]
    max_valid_tokens = hparams["max_valid_tokens"]
    if max_valid_tokens == -1:
        hparams["max_valid_tokens"] = max_valid_tokens = max_tokens
    max_valid_sentences = hparams["max_valid_sentences"]
    if max_valid_sentences == -1:
        hparams["max_valid_sentences"] = max_valid_sentences = max_sentences
    return max_tokens, max_sentences, max_valid_tokens, max_valid_sentences
