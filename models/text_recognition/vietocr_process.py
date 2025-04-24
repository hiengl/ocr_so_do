import cv2
import torch
import numpy as np
import math
from collections import defaultdict
from PIL import Image
from torch.nn.functional import log_softmax, softmax

from .vietocr.transformerocr import VietOCR
from .vietocr.vocab import Vocab
from .vietocr.beam import Beam


def batch_translate_beam_search(
    img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2
):
    # img: NxCxHxW
    model.eval()
    device = img.device
    sents = []

    with torch.no_grad():
        src = model.cnn(img)
        print(src.shap)
        memories = model.transformer.forward_encoder(src)
        for i in range(src.size(0)):
            #            memory = memories[:,i,:].repeat(1, beam_size, 1) # TxNxE
            memory = model.transformer.get_memory(memories, i)
            sent = beamsearch(
                memory,
                model,
                device,
                beam_size,
                candidates,
                max_seq_length,
                sos_token,
                eos_token,
            )
            sents.append(sent)

    sents = np.asarray(sents)

    return sents


def translate_beam_search(
    img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2
):
    # img: 1xCxHxW
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)  # TxNxE
        sent = beamsearch(
            memory,
            model,
            device,
            beam_size,
            candidates,
            max_seq_length,
            sos_token,
            eos_token,
        )

    return sent


def beamsearch(
    memory,
    model,
    device,
    beam_size=4,
    candidates=1,
    max_seq_length=128,
    sos_token=1,
    eos_token=2,
):
    # memory: Tx1xE
    model.eval()

    beam = Beam(
        beam_size=beam_size,
        min_length=0,
        n_top=candidates,
        ranker=None,
        start_token_id=sos_token,
        end_token_id=eos_token,
    )

    with torch.no_grad():
        #        memory = memory.repeat(1, beam_size, 1) # TxNxE
        memory = model.transformer.expand_memory(memory, beam_size)

        for _ in range(max_seq_length):

            tgt_inp = beam.get_current_state().transpose(0, 1).to(device)  # TxN
            decoder_outputs, memory = model.transformer.forward_decoder(tgt_inp, memory)

            log_prob = log_softmax(decoder_outputs[:, -1, :].squeeze(0), dim=-1)
            beam.advance(log_prob.cpu())

            if beam.done():
                break

        scores, ks = beam.sort_finished(minimum=1)

        hypothesises = []
        for i, (times, k) in enumerate(ks[:candidates]):
            hypothesis = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)

    return [1] + [int(i) for i in hypothesises[0][:-1]]


def translate(img, model, max_seq_length=128, sos_token=1, eos_token=2):
    "data: BxCXHxW"
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)

        translated_sentence = [[sos_token] * len(img)]
        char_probs = [[1] * len(img)]

        max_length = 0

        while max_length <= max_seq_length and not all(
            np.any(np.asarray(translated_sentence).T == eos_token, axis=1)
        ):

            tgt_inp = torch.LongTensor(translated_sentence).to(device)

            #            output = vietocr(img, tgt_inp, tgt_key_padding_mask=None)
            #            output = vietocr.transformer(src, tgt_inp, tgt_key_padding_mask=None)
            output, memory = model.transformer.forward_decoder(tgt_inp, memory)
            output = softmax(output, dim=-1)
            output = output.to("cpu")

            values, indices = torch.topk(output, 5)

            indices = indices[:, -1, 0]
            indices = indices.tolist()

            values = values[:, -1, 0]
            values = values.tolist()
            char_probs.append(values)

            translated_sentence.append(indices)
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T

        char_probs = np.asarray(char_probs).T
        char_probs = np.multiply(char_probs, translated_sentence > 3)
        char_probs = np.sum(char_probs, axis=-1) / (char_probs > 0).sum(-1)

    return translated_sentence, char_probs


def load_model(config, device='cpu'):
    vocab = Vocab(config["vocab"])

    model = VietOCR(
        len(vocab),
        config['models']["backbone"],
        config['models']["cnn"],
        config['models']["transformer"],
        config['models']["seq_modeling"],
    )

    model = model.to(device)
    model.load_state_dict(torch.load(config['weights_path'], map_location=torch.device(device)))

    return model, vocab


def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w / round_to) * round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height


def process_image(image, image_height, image_min_width, image_max_width):
    # img = image.convert("RGB")
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    w, h = img.size
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)

    img = img.resize((new_w, image_height), Image.LANCZOS)

    img = np.asarray(img).transpose(2, 0, 1)
    img = img / 255
    return img


def process_input(image, image_height, image_min_width, image_max_width):
    img = process_image(image, image_height, image_min_width, image_max_width)
    img = img[np.newaxis, ...]
    img = torch.FloatTensor(img)
    return img


class Predictor:
    def __init__(self, config, device='cpu'):
        model, vocab = load_model(config, device=device)
        self.device = device
        self.config = config
        self.model = model
        self.vocab = vocab

    def predict(self, img, return_prob=False):
        img = process_input(
            img,
            self.config["dataset"]["image_height"],
            self.config["dataset"]["image_min_width"],
            self.config["dataset"]["image_max_width"],
        )
        img = img.to(self.device)

        if self.config.get("predictor", {}).get("beamsearch"):
            sent = translate_beam_search(img, self.model)
            s = sent
            prob = None
        else:
            s, prob = translate(img, self.model)
            s = s[0].tolist()
            prob = prob[0]

        s = self.vocab.decode(s)

        if return_prob:
            return s, prob
        else:
            return s

    def predict_batch(self, imgs, return_prob=False):
        bucket = defaultdict(list)
        bucket_idx = defaultdict(list)
        bucket_pred = {}

        sents, probs = [0] * len(imgs), [0] * len(imgs)

        for i, img in enumerate(imgs):
            img = process_input(
                img,
                self.config["dataset"]["image_height"],
                self.config["dataset"]["image_min_width"],
                self.config["dataset"]["image_max_width"],
            )

            bucket[img.shape[-1]].append(img)
            bucket_idx[img.shape[-1]].append(i)

        for k, batch in bucket.items():
            batch = torch.cat(batch, 0).to(self.device)
            s, prob = translate(batch, self.model)
            prob = prob.tolist()

            s = s.tolist()
            s = self.vocab.batch_decode(s)

            bucket_pred[k] = (s, prob)

        for k in bucket_pred:
            idx = bucket_idx[k]
            sent, prob = bucket_pred[k]
            for i, j in enumerate(idx):
                sents[j] = sent[i]
                probs[j] = prob[i]

        if return_prob:
            return sents, probs
        else:
            return sents
