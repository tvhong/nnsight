from typing import Callable, List

import torch

from nnsight import LanguageModel


def logitlens(model: LanguageModel, prompt: str, layers: List, decoding_fn: Callable):
    probs_layers = []

    with model.forward(validate=False) as runner:
        with runner.invoke(prompt, scan=False) as invoker:
            for layer in layers:

                probs = torch.nn.functional.softmax(
                    decoding_fn(layer.output), dim=-1
                ).save()

                probs_layers.append(probs)

    probs = torch.concatenate([probs.value for probs in probs_layers])

    max_probs, tokens = probs.max(dim=-1)

    words = [[model.tokenizer.decode(t).encode("unicode_escape").decode() for t in layer_tokens] for layer_tokens in tokens]

    input_words = [model.tokenizer.decode(t) for t in invoker.input["input_ids"][0]]

    return words, max_probs, input_words
