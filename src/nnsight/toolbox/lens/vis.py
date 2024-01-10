from typing import List

import baukit


def vis(
    words: List[List[str]],
    probs: List[List[float]],
    input_words: List[str],
    color=[50, 168, 123],
):
    header_line = [  # header line
        [[" "]]
        + [
            [
                baukit.show.style(fontWeight="bold", width="50px"),
                baukit.show.attr(title=f"Token {i}"),
                t,
            ]
            for i, t in enumerate(input_words)
        ]
    ]

    def color_fn(p):
        a = [int(255 * (1 - p) + c * p) for c in color]
        return baukit.show.style(background=f"rgb({a[0]}, {a[1]}, {a[2]})")

    layer_logits = [
        # first column
        [[baukit.show.style(fontWeight="bold", width="50px"), f"L{layer_idx}"]]
        + [
            # subsequent columns
            [
                color_fn(token_prob),
                baukit.show.style(overflowX="hide", color="black"),
                f"{token_word}",
            ]
            for token_word, token_prob in zip(layer_words, layer_probs)
        ]
        for layer_idx, (layer_words, layer_probs) in enumerate(zip(words, probs))
    ]

    baukit.show(header_line + layer_logits + header_line)
