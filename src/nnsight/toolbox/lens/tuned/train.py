from copy import deepcopy
from typing import List

import torch
from torch.utils.data import DataLoader
from tuned_lens.scripts.ingredients import Data

from nnsight.models.Mamba import Mamba

device = "cuda"

model = Mamba("state-spaces/mamba-2.8b-slimpj", device=device, dispatch=True)


class TunedLens(torch.nn.Module):
    def __init__(self, layers: List, d_model: int) -> None:
        super().__init__()

        translator = torch.nn.Linear(d_model, d_model, bias=True)
        translator.weight.data.zero_()
        translator.bias.data.zero_()

        self.layer_translators = torch.nn.ModuleList(
            [deepcopy(translator) for _ in range(len(layers) - 1)]
        )


d_model_hidden_states = model.backbone.layers[0].output_shape[0][-1]
d_model_residual = d_model_hidden_states

hidden_states_lens = TunedLens(model.backbone.layers, d_model_hidden_states).to(device)
residual_lens = TunedLens(model.backbone.layers, d_model_residual).to(device)


data, _ = Data(["JeanKaddour/minipile", "default"], split="train").load(model.tokenizer)
dataloader = DataLoader(data)

params = list(hidden_states_lens.parameters()) + list(residual_lens.parameters())

optimizer = torch.optim.Adam(params, lr=1e-3)

for batch_idx, batch in enumerate(dataloader):
    optimizer.zero_grad()

    with model.forward(validate=False, inference=False) as runner:
        with runner.invoke(batch, scan=False):
            total_loss = 0

            last_output = torch.stack(
                [
                    model.backbone.layers[-1].output[0],
                    model.backbone.layers[-1].output[1],
                ],
                dim=1,
            )

            for layer_idx, layer in enumerate(model.backbone.layers[:-1]):
                hidden_state_prediction = hidden_states_lens.layer_translators[
                    layer_idx
                ](layer.output[0])
                residual_prediction = residual_lens.layer_translators[layer_idx](
                    layer.output[1]
                )

                prediction = torch.stack(
                    [hidden_state_prediction, residual_prediction], dim=1
                )

                loss = torch.nn.functional.cross_entropy(prediction, last_output)

                total_loss += loss

            total_loss.backward()

            total_loss.save()

optimizer.step()

print(total_loss.value)
