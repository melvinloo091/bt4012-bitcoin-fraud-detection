
from pathlib import Path
import json
import sys
import time
import numpy as np
import torch
from torch import nn

configuration = json.loads(sys.argv[1])
torch.set_num_threads(configuration['threads'])
torch.set_num_interop_threads(configuration['threads'])
torch.use_deterministic_algorithms(True)

class TransactionAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(165, 128), nn.ReLU(), nn.Linear(128, 32))
        self.decoder = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 165))

    def forward(self, noisy_features):
        return self.decoder(self.encoder(noisy_features))

def encode(clean, model):
    batches = []
    model.eval()
    with torch.no_grad():
        for first in range(0, len(clean), configuration['batch_size']):
            inputs = torch.from_numpy(clean[first:first + configuration['batch_size']])
            latent = model.encoder(inputs)
            error = (inputs - model.decoder(latent)).square().mean(dim=1, keepdim=True)
            batches.append(torch.cat([latent, error], dim=1).numpy())
    return np.concatenate(batches, axis=0)

clean_array = np.load(configuration['input_path'], allow_pickle=False)
assert clean_array.dtype == np.float32 and np.isfinite(clean_array).all()
started = time.perf_counter()
if configuration['action'] == 'fit':
    torch.manual_seed(configuration['seed'])
    model = TransactionAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration['learning_rate'],
                                 weight_decay=configuration['weight_decay'])
    shuffle_generator = torch.Generator().manual_seed(configuration['seed'])
    clean = torch.from_numpy(clean_array)
    curve = []
    for epoch in range(1, configuration['epochs'] + 1):
        model.train()
        order = torch.randperm(len(clean), generator=shuffle_generator)
        total_loss = 0.0
        for first in range(0, len(clean), configuration['batch_size']):
            batch = clean[order[first:first + configuration['batch_size']]]
            noisy = batch + configuration['noise_sd'] * torch.randn_like(batch)
            optimizer.zero_grad()
            loss = (model(noisy) - batch).square().mean()
            assert torch.isfinite(loss), 'Non-finite reconstruction loss.'
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
        curve.append(total_loss / len(clean))
        progress_path = Path(configuration['progress_path'])
        temporary = progress_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(
            phase='encoder pretraining', family=configuration['family'],
            cutoff=configuration['end'], stage=configuration['stage'],
            epoch=epoch, epochs=configuration['epochs'], rows=len(clean),
            seconds=round(time.perf_counter() - started, 2))))
        temporary.replace(progress_path)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    torch.save(model.state_dict(), configuration['weights_path'])
    restored = TransactionAutoencoder()
    restored.load_state_dict(torch.load(configuration['weights_path'], map_location='cpu', weights_only=True))
    probe = clean_array[:min(len(clean_array), 1024)]
    assert np.array_equal(encode(probe, model), encode(probe, restored))
    print(json.dumps(dict(torch_version=torch.__version__, curve=curve,
                          saved_encoder_reload_verified=True)))
else:
    assert configuration['action'] == 'encode'
    model = TransactionAutoencoder()
    model.load_state_dict(torch.load(configuration['weights_path'], map_location='cpu', weights_only=True))
    summaries = encode(clean_array, model)
    assert summaries.shape == (len(clean_array), 33) and np.isfinite(summaries).all()
    np.save(configuration['output_path'], summaries, allow_pickle=False)
    print(json.dumps(dict(torch_version=torch.__version__)))
