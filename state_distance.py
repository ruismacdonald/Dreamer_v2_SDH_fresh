import os
import numpy as np
import torch
from torch import nn

from torch.utils.data import Dataset, DataLoader


class ContrastiveStateDistanceDataset(Dataset):
    def __init__(self, observation_pairs, num_negative_samples=128, seed=0,transform=None):
        self.data = observation_pairs
        self.num_negative_samples = num_negative_samples
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        obs, positive = self.data[idx]

        # TODO use a proper seeding
        indices = np.random.randint(len(self.data), size=self.num_negative_samples)
        negatives = np.array([self.data[ind][0] for ind in indices])

        return obs, positive, negatives


def calculate_output_dim(net, input_shape):
    if isinstance(input_shape, int):
        input_shape = (input_shape,)
    placeholder = torch.zeros((1,) + tuple(input_shape))
    output = net(placeholder)
    return output.size()[1:]


def tstats(x: torch.Tensor, prefix: str) -> dict:
    x = x.detach()
    return {
        f"{prefix}_shape0": float(x.shape[0]),
        f"{prefix}_dtype": str(x.dtype),
        f"{prefix}_min": float(x.min().item()),
        f"{prefix}_max": float(x.max().item()),
        f"{prefix}_mean": float(x.float().mean().item()),
        f"{prefix}_std": float(x.float().std(unbiased=False).item()),
    }


def preprocess_uint8_batch(x: torch.Tensor) -> torch.Tensor:
    # x: uint8 BCHW
    return x.to(torch.float32).div_(255.0).sub_(0.5)


class ContrastiveStateDistanceNet(nn.Module):
    def __init__(
        self,
        in_dim=[3, 64, 64],
        channels=[32, 64, 128, 256],
        mlp_layers=[512, 64, 32],
        kernel_sizes=[4, 4, 4, 4],
        strides=[2, 2, 2, 2],
        paddings=[0, 0, 0, 0],
    ):
        super().__init__()

        channels.insert(0, in_dim[0])
        conv_seq = []
        for i in range(0, len(channels) - 1):
            conv_seq.append(
                nn.Conv2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                )
            )
            conv_seq.append(torch.nn.ReLU())
        self.conv = torch.nn.Sequential(*conv_seq)

        mlp_in_dim = calculate_output_dim(self.conv, in_dim)
        mlp_modules = [nn.Linear(np.prod(mlp_in_dim), mlp_layers[0]), torch.nn.ReLU()]
        for i in range(len(mlp_layers) - 2):
            mlp_modules.append(nn.Linear(mlp_layers[i], mlp_layers[i + 1]))
            mlp_modules.append(torch.nn.ReLU())
        mlp_modules.append(nn.Linear(mlp_layers[-2], mlp_layers[-1]))
        self.mlp = torch.nn.Sequential(*mlp_modules)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.mlp(x)
        return x


def create_init_weights_fn(initialization_fn):
    if initialization_fn is not None:

        def init_weights(m):
            if hasattr(m, "weight"):
                initialization_fn(m.weight)

        return init_weights
    else:
        return lambda m: None


class SimpleContrastiveStateDistanceModel:
    """A simple distance model based on contrastive loss."""

    def __init__(
        self,
        in_dim,
        optimizer_fn,
        seed,
        init_fn=torch.nn.init.kaiming_uniform_,
        negative_distance_target=50.0,
        negative_loss_ratio=0.1,
        num_negative_samples=128,
        num_training_epochs=5,
        batch_size=32,
        device: str = "cuda",
        normalize_representations: bool = True,
    ):

        self._device = torch.device("cpu" if not torch.cuda.is_available() else device)
        self._representation_net = ContrastiveStateDistanceNet(in_dim)
        init_fn = create_init_weights_fn(init_fn)
        self._representation_net = self._representation_net.apply(init_fn).to(
            self._device
        )

        if optimizer_fn is None:
            optimizer_fn = torch.optim.Adam
        self._optimizer = optimizer_fn(self._representation_net.parameters(), lr=1e-4)

        self._negative_distance_target = negative_distance_target
        self._negative_loss_ratio = negative_loss_ratio

        self._num_negative_samples = int(num_negative_samples)
        self._num_training_epochs = int(num_training_epochs)
        self._batch_size = int(batch_size)

        self._normalize_representations = bool(normalize_representations)
        self._repr_mean = None
        self._repr_std = None

        self._seed = int(seed)

    def prepare_train_loader(self, data):
        observation_pairs = []
        for i in range(data["observation"].shape[0] - 1):
            if data["terminal"][i]:
                continue
            observation_pairs.append((data["observation"][i], data["observation"][i + 1]))

        train_dataset = ContrastiveStateDistanceDataset(
            observation_pairs,
            num_negative_samples=self._num_negative_samples,
            seed=self._seed,
        )

        g = torch.Generator()
        g.manual_seed(self._seed)

        def seed_worker(worker_id: int):
            info = torch.utils.data.get_worker_info()
            if info is None:
                return
            ds = info.dataset
            seed = (self._seed + 1009 * worker_id) % (2**32 - 1)
            if hasattr(ds, "_rng"):
                ds._rng = np.random.default_rng(seed)

        return DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            generator=g,
            pin_memory=True,
            num_workers=min(4, os.cpu_count() or 1),
            persistent_workers=True,
            worker_init_fn=seed_worker,
        )


    def calculate_loss_terms(self, obs, positive, negatives):
        """
        obs: (B,3,64,64) float in [-0.5, 0.5]
        positive: (B,3,64,64) float in [-0.5, 0.5]
        negatives: (B,K,3,64,64) float in [-0.5, 0.5]

        Positive term: Compute the squared distance between the embedding of the current observation 
        and its true next (consecutive) observation. Minimizing this term encourages temporally 
        adjacent states to have similar representations.

        Negative term: Compute the average squared distance between the embedding of the current 
        observation and a set of randomly sampled (non-consecutive) observations. Penalize the 
        model if this average distance is not close to a predefined target. This pushes unrelated 
        states apart and prevents representation collapse.
        """
        # Reprs
        obs_repr = self._representation_net(obs)  # (B,D)
        pos_repr = self._representation_net(positive)  # (B,D)

        B, K = negatives.shape[0], negatives.shape[1]
        neg_repr = self._representation_net(
            negatives.view(B * K, *negatives.shape[2:])  # (B*K,C,H,W)
        )
        neg_repr = neg_repr.view(B, K, -1).permute(1, 0, 2)  # (K,B,D)

        # Loss
        pos_term = (obs_repr - pos_repr).pow(2).sum()
        neg_mean_dist = (obs_repr - neg_repr).pow(2).sum(dim=[0, 2]).mean()
        neg_term = self._negative_loss_ratio * (self._negative_distance_target - neg_mean_dist).pow(2)

        loss = pos_term + neg_term

        # Stats (extra, doesn’t affect gradients)
        obs_norm = obs_repr.norm(dim=1).mean()
        obs_std = obs_repr.std(dim=0, unbiased=False).mean()
        stats = {
            "sdm_loss": float(loss.detach().cpu().item()),
            "sdm_pos_term": float(pos_term.detach().cpu().item()),
            "sdm_neg_mean_dist": float(neg_mean_dist.detach().cpu().item()),
            "sdm_neg_term": float(neg_term.detach().cpu().item()),
            "sdm_repr_norm_mean": float(obs_norm.detach().cpu().item()),
            "sdm_repr_std_mean": float(obs_std.detach().cpu().item()),
            "sdm_obs_repr_min": float(obs_repr.min().detach().cpu().item()),
            "sdm_obs_repr_max": float(obs_repr.max().detach().cpu().item()),
        }
        return loss, stats


    def train(self, buffer_data):
        train_loader = self.prepare_train_loader(buffer_data)
        self._representation_net.train()

        epoch_stats = {}
        for _epoch in range(self._num_training_epochs):
            running = {}
            count = 0
            did_input_log = False

            for _i, batch in enumerate(train_loader, 0):
                obs, positive, negatives = batch  # obs:(B,3,64,64) u8, neg:(B,K,3,64,64) u8

                if not did_input_log:
                    input_stats = {}
                    input_stats.update(tstats(obs, "sdm_in_obs_u8"))
                    input_stats.update(tstats(positive, "sdm_in_pos_u8"))
                    input_stats.update(
                        tstats(negatives.view(-1, *negatives.shape[2:]), "sdm_in_neg_u8_flat")
                    )

                obs = obs.to(self._device, non_blocking=True)
                positive = positive.to(self._device, non_blocking=True)
                negatives = negatives.to(self._device, non_blocking=True)

                obs_f = preprocess_uint8_batch(obs)
                pos_f = preprocess_uint8_batch(positive)
                neg_f = negatives.to(torch.float32).div_(255.0).sub_(0.5)

                if not did_input_log:
                    input_stats.update(tstats(obs_f, "sdm_in_obs_pre"))
                    input_stats.update(tstats(pos_f, "sdm_in_pos_pre"))
                    input_stats.update(
                        tstats(neg_f.view(-1, *neg_f.shape[2:]), "sdm_in_neg_pre_flat")
                    )
                    bs0 = int(obs.shape[0])
                    for k, v in input_stats.items():
                        running[k] = running.get(k, 0.0) + float(v) * bs0
                    did_input_log = True

                self._optimizer.zero_grad(set_to_none=True)
                loss, stats = self.calculate_loss_terms(obs_f, pos_f, neg_f)
                loss.backward()
                self._optimizer.step()

                bs = int(obs_f.shape[0])
                for k, v in stats.items():
                    running[k] = running.get(k, 0.0) + float(v) * bs
                count += bs

            epoch_stats = {k: v / max(count, 1) for k, v in running.items()}

        self._representation_net.eval()
        sdm_out_stats = self.learn_representation_stats(buffer_data)

        return {
            **epoch_stats,
            **sdm_out_stats,
            "sdm_repr_mean_set": float(self._repr_mean is not None),
            "sdm_repr_std_set": float(self._repr_std is not None),
            "sdm_repr_mean_abs_mean": float(np.mean(np.abs(self._repr_mean))) if self._repr_mean is not None else 0.0,
            "sdm_repr_std_mean": float(np.mean(self._repr_std)) if self._repr_std is not None else 0.0,
        }

    @torch.no_grad()
    def learn_representation_stats(self, data):
        """Compute mean/std of representations for normalization."""
        obs = torch.as_tensor(data["observation"], device=self._device)  # uint8
        if obs.ndim == 4 and obs.shape[-1] in (1, 3):
            obs = obs.permute(0, 3, 1, 2)
        obs = obs.to(torch.float32).div_(255.0).sub_(0.5)

        self._representation_net.eval()
        reprs = self._representation_net(obs).detach().cpu().numpy()  # (N, D)

        stats = {
            "sdm_out_min": float(reprs.min()),
            "sdm_out_max": float(reprs.max()),
            "sdm_out_mean": float(reprs.mean()),
            "sdm_out_std": float(reprs.std()),
            "sdm_out_abs_mean": float(np.mean(np.abs(reprs))),
            "sdm_out_p01": float(np.quantile(reprs, 0.01)),
            "sdm_out_p50": float(np.quantile(reprs, 0.50)),
            "sdm_out_p99": float(np.quantile(reprs, 0.99)),
            "sdm_out_norm_mean": float(np.mean(np.linalg.norm(reprs, axis=1))),
            "sdm_out_norm_p99": float(np.quantile(np.linalg.norm(reprs, axis=1), 0.99)),
        }

        if self._normalize_representations:
            self._repr_mean = reprs.mean(axis=0)
            self._repr_std = reprs.std(axis=0) + 1e-8

            z = (reprs - self._repr_mean) / self._repr_std
            stats.update({
                "sdm_out_normed_min": float(z.min()),
                "sdm_out_normed_max": float(z.max()),
                "sdm_out_normed_mean": float(z.mean()),
                "sdm_out_normed_std": float(z.std()),
                "sdm_out_normed_p01": float(np.quantile(z, 0.01)),
                "sdm_out_normed_p99": float(np.quantile(z, 0.99)),
            })

        return stats

    @torch.no_grad()
    def get_representation(self, obs):
        obs = obs.to(self._device, non_blocking=True)
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)

        if obs.dtype == torch.uint8:
            obs = obs.to(torch.float32).div_(255.0).sub_(0.5)
        else:
            obs = obs.to(torch.float32)  # already in [-0.5, 0.5] from preprocess_obs

        reprs = self._representation_net(obs).squeeze(0).detach().cpu().numpy()

        if self._normalize_representations and (self._repr_mean is not None):
            reprs = (reprs - self._repr_mean) / self._repr_std
        return reprs

    def save(self, dname):
        torch.save(
            {
                "representation_net": self._representation_net.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "repr_mean": self._repr_mean,
                "repr_std": self._repr_std,
                "normalize_representations": self._normalize_representations,
            },
            os.path.join(dname, "distance_model.pt"),
        )

    def load(self, dname):
        checkpoint = torch.load(os.path.join(dname, "distance_model.pt"), map_location=self._device)
        self._representation_net.load_state_dict(checkpoint["representation_net"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])

        self._repr_mean = checkpoint.get("repr_mean", None)
        self._repr_std = checkpoint.get("repr_std", None)
        self._normalize_representations = checkpoint.get(
            "normalize_representations", self._normalize_representations
        )