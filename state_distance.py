import os
import numpy as np
import torch
from torch import nn

from torch.utils.data import Dataset, DataLoader


class ContrastiveStateDistanceDataset(Dataset):
    def __init__(self, observation_pairs, num_negative_samples=128, transform=None):
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
        # indices = self._rng.integers(len(self.data), size=self.num_negative_samples)
        indices = np.random.randint(len(self.data), size=self.num_negative_samples)
        negatives = np.array([self.data[ind][0] for ind in indices])

        return obs, positive, negatives


def calculate_output_dim(net, input_shape):
    if isinstance(input_shape, int):
        input_shape = (input_shape,)
    placeholder = torch.zeros((0,) + tuple(input_shape))
    output = net(placeholder)
    return output.size()[1:]


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

    def prepare_train_loader(self, data):
        observation_pairs = []
        for i in range(data["observation"].shape[0] - 1):
            if data["terminal"][i]:
                continue
            observation_pairs.append(
                (data["observation"][i], data["observation"][i + 1])
            )
        train_dataset = ContrastiveStateDistanceDataset(
            observation_pairs, self._num_negative_samples
        )
        return DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True)

    def calculate_loss(self, obs, positive, negatives):
        obs_repr = self._representation_net(obs)
        positive_repr = self._representation_net(positive)
        negatives_repr = self._representation_net(
            negatives.view(
                (negatives.shape[0] * negatives.shape[1], *negatives.shape[2:])
            )
        )
        negatives_repr = negatives_repr.view(
            (negatives.shape[0], negatives.shape[1], *negatives_repr.shape[1:])
        ).permute((1, 0, 2))

        loss = (obs_repr - positive_repr).pow(2).sum()
        loss += (
            self._negative_loss_ratio
            * (
                self._negative_distance_target
                - (obs_repr - negatives_repr).pow(2).sum(dim=[0, 2]).mean()
            )
            ** 2
        )
        return loss

    def train(self, buffer_data):
        train_loader = self.prepare_train_loader(buffer_data)
        self._representation_net.train()
        for _ in range(self._num_training_epochs):
            running_loss, dataset_size = 0, 0
            for i, data in enumerate(train_loader, 0):
                obs, positive, negatives = (
                    data[0].float().to(self._device),
                    data[1].float().to(self._device),
                    data[2].float().to(self._device),
                )
                self._optimizer.zero_grad()
                loss = self.calculate_loss(obs, positive, negatives)
                loss.backward()
                self._optimizer.step()

                running_loss += loss.item() * obs.shape[0]
                dataset_size += obs.shape[0]

            # TODO log things somehow :D

        self._representation_net.eval()
        self.learn_representation_stats(buffer_data)

    @torch.no_grad()
    def learn_representation_stats(self, data):
        """Compute mean/std of representations for normalization."""
        if not self._normalize_representations:
            return

        obs = torch.as_tensor(data["observation"], device=self._device).float()
        if obs.ndim == 4 and obs.shape[-1] in (1, 3):
            obs = obs.permute(0, 3, 1, 2)

        self._representation_net.eval()
        reprs = self._representation_net(obs).detach().cpu().numpy()
        self._repr_mean = reprs.mean(axis=0)
        self._repr_std = reprs.std(axis=0) + 1e-8

    @torch.no_grad()
    def get_representation(self, obs):
        obs = obs.to(self._device).float()
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
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
