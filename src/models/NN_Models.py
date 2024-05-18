import torch.nn as nn
import torch
from utils.ann_utils import train
import torch.nn.functional as F


class NeighborNN(nn.Module):
    """
    Class of Neighbor NN, it takes the features of the closest neighbors and concatenates them to
    the features of the x you are trying to predict.
    """

    def __init__(
        self,
        n_neighbors,
        hidden=126,
        activation_function="tanh",
        n_layers: int = 1,
        p=2,
    ) -> None:
        super().__init__()
        self.X_data = None
        self.y = None
        self.n_neighbors = n_neighbors
        self.p = p
        self.hidden = hidden
        self.n_features = None  # X.shape[1]
        self.l1 = None  # nn.Linear((n_features + 1) * n_neighbors + n_features, hidden)
        self.activation_function = (
            nn.Tanh() if activation_function == "tanh" else nn.ReLU()
        )
        self.n_layers = n_layers
        if self.n_layers > 1:
            self.hidden_layers = nn.Sequential(
                *[
                    (
                        nn.Linear(hidden, hidden)
                        if i % 2 == 0
                        else self.activation_function
                    )
                    for i in range(2 * (self.n_layers - 1))
                ]
            )

        self.last_layer = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # If it is training, we don't want to use itself as a neighbor
        # So we skip the closest neighbor (itself) because it has a distance of 0
        if self.training:
            displacement = 1
        else:
            displacement = 0

        # Calculate distances
        distances = torch.cdist(x, self.X_data, p=self.p)

        # Get nearest neighbors
        k_nearest_neighbours_index = torch.argsort(distances, 1)[
            :, displacement : self.n_neighbors + displacement
        ]
        neighbors = self.X_data[k_nearest_neighbours_index]
        neighbors_y = torch.unsqueeze(self.y[k_nearest_neighbours_index], 2)

        neighbors_features = torch.cat((neighbors, neighbors_y), 2)
        # neighbors_permuted = torch.permute(neighbors_features, (1, 0, 2))

        # Feed forward
        o = torch.cat((x, neighbors_features.flatten(1, 2)), 1)
        o = self.l1(o)
        o = self.activation_function(o)
        if self.n_layers > 1:
            o = self.hidden_layers(o)
        o = self.last_layer(o)
        o = self.sigmoid(o)
        return o

    def fit(
        self, X, y, epochs=50, batch_size=64, device="cpu", lr=0.01, l2_lambda=1e-3
    ) -> None:

        self.n_features = X.shape[1]
        self.X_data = X.to(device)
        self.y = y.to(device)

        self.l1 = nn.Linear(
            (self.n_features + 1) * self.n_neighbors + self.n_features, self.hidden
        ).to(device)

        criterion = F.binary_cross_entropy
        self.train()
        train(
            X_train=X,
            y_train=y,
            model=self,
            criterion=criterion,
            learning_rate=lr,
            l2_lambda=l2_lambda,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            X_val=None,
            y_val=None,
        )

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return (self(x) >= 0.5).squeeze().float().cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        return self(x).squeeze().float().cpu().numpy()


class MLP(nn.Module):
    """
    A simple multilayer perceptron
    """

    def __init__(
        self, hidden=5, activation_function: str = "tanh", n_layers: int = 1
    ) -> None:
        super(MLP, self).__init__()
        self.fan_in = None
        self.hidden = hidden
        self.l1 = None  # nn.Linear(fan_in, hidden)

        self.activation_function = (
            nn.Tanh() if activation_function == "tanh" else nn.ReLU()
        )
        self.n_layers = n_layers
        self.hidden_layers = None
        if self.n_layers > 1:
            self.hidden_layers = nn.Sequential(
                *[
                    (
                        nn.Linear(hidden, hidden)
                        if i % 2 == 0
                        else self.activation_function
                    )
                    for i in range(2 * (self.n_layers - 1))
                ]
            )

        self.last_layer = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.activation_function(x)
        if self.n_layers > 1:
            x = self.hidden_layers(x)
        x = self.last_layer(x)
        o = self.sigmoid(x)
        return o

    def fit(
        self, X, y, epochs=50, batch_size=64, device="cpu", lr=0.01, l2_lambda=1e-3
    ) -> None:
        criterion = F.binary_cross_entropy
        self.fan_in = X.shape[1]
        self.l1 = nn.Linear(self.fan_in, self.hidden).to(device)

        train(
            X_train=X,
            y_train=y,
            model=self,
            criterion=criterion,
            learning_rate=lr,
            l2_lambda=l2_lambda,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            X_val=None,
            y_val=None,
        )

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return (self.forward(x) >= 0.5).float().cpu().numpy().reshape((-1))

    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        return self.forward(x).float().cpu().numpy().reshape((-1))


class NeighborCompressedNN(nn.Module):
    """
    It takes the closes neighbors of the x you are trying to predict on and
    and calculates a feature vector based on their features that is then concatenated
    to de features of the current x.
    """

    def __init__(
        self,
        n_neighbors,
        compression_size=10,
        hidden=126,
        activation_function="tanh",
        n_layers: int = 1,
        p=2,
    ) -> None:
        super().__init__()
        self.X_data = None
        self.y = None
        self.n_neighbors = n_neighbors
        self.hidden = hidden
        self.p = p
        self.n_features = None
        self.compression_size = compression_size
        self.neighbors_gate = None  # nn.Linear(n_features + 1, compression_size)
        self.l1 = None  # nn.Linear(compression_size + n_features, hidden)
        self.activation_function = (
            nn.Tanh() if activation_function == "tanh" else nn.ReLU()
        )

        self.n_layers = n_layers
        self.hidden_layers = None
        if self.n_layers > 1:
            self.hidden_layers = nn.Sequential(
                *[
                    (
                        nn.Linear(hidden, hidden)
                        if i % 2 == 0
                        else self.activation_function
                    )
                    for i in range(2 * (self.n_layers - 1))
                ]
            )

        self.last_layer = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):

        # If it is training, we don't want to use itself as a neighbor
        # So we skip the closest neighbor (itself) because it has a distance of 0
        if self.training:
            displacement = 1
        else:
            displacement = 0

        # Calculate distances
        distances = torch.cdist(x, self.X_data, p=self.p)

        # Get nearest neighbors
        k_nearest_neighbours_index = torch.argsort(distances, 1)[
            :, displacement : self.n_neighbors + displacement
        ]
        neighbors = self.X_data[k_nearest_neighbours_index]
        neighbors_y = torch.unsqueeze(self.y[k_nearest_neighbours_index], 2)

        neighbors_features = torch.cat((neighbors, neighbors_y), 2)
        neighbors_features = self.neighbors_gate(neighbors_features)
        neighbors_features = torch.sum(self.tanh(neighbors_features), dim=1)

        # Feed forward
        o = torch.cat((x, neighbors_features), 1)
        o = self.l1(o)
        o = self.activation_function(o)

        if self.n_layers > 1:
            x = self.hidden_layers(o)

        o = self.last_layer(o)
        o = self.sigmoid(o)
        return o

    def fit(
        self, X, y, epochs=50, batch_size=64, device="cpu", lr=0.01, l2_lambda=1e-3
    ) -> None:

        self.n_features = X.shape[1]
        self.X_data = X.to(device)
        self.y = y.to(device)
        self.neighbors_gate = nn.Linear(self.n_features + 1, self.compression_size).to(
            device
        )
        self.l1 = nn.Linear(self.compression_size + self.n_features, self.hidden).to(
            device
        )

        criterion = F.binary_cross_entropy
        self.train()
        train(
            X_train=X,
            y_train=y,
            model=self,
            criterion=criterion,
            learning_rate=lr,
            l2_lambda=l2_lambda,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            X_val=None,
            y_val=None,
        )

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return (self(x) >= 0.5).squeeze().float().cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        return self(x).squeeze().float().cpu().numpy()


class DNeighborCompressedNN(nn.Module):
    """
    It takes the closes neighbors of the x you are trying to predict on and
    and calculates a feature vector based on their features that is then concatenated
    to de features of the current x.
    """

    def __init__(
        self,
        n_neighbors,
        compression_size=10,
        neighbor_size=10,
        hidden=126,
        activation_function="tanh",
        n_layers: int = 1,
        p=2,
    ) -> None:
        super().__init__()
        self.X_data = None
        self.y = None
        self.n_neighbors = n_neighbors
        self.hidden = hidden
        self.p = p
        self.n_features = None
        self.compression_size = compression_size
        self.neighbor_size = neighbor_size
        self.neighbors_gate = None  # nn.Linear(n_features + 1, compression_size)
        self.l1 = None  # nn.Linear(compression_size + n_features, hidden)
        self.activation_function = (
            nn.Tanh() if activation_function == "tanh" else nn.ReLU()
        )

        self.n_layers = n_layers
        self.hidden_layers = None
        if self.n_layers > 1:
            self.hidden_layers = nn.Sequential(
                *[
                    (
                        nn.Linear(hidden, hidden)
                        if i % 2 == 0
                        else self.activation_function
                    )
                    for i in range(2 * (self.n_layers - 1))
                ]
            )

        self.last_layer = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.middle_neighbor = nn.Linear(neighbor_size, neighbor_size)

        self.middle_neighbor2 = nn.Linear(neighbor_size, compression_size)

    def forward(self, x):

        # If it is training, we don't want to use itself as a neighbor
        # So we skip the closest neighbor (itself) because it has a distance of 0
        if self.training:
            displacement = 1
        else:
            displacement = 0

        # Calculate distances
        distances = torch.cdist(x, self.X_data, p=self.p)

        # Get nearest neighbors
        k_nearest_neighbours_index = torch.argsort(distances, 1)[
            :, displacement : self.n_neighbors + displacement
        ]
        neighbors = self.X_data[k_nearest_neighbours_index]
        neighbors_y = torch.unsqueeze(self.y[k_nearest_neighbours_index], 2)

        neighbors_features = torch.cat((neighbors, neighbors_y), 2)
        neighbors_features = self.neighbors_gate(neighbors_features)
        neighbors_features = torch.sum(self.tanh(neighbors_features), dim=1)

        neighbors_features = self.middle_neighbor(neighbors_features)
        neighbors_features = self.middle_neighbor2(neighbors_features)

        # Feed forward
        o = torch.cat((x, neighbors_features), 1)
        o = self.l1(o)
        o = self.activation_function(o)

        if self.n_layers > 1:
            x = self.hidden_layers(o)

        o = self.last_layer(o)
        o = self.sigmoid(o)
        return o

    def fit(
        self, X, y, epochs=50, batch_size=64, device="cpu", lr=0.01, l2_lambda=1e-3
    ) -> None:

        self.n_features = X.shape[1]
        self.X_data = X.to(device)
        self.y = y.to(device)
        self.neighbors_gate = nn.Linear(self.n_features + 1, self.neighbor_size).to(
            device
        )

        self.l1 = nn.Linear(self.compression_size + self.n_features, self.hidden).to(
            device
        )

        criterion = F.binary_cross_entropy
        self.train()
        train(
            X_train=X,
            y_train=y,
            model=self,
            criterion=criterion,
            learning_rate=lr,
            l2_lambda=l2_lambda,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            X_val=None,
            y_val=None,
        )

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return (self(x) >= 0.5).squeeze().float().cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        return self(x).squeeze().float().cpu().numpy()
