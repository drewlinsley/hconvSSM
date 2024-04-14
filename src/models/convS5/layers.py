# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ConvSSM/blob/main/LICENSE
#
# Written by Jimmy Smith
# ------------------------------------------------------------------------------


from flax import linen as nn


class SequenceLayer(nn.Module):
    """Defines a single layer with activation,
       layer/batch norm, pre/postnorm, dropout, etc"""
    ssm: nn.Module
    training: bool
    parallel: bool
    activation_fn: str = "gelu"
    dropout: float = 0.0
    use_norm: bool = True
    prenorm: bool = False
    per_layer_skip: bool = True
    num_groups: int = 32
    squeeze_excite: bool = False
    d_model: int = 0

    def setup(self):
        if self.activation_fn in ["relu"]:
            self.activation = nn.relu
        elif self.activation_fn in ["gelu"]:
            self.activation = nn.gelu
        elif self.activation_fn in ["swish"]:
            self.activation = nn.swish
        elif self.activation_fn in ["elu"]:
            self.activation = nn.elu
        elif self.activation_fn in ["softplus"]:
            self.activation = nn.softplus
        else:
            raise NotImplementedError(self.activation_fn)

        self.seq = self.ssm(parallel=self.parallel,
                            activation=self.activation,
                            num_groups=self.num_groups,
                            squeeze_excite=self.squeeze_excite)

        if self.use_norm:
            # self.norm = nn.LayerNorm()
            self.norm = nn.BatchNorm()

        # TODO: Need to figure out dropout strategy, maybe drop whole channels?
        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, u, x0):
        if self.per_layer_skip:
            skip = u
        else:
            skip = 0
        # Apply pre-norm if necessary
        if self.use_norm:
            if self.prenorm:
                # u = self.norm(u)
                u = self.norm(u, use_running_average=~self.training)
        x_L, u = self.seq(u, x0)
        u = self.drop(u)
        u = skip + u
        if self.use_norm:
            if not self.prenorm:
                # u = self.norm(u)
                u = self.norm(u, use_running_average=~self.training)
        return x_L, u


class HSequenceLayer(nn.Module):
    """Defines a single layer with activation,
       layer/batch norm, pre/postnorm, dropout, etc"""
    ssm: nn.Module
    training: bool
    parallel: bool
    activation_fn: str = "gelu"
    dropout: float = 0.0
    use_norm: bool = True
    prenorm: bool = False
    per_layer_skip: bool = True
    num_groups: int = 32
    squeeze_excite: bool = False
    d_model: int = 0

    def setup(self):
        if self.activation_fn in ["relu"]:
            self.activation = nn.relu
        elif self.activation_fn in ["gelu"]:
            self.activation = nn.gelu
        elif self.activation_fn in ["swish"]:
            self.activation = nn.swish
        elif self.activation_fn in ["elu"]:
            self.activation = nn.elu
        elif self.activation_fn in ["softplus"]:
            self.activation = nn.softplus
        else:
            raise NotImplementedError(self.activation_fn)

        assert self.d_model > 0, "Pass model dimensionality."

        self.i_neurons = self.ssm(parallel=self.parallel,
                            activation=self.activation,
                            num_groups=self.num_groups,
                            squeeze_excite=self.squeeze_excite)
        self.e_neurons = self.ssm(parallel=self.parallel,
                            activation=self.activation,
                            num_groups=self.num_groups,
                            squeeze_excite=self.squeeze_excite)
        self.projection = nn.Conv(features=self.d_model, kernel_size=(1, 1, 1), padding="SAME")

        if self.use_norm:
            self.i_norm = nn.BatchNorm()
            self.e_norm = nn.BatchNorm()

        # TODO: Need to figure out dropout strategy, maybe drop whole channels?
        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, u, x0):
        if self.per_layer_skip:
            skip = u
        else:
            skip = 0

        # Normalize FF drive
        if self.use_norm:
            if self.prenorm:
                # u = self.e_norm(u)
                u = self.e_norm(u, use_running_average=~self.training)

        e0, i0 = x0

        # Compute I response (negate u)
        u = self.activation(u)
        i_t, u = self.i_neurons(-u, e0)
        u = self.projection(u)
        # u = self.i_norm(u)
        u = self.i_norm(u, use_running_average=~self.training)
        u = self.activation(u)

        # Compute E response
        e_t, u = self.e_neurons(u, i0)

        # Package states
        x_L = [e_t, i_t]

        # Add the SSM dropout and skip
        u = self.drop(u)
        u = skip + u

        # Normalize recurrent output
        if self.use_norm:
            if not self.prenorm:
                # u = self.e_norm(u)
                u = self.e_norm(u, use_running_average=~self.training)
        return x_L, u


class StackedLayers(nn.Module):
    """Stacks S5 layers
     output: outputs LxbszxH_uxW_uxU sequence of outputs and
             a list containing the last state of each layer"""
    ssm: nn.Module
    n_layers: int
    training: bool
    parallel: bool
    layer_activation: str = "gelu"
    dropout: float = 0.0
    use_norm: bool = False
    prenorm: bool = False
    skip_connections: bool = False
    per_layer_skip: bool = True
    num_groups: int = 32
    squeeze_excite: bool = False
    horizontal_connections: bool = False
    d_model: int = 0

    def setup(self):

        if self.horizontal_connections:
            sl_class = HSequenceLayer
        else:
            sl_class = SequenceLayer

        self.layers = [
            sl_class(
                ssm=self.ssm,
                activation_fn=self.layer_activation,
                dropout=self.dropout,
                training=self.training,
                parallel=self.parallel,
                use_norm=self.use_norm,
                prenorm=self.prenorm,
                per_layer_skip=self.per_layer_skip,
                num_groups=self.num_groups,
                squeeze_excite=self.squeeze_excite,
                d_model=self.d_model
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, u, initial_states):
        # u is shape (L, bsz, d_in, im_H, im_W)
        # x0s is a list of initial arrays each of shape (bsz, d_model, im_H, im_W)
        last_states = []
        for i in range(len(self.layers)):
            if self.skip_connections:
                if i == 3:
                    layer9_in = u
                elif i == 6:
                    layer12_in = u

                if i == 8:
                    u = u + layer9_in
                elif i == 11:
                    u = u + layer12_in
            x_L, u = self.layers[i](u, initial_states[i])
            last_states.append(x_L)  # keep last state of each layer
        return last_states, u

