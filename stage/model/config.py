"""Architecture hyperparameters of STAGEEncoder and CrossAttnPosHead."""

# Encoder
HIDDEN_DIM = 64           # input projection width
EMBED_DIM = 64            # spatial/temporal embedding width
N_HEADS = 4               # heads of every encoder attention layer (GATv2Conv and temporal)
N_SPATIAL_LAYERS = 2      # GATv2Conv layers
N_TEMPORAL_LAYERS = 2     # temporal self-attention layers
MAX_WINDOW_SIZE = 100     # length of the temporal positional encoding table

# Position decoder
DECODER_HIDDEN_DIM = 256  # feedforward width of the decoder layers and the position MLP
DECODER_N_HEADS = 4
DECODER_N_LAYERS = 2
