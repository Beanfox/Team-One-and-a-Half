import torch
import torch.nn as nn
import torch.nn.functional as F

CONTEXT_LEN = 20  # Must match train.py

class TrafficDecisionTransformer(nn.Module):
    def __init__(self, state_dim=10, act_dim=2, hidden_size=128, max_length=CONTEXT_LEN, num_layers=3, num_heads=4):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        # 1. Linear Embeddings
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Embedding(act_dim, hidden_size) 
        self.embed_return = nn.Linear(1, hidden_size)
        
        # 2. Time/Positional Embedding
        # Use max_length for the positional embedding (matches context window)
        self.embed_timestep = nn.Embedding(max_length, hidden_size)

        # 3. LayerNorm on the input embeddings (stabilizes training)
        self.embed_ln = nn.LayerNorm(hidden_size)

        # 4. The Core Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Action Prediction Head
        self.predict_action = nn.Linear(hidden_size, act_dim)

    def forward(self, states, actions, returns_to_go, timesteps):
        """
        states: (Batch, K, 10)
        actions: (Batch, K)
        returns_to_go: (Batch, K, 1)
        timesteps: (Batch, K) - values must be in [0, max_length)
        """
        batch_size, seq_length = states.shape[0], states.shape[1]

        # Clamp timesteps to valid range for the embedding table
        timesteps = timesteps.clamp(0, self.max_length - 1)

        # 1. Embed everything
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # 2. Add time embeddings to each modality
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # 3. Interleave tokens: (R_1, s_1, a_1, R_2, s_2, a_2 ...)
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=2
        )
        # Reshape to (Batch, K * 3, hidden_size)
        sequence = stacked_inputs.reshape(batch_size, seq_length * 3, self.hidden_size)
        
        # Apply LayerNorm
        sequence = self.embed_ln(sequence)

        # 4. Create Causal Mask 
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_length * 3).to(states.device)

        # 5. Pass through Transformer
        transformer_outputs = self.transformer(sequence, mask=causal_mask, is_causal=True)

        # 6. Extract the state representations to predict the next action
        # The state token is the 2nd token in every 3-token block (index 1, 4, 7...)
        state_outputs = transformer_outputs[:, 1::3, :]

        # 7. Predict action logits
        action_logits = self.predict_action(state_outputs)
        
        return action_logits

    @torch.no_grad()
    def get_action(self, states, actions, returns_to_go, timesteps):
        """
        Autoregressive action selection for a single environment step.
        Input sequences can be any length; we truncate to max_length (context window).
        """
        # Truncate to context window
        states = states[:, -self.max_length:]
        actions = actions[:, -self.max_length:]
        returns_to_go = returns_to_go[:, -self.max_length:]
        timesteps = timesteps[:, -self.max_length:]

        seq_length = states.shape[1]
        
        # Pad on the LEFT if shorter than context window
        if seq_length < self.max_length:
            pad_len = self.max_length - seq_length
            states = F.pad(states, (0, 0, pad_len, 0), value=0.0)
            actions = F.pad(actions, (pad_len, 0), value=0)
            returns_to_go = F.pad(returns_to_go, (0, 0, pad_len, 0), value=0.0)
            timesteps = F.pad(timesteps, (pad_len, 0), value=0)

        # Forward pass
        action_logits = self.forward(states, actions, returns_to_go, timesteps)
        
        # Get the logits for the last (most recent) position
        latest_logits = action_logits[0, -1, :]
        
        # Greedy action selection
        action = torch.argmax(latest_logits).item()
        
        return action
