"""
Turns a saved checkpoint into a policy function obs -> action, so scripts
(the viewer) can run trained agents without importing the notebooks.
"""
import torch
import torch.nn as nn

def load_policy(path):
    """Load a PPO checkpoint and return a deterministic policy obs -> action (Gaussian mean)."""
    ckpt = torch.load(path, map_location="cpu")
    h1, h2 = ckpt["hparams"]["hidden"]
    obs_dim, act_dim = ckpt["obs_dim"], ckpt["act_dim"]
    actor = nn.Sequential(
        nn.Linear(obs_dim, h1), nn.Tanh(),
        nn.Linear(h1, h2), nn.Tanh(),
        nn.Linear(h2, act_dim),
    )

    prefix = "actor_mean."
    weights = {k[len(prefix):]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)}
    actor.load_state_dict(weights)
    actor.eval()

    def policy(obs):
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)   # (14,) -> (1, 14)
        with torch.no_grad():
            action = actor(x)                                      # mean of the Gaussian, no sampling
        return action.squeeze(0).numpy()                         # (1, 4) -> (4,)

    return policy
