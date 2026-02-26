'''
GC PPO — Graph coloring env: PyG Data inside, NetworkX only for viz.
'''
import networkx as nx
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, erdos_renyi_graph
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

class GraphColoringEnv:
    def __init__(self, data, num_colors):
        """
        Args:
            data: torch_geometric.data.Data with .edge_index and .num_nodes (or inferred).
            num_colors: int, number of colors k.
        """
        self.data = data
        self.n_nodes = data.num_nodes
        self.num_colors = num_colors

    def to_networkx(self, node_attr=None):
        """Export to NetworkX for visualization. Optionally pass node attributes (e.g. coloring)."""
        g = to_networkx(self.data, to_undirected=True, remove_self_loops=True)
        if node_attr is not None:
            nx.set_node_attributes(g, node_attr, name="color")
        return g

    def reset(self):
        # State: all nodes uncolored (-1 = uncolored)
        self.coloring = torch.full((self.n_nodes,), -1, dtype=torch.long)
        self.step_count = 0
        self.done = False

        # Obs: PyG Data with node features = current color encoding (one-hot: uncolored + k colors)
        # uncolored -> [1,0,...,0], color c -> zeros with 1 at index c+1
        x = torch.zeros(self.n_nodes, self.num_colors + 1)
        x[:, 0] = 1.0  # all uncolored
        obs_data = Data(edge_index=self.data.edge_index, x=x)
        obs_data.num_nodes = self.n_nodes

        # Legal actions: (node, color) with node uncolored -> all nodes uncolored, so all (node, color) valid
        action_mask = self._get_action_mask()

        info = {"action_mask": action_mask}
        return obs_data, info

    def _get_action_mask(self):
        """Boolean [n_nodes, num_colors]: True if (node, color) is legal (node uncolored)."""
        uncolored = self.coloring == -1
        return uncolored.unsqueeze(1).expand(-1, self.num_colors)

    def _coloring_to_x(self):
        """Node features from current coloring: one-hot [uncolored, color0, ..., color_k-1]."""
        x = torch.zeros(self.n_nodes, self.num_colors + 1)
        uncolored = self.coloring == -1
        x[uncolored, 0] = 1.0
        for c in range(self.num_colors):
            x[self.coloring == c, c + 1] = 1.0
        return x

    def step(self, action):
        node, color = action
        # Update state: assign color to node
        self.coloring[node] = color
        self.step_count += 1

        # Reward: dense = -number of conflicts this assignment causes (neighbors already with same color)
        row, col = self.data.edge_index[0], self.data.edge_index[1]
        neighbors_of_node = col[row == node].unique()
        conflicts = (self.coloring[neighbors_of_node] == color).sum().item()
        reward = -float(conflicts)

        # Done when all nodes colored
        self.done = self.step_count >= self.n_nodes
        terminated = self.done
        truncated = False

        # Next obs: same graph, updated node features
        x = self._coloring_to_x()
        obs_data = Data(edge_index=self.data.edge_index, x=x)
        obs_data.num_nodes = self.n_nodes
        action_mask = self._get_action_mask()
        info = {"action_mask": action_mask, "conflicts": conflicts}

        return obs_data, reward, terminated, truncated, info
    
class GraphColoringPolicy(nn.Module):
    """GNN policy: obs (Data) + action_mask -> logits over flat (node, color), sample with mask."""

    def __init__(self, in_dim, num_colors, hidden_dim=64):
        super().__init__()
        self.num_colors = num_colors
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_colors)  # per-node logits over colors

    def forward(self, data, action_mask):
        # data.x: [n_nodes, in_dim], data.edge_index: [2, E]
        x = self.conv1(data.x, data.edge_index).relu()
        x = self.conv2(x, data.edge_index).relu()   # [n_nodes, hidden_dim]
        logits = self.head(x)                      # [n_nodes, num_colors]
        logits_flat = logits.view(-1)              # [n_nodes * num_colors]
        mask_flat = action_mask.reshape(-1)        # .reshape handles non-contiguous (e.g. from .expand)
        logits_flat = torch.where(
            mask_flat,
            logits_flat,
            torch.finfo(logits_flat.dtype).min,
        )
        return logits_flat

    def act(self, data, action_mask, deterministic=False):
        logits = self.forward(data, action_mask)
        dist = Categorical(logits=logits)
        if deterministic:
            action_flat = logits.argmax(dim=-1).item()
        else:
            action_flat = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action_flat, device=logits.device))
        node = action_flat // self.num_colors
        color = action_flat % self.num_colors
        return (node, color), log_prob
    
def rollout(env, policy, device, max_steps=None, deterministic=False):
    """Run one episode; return trajectory for PPO (obs, action_flat, log_prob, reward)."""
    if max_steps is None:
        max_steps = env.n_nodes
    obs, info = env.reset()
    obs = obs.to(device)
    action_mask = info["action_mask"].to(device)

    trajectory = []
    for _ in range(max_steps):
        (node, color), log_prob = policy.act(obs, action_mask, deterministic=deterministic)
        action_flat = node * env.num_colors + color
        next_obs, reward, terminated, truncated, info = env.step((node, color))
        trajectory.append({
            "obs": obs,
            "action_flat": action_flat,
            "action_mask": action_mask,
            "log_prob": log_prob,
            "reward": reward,
        })

        if terminated or truncated:
            break
        obs = next_obs.to(device)
        action_mask = info["action_mask"].to(device)

    return trajectory

def compute_returns(rewards, gamma, device=None):
    """Compute returns for PPO: discounted sum of rewards.
    rewards: list of floats or 1D tensor. If list, pass device.
    """
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device or torch.device("cpu"))
    T = len(rewards)
    returns = torch.zeros(T, device=rewards.device, dtype=rewards.dtype)
    R_next = 0.0
    for t in range(T - 1, -1, -1):
        R_t = rewards[t].item() + gamma * R_next
        returns[t] = R_t
        R_next = R_t
    return returns

def advantages(returns):
    """Compute advantages for PPO: returns - value."""
    if not isinstance(returns, torch.Tensor):
        returns = torch.as_tensor(returns)
    returns_tensor = returns.detach().clone()
    mean_return = returns_tensor.mean()
    advantages = returns_tensor - mean_return
    std_adv = advantages.std()
    if std_adv > 0:
        advantages = advantages / (std_adv + 1e-8)
    return advantages
    

def ppo_update(policy, optimizer, trajectory, returns, advantages, epsilon=0.2, entropy_coef=0.01):
    """PPO update: policy gradient with clipped surrogate + entropy bonus."""
    policy_loss_total = 0.0
    entropy_total = 0.0
    for t in range(len(trajectory)):
        obs = trajectory[t]['obs']
        action_flat = trajectory[t]['action_flat']
        action_mask = trajectory[t]['action_mask']
        old_log_prob = trajectory[t]['log_prob'].detach()
        adv_t = advantages[t]

        logits = policy.forward(obs, action_mask)
        dist = Categorical(logits=logits)
        action_flat_t = torch.tensor(action_flat, device=logits.device, dtype=torch.long)
        new_log_prob = dist.log_prob(action_flat_t)
        entropy = dist.entropy()

        ratio = torch.exp(new_log_prob - old_log_prob)
        surr1 = ratio * adv_t
        surr2 = torch.clip(ratio, 1 - epsilon, 1 + epsilon) * adv_t
        policy_loss_t = -torch.min(surr1, surr2) - entropy_coef * entropy
        policy_loss_total += policy_loss_t
        entropy_total += entropy.item()

    policy_loss = policy_loss_total / len(trajectory)
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    return policy_loss.item(), entropy_total / len(trajectory)      
        
def train_ppo(env, policy, optimizer, device, num_episodes=3000, max_steps=None, gamma=0.99, epsilon=0.5, log_every=100, reward_scale=10.0, entropy_coef=0.001, ppo_epochs=4):
    """Train PPO policy on graph coloring env.

    Each episode: rollout -> scale rewards -> returns -> advantages -> multiple PPO updates (ppo_epochs).
    Every num_episodes/5 episodes, runs a deterministic rollout and stores (episode, coloring, conflicts)
    for later animation. Returns (policy, snapshots).
    """
    policy.to(device)
    checkpoint_interval = max(1, num_episodes // 5)
    snapshots = []  # list of (episode, coloring_tensor, conflicts)

    for episode in range(num_episodes):
        trajectory = rollout(env, policy, device, max_steps=max_steps)
        if len(trajectory) == 0:
            continue

        rewards = [t["reward"] for t in trajectory]
        rewards_scaled = [r * reward_scale for r in rewards]
        returns = compute_returns(rewards_scaled, gamma, device)
        advs = advantages(returns)

        for _ in range(ppo_epochs):
            loss, mean_entropy = ppo_update(policy, optimizer, trajectory, returns, advs, epsilon=epsilon, entropy_coef=entropy_coef)

        total_reward = sum(rewards)
        if (episode + 1) % log_every == 0 or episode == 0:
            adv_mean = advs.mean().item()
            adv_max = advs.max().item()
            print(f"Ep {episode + 1}/{num_episodes}  reward={total_reward:.1f}  loss={loss:.4f}  entropy={mean_entropy:.4f}  adv_mean={adv_mean:.4f}  adv_max={adv_max:.4f}")

        # Every num_episodes/5, run deterministic policy and store coloring for progression animation
        if (episode + 1) % checkpoint_interval == 0 or episode == 0:
            _ = rollout(env, policy, device, deterministic=True)
            snapshots.append((episode + 1, env.coloring.clone().cpu(), count_conflicts(env)))

    return policy, snapshots


def make_synthetic_graph(n_nodes, edge_prob=0.3, seed=0):
    """Create a random graph as PyG Data (Erdos-Renyi). Ensures at least one edge."""
    torch.manual_seed(seed)
    edge_index = erdos_renyi_graph(n_nodes, edge_prob)
    if edge_index.numel() == 0 and n_nodes >= 2:
        # ensure at least one edge so GNN ops don't hit empty-edge edge cases
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    return Data(edge_index=edge_index, num_nodes=n_nodes)


def count_conflicts(env):
    """Return number of edges whose endpoints have the same color (only valid after full coloring)."""
    row, col = env.data.edge_index[0], env.data.edge_index[1]
    same = env.coloring[row] == env.coloring[col]
    return same.sum().item()


def visualize_coloring(env, save_path=None, title="Graph coloring"):
    """Draw the graph with node colors. Requires env.coloring to be set (e.g. after a rollout)."""
    g = env.to_networkx(node_attr={i: int(env.coloring[i].item()) for i in range(env.n_nodes)})
    color_ids = [g.nodes[i]["color"] for i in range(env.n_nodes)]
    palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62"]
    node_colors = [palette[c % len(palette)] for c in color_ids]
    pos = nx.spring_layout(g, seed=42)
    nx.draw(g, pos, node_color=node_colors, with_labels=True, font_weight="bold")
    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.tight_layout()
    plt.show()


_PALETTE = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62"]


def animate_coloring_progression(env, snapshots, save_path=None, interval=800, figsize=(6, 5)):
    """Animate the progression of graph colorings over training checkpoints.

    snapshots: list of (episode, coloring_tensor, conflicts) from train_ppo.
    save_path: if set (e.g. 'progression.gif'), save the animation to file.
    interval: milliseconds between frames.
    """
    if not snapshots:
        print("No snapshots to animate.")
        return
    g = env.to_networkx()
    pos = nx.spring_layout(g, seed=42)

    fig, ax = plt.subplots(figsize=figsize)

    def get_node_colors(coloring):
        return [_PALETTE[c % len(_PALETTE)] for c in coloring.tolist()]

    def update(frame):
        ax.clear()
        ep, coloring, conflicts = snapshots[frame]
        node_colors = get_node_colors(coloring)
        nx.draw(g, pos, node_color=node_colors, with_labels=True, font_weight="bold", ax=ax)
        ax.set_title(f"Episode {ep}  |  Conflicts: {conflicts}")
        return ax,

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=interval, blit=False, repeat=True)
    plt.tight_layout()
    if save_path:
        path = save_path if save_path.endswith(".gif") else save_path + ".gif"
        writer = PillowWriter(fps=1000 // interval)
        anim.save(path, writer=writer)
        print(f"Saved animation to {path}")
    else:
        plt.show()
    return anim


def main():
    # 1. Denser synthetic graph and env (more edges = stronger learning signal)
    n_nodes, num_colors = 10, 3
    data = make_synthetic_graph(n_nodes, edge_prob=0.5, seed=42)
    env = GraphColoringEnv(data, num_colors)
    num_edges = data.edge_index.size(1)
    print(f"Graph: {n_nodes} nodes, {num_edges} edges, {num_colors} colors")

    # 2. Policy and optimizer (higher lr so policy can actually move)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = GraphColoringPolicy(in_dim=num_colors + 1, num_colors=num_colors, hidden_dim=64)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # 3. Train: looser clip (epsilon=0.5), lower entropy so policy can specialize, multiple PPO epochs per trajectory
    print("Training PPO...")
    policy, snapshots = train_ppo(
        env, policy, optimizer, device,
        num_episodes=3000,
        gamma=0.99,
        epsilon=0.5,
        log_every=100,
        reward_scale=10.0,
        entropy_coef=0.001,
        ppo_epochs=4,
    )
    print(f"Stored {len(snapshots)} checkpoints for progression animation (every num_episodes/5).")

    # 4. Test: deterministic (greedy) and stochastic rollouts
    print("\nRunning deterministic rollout (greedy)...")
    _ = rollout(env, policy, device, deterministic=True)
    conflicts_det = count_conflicts(env)
    print(f"Deterministic: coloring={env.coloring.tolist()}  conflicts={conflicts_det}")

    # A few stochastic rollouts to compare
    conflict_list = []
    for _ in range(5):
        _ = rollout(env, policy, device, deterministic=False)
        conflict_list.append(count_conflicts(env))
    print(f"Stochastic (5 runs) conflicts: {conflict_list}  mean={sum(conflict_list)/len(conflict_list):.1f}")

    # 5. Visualize final deterministic result
    _ = rollout(env, policy, device, deterministic=True)
    conflicts = count_conflicts(env)
    visualize_coloring(env, save_path="gc_result.png", title=f"Graph coloring (conflicts={conflicts})")

    # 6. Run progression animation (policy improving over training)
    print("\nProgression animation (checkpoints every num_episodes/5)...")
    animate_coloring_progression(env, snapshots, save_path="gc_progression.gif", interval=800)


if __name__ == "__main__":
    main()