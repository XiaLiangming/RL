import torch

'''Experience Replay Buffer'''
class ReplayBuffer:
    def __init__(self,
                 capacity: int,
                 observation_space: tuple,
                 frame_stacking: int,
                 device: torch.device):
        self._size = 0
        self._capacity = capacity
        self._observation_space = observation_space
        self._frame_stacking = frame_stacking
        self._device = device
        self._idx = 0
        self._frames = torch.empty((self._capacity, *self._observation_space), dtype=torch.uint8, device=self._device, memory_format=torch.contiguous_format)
        self._actions = torch.empty((self._capacity, 1), dtype=torch.uint8, device=self._device)
        self._rewards = torch.empty((self._capacity, 1), dtype=torch.float32, device=self._device)
        self._dones = torch.empty((self._capacity, 1), dtype=torch.bool, device=self._device)
        self._valid_idx = torch.zeros(capacity, dtype=torch.bool, device=device)
        self._idx_offsets = torch.arange(-frame_stacking, 0, 1, device=device).view(1, -1)
        self._c = 0

    def push(self,
             frame: torch.Tensor,
             action: torch.Tensor,
             reward: torch.Tensor,
             done: torch.Tensor):
        self._frames[self._idx] = frame
        self._actions[self._idx] = action
        self._rewards[self._idx] = reward
        self._dones[self._idx] = done
        if self._c == self._frame_stacking:
            self._valid_idx[self._idx] = True
        if done:
            self._c = 0
        else:
            self._c = min(self._c + 1, self._frame_stacking)
        if self._size == self._capacity:
            self._valid_idx[(self._idx + self._frame_stacking) % self._capacity] = False
        self._idx = (self._idx + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int):
        valid_idx = torch.nonzero(self._valid_idx).view(-1)
        assert batch_size <= valid_idx.numel()
        indices = valid_idx[torch.randint(0, valid_idx.numel(), (batch_size,), device=self._device)]
        state_idx = ((indices.view(-1,1) + self._idx_offsets) % self._capacity).view(-1)
        next_state_idx = (state_idx + 1) % self._capacity
        states = self._frames[state_idx].view(batch_size, self._frame_stacking, *self._observation_space)
        next_states = self._frames[next_state_idx].view(batch_size, self._frame_stacking, *self._observation_space)
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        dones = self._dones[indices]

        return states.float(), actions.long(), rewards.float(), next_states.float(), dones.long()

    def size(self):
        return self._size

'''Prioritized Experience Replay Buffer'''
class PrioritizedReplayBuffer:
    def __init__(self,
                 capacity: int,
                 observation_space: tuple,
                 frame_stacking: int,
                 max_steps: int,
                 alpha: float,
                 beta: float,
                 device: torch.device,
                 min_priority: float = 1e-5):
        self._size = 0
        self._capacity = capacity
        self._observation_space = observation_space
        self._frame_stacking = frame_stacking
        self._delta_beta = (1.0 - beta) / max_steps
        self._alpha = alpha
        self._beta = beta
        self._min_priority = min_priority
        self._device = device
        self._idx = 0
        self._frames = torch.empty((self._capacity, *self._observation_space), dtype=torch.uint8, device=self._device, memory_format=torch.contiguous_format)
        self._actions = torch.empty((self._capacity, 1), dtype=torch.uint8, device=self._device)
        self._rewards = torch.empty((self._capacity, 1), dtype=torch.float32, device=self._device)
        self._dones = torch.empty((self._capacity, 1), dtype=torch.bool, device=self._device)
        self._priorities = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self._max_priority = 1.0
        self._valid_idx = torch.zeros(capacity, dtype=torch.bool, device=device)
        self._idx_offsets = torch.arange(-frame_stacking, 0, 1, device=device).view(1, -1)
        self._indices = None
        self._c = 0
        
    def push(self,
             frame: torch.Tensor,
             action: torch.Tensor,
             reward: torch.Tensor,
             done: torch.Tensor):
        self._frames[self._idx] = frame
        self._actions[self._idx] = action
        self._rewards[self._idx] = reward
        self._dones[self._idx] = done
        if self._c == self._frame_stacking:
            self._valid_idx[self._idx] = True
            self._priorities[self._idx] = max(self._max_priority, self._min_priority)
        if done:
            self._c = 0
        else:
            self._c = min(self._c + 1, self._frame_stacking)
        if self._size == self._capacity:
            idx = (self._idx + self._frame_stacking) % self._capacity
            self._valid_idx[idx] = False
            self._priorities[idx] = 0.0
        self._idx = (self._idx + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

        self._beta = min(1.0, self._beta + self._delta_beta)

    def sample(self, batch_size: int):
        assert (self._priorities > 0).sum() >= batch_size
        # Compute sampling probabilities
        probabilities = (self._priorities * self._valid_idx) ** self._alpha
        probabilities /= probabilities.sum()
        # Sample indices based on probabilities
        self._indices = torch.multinomial(probabilities, batch_size, replacement=True)
        state_idx = ((self._indices.view(-1,1) + self._idx_offsets) % self._capacity).view(-1)
        next_state_idx = (state_idx + 1) % self._capacity
        states = self._frames[state_idx].view(batch_size, self._frame_stacking, *self._observation_space)
        next_states = self._frames[next_state_idx].view(batch_size, self._frame_stacking, *self._observation_space)
        actions = self._actions[self._indices]
        rewards = self._rewards[self._indices]
        dones = self._dones[self._indices]
        
        weights = (self._size * probabilities[self._indices]) ** (-self._beta)
        weights = weights / weights.max()
        
        return states.float(), actions.long(), rewards.float(), next_states.float(), dones.long(), weights.unsqueeze(1).float()

    def update_priorities(self, td_errors: torch.Tensor):
        td_errors = td_errors.abs().clamp(min=self._min_priority)
        self._max_priority = max(self._max_priority, td_errors.max().item())
        self._priorities[self._indices] = td_errors.squeeze()

    def size(self):
        return self._size