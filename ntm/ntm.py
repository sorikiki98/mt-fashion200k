import torch
from torch import nn
import torch.nn.functional as F
from ntm.controller import Controller
from ntm.memory import Memory
from ntm.head import ReadHead, WriteHead


class CM_NTM(nn.Module):
    def __init__(self, num_ntms, embed_dim, vector_length, hidden_size, memory_size, lstm_controller=True):
        super(CM_NTM, self).__init__()
        self.num_ntms = num_ntms
        self.embed_dim = embed_dim
        self.vector_length = vector_length

        self.input_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, vector_length),
                nn.Tanh()  # Normalize output to [-1, 1]
            ) for _ in range(num_ntms)
        ])

        self.ntms = nn.ModuleList([
            NTM(embed_dim, vector_length, hidden_size, memory_size, lstm_controller)
            for _ in range(num_ntms)
        ])

    def get_initial_state(self, batch_size=1):
        states = []
        read_vectors = []

        for ntm in self.ntms:
            state = ntm.get_initial_state(batch_size)
            states.append(state)
            read_vectors.append(state[0])  # read vector

        return states, read_vectors

    def forward(self, inputs, states_and_reads):
        """
        inputs: list of tensors, one per NTM. Each is (batch_size, embed_dim)
        states_and_reads: tuple (states, read_vectors)
            - states: list of previous states for each NTM
            - read_vectors: list of previous read vectors for each NTM
        """
        states, read_vectors = states_and_reads
        new_states = []
        outputs = []
        projected_inputs = [
            self.input_projections[i](inputs[i])
            for i in range(self.num_ntms)
        ]  # (batch_size, vector_length)

        for i, ntm in enumerate(self.ntms):
            prev_idx = (i - 1) % self.num_ntms
            prev_read = read_vectors[prev_idx]

            ntm_input = torch.cat([projected_inputs[i], prev_read], dim=1)

            output, new_state = ntm(ntm_input, states[i])

            outputs.append(output)
            new_states.append(new_state)
            read_vectors[i] = new_state[0]

        return outputs, (new_states, read_vectors)


class NTM(nn.Module):
    def __init__(self, embed_dim, vector_length, hidden_size, memory_size, lstm_controller=True):
        super(NTM, self).__init__()
        self.controller = Controller(lstm_controller, vector_length + memory_size[1], hidden_size)
        self.memory = Memory(memory_size)
        self.read_head = ReadHead(self.memory, hidden_size)
        self.write_head = WriteHead(self.memory, hidden_size)
        self.fc = nn.Linear(hidden_size + memory_size[1], embed_dim)
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.normal_(self.fc.bias, std=0.01)

    def get_initial_state(self, batch_size=1):
        self.memory.reset(batch_size)
        controller_state = self.controller.get_initial_state(batch_size)
        read = self.memory.get_initial_read(batch_size)
        read_head_state = self.read_head.get_initial_state(batch_size)
        write_head_state = self.write_head.get_initial_state(batch_size)
        return (read, read_head_state, write_head_state, controller_state)

    def forward(self, x, previous_state):
        previous_read, previous_read_head_state, previous_write_head_state, previous_controller_state = previous_state
        controller_output, controller_state = self.controller(x, previous_controller_state)
        # Read
        read_head_output, read_head_state = self.read_head(controller_output, previous_read_head_state)
        # Write
        write_head_state = self.write_head(controller_output, previous_write_head_state)
        fc_input = torch.cat((controller_output, read_head_output), dim=1)
        state = (read_head_output, read_head_state, write_head_state, controller_state)
        return F.sigmoid(self.fc(fc_input)), state
