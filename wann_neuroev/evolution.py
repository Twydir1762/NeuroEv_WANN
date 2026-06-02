import math
from dataclasses import dataclass
import random
import json

# ТИПЫ
INPUT = 0
OUTPUT = 1
HIDDEN = 2

""" ========= Функции активации ========= """
@dataclass
class Act:
    linear = 'linear'
    tanh = 'tanh'
    relu = 'relu'
    sigmoid = 'sigm'
    sin = 'sin'

# Все
Act.ALL = tuple(v for k, v in Act.__dict__.items() if not k.startswith('_') and isinstance(v, str))

ACTIVATIONS = {
    Act.linear: lambda x: x,
    Act.tanh: math.tanh,
    Act.relu: lambda x: max(0, x),
    Act.sigmoid: lambda x: 1 / (1 + math.exp(max(-500, min(x, 500)))),
    Act.sin: math.sin
}

def activate(x, name):
    return ACTIVATIONS[name](x)

""" ========= База ========= """
@dataclass
class Node:
    type: int
    activation: str

class Individual:
    def __init__(self, n_inputs, n_outputs, inp_act=Act.linear, out_act=Act.sigmoid):
        self.inputs = n_inputs
        self.outputs = n_outputs
        self.nodes: dict[int, Node] = {} # Нейроны - {node_id: Node(type: int, activation: str)}
        self.connections: set[tuple] = set() # Связи между нейронами

        self._born(inp_act, out_act)

    def __repr__(self):
        return (f"Individual(ins={self.inputs}, outs={self.outputs}, "
                f"n: {len(self.nodes)}, cons: {len(self.connections)})")

    def _add_node(self, node_type, activation):
        n_id = len(self.nodes)
        self.nodes[n_id] = Node(type=node_type, activation=activation)
        return n_id

    def _add_connection(self, input_idx, output_idx):
        self.connections.add((input_idx, output_idx))

    def _is_cycled(self, node1, node2):
        checked = set()
        queue = [node2]
        while queue:
            current = queue.pop()
            if current == node1:
                return True
            if current in checked:
                continue
            checked.add(current)
            for (frm, to) in self.connections:
                if frm == current:
                    queue.append(to)

        return False

    def _born(self, input_activation, output_activation):
        # Входы
        for _ in range(self.inputs):
            self._add_node(INPUT, input_activation)
        # Выходы
        for _ in range(self.outputs):
            self._add_node(OUTPUT, output_activation) # Чтобы не было за [-1, 1]

    def get_node_ids(self, node_type, exclude=False):
        if not exclude:
            return tuple(sorted(n_id for n_id, node in self.nodes.items() if node.type == node_type))
        return tuple(sorted(n_id for n_id, node in self.nodes.items() if node.type != node_type))

    def _calc_node(self, node_id, w, values_dict):
        # Если уже есть в values_dict (входной/посчитанный)
        if node_id in values_dict:
            return values_dict[node_id]

        # Значения нейронов, которые входят в этот (node_id) нейрон
        income_values = [self._calc_node(frm, w, values_dict)
                         for (frm, to) in self.connections if to == node_id]

        node_activation = self.nodes[node_id].activation
        res = activate(sum(income_values) * w, node_activation)
        values_dict[node_id] = res

        return res

    # Прямое распространение
    def forward(self, x: tuple, w: float):
        input_ids = self.get_node_ids(INPUT) # id ВХОДОВ
        node_values = {n_id: val for n_id, val in zip(input_ids, x)} # Значения нейронов

        out_ids = self.get_node_ids(OUTPUT)
        out_values = [self._calc_node(out_id, w, node_values) for out_id in out_ids]

        # Значения на выходе
        return tuple(out_values)


    # МУТАЦИИ
    # Добавляем связь между 2-мя рандомными нейронами (вх не может быть "в", вых. не может быть "из")
    def mutate_add_con(self):
        candidates_from = self.get_node_ids(OUTPUT, exclude=True) # ТОЛЬКО ВХОДЫ + СКРЫТЫЕ
        candidates_to = self.get_node_ids(INPUT, exclude=True) # ТОЛЬКО СКРЫТЫЕ + ВЫХОДЫ

        node1 = random.choice(candidates_from) # ПЕРВАЯ ТОЧКА СВЯЗИ

        p1_outcome = {to for (frm, to) in self.connections if frm == node1} # Уже вытекают из p1
        valid_to = set(candidates_to) - {node1} - p1_outcome
        valid_to = {n for n in valid_to if not self._is_cycled(node1, n)}
        # Если нет возможностей - скип
        if not valid_to:
            return

        node2 = random.choice(tuple(valid_to)) # ВТОРАЯ ТОЧКА СВЯЗИ

        self._add_connection(node1, node2)

    # Добавляем новый нейрон между двумя связанными
    def mutate_add_node(self):
        # Если вся связи в мире исчезнут...
        if not self.connections:
            return

        from_node, to_node = random.choice(tuple(self.connections))
        # Новый нейрон - новые связи
        new_node = self._add_node(HIDDEN, random.choice(Act.ALL))
        self._add_connection(from_node, new_node)
        self._add_connection(new_node, to_node)

        # Старой связи больше нет
        self.connections.remove((from_node, to_node))

    # Меняем функцию активации случайного нейрона
    # ЛИНЕЙНАЯ ТОЖЕ МОЖЕТ ВЫПАСТЬ
    def mutate_activation(self):
        to_mutate = self.get_node_ids(HIDDEN)
        if not to_mutate:
            return

        to_mutate = random.choice(to_mutate)
        self.nodes[to_mutate].activation = random.choice(Act.ALL)

""" ========= Сохранение/Загрузка ========= """
def save_ind(ind, path):
    data = {
        'inputs': ind.inputs,
        'outputs': ind.outputs,
        'nodes': {n_id: {'type': node.type, 'activation': node.activation}
                  for n_id, node in ind.nodes.items()},
        'connections': [list(con) for con in ind.connections],
    }

    with open(path, 'w') as f:
        json.dump(data, f)

def load_ind(path):
    with open(path, 'r') as f:
        data = json.load(f)

    ind = Individual.__new__(Individual)
    ind.inputs = data['inputs']
    ind.outputs = data['outputs']
    ind.nodes = {int(n_id): Node(type=n_data['type'], activation=n_data['activation'])
                 for n_id, n_data in data['nodes'].items()}
    ind.connections = {tuple(con) for con in data['connections']}

    return ind



