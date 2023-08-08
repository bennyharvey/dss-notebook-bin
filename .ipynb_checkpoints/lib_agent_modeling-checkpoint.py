import math
import random
from enum import IntEnum
import numpy
import names as random_names
import typing
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
from collections.abc import Sequence


class AgentPDBehaviour(IntEnum):
    TFT = 1
    ALLD = 2
    RND = 3


class PDConfig:
    def __init__(
            self,
            max_history: int = 200
    ) -> None:
        self.max_history = max_history


class PDAction(IntEnum):
    Defect = 0
    Cooperate = 1


class PDGame:
    def __init__(self):
        pass


class AgentModelConfig:
    def __init__(
            self,
            pd: PDConfig = PDConfig(),
            random_agent_amount: int = 100,
            weight_amount_speed_rate: float = 0.6,
            children_y_margin: int = 50,
            children_x_margin: int = 50,
            graph_height: int = 1000,
            graph_width: int = 1000,
            hierarchy_levels: int = 6,
            max_transfer_iterations: int = 1000,
            transfer_addition_rate: int = 10,
            lowest_level_percent: int = 60,
            debug_mode: bool = False,
    ) -> None:
        self.pd = pd
        self.children_y_margin = children_y_margin
        self.children_x_margin = children_x_margin
        self.graph_height = graph_height
        self.graph_width = graph_width
        self.hierarchy_levels = hierarchy_levels
        self.weight_amount_speed_rate = weight_amount_speed_rate
        self.random_agent_amount = random_agent_amount
        self.max_transfer_iterations = max_transfer_iterations
        self.transfer_addition_rate = transfer_addition_rate
        self.lowest_level_percent = lowest_level_percent
        self.debug_mode = debug_mode

    def get_random_PD_behaviour(self):
        return list(AgentPDBehaviour)[random.randint(0, len(AgentPDBehaviour) - 1)]


class AgentExperience:
    def __init__(
            self,
            new_information_grasp: float = 0
    ) -> None:
        self.new_information_grasp = new_information_grasp


class Agent:
    def __init__(
            self,
            index: int,
            name: str,
            PD_behaviour: AgentPDBehaviour,
            experience: AgentExperience,
            tier: int = 0,
            parent = None
    ) -> None:
        self.index = index
        self.name = name
        self.plot_x = 0
        self.plot_y = 0
        self.PD_behaviour = PD_behaviour
        self.experience = experience
        self.tier = tier
        self.parent: Agent | None = parent
        self.PD_history: dict[int, list[PDAction]] | dict[None] = {}

    def __hash__(self) -> int:
        return hash(self.index)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Agent):
            return False
        return self.index == other.index


class AgentNetwork:
    def __init__(self, config: AgentModelConfig, agents: Sequence[Agent] = None) -> None:
        self.config = config
        self.agents = agents
        self.hydrate()
        self.weighted_agent_amounts = self.get_weighted_agent_amounts()
        if config.debug_mode:
            print(self.weighted_agent_amounts)
            # print(sum(self.weighted_agent_amounts))
        self.assign_parents()
        self.generate_random_trace()

    def __getitem__(self, items) -> typing.Union[Agent, Sequence[Agent]]:
        return self.agents[items]

    def __len__(self) -> int:
        return len(self.agents)

    def hydrate(self) -> None:
        if not self.agents is None:
            return
        self.agents = [Agent(
            index = i,
            name = random_names.get_full_name(),
            PD_behaviour = AgentPDBehaviour.TFT if i == 0 else self.config.get_random_PD_behaviour(),
            experience = AgentExperience()
        ) for i in range(self.config.random_agent_amount)]

    def assign_parents(self) -> None:
        pointer = 1
        parent_agents = []
        current_agents = []
        self.agents[0].parent = self.agents[0]
        for id_wam, weighted_agent_amount in enumerate(self.weighted_agent_amounts):
            for i in range(weighted_agent_amount):
                if pointer == len(self.agents):
                    continue
                current_agents.append(self.agents[pointer])
                self.agents[pointer].tier = id_wam + 1
                if id_wam == 0:
                    self.agents[pointer].parent = self.agents[0]
                else:
                    self.agents[pointer].parent = parent_agents[random.randint(0, len(parent_agents) - 1)]
                pointer += 1
            parent_agents = current_agents
            current_agents = []

    def generate_random_trace(self) -> None:
        for i in range(len(self.agents)):
            if self.agents[i].tier == 0:
                self.agents[i].plot_x = round(self.config.graph_width / 2)
                self.agents[i].plot_y = self.config.graph_height - 10
            self.agents[i].plot_x = random.randint(
                self.agents[i].parent.plot_x - self.config.children_x_margin,
                self.agents[i].parent.plot_x + self.config.children_x_margin
            )
            self.agents[i].plot_y = self.agents[i].parent.plot_y - self.config.children_y_margin

    def reset_agents_experience(self) -> None:
        for agent in self.agents[1:]:
            agent.experience.new_information_grasp = 0

    def get_min_grasp(self) -> float:
        min_grasp = 1
        for agent in self.agents:
            if agent.experience.new_information_grasp < min_grasp:
                min_grasp = agent.experience.new_information_grasp
        return min_grasp

    def get_nx_edges(self) -> list[tuple[Agent, Agent, float]]:
        result = []
        for agent in self.agents:
            if agent.index == 0:
                continue
            result.append((agent, agent.parent, random.random()))
        return result

    def test_print_agents(self) -> None:
        for agent in self.agents:
            try:
                print(
                    f'i={agent.index}, tier={agent.tier}, parent={agent.parent.index}, x={agent.plot_x}, y={agent.plot_y}, beh={agent.PD_behaviour}, exp={agent.experience.new_information_grasp}')
            except AttributeError:
                print(f'i={agent.index}, tier={agent.tier}, parent=None, x={agent.plot_x}, y={agent.plot_y}')

    def get_weighted_agent_amounts(self) -> list[int]:
        """
        Very naive implementation of weighted distribution. Returns ordered (n-1) list of amounts.

        This function assumes that first hierarchy level always contains single agent.
        """
        total_parts = sum(range(self.config.hierarchy_levels))
        agent_amount = len(self.agents)
        part_size = round(agent_amount / total_parts / 2)
        amounts = []
        for level in range(self.config.hierarchy_levels - 1):
            weighted_count = round(pow(part_size, (level + 1) * self.config.weight_amount_speed_rate))
            if level == self.config.hierarchy_levels - 2:
                amounts.append(weighted_count - (sum(amounts, weighted_count) - agent_amount))
            else:
                amounts.append(weighted_count)
        return amounts


class AgentGraph:
    def __init__(self, agent_network: AgentNetwork) -> None:
        self.agent_network = agent_network
        self.nx_graph = nx.Graph()
        self.hydrate_nx_graph()

    def hydrate_nx_graph(self) -> None:
        self.nx_graph.add_nodes_from(self.agent_network)
        self.nx_graph.add_weighted_edges_from(self.agent_network.get_nx_edges())

    def convert_from_nx_to_ploty(self):
        edge_x = []
        edge_y = []
        for edge in self.nx_graph.edges():
            edge_x.append(edge[0].plot_x)
            edge_x.append(edge[1].plot_x)
            edge_x.append(None)
            edge_y.append(edge[0].plot_y)
            edge_y.append(edge[1].plot_y)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        text = []
        for node in self.nx_graph.nodes():
            node_x.append(node.plot_x)
            node_y.append(node.plot_y)
            text.append(f'name: {node.name}, x:{node.plot_x}, y:{node.plot_y}')

        node_adjacencies = []
        for node, adjacencies in enumerate(self.nx_graph.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))

        node_tiers = []
        for node in self.nx_graph.nodes():
            node_tiers.append(node.tier)

        colors = node_adjacencies
        # node_text.append('# of connections: '+str(len(adjacencies[1])))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            text=text,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=colors,
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2)
        )
        return [edge_trace, node_trace]

    def draw_ploty_graph(self) -> None:
        fig = go.Figure(
            data=self.convert_from_nx_to_ploty(),
            layout=go.Layout(
                title='Agent Network',
                # titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(
                    text="",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002)],
                xaxis=dict(showgrid=True, zeroline=True, showticklabels=True),
                yaxis=dict(showgrid=True, zeroline=True, showticklabels=True))
        )
        # fig.update_layout(
        #     updatemenus=[
        #         dict(
        #             type="buttons",
        #             direction="left",
        #             buttons=list([
        #                 dict(
        #                     args=["type", "surface"],
        #                     label="3D Surface",
        #                     method="restyle"
        #                 ),
        #                 dict(
        #                     args=["type", "heatmap"],
        #                     label="Heatmap",
        #                     method="restyle"
        #                 )
        #             ]),
        #             pad={"r": 10, "t": 10},
        #             showactive=True,
        #             x=0.11,
        #             xanchor="left",
        #             y=1.1,
        #             yanchor="top"
        #         ),
        #     ]
        # )
        if self.agent_network.config.debug_mode:
            print(fig.data)

        fig.show()

    def draw_visjs_graph(self) -> None:
        net = Network('1000px', '1000px')
        for node in self.nx_graph.nodes():
            net.add_node(node.index, label=node.name)
        for edge in self.nx_graph.edges():
            net.add_edge(edge[0].index, edge[1].index)
        # net.from_nx(nx_graph)
        net.toggle_physics(False)
        net.show_buttons(filter_=[
            'physics',
            'layout',
            # 'edges'
        ])
        net.show('graph.html')

class SimulationConfig:
    def __init__(
            self,
            grasp_threshold: float = 0.75
    ):
        self.grasp_threshold = grasp_threshold


class SimulationType(IntEnum):
    GraspTransfer = 1


class Simulation:
    completed_in: int = 0
    type: SimulationType | None = None

    def __init__(self, agent_network: AgentNetwork, config: SimulationConfig = SimulationConfig()):
        self.agent_network = agent_network
        self.config = config

    def set_type(self, type: SimulationType) -> None:
        self.type = type

    def get_pd_round_response(self, player: Agent, opponent: Agent) -> PDAction:
        match player.PD_behaviour:
            case AgentPDBehaviour.TFT:
                try:
                    player.PD_history[opponent.index]
                except:
                    return PDAction.Cooperate
                match player.PD_history[opponent.index][-1]:
                    case PDAction.Defect: return PDAction.Defect
                    case PDAction.Cooperate: return PDAction.Cooperate
            case AgentPDBehaviour.ALLD:
                return PDAction.Defect
            case AgentPDBehaviour.RND:
                return list(PDAction)[random.randint(0, len(PDAction) - 1)]
            case _:
                print(player.index)
                print(id(player.PD_behaviour))
                print(id(AgentPDBehaviour.ALLD))
                print(player.PD_behaviour == AgentPDBehaviour.ALLD)
                raise AttributeError('Unknown agent PD behaviour')

    def get_pd_score(self, first_action: PDAction, second_action: PDAction) -> list[int]:
        if (first_action == PDAction.Cooperate) and (second_action == PDAction.Cooperate):  return [3, 3]
        if (first_action == PDAction.Cooperate) and (second_action == PDAction.Defect):     return [0, 5]
        if (first_action == PDAction.Defect) and (second_action == PDAction.Cooperate):     return [5, 0]
        if (first_action == PDAction.Defect) and (second_action == PDAction.Defect):        return [1, 1]

    def play_pd_round(self, player: Agent, opponent: Agent) -> int:
        player_responce = self.get_pd_round_response(player, opponent)
        opponent_responce = self.get_pd_round_response(opponent, player)
        player.PD_history.setdefault(opponent.index, []).append(opponent_responce)
        opponent.PD_history.setdefault(player.index, []).append(player_responce)
        return sum(self.get_pd_score(player_responce, opponent_responce))

    def get_transfer_rate(self, pd_round_result: int) -> float:
        match pd_round_result:
            case 6: return 1
            case 5: return 0.5
            case 2: return 0.1

    def run(self):
        match self.type:
            case SimulationType.GraspTransfer:
                self.simulate_grasp_transfer()
            case _:
                raise AttributeError('Unknown simulation type')

    def simulate_grasp_transfer(self):
        self.agent_network[0].experience.new_information_grasp = 1
        for i in range(self.agent_network.config.max_transfer_iterations):
            for agent in self.agent_network[1:]:
                score = self.play_pd_round(agent, agent.parent)
                transfer_rate = self.get_transfer_rate(score)
                if agent.experience.new_information_grasp == 0:
                    agent.experience.new_information_grasp = agent.parent.experience.new_information_grasp * \
                                                             transfer_rate
                else:
                    if transfer_rate == 1:
                        agent.experience.new_information_grasp = agent.parent.experience.new_information_grasp
                    else :
                        agent.experience.new_information_grasp = agent.experience.new_information_grasp + \
                                                                 agent.parent.experience.new_information_grasp * \
                                                                 transfer_rate / \
                                                                 self.agent_network.config.transfer_addition_rate
                if agent.experience.new_information_grasp > 1: agent.experience.new_information_grasp = 1

            if self.agent_network.get_min_grasp() > self.config.grasp_threshold:
                self.completed_in = i
                break


def main():
    conf = AgentModelConfig()
    agent_network = AgentNetwork(conf)
    graph = AgentGraph(agent_network)

    graph.draw_ploty_graph()
    graph.draw_visjs_graph()

    # agent_network[0].name = 'asd'
    # for node in graph.nx_graph.nodes():
    #     print(node.name)


if __name__ == "__main__":
    main()
