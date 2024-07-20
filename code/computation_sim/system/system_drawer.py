from typing import Tuple

import networkx as nx
import plotly.graph_objects as go


def default_layout() -> dict:
    return {
        "color": "#1f77b4",
        "symbol": "circle",
        "hovertext": "",
    }


class SystemDrawer(object):
    def __init__(self, edge_color="#888", edge_width=0.5):
        self._edge_color = edge_color
        self._edge_width = edge_width
        pass

    def build(self, graph: nx.DiGraph) -> None:
        self._node_positions = self._compute_node_positions(graph)
        edges_x, edges_y = self._compute_edges(graph)

        self.fw = go.FigureWidget(
            layout=go.Layout(
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                coloraxis=dict(colorbar=dict(thickness=0, ticklen=0)),
            )
        )
        self.fw.update_layout(coloraxis_showscale=False)
        self.fw.add_scatter(
            x=edges_x,
            y=edges_y,
            line=dict(width=self._edge_width, color=self._edge_color),
            hoverinfo="none",
            mode="lines",
            showlegend=False,
        )

        nodes_x, nodes_y = zip(*[self._node_positions[node] for node in graph.nodes])
        options = self._collect_opts(graph)
        self.fw.add_scatter(
            x=nodes_x,
            y=nodes_y,
            mode="markers+text",
            textposition="top center",
            marker=dict(
                showscale=False,
                color=options["color"],
                symbol=options["symbol"],
                size=100,
                line_width=2,
                coloraxis="coloraxis",
            ),
            text=options["text"],
            hovertext=options["hovertext"],
            showlegend=False,
        )

    def _compute_node_positions(self, graph: nx.DiGraph) -> dict:
        layers = dict()
        for layer, node in enumerate(nx.topological_generations(graph)):
            layers[layer] = node
        return nx.multipartite_layout(graph, subset_key=layers)

    def _compute_edges(self, graph: nx.DiGraph) -> Tuple[Tuple[float, float, None], Tuple[float, float, None]]:
        edges_x = []
        edges_y = []
        for edge in graph.edges():
            x0, y0 = self._node_positions[edge[0]]
            x1, y1 = self._node_positions[edge[1]]
            edges_x.extend((x0, x1, None))
            edges_y.extend((y0, y1, None))
        return edges_x, edges_y

    def _collect_opts(self, graph: nx.DiGraph) -> dict:
        opts = []
        for node in graph.nodes:
            opt = node.draw_options
            opt["text"] = node.id
            for key, val in default_layout().items():
                opt.setdefault(key, val)
            opts.append(opt)

        common_keys = set.intersection(*map(set, opts))
        return {k: [dic[k] for dic in opts] for k in common_keys}

    def update(self, graph: nx.DiGraph) -> None:
        options = self._collect_opts(graph)
        node_scatter = self.fw.data[1]
        node_scatter.marker.color = options["color"]
        node_scatter.marker.symbol = options["symbol"]
        node_scatter.hovertext = options["hovertext"]
        self.fw
