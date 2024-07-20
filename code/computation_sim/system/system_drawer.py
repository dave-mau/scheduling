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

    def build_edges(self, graph: nx.DiGraph) -> None:
        self._layers = dict()
        for layer, node in enumerate(nx.topological_generations(graph)):
            self._layers[layer] = node
        self._positions = nx.multipartite_layout(graph, subset_key=self._layers)

        self._edges_x = []
        self._edges_y = []
        for edge in graph.edges():
            x0, y0 = self._positions[edge[0]]
            x1, y1 = self._positions[edge[1]]
            self._edges_x.extend((x0, x1, None))
            self._edges_y.extend((y0, y1, None))
        self._edges = go.Scatter(
            x=self._edges_x,
            y=self._edges_y,
            line=dict(width=self._edge_width, color=self._edge_color),
            hoverinfo="none",
            mode="lines",
            showlegend=False,
        )

    def build_nodes(self, graph: nx.DiGraph) -> None:
        x, y = zip(*[self._positions[node] for node in graph.nodes])
        opts = []
        for node in graph.nodes:
            opt = node.draw_options
            opt["text"] = node.id
            for key, val in default_layout().items():
                opt.setdefault(key, val)
            opts.append(opt)

        common_keys = set.intersection(*map(set, opts))
        v = {k: [dic[k] for dic in opts] for k in common_keys}

        self._nodes = go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            textposition="top center",
            marker=dict(
                showscale=True,
                color=v["color"],
                symbol=v["symbol"],
                size=100,
                line_width=2,
                coloraxis="coloraxis",
            ),
            text=v["text"],
            hovertext=v["hovertext"],
            showlegend=False,
        )

    def update(self, graph: nx.DiGraph) -> None:
        pass

    def close(self, graph: nx.DiGraph) -> None:
        pass

    def build_figure(self) -> None:
        self._fig = go.Figure(
            data=[self._edges, self._nodes],
            layout=go.Layout(
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                coloraxis=dict(colorbar=dict(thickness=0, ticklen=0)),
            ),
        )
        self._fig.update_layout(coloraxis_showscale=False)

    def show(self) -> None:
        self._fig.show()
