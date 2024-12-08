import pytest
from environments.hierarchical import  HierarchicalSystemBuilder
from computation_sim.time import Clock, FixedDuration
from unittest.mock import MagicMock
import networkx as nx


@pytest.fixture
def build_simple_tree():
    clock = Clock(0)
    builder = HierarchicalSystemBuilder(clock)

    # Set-up the sensor chains
    s0 = [
        builder.add_sensor_chain("0", 0, 100, FixedDuration(100), FixedDuration(10)),
        builder.add_sensor_chain("1", 0, 100, FixedDuration(100), FixedDuration(10)),
    ]
    s1 = [
        builder.add_sensor_chain("2", 0, 100, FixedDuration(100), FixedDuration(10)),
        builder.add_sensor_chain("3", 0, 100, FixedDuration(100), FixedDuration(10)),
    ]

    # Set-up the edge nodes
    m = [
        builder.add_edge_compute("0", s0, FixedDuration(10), 90.0),
        builder.add_edge_compute("1", s1, FixedDuration(10), 90.0),
    ]

    # Set-up the output node
    builder.add_output_compute(m, FixedDuration(10), 90.0)
    return builder

def test_simple_tree_builds(build_simple_tree):
    builder = build_simple_tree
    builder.build()

def test_simple_tree_graph(build_simple_tree):
    builder = build_simple_tree
    builder.build()

    node_graph = builder.system_collection.system.node_graph
    assert node_graph is not None
    assert nx.is_directed_acyclic_graph(node_graph)
    assert nx.is_weakly_connected(node_graph)   

def test_simple_tree_collection(build_simple_tree):
    builder = build_simple_tree
    builder.build()

    collection = builder.system_collection
    assert collection.system is not None
     
    # Sources
    assert len(collection.sources) == 4
    for node in collection.sources:
        assert len(list(collection.system.node_graph.predecessors(node))) == 0

    # Sinks
    assert len(collection.sinks) == 3
    for sink in collection.sinks:
        assert len(list(collection.system.node_graph.successors(sink))) == 0

    # Action Collections
    assert len(collection.action_collections) == 3
    for action in collection.action_collections:
        assert action.node in collection.system.node_graph.nodes
        for input in action.input_buffers:
            assert input in collection.system.node_graph.nodes
            assert collection.system.node_graph.has_edge(input, action.node)

    # Output
    assert len(list(collection.system.node_graph.successors(collection.output))) == 0

    # Samplers
    assert len(collection.samplers) == 11

@pytest.fixture
def build_four_stages():
    clock = Clock(0)
    builder = HierarchicalSystemBuilder(clock)

    # Set-up the sensor chains
    s0 = [
        builder.add_sensor_chain("0", 0, 100, FixedDuration(100), FixedDuration(10)),
        builder.add_sensor_chain("1", 0, 100, FixedDuration(100), FixedDuration(10)),
    ]
    s1 = [
        builder.add_sensor_chain("2", 0, 100, FixedDuration(100), FixedDuration(10)),
        builder.add_sensor_chain("3", 0, 100, FixedDuration(100), FixedDuration(10)),
    ]

    # Set-up the edge nodes
    m0 = [
        builder.add_edge_compute("0", s0, FixedDuration(10), 90.0),
    ]

    m1 = [
        builder.add_edge_compute("2", s1, FixedDuration(10), 90.0),
        builder.add_edge_compute("3", m0, FixedDuration(10), 90.0),
    ]

    # Set-up the output node
    builder.add_output_compute(m1, FixedDuration(10), 90.0)
    return builder

def test_four_stages_builds(build_four_stages):
    builder = build_four_stages
    builder.build()

def test_four_stages_graph(build_four_stages):
    builder = build_four_stages
    builder.build()

    node_graph = builder.system_collection.system.node_graph
    assert node_graph is not None
    assert nx.is_directed_acyclic_graph(node_graph)
    assert nx.is_weakly_connected(node_graph)   

def test_four_stages_collection(build_four_stages):
    builder = build_four_stages
    builder.build()

    collection = builder.system_collection
    assert collection.system is not None
     
    # Sources
    assert len(collection.sources) == 4
    for node in collection.sources:
        assert len(list(collection.system.node_graph.predecessors(node))) == 0

    # Sinks
    assert len(collection.sinks) == 3
    for sink in collection.sinks:
        assert len(list(collection.system.node_graph.successors(sink))) == 0

    # Action Collections
    assert len(collection.action_collections) == 4
    for action in collection.action_collections:
        assert action.node in collection.system.node_graph.nodes
        for input in action.input_buffers:
            assert input in collection.system.node_graph.nodes
            assert collection.system.node_graph.has_edge(input, action.node)

    # Output
    assert len(list(collection.system.node_graph.successors(collection.output))) == 0

    # Samplers
    assert len(collection.samplers) == 12
