# File: graph/engine.py
# Description: Provides the public interface for running and managing the QA workflow graph.
# Author Name: Debleena Nandy
# Date: 07-10-2026
"""Public entry point for constructing and executing the QA workflow graph."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

START, END = "__start__", "__end__"

try:  # LangGraph is the primary engine; the local engine keeps the platform runnable without it.
    from langgraph.graph import END as LG_END
    from langgraph.graph import START as LG_START
    from langgraph.graph import StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover
    # FIX: names were "possibly unbound" when LangGraph is not installed.
    LG_START, LG_END = START, END
    StateGraph = None  # type: ignore[assignment,misc]
    LANGGRAPH_AVAILABLE = False

State = Dict[str, Any]
# FIX: nodes/routers take the workflow's TypedDict (GraphState), not Dict[str, Any].
# Typing the parameter as Any keeps GraphSpec reusable and accepts `(state: GraphState) -> ...`.
Node = Callable[[Any], Any]
Router = Callable[[Any], str]
Compiled = Callable[[State], State]
Mapping = Dict[str, str]
Branch = Tuple[Router, Mapping]
Lines = List[str]

MAX_STEPS = 100


class GraphSpec:
    """Declarative graph (nodes, edges, conditional edges) compiled to LangGraph or a minimal local engine."""

    def __init__(self, state_type: type):
        self.state_type = state_type
        self.nodes: Dict[str, Node] = {}
        self.edges: Mapping = {}
        self.conditional: Dict[str, Branch] = {}
        self.entry: Optional[str] = None

    def add_node(self, name: str, fn: Node) -> GraphSpec:
        self.nodes[name] = fn
        return self

    def set_entry(self, name: str) -> GraphSpec:
        self.entry = name
        return self

    def add_edge(self, source: str, target: str) -> GraphSpec:
        self.edges[source] = target
        return self

    def add_conditional_edges(self, source: str, router: Router, mapping: Mapping) -> GraphSpec:
        self.conditional[source] = (router, mapping)
        return self

    def describe(self) -> Lines:
        lines = [f"START -> {self.entry}"]
        for source, target in self.edges.items():
            lines.append(f"{source} -> {'END' if target == END else target}")
        for source, (_, mapping) in self.conditional.items():
            for label, target in mapping.items():
                lines.append(f"{source} -[{label}]-> {'END' if target == END else target}")
        return lines

    def compile(self, prefer_langgraph: bool = True) -> Compiled:
        if self.entry is None:
            raise ValueError("Graph has no entry node.")
        if prefer_langgraph and LANGGRAPH_AVAILABLE:
            return self._compile_langgraph(self.entry)
        return self._local_invoke

    def _compile_langgraph(self, entry: str) -> Compiled:
        if StateGraph is None:  # pragma: no cover - guarded by LANGGRAPH_AVAILABLE in compile()
            raise RuntimeError("LangGraph is not installed.")
        graph = StateGraph(self.state_type)
        for name, fn in self.nodes.items():
            graph.add_node(name, fn)
        graph.add_edge(LG_START, entry)
        for source, target in self.edges.items():
            graph.add_edge(source, LG_END if target == END else target)
        for source, (router, mapping) in self.conditional.items():
            graph.add_conditional_edges(source, router, {k: (LG_END if v == END else v) for k, v in mapping.items()})
        compiled = graph.compile()
        return lambda state: dict(compiled.invoke(state))

    def _local_invoke(self, initial: State) -> State:
        state = dict(initial)
        current: Optional[str] = self.entry
        steps = 0
        while current and current != END:
            steps += 1
            if steps > MAX_STEPS:
                raise RuntimeError("Graph exceeded the maximum number of steps; possible cycle.")
            # FIX: was `self.nodesstate` (typo) - call the current node with the state.
            state.update(self.nodes[current](state) or {})
            if current in self.conditional:
                router, mapping = self.conditional[current]
                current = mapping[router(state)]
            else:
                current = self.edges.get(current, END)
        return state