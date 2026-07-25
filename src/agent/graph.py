"""Build and run the LangGraph state machine for a review.

Topology::

    START -> classify -> (route_by_emotion) -> escalate ┐
                                             -> draft    ├-> END
                                             -> archive  ┘

The graph is the single source of truth for the flow. ``run_review`` compiles
and invokes it; if LangGraph is not installed it falls back to an equivalent
sequential executor over the same nodes and router, so the agent degrades
gracefully instead of failing to import (a project design principle).
"""

from __future__ import annotations

from src.agent import nodes
from src.agent.config import AgentConfig
from src.agent.nodes import AgentDeps
from src.agent.router import RouterConfig, route_by_emotion
from src.agent.state import ReviewState

# branch name (also the router return value) -> node function.
_BRANCH_NODES = {
    "escalate": nodes.escalate_node,
    "draft": nodes.draft_empathetic_node,
    "archive": nodes.archive_node,
}

# LangGraph node ids must not collide with ``ReviewState`` keys, so the "draft"
# branch node is registered under a distinct id ("draft" is also a state field).
_NODE_IDS = {"escalate": "escalate", "draft": "draft_reply", "archive": "archive"}


def build_graph(deps: AgentDeps, router_config: RouterConfig):
    """Compile the LangGraph ``StateGraph``. Raises if LangGraph is unavailable."""
    from langgraph.graph import END, START, StateGraph

    # Single-arg closures: LangGraph reserves a param named ``config`` for its
    # own RunnableConfig, so nodes/router are wrapped to take only ``state``.
    def _classify(state: ReviewState) -> dict:
        return nodes.classify_node(state, deps)

    def _make_branch(fn):
        return lambda state: fn(state, deps)

    def _router(state: ReviewState) -> str:
        return route_by_emotion(state, router_config)

    graph = StateGraph(ReviewState)
    graph.add_node("classify", _classify)
    for branch, fn in _BRANCH_NODES.items():
        graph.add_node(_NODE_IDS[branch], _make_branch(fn))

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        _router,
        {branch: _NODE_IDS[branch] for branch in _BRANCH_NODES},
    )
    for node_id in _NODE_IDS.values():
        graph.add_edge(node_id, END)
    return graph.compile()


class AgentRuntime:
    """Runs one review through the graph, reusing a compiled app when possible."""

    def __init__(self, deps: AgentDeps, config: AgentConfig):
        self.deps = deps
        self.config = config
        self._app = None
        try:
            self._app = build_graph(deps, config.router)
        except Exception:  # noqa: BLE001 - langgraph missing/incompatible -> fallback
            self._app = None

    def run(self, review_text: str) -> ReviewState:
        """Classify, route, and act on ``review_text``; return the final state."""
        initial: ReviewState = {"review_text": review_text}
        if self._app is not None:
            return dict(self._app.invoke(initial))  # type: ignore[return-value]
        return self._run_sequential(initial)

    def _run_sequential(self, state: ReviewState) -> ReviewState:
        """LangGraph-free equivalent: classify -> route -> branch node."""
        state = {**state, **nodes.classify_node(state, self.deps)}
        branch = route_by_emotion(state, self.config.router)
        state = {**state, **_BRANCH_NODES[branch](state, self.deps)}
        return state
