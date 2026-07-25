"""Agentic layer (Fase 6).

Turns the passive emotion classifier into one tool of a decision-making agent:
a LangGraph state machine that classifies an incoming review, routes it by
emotion + confidence, and either escalates it, drafts a grounded empathetic
reply for human approval, or archives it. Reuses the existing classifier,
hybrid RAG retriever, and provider-agnostic LLM client.
"""
