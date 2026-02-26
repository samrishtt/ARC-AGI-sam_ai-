"""
Cognitive Synthesis Architecture Core Module.
Contains Meta-Controller and Routing.
"""
from .models import TaskDomain, TaskComplexity, RouteDecision
from .router import IntentRouter
from .meta_controller import MetaController
