class LLMError(RuntimeError):
    pass


class ContextLengthError(LLMError):
    """The provider explicitly rejected the context length, before any answer."""


class ContextBudgetError(LLMError):
    """Required content cannot fit the configured local request budget."""
