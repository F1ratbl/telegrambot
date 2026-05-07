MARKET_TOOL_DECLARATION = {
    "name": "get_market_snapshot",
    "description": (
        "Fetch the latest available market snapshots for stock indices, FX pairs, "
        "commodities, and crypto assets. Use this for questions about current "
        "levels, daily moves, index performance, exchange rates, gold, oil, or BTC. "
        "Accepted aliases include BIST100, SP500, NASDAQ, DOWJONES, DAX, FTSE100, "
        "NIKKEI, VIX, USDTRY, EURTRY, EURUSD, GOLD, BRENT, BTCUSD. If symbols is "
        "empty, a default economy dashboard is returned. For gold requests, the tool "
        "may also include derived metrics such as estimated gram/TL and ounce/TL."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "symbols": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Market aliases or Yahoo-style symbols requested by the user.",
            }
        },
    },
}


KNOWLEDGE_BASE_TOOL_DECLARATION = {
    "name": "search_knowledge_base",
    "description": (
        "Search the private Qdrant knowledge base for economy and finance content. "
        "Use this when the user asks about concepts, internal methodology, uploaded "
        "documents, local notes, or context that may exist in the project knowledge base."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The economy/finance question or search query.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of knowledge chunks to return. Default is 5.",
            },
        },
        "required": ["query"],
    },
}


FUNCTION_DECLARATIONS = [
    MARKET_TOOL_DECLARATION,
    KNOWLEDGE_BASE_TOOL_DECLARATION,
]
