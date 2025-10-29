class AttrDict(dict):
    """
    Dict with attribute-style access.
    Example:
        h = AttrDict({"foo": 1}); print(h.foo)  # 1
    """
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value
