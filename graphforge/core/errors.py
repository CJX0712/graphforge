"""graphForge 错误码体系（E100~E400）。

作者：晨星
"""


class GraphForgeError(Exception):
    """graphForge 基类错误。"""

    code = "E000"

    def __init__(self, message: str, code: str | None = None):
        self.code = code or self.code
        super().__init__(f"[{self.code}] {message}")


class E100_INVALID_GRAPH(GraphForgeError):
    """图结构非法（形状、自环、孤立节点等）。"""

    code = "E100"


class E101_NOT_SYMMETRIC(GraphForgeError):
    """邻接矩阵非对称。"""

    code = "E101"


class E102_NEGATIVE_WEIGHT(GraphForgeError):
    """出现负边权。"""

    code = "E102"


class E200_EMPTY_GRAPH(GraphForgeError):
    """空图（0 节点 / 0 边）。"""

    code = "E200"


class E300_NOT_CONNECTED(GraphForgeError):
    """图不连通（某些算法要求连通分量）。"""

    code = "E300"


class E400_CONVERGENCE(GraphForgeError):
    """迭代算法未在规定步数内收敛。"""

    code = "E400"
