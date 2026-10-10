"""pytest 共用 fixtures。

作者：晨星
"""

import os
import sys

# 保证仓库根在 sys.path（pytest rootdir 可能因 CI 不同而变化）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
