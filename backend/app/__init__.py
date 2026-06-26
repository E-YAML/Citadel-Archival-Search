# App Package Initialization
__version__ = "0.1.0"

import os
import sys

# Ensure the parent directory ('backend') is in sys.path to allow absolute imports of 'app'
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Workaround for Windows Application Control blocking grpc cygrpc DLL
from unittest.mock import MagicMock

class Dummy1: pass
class Dummy2: pass
class Dummy3: pass
class Dummy4: pass
class Dummy5: pass
class Dummy6: pass
class Dummy7: pass
class Dummy8: pass
class Dummy9: pass
class Dummy10: pass

class MockAio:
    UnaryUnaryClientInterceptor = Dummy5
    UnaryStreamClientInterceptor = Dummy6
    StreamUnaryClientInterceptor = Dummy7
    StreamStreamClientInterceptor = Dummy8
    ClientCallDetails = Dummy9
    def __getattr__(self, name): return MagicMock()

class MockGrpc:
    UnaryUnaryClientInterceptor = Dummy1
    UnaryStreamClientInterceptor = Dummy2
    StreamUnaryClientInterceptor = Dummy3
    StreamStreamClientInterceptor = Dummy4
    ClientCallDetails = Dummy10
    RpcError = Exception
    aio = MockAio()
    def __getattr__(self, name): return MagicMock()

sys.modules['grpc'] = MockGrpc()
sys.modules['grpc.beta'] = MagicMock()
sys.modules['grpc._cython'] = MagicMock()

