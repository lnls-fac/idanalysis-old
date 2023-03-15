from .deltadata import DeltaData
from .epudata import EPUData
from .trajectory import IDTrajectory
from .kickmaps import IDKickMap
from .analysiswip import AnalysisFromRadia

# This has to be defined before using the library.
FOLDER_BASE = None


import os as _os
with open(_os.path.join(__path__[0], 'VERSION'), 'r') as _f:
    __version__ = _f.read().strip()
