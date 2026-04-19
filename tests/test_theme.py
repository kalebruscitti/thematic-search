"""
Correctness tests for theme finding 
"""
import numpy as np
import pandas as pd
import pytest

from thematic_search.softclustertree import SoftClusterTree, Cluster
from thematic_search.topicdatabase import TopicDatabase
from thematic_search.queries import  SampleQuery, TopicQuery
from thematic_search.utilities import topic_uid, uid_to_ints