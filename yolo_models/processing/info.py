import enum

# Sync with touch designer code
@enum.unique
class BufferStates(enum.IntEnum):
    SERVER = 0
    CLIENT = 1
    SERVER_ALIVE = 2

@enum.unique
class States(bytes, enum.Enum):
    NULL_STATE = b'0'
    READY_SERVER_MESSAGE = b'1'
    READY_CLIENT_MESSAGE = b'2'
    IS_SERVER_ALIVE = b'3'

@enum.unique
class ParamsIndex(enum.IntEnum):
    IOU_THRESH = 0
    SCORE_THRESH = 1
    TOP_K = 2
    ETA = 3
    IMAGE_WIDTH = 4
    IMAGE_HEIGHT = 5
    IMAGE_CHANNELS = 6
    SHARED_ARRAY_MEM_NAME = 7
    SHARD_STATE_MEM_NAME = 8
    IMAGE_DTYPE = 9
