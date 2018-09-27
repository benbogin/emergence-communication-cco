from enum import Enum


class AgentType(Enum):
    speaker = 0
    listener = 1

    @classmethod
    def from_name(cls, name):
        if name == 'speaker':
            return cls.speaker
        if name == 'listener':
            return cls.listener
        else:
            raise(Exception("No such agent type: %s" % name))
