from pydantic import BaseModel, Field
from typing import Union
from pydantic.functional_validators import AfterValidator
from bson import ObjectId

class PydanticObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not isinstance(v, ObjectId):
            raise TypeError('ObjectId required')
        return str(v)


class IdeaObj():
  def __init__(self, problem='', solution='') -> None:
    self.problem = problem
    self.solution = solution

# class Idea(BaseModel):
#   id: Union[PydanticObjectId, str] = Field(alias="_id", default=None)
#   problem: str
#   solution: str

class Idea(BaseModel):
  id: str = Field(alias="_id", default=None)
  problem: str
  solution: str

class Requirement(BaseModel):
  id: str = Field(alias="_id", default=None)
  problem: str
  acceptance_criteria: str
  top: int = 20