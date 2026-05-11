import os
import json
from typing import List, Dict
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0.3,
)

class Coordinates(BaseModel):
    x: float = Field(description = "the x coordinate of the point")
    y: float = Field(description = "the y coordinate of the point")

class Point(BaseModel):
    name: str = Field(description = "the name of the point")
    manhattan_norm: float = Field(description = "the manhattan norm of the point")
    euclidean_norm: float = Field(description = "the euclidean norm of the point")
    coordinates: Coordinates = Field(description = "the coordinates of the point")

class QuerySec(BaseModel):
    findCoor: str = Field(description = "the user's request about the Coordinate of the Point")
    Move: str = Field(description = "the user's request about how the point should be moved")


@tool
def move_x(x: float) -> str:
    """Move a point horizontally on the x-axis."""
    if x > 0:
        return f"move the point D by {x} units to the right"
    elif x < 0:
        return f"move the point D by {abs(x)} units to the left"
    return "do not move on the x-axis"

@tool
def move_y(y: float) -> str:
    """Move a point vertically on the y-axis."""
    if y > 0:
        return f"move the point D by {y} units up"
    elif y < 0:
        return f"move the point D by {abs(y)} units down"
    return "do not move on the y-axis"

structured_llm_1 = llm.with_structured_output(
    QuerySec,
    method = "json_schema",
)

structured_llm_2 = llm.with_structured_output(
    Point,
    method = "json_schema",
)
tools = [move_x, move_y]
tool_map = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

user_query = "what is the coordinate of the point D satisfying that the quadrilateral ABCD is a rectangle where A(0, 0), B(0, 5), C(10, 5); and how to move the point D to the origin then move it to the point I which is the middle of A and B"

query_classify = structured_llm_1.invoke(
    f"Analyze the following query and split it into two parts: "
    f"1. Information about calculating the coordinate of the point. "
    f"2. Information about how to move the point. \n"
    f"Query: {user_query}"
)
how_coor = query_classify.findCoor
how_move = query_classify.Move
print(how_move)

point_coor_prompt = (
    "Solve the following geometry problem to find the coordinates of point D. "
    "Also calculate its Manhattan norm (abs(x) + abs(y)) and Euclidean norm (sqrt(x^2 + y^2)). "
    f"Problem: {how_coor}"
)
point_coor = structured_llm_2.invoke(point_coor_prompt)
print(point_coor)

prompt_for_move = (
    f"Original context: {user_query}\n"
    f"Point D's calculated properties: {point_coor.model_dump_json()}\n"
    f"Action required: {how_move}\n"
    "Please calculate the exact numeric distances (delta x and delta y) needed to perform the required movements. "
    "Then call the move_x and move_y tools with those numeric values."
)
move = llm_with_tools.invoke(prompt_for_move)

if move.tool_calls:
    for tool_call in move.tool_calls:
        func_name = tool_call["name"]
        args = tool_call["args"]
        tool_id = tool_call["id"]
        print(f"  -> Executing Tool: {func_name}({args})")
        
        if func_name in tool_map:
            result = tool_map[func_name].invoke(args)
            print(f"  <- Tool Result: {result}")
