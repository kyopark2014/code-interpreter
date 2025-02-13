# Code Interpreter

[Code generation with RAG and self-correction](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/code_assistant/langgraph_code_assistant.ipynb)를 참조합니다.

![image](https://github.com/user-attachments/assets/fc509eda-97ca-4994-8e47-6252764e4413)



## E2B

[LangGraph with Code Interpreting](https://www.linkedin.com/feed/update/urn:li:activity:7191459920251109377/?commentUrn=urn%3Ali%3Acomment%3A(activity%3A7191459920251109377%2C7295624350970363904)&dashCommentUrn=urn%3Ali%3Afsd_comment%3A(7295624350970363904%2Curn%3Ali%3Aactivity%3A7191459920251109377))을 이용해 agent에 tool로 등록을 수행하려고 합니다.

설치한 패키지는 아래와 같습니다.

```python
pip install e2b_code_interpreter
```

이후 아래와 같은 코드를 추가하였습니다.

```python
from e2b_code_interpreter import Sandbox
from langchain.agents.output_parsers.tools import (
    ToolAgentAction,
)
from langchain_core.tools import Tool
from pydantic.v1 import BaseModel, Field
from langchain_core.messages import BaseMessage, ToolMessage
from typing import Any, List

class LangchainCodeInterpreterToolInput(BaseModel):
    code: str = Field(description="Python code to execute.")

class CodeInterpreterFunctionTool:
    """
    This class calls arbitrary code against a Python Jupyter notebook.
    It requires an E2B_API_KEY to create a sandbox.
    """

    tool_name: str = "code_interpreter"

    def __init__(self):
        # Instantiate the E2B sandbox - this is a long lived object
        # that's pinging E2B cloud to keep the sandbox alive.
        os.environ["E2B_API_KEY"] = "e2b_xxxxx....xxxx"
        if "E2B_API_KEY" not in os.environ:
            raise Exception(
                "Code Interpreter tool called while E2B_API_KEY environment variable is not set. Please get your E2B api key here https://e2b.dev/docs and set the E2B_API_KEY environment variable."
            )
        self.code_interpreter = Sandbox()

    def call(self, parameters: dict, **kwargs: Any):
        code = parameters.get("code", "")
        print(f"***Code Interpreting...\n{code}\n====")
        execution = self.code_interpreter.notebook.exec_cell(code)
        return {
            "results": execution.results,
            "stdout": execution.logs.stdout,
            "stderr": execution.logs.stderr,
            "error": execution.error,
        }

    def close(self):
        self.code_interpreter.close()

    # langchain does not return a dict as a parameter, only a code string
    def langchain_call(self, code: str):
        return self.call({"code": code})

    def to_langchain_tool(self) -> Tool:
        tool = Tool(
            name=self.tool_name,
            description="Execute python code in a Jupyter notebook cell and returns any rich data (eg charts), stdout, stderr, and error.",
            func=self.langchain_call,
        )
        tool.args_schema = LangchainCodeInterpreterToolInput
        return tool

    @staticmethod
    def format_to_tool_message(
        agent_action: ToolAgentAction,
        observation: dict,
    ) -> List[BaseMessage]:
        """
        Format the output of the CodeInterpreter tool to be returned as a ToolMessage.
        """
        new_messages = list(agent_action.message_log)

        # TODO: Add info about the results for the LLM
        content = json.dumps(
            {k: v for k, v in observation.items() if k not in ("results")}, indent=2
        )
        new_messages.append(
            ToolMessage(content=content, tool_call_id=agent_action.tool_call_id)
        )

        return new_messages

code_interpreter = CodeInterpreterFunctionTool()
code_interpreter_tool = code_interpreter.to_langchain_tool()
```

이후 아래처럼 tools에 등록후 실행하였습니다.

```python
tools = [get_current_time, get_book_list, get_weather_info, search_by_tavily, stock_data_lookup, code_interpreter_tool]        
```

이때의 결과는 아래와 같이 tool은 실행이 되었지만 주피터 노트북을 별도로 설치하야 한다고 합니다.

![image](https://github.com/user-attachments/assets/69acd506-8e1d-4db9-b256-ccb7b2c346c6)

#### 관련 링크들

- [LangGraph + Function Calling + E2B Code Interpreter](https://github.com/e2b-dev/e2b-cookbook/blob/main/examples/langgraph-python/langgraph_code_interpreter.ipynb)

- [E2B](https://e2b.dev/)

결과적으로 Sandbox 환경 구성이 먼저이고, code 생성을 위해 E2B를 써야할 이유는 없는것으로 보여집니다. 



## Reference

[Code Interpreter API](https://blog.langchain.dev/code-interpreter-api/): ChatGPT 기반의 code interpreter

[Code Interpreter API - Github](https://github.com/shroominic/codeinterpreter-api/tree/main): Prompt등 다수의 예제 활용 가능


[Build a coding agent with Modal Sandboxes and LangGraph](https://modal.com/docs/examples/agent)

[Creating a Clever Code Interpreter Tool With Langchain agents+Advanced Prompt Techniques](https://medium.com/latinxinai/creating-a-clever-code-interpreter-tool-with-langchain-agents-advanced-prompt-techniques-3d7b493cc580)

[E2B - Give LangGraph code execution capabilities](https://e2b-blog.framer.website/blog/langgraph-with-code-interpreter-guide-with-code)

[LangGraph for Code Generation](https://blog.langchain.dev/code-execution-with-langgraph/)

[Code generation with RAG and self-correction](https://langchain-ai.github.io/langgraph/tutorials/code_assistant/langgraph_code_assistant/)

[Building a Self-Correcting Coding Assistant with LangChain and LangGraph: A Hands-on Guide](https://medium.com/@mauryaanoop3/building-a-self-correcting-coding-assistant-with-langchain-and-langgraph-a-hands-on-guide-3ea7424655be)

