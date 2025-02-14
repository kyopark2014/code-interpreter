# Code Interpreter

[Code generation with RAG and self-correction](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/code_assistant/langgraph_code_assistant.ipynb)를 참조합니다.

![image](https://github.com/user-attachments/assets/fc509eda-97ca-4994-8e47-6252764e4413)


## Riza

[Riza](https://docs.riza.io/introduction)를 참조해서 Code Interpreter를 이용할 수 있습니다.
[Riza - dashboard](https://dashboard.riza.io/)에 접속해서 credential을 발급 받습니다.

필요한 패키지를 등록하고 credential을 등록합니다. 

```text
npm install @riza-io/api
export RIZA_API_KEY=riza_01JM27....
```

참고할 코드는 [Hello. World!](https://docs.riza.io/guides/hello-world)에서 확인합니다.

```python
CODE = """
import sys
import os

print("stdin", sys.stdin.read())
print("args", sys.argv)
print("env", dict(os.environ))

a = 1
b = 2
c = a+b
print(f"c = {c}")
"""

셈플 코드는 아래와 같습니다.

```python
client = Riza()

resp = client.command.exec(
    language="python",
    code=CODE,
    stdin="Hello",
    args=["one", "two"],
    env={
        "DEBUG": "true",
    }
)

print(resp.stdout)
```

아래와 같이 실행할 수 있습니다.

```python
python3 riza2.py
stdin Hello
args ['python', '/src/code.py', 'one', 'two']
env {'DEBUG': 'true'}
c = 3
```

### LangChain에서 활용

설치할 패키지는 아래와 같습니다.

```text
pip install --upgrade --quiet langchain-community rizaio
```

LangChain과 연결은 [Riza Code Interpreter](https://python.langchain.com/docs/integrations/tools/riza/)을 참조합니다.

아래와 같이 tool로 등록합니다. 

```python
from langchain_community.tools.riza.command import ExecPython
tools = [ExecPython()]
```

```python
tools = [
    {
        "name": "execute_python",
        "description": "Execute a Python script. The Python runtime does not have filesystem access, but does include the entire standard library. Make HTTP requests with the httpx or requests libraries. Read input from stdin and write output to stdout.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute",
                }
            },
            "required": ["code"],
        },
    },
]
```

비용은 30초에 한번은 hobby로 무료입니다. 이후로는 $200에 400 hours의 비용을 부과합니다. 또한 [self host](https://docs.riza.io/enterprise/quickstart)로도 사용 가능합니다. 

```text
docker pull rizaio/code-interpreter
docker run --rm -p3003:3003 -it rizaio/code-interpreter
```

```text
docker run -p3003:3003 -e RIZA_LICENSE_KEY=riza_license_xxx --rm -it rizaio/code-interpreter
```



## Codebox

명령어 입력 방식이 LLM에 맞지 않아서 제외합니다.

```python
pip install codeboxapi ipython matplotlib
export CODEBOX_API_KEY=local
```

sample code는 아래와 같습니다.

```python
from codeboxapi import CodeBox

# create a new codebox
codebox = CodeBox()

# run some code
codebox.exec("a = 'Hello'")
codebox.exec("b = 'World!'")
codebox.exec("x = a + ', ' + b")
result = codebox.exec("print(x)")

print(result)
```

## LLama Index의 code interpreter

[code_interpreter.ipynb](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-code-interpreter/examples/code_interpreter.ipynb)을 참조합니다. tools에 등록된 tool들만 code interpreter를 쓸수 있는것으로 보여집니다.

```text
pip install llama-index-tools-code-interpreter
```

```python
from llama_index.tools.code_interpreter.base import CodeInterpreterToolSpec

# Initialize the code interpreter tool
code_interpreter = CodeInterpreterToolSpec()

tools = code_spec.to_tool_list()
# Create the Agent with our tools
agent = OpenAIAgent.from_tools(tools, verbose=True)
```


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

