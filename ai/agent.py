"""ai大模型总入口"""

from ai.client import ask_deepseek       # 从 ai 包的 client.py 文件中导入
# ask_deepseek：用于普通JSON模式调用大模型。agent.py使用它完成用户意图识别。

from ai.prompt import intent_prompt     #从ai.prompt模块导入意图识别提示词函数
# intent_prompt()负责创建“判断用户操作(add/update/delete/search)”的PromptTemplate模板。
#真正的用户输入(message)会在下面format()时填入模板。


import json     #为了把json格式转为dict字典格式

from schema import AIAction

from ai.todo import (       # 从ai.todo模块导入 AI操作Todo数据的业务函数。
    addtodo_by_AI,          #根据用户自然语言，调用AI解析后新增Todo。
    updatetodo_by_AI,       #根据用户自然语言，调用AI解析后修改Todo。
    deletetodo_by_AI        #根据用户自然语言，调用AI匹配后删除Todo。
)
from ai.search import searchtodo_by_AI  #根据用户自然语言，调用AI匹配后查询Todo。

"""
项目中的两种AI调用方式：

1、普通JSON模式：
    ask_deepseek()
            |
            ↓
    返回固定JSON格式
            |
            ↓
    用于：
        - 用户意图识别(agent.py)
        - 新增Todo(todo.py)
        - 修改Todo(todo.py)
        - 删除Todo(todo.py)


2、Tool Calling模式：
    llm.bind_tools(ai_tools)
            |
            ↓
    AI返回tool_calls
            |
            ↓
    用于：
        - 查询Todo(search.py)
  
"""

### 1、定义AI判断用户意图函数
def ai_chat(message):
    prompt_mod = intent_prompt()                    #获取意图识别的Prompt模板（只是拿模板）
    #注意这里拿到的不是最终发送给AI的文本，而是一个包含 {message} 占位符的PromptTemplate对象。

    prompt = prompt_mod.format(message = message)   #使用真实用户输入填充Prompt模板（填充模板）
#.format()作用是：给模板里面的变量填入真实数据，左边的“message=”表示：我要给模板里的{message}这个变量赋值。右边的是ai_chat传入的参数
#如果PromptTemplate()里面的参数input_variables 又多个，比如input_variables=["time","message"]
# 调用prompt.format(time="2026-09-26",message="明天提醒我买牛奶")，两个变量都会被替换。

    result = ask_deepseek(prompt)            #调用普通JSON模式的大模型。AI根据intent_prompt判断用户想执行：增/删/改/查


    result = json.loads(result)     #把AI返回的是JSON字符串转换成Python字典
    #如JSON字符串格式'{"action":"search","content":"查询未完成任务"}'转成字典格式{"action":"search","content":"查询未完成任务"}

    action = AIAction(**result)     #使用Pydantic模型验证AI返回格式。AIAction要求：{action: str,content: str}
    #校验AI输出的格式是否满足AIAction，如action={"action": "search","content": "帮我查一下没完成的任务"}
    #并把输出结果变成AIAction对象AIAction("action": "search","content": "帮我查一下没完成的任务")
    #⭐前3行代码格式的转化流程为：DeepSeek  →  JSON字符串 → json.loads() →  dict →  AIAction(**dict) →  AIAction对象

    #后续通过解析器 parser = PydanticOutputParser(pydantic_object=AIAction)直接代替上面3行代码
    return action


### 2、AI总入口函数
def todo_by_AI(message):
    action = ai_chat(message)   #调用意图识别函数。AI判断意图，得出用户想操作的动作（增/删/改/查），并返回AIAction对象。
    # 例如ai_chat(message)输出：action = {"action": "search", "content": "帮我查一下没完成的任务"}

    # action = add / update / delete / search
    if action.action == "add":
        result = addtodo_by_AI(action.content)      #AI增
        #把用户输入的消息，当作参数传给addtodo_by_AI(用户输入的消息)

    elif action.action == "update":
        result = updatetodo_by_AI(action.content)   #AI改

    elif action.action == "delete":
        result = deletetodo_by_AI(action.content)   #AI删

    elif action.action == "search":
        result = searchtodo_by_AI(action.content)  #AI查

    else:
        raise ValueError("unknown action")


    return result

"""
   用户输入自然语言
        ↓
   todo_by_AI()
        ↓
   ai_chat()
        ↓
   ask_deepseek()
        ↓
   AI返回JSON
        ↓
   AIAction校验
        ↓
   根据action分发业务

add → addtodo_by_AI()
update → updatetodo_by_AI()
delete → deletetodo_by_AI()
search → searchtodo_by_AI()

"""
















