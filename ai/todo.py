"""AI操作单个Todo数据的业务逻辑: AI新增/修改/删除"""

from ai.client import ask_deepseek,llm
from db import (
    add_todo,
    update_all,
    delete_todo,
    get_todo_id,
    get_all_titles
)

from ai.prompt import (     #从prompt模块导入AI提示词生成函数
    add_prompt,             #生成AI新增Todo的提示词
    update_prompt,          #生成AI提取修改字段的提示词
    match_todo_prompt       #生成AI匹配目标Todo id的提示词
)


from datetime import datetime   #导入时间，让AI知道现在的时间，助于其理解我们说的明天、后天
import json     #为了把json格式转为dict字典格式
from pydantic import ValidationError
from fastapi import HTTPException


from schema import TodoAdd, AITodoUpdate


### 1、定义AI添加 Todo/数据 的业务函数
def addtodo_by_AI(message):
    try:
        now_time = datetime.now()

        prompt_mod = add_prompt()       #AI新增Todo的Prompt模板,add_prompt()现在不接收参数，只负责返回PromptTemplate对象
        prompt = prompt_mod.format(now_time=now_time,message=message)   #把当前时间和用户输入传入模板,给Prompt模板填充真实数据

        result = ask_deepseek(prompt)    #把生成好的完整prompt发送给DeepSeek
        result = json.loads(result)     #将上面“response_format=”指定的返回JSON字符串格式的结果，转化为dict字典格式
        #转为字典格式，就可以根据键取值了

        todo = TodoAdd(**result)     #把AI给出的 title和deadline，按照 TodoAdd 这个规则创建一个数据对象。
        # 字典 -> Pydantic对象,这里会校验AI输出的数据格式。todo是基于自定义类TodoAdd实例化出来的一个模型对象实例
        # ⭐上3行代码格式转化流程是：DeepSeek → JSON字符串 → json.loads() → dict → TodoAdd(**dict) → TodoAdd对象

        #后续通过解析器 parser = PydanticOutputParser(pydantic_object=TodoAdd)直接代替上面3行代码

        """
        ** 作用：把字典里的键和值，拆开作为函数/类的关键字参数传进去
        TodoAdd(**result)等价于：
            TodoAdd(                    #假如用户上传的title是买面包，deadline是2026-09-18 16:00:00
                title="买面包",
                deadline="2026-09-18 16:00:00")
        """
        new_id = add_todo(todo.title, todo.deadline)     #添加到数据库（上行代码创建的TodoAdd对象可以通过.的方式来取响应的值）

        return get_todo_id(new_id)      #返回新增后的完整数据

    #AI输出的数据不符合TodoAdd要求。 例如：{"title":"买牛奶"}，缺少deadline
    except ValidationError as e:
        raise Exception("ai output data validation failed") from e


    #其他未知错误.例如：DeepSeek接口失败、json解析失败、MySQL失败
    except Exception as e:
        # raise Exception(f"ai创建todo失败：{e}")    如果错误里面包含：api key 用户信息 数据库信息 可能泄露。
        raise Exception("ai create todo failed") from e
"""
AI新增流程：
    用户输入：
    "明天上午10点提醒我买牛奶"
            ↓
        agent.py
            ↓
      addtodo_by_AI()
            ↓
      add_prompt()
            ↓
    得到PromptTemplate
            ↓
    format(now_time,message)
            ↓
      生成完整prompt
            ↓
      ask_deepseek()
            ↓
        DeepSeek
            ↓
          JSON
            ↓
       json.loads()
            ↓
    TodoAdd(**result)
            ↓
        add_todo()
            ↓
          MySQL
"""

### 2、定义AI修改 Todo/数据 的业务函数
def updatetodo_by_AI(message):
    try:
        now_time = datetime.now()

        update1_prompt_mod = update_prompt()        #创建修改Todo的Prompt模板
        update1_prompt = update1_prompt_mod.format(now_time=now_time,message=message)   #填充模板变量
        result = ask_deepseek(update1_prompt)   #调用DeepSeek提取修改字段
        # 第一次调用AI：作用：只负责理解用户想修改什么字段,不负责判断修改哪一个todo

        # DeepSeek返回的是JSON字符串,例如：'{"status":"completed"}'。json.loads()将其转换成Python字典：{"status":"completed"}
        result = json.loads(result)

        # 字典解包给Pydantic模型。例如：AITodoUpdate(status="completed")这里会检查AI返回的数据格式是否正确
        todo = AITodoUpdate(**result)       #todo基于自定义类 AITodoUpdate 实例化出来的一个模型对象实例
        """
        为什么这里需要Pydantic？
        因为：
        LLM输出
            ↓
        可能格式错误的数据
            ↓
        Pydantic校验
            ↓
        业务代码
            ↓
        MySQL
        LLM不能直接相信。
        """

        # 第二次调用AI：作用：根据用户描述 + 数据库已有todo，找到用户真正想修改的是哪一条数据
        titles = get_all_titles()    #获取数据库所有title，为了AI匹配

        update2_prompt_mod = match_todo_prompt()        #创建匹配Todo的Prompt模板
        update2_prompt = update2_prompt_mod.format(message=message,todos=titles)        #填充模板
        match_result = ask_deepseek(update2_prompt)     #调用DeepSeek匹配目标Todo id

        # AI返回JSON字符串，例如：'{"id": 3}'，json.loads()将其转换成Python字典：{"id": 3}
        match_result = json.loads(match_result)
        todo_id = int(match_result["id"])
        #从字典中取出id，并转换成int类型，例如：3

        # 根据AI找到的id，利用查询数据函数找到对应的数据
        todo_data = get_todo_id(todo_id)    #返回找到的数据

        if todo_data is None:
            raise HTTPException(
                status_code=404,
                detail="todo not found"
            )

        # 如果AI返回了新的status，就使用AI的；如果AI没有返回status，就保留数据库原来的status
        status = (
            todo.status
            if todo.status is not None
            else todo_data["status"]
        )

        # 如果AI返回了新的deadline，就使用AI的；如果AI没有返回deadline，就保留数据库原来的deadline
        deadline = (
            todo.deadline
            if todo.deadline is not None
            else todo_data["deadline"]
        )


        # 根据数据库真实id修改数据
        result = update_all(        #update_all()返回修改了几条数据
            todo_data["id"],
            status,
            deadline
        )

        if result == 0:
            raise Exception("todo update failed")

        # 修改完成后重新查询最新数据返回
        return get_todo_id(todo_data["id"])

    # HTTPException需要原样抛出
    except HTTPException:
        raise

    # AI返回格式错误
    except ValidationError as e:
        raise Exception(
            "ai output data validation failed"
        ) from e

    # 其他错误
    except Exception as e:
        raise Exception("ai update todo failed") from e
"""
AI修改Todo流程：

    用户自然语言
        ↓
    updatetodo_by_AI()
        ↓
    第一次DeepSeek
        ↓
    提取用户想修改的字段(title/deadline/status)
        ↓
    AITodoUpdate校验AI输出
        ↓
    获取数据库已有Todo标题列表
        ↓
    第二次DeepSeek匹配目标Todo id
        ↓
    根据id查询原Todo数据
        ↓
    合并新字段和旧字段
        ↓
    update_all()更新数据库
        ↓
    重新查询并返回修改后的Todo
"""

"""
例如用户：
    “把买牛肉改成明天下午3点”
流程：
用户自然语言
        ↓
第一次DeepSeek
        ↓
提取需要修改的字段
        ↓
{
    "deadline":"2026-09-20 15:00:00"
}
        ↓
获取数据库所有todo标题
        ↓
第二次DeepSeek匹配目标todo
        ↓
返回todo对应id
        ↓
根据id查询数据库原数据
        ↓
合并新数据和原数据：
新的deadline/status + 原来的其他字段
        ↓
update_all(id, status, deadline)
        ↓
get_todo_id(id)
        ↓
返回修改后的完整todo
"""


### 11、定义AI删除数据函数
def deletetodo_by_AI(message):
    try:
        # 获取数据库已有Todo标题列表，用于AI匹配目标Todo
        todos = get_all_titles()

        prompt_mod = match_todo_prompt()     #创建匹配Todo的Prompt模板
        prompt = prompt_mod.format(message=message,todos=todos)     #填充模板变量

        #调用DeepSeek，根据用户描述匹配目标Todo id
        match_result = ask_deepseek(prompt)


        match_result = json.loads(match_result)
        todo_id = int(match_result["id"])

        # 根据AI找到的id，删除数据库中的todo
        result = delete_todo(todo_id)       #delete_todo()返回删除了几条数据

        # 如果删除数量为0，说明没有找到对应数据
        if result == 0:
            raise HTTPException(
                status_code=404,
                detail="todo not found"
            )

        # 返回删除结果
        return {
            "id": todo_id,
            "message": "todo deleted"
        }

    except HTTPException:
        raise

    except Exception as e:
        raise Exception("ai delete todo failed") from e
"""
AI删除Todo流程：

    用户自然语言
        ↓
    deletetodo_by_AI()
        ↓
    获取数据库已有Todo标题列表
        ↓
    创建match_todo_prompt模板
        ↓
    format填充用户描述和Todo列表
        ↓
    DeepSeek匹配目标Todo id
        ↓
    json.loads()
        ↓
    delete_todo()
        ↓
    返回删除结果
"""

