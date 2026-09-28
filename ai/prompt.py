"""各种prompt"""
"""
PromptTemplate = 专门给大模型设计的“可重复使用的提示词模板”。
它解决的是：固定提示词 + 动态用户输入 + 统一管理

后续生成提示词流程：
    函数
     ↓
    创建PromptTemplate对象
     ↓
    调用format()
     ↓
    生成最终Prompt
"""

from langchain_core.prompts import PromptTemplate
#导入LangChain的提示词模板类。作用：把以前手写字符串prompt，改成可复用的模板

"""⭐下面定义的函数均不直接接收参数。因为LangChain的设计是：先创建模板 → 后面调用format()填充真实数据"""
#~~~~~~~~~~~~~~~~~~意图识别类提示词~~~~~~~~~~~~~~~~~~
### 1、创建“判断用户操作(add/update/delete/search)”的提示词模板
# 作用：创建一个LangChain PromptTemplate对象。模板用于告诉AI：用户想执行什么操作
#模板中预留 {message} 变量,真正的用户输入会在调用模板时传入。返回格式 {"action":"add","content":"用户原话"}
def intent_prompt():     #创建一个LangChain提示词模板对象
    return PromptTemplate(      #PromptTemplate 是构建动态提示词的核心组件
        template=("你是一个todo助手。"  #template是参1：提示词模板的文本字符串。其中包含用花括号（例如 {variable}）包裹的占位符
            "判断用户想执行什么操作。"
            "只能返回json。"
            "action只能是add、update、delete、search。"
            "content返回用户原话。"
            "用户输入：{message}" ),     #这里的{message}是变量占位符，后续调用format()时，会把真实用户输入填入这里

        input_variables=["message"]     #input_variables是参2：输入变量名列表。指定模板中所有需要动态替换的变量名称
        )
# 函数 → 返回prompt模板 → 调用时传入用户输入 → 生成最终prompt    （所以函数本身不接收 message。）
# 调用时会给模板填入真实用户内容，text = prompt.format(message="明天上午提醒我买牛奶")，这时候是填充模板生成具体的字符串提示词



#~~~~~~~~~~~~~~~~~~查询类提示词~~~~~~~~~~~~~~~~~~
### 2、负责生成“AI查询Todo时选择工具”的提示词模板。
#作用：创建一个PromptTemplate。这个模板告诉AI：你有哪些查询工具，用户提出查询需求时应该选择哪个工具。
#这里不查询数据库。这里只负责告诉AI如何选择工具。
def search_prompt():
    return PromptTemplate(
        template="""
            你是一个Todo查询助手，根据用户需求选择合适的查询工具。
            
            规则：
                1. 如果用户查询标题关键词，调用 search_todos_by_title。
                2. 如果用户查询全部任务，调用 search_all_todos。
                3. 如果用户查询状态，例如未完成、已完成，调用 search_todos_by_status。
                4. 如果用户查询某个时间范围，例如：
                   - 今天有哪些任务
                   - 明天有哪些任务
                   - 后天有哪些任务
                   - 某日期有哪些任务
                    必须调用 search_todos_by_deadline。
                5. 如果用户同时提供多个查询条件：
                   调用 search_todos_advanced。

                   例如：
                   - 查询未完成的买牛肉任务
                   - 查询明天未完成任务
                   - 查询9月份pending任务
                
                deadline 查询需要生成：
                start_time:
                开始时间 YYYY-MM-DD 00:00:00
                
                end_time:
                结束时间 YYYY-MM-DD 23:59:59
                
                不要直接回答用户。
                必须选择工具。
            """,
        input_variables=[] )     #这个模板没有变量,所以填写空列表。


### 3、生成本次AI查询的用户提示词模板
#作用：把当前时间和用户的问题传给AI，帮助AI理解“今天、明天、后天”等时间表达
def search_user_prompt():
    # ⭐这里不直接接收message和now_time。因为LangChain的设计是：先创建模板 → 后面调用format()填充真实数据。

    return PromptTemplate(
        template="""
            你是一个Todo查询助手。
            
            当前时间：
            {now_time}
    
            请根据当前时间理解用户的日期表达。
            例如：
            明天 = 当前日期的下一天。
        
            用户说：
            {message}
            """,
        input_variables=["now_time","message"] )     #指定模板里面有哪些变量
# 后续调用：prompt.format(now_time="2026-09-26 21:00:00",message="帮我查明天的任务")
#LangChain会自动替换：{now_time} → 当前时间 ；{message} → 用户问题


### 4、查询结果返回Prompt模板
# 工具执行完成后，告诉AI如何根据查询结果生成最终回答
def search_result_prompt():
    return PromptTemplate(
        template ="""
        工具已经执行完成。
        不要再次调用任何工具，只根据工具返回的数据回答用户。
        如果 total 为 0，明确告诉用户没有找到符合条件的待办事项。
         """,       #这里没有动态变量,所以下面是空列表。
        input_variables=[] )
#后续调用：search_result_prompt().format() 只是把PromptTemplate转换成普通字符串。

#~~~~~~~~~~~~~~~~~~Todo操作类提示词(增、删、改)~~~~~~~~~~~~~~~~~~
### 5、负责生成AI新增Todo提示词模板
#作用：让AI从用户自然语言中提取：title deadline，后续交给TodoAdd校验，然后写入数据库。
def add_prompt():
    return PromptTemplate(
        template="""
        请根据用户的要求创建一个todo。
        
        返回json，必须包含title和deadline两个字段。
        deadline必须返回完整的日期时间，格式必须是YYYY-MM-DD HH:MM:SS。
        deadline禁止返回：
        明天
        下午3点
        后天
        
        当前时间是：{now_time}
        用户要求：{message}""",
        input_variables=["now_time","message"]      #指定模板里的变量。
    )


### 6、负责生成AI修改Todo提示词模板
#作用：第一次调用AI。只负责提取用户想修改的字段。
#不负责判断修改哪一条todo。找具体todo由第二次AI完成。
def update_prompt():
    return PromptTemplate(
        template="""
            请根据用户要求，提取需要修改的todo字段。

            只返回json格式。
            
            可以修改的字段：
            
            title:
            任务名称
            
            deadline:
            任务截止时间，格式必须为YYYY-MM-DD HH:MM:SS
            
            status:
            任务状态，只能是pending或者completed
            
            注意：
            1. 用户没有要求修改的字段，不要返回。
            2. 不要返回解释。
            3. deadline需要根据当前时间理解“今天、明天、下午”等时间表达。

            当前时间：
            {now_time}


            用户要求：
            {message}""",       #这里只创建模板。不负责传入用户输入
                            #真正传入数据的位置：todo.py里面调用：prompt = add_prompt().format( message=message,now_time=now_time)

        input_variables=["now_time","message"]  #模板变量：{now_time} 当前时间，让AI理解：今天、明天、后天等时间表达。
    )                                           #{message} 用户原始修改要求。


### 7、负责生成AI匹配Todo提示词模板（可充当删除提示词模板）
#作用：修改Todo、删除Todo时，根据用户描述，从数据库已有Todo列表里面找到目标id。
#这里只负责生成匹配提示词。不查询数据库。不删除、不修改数据。
def match_todo_prompt():
    return PromptTemplate(
        template="""
            你是一个todo匹配助手。

            根据用户描述，
            从todo列表中找到用户想操作的那一个。

            只返回json格式。
            返回格式：{{\"id\":数字}}   
                                 
            不要返回解释。
            
            用户描述：{message}

            todo列表：{todos}""",
        input_variables=["message","todos"]     #{message} ：用户想删除/修改的描述。
    )   #{todos}：数据库查询出来的Todo列表。

 #上面的{{ }}是因为PromptTemplate使用{}作为变量标记。（转意的功能）
#两个大括号表示：我想让AI看到一个普通的大括号，不是模板变量。（回归大括号本意）


