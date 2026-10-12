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
    
V1 主要通过普通 Python 字符串构造提示词，V2 使用 LangChain 的 PromptTemplate 统一管理模板
"""

from langchain_core.prompts import PromptTemplate
#导入LangChain的提示词模板类。作用：把以前手写字符串prompt，改成可复用的模板

"""⭐下面定义的函数均不直接接收参数。因为LangChain的设计是：先创建模板 → 后面调用format()填充真实数据"""
#~~~~~~~~~~~~~~~~~~意图识别类提示词~~~~~~~~~~~~~~~~~~
### 1、创建“判断用户操作(add/update/delete/search)”的提示词模板
# 作用：创建一个LangChain PromptTemplate对象。用于判断用户想执行什么操作。
# from_template() 会根据模板字符串自动识别输入变量，（相较于最初版本）不需要手动填写 input_variables。
#模板中预留 {message} 变量占位符,后续调用 format() 时会被真实的用户输入替换。返回格式 {"action":"add","content":"用户原话"}
"""
最初版本（使用构造方法：PromptTemplate）：
    def intent_prompt():     #创建一个LangChain提示词模板对象。返回的是PromptTemplate 对象
        return PromptTemplate(            #PromptTemplate 是构建动态提示词的核心组件
            template=("你是一个todo助手。"  #template是参1：提示词模板的文本字符串。其中包含用花括号（例如 {variable}）包裹的占位符
                "判断用户想执行什么操作。"
                "只能返回json。"
                "action只能是add、update、delete、search。"
                "content返回用户原话。"
                "用户输入：{message}" ),     #这里的{message}是变量占位符，后续调用format()时，会把真实用户输入填入这里
    
            input_variables=["message"]     #input_variables是参2：输入变量名列表。指定模板中所有需要动态替换的变量名称
            )
"""
#现在版本使用PromptTemplate.from_template（更简洁常用）
def intent_prompt():     #创建一个LangChain提示词模板对象。返回的是PromptTemplate 对象
    return PromptTemplate.from_template(            #PromptTemplate 是构建动态提示词的核心组件
            "你是一个todo助手。"           # 固定提示词：说明 AI 的角色
            "判断用户想执行什么操作。"       #告诉 AI 需要识别用户的操作意图
            "只能返回json。"               # 要求 AI 只返回 JSON 格式
            "action只能是add、update、delete、search。"    # 限定 action 的可选值
            "content返回用户原话。"        # 要求 content 保存用户原始请求
            "用户输入：{message}" )     #这里的{message}是变量占位符，后续调用format()时，会把真实用户输入填入这里


# 函数 → 返回prompt模板 → 调用时传入用户输入 → 生成最终prompt    （所以函数本身不接收 message。）
# prompt = intent_prompt()，text = prompt.format(message="明天上午提醒我买牛奶")，这时候是填充模板生成具体的字符串提示词


#~~~~~~~~~~~~~~~~~~查询类提示词~~~~~~~~~~~~~~~~~~
### 2、负责生成“AI查询Todo时选择工具”的提示词模板。
# 作用：创建一个 PromptTemplate 对象，告诉 AI 应该根据用户需求选择哪个查询工具。
# 这里不查询数据库，只负责指导 AI 选择工具。
# 这个模板没有动态变量，因此不需要在模板中填写 {变量名}。
# from_template() 会自动识别模板变量；没有变量时，可以直接传入固定的提示词字符串。
def search_prompt():
    return PromptTemplate.from_template(
            """
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
            """)     #这个模板没有变量


### 3、生成本次AI查询的用户提示词模板
#作用：把当前时间和用户的问题传给 AI，帮助 AI 理解“今天、明天、后天”等时间表达。
# 这里不直接接收 now_time 和 message，而是先创建模板对象，后续再填入真实数据。
# from_template() 会自动识别模板中的 {now_time} 和 {message} 两个变量。
def search_user_prompt():
    # ⭐这里不直接接收message和now_time。因为LangChain的设计是：先创建模板 → 后面调用format()填充真实数据。
    return PromptTemplate.from_template(
            """
            你是一个Todo查询助手。
            
            当前时间：
            {now_time}
    
            请根据当前时间理解用户的日期表达。
            例如：
            明天 = 当前日期的下一天。
        
            用户说：
            {message}
            """
            )
"""
 调用示例：
 prompt = search_user_prompt()  # 创建提示词模板对象
 text = prompt.format(now_time="2026-09-26 21:00:00",message="帮我查明天的任务")
 填入两个变量，生成最终提示词字符串
LangChain会自动替换：{now_time} → 当前时间 ；{message} → 用户问题
"""


### 4、查询结果返回Prompt模板
# 作用：工具执行完成后，告诉 AI 如何根据查询结果生成最终回答。
# 这个模板只有固定提示词，没有动态变量。
# from_template() 会自动识别模板变量，因此这里不需要手动填写 input_variables。
def search_result_prompt():
    return PromptTemplate.from_template(
        """
        工具已经执行完成。
        不要再次调用任何工具，只根据工具返回的数据回答用户。
        如果 total 为 0，明确告诉用户没有找到符合条件的待办事项。
        """
    )
"""
 调用示例：
    prompt = search_result_prompt()  # 创建提示词模板对象
    text = prompt.format()  # 没有动态变量，直接生成提示词字符串
"""

#~~~~~~~~~~~~~~~~~~Todo操作类提示词(增、删、改)~~~~~~~~~~~~~~~~~~
### 5、负责生成AI新增Todo提示词模板
# 作用：让 AI 从用户的自然语言中提取 title 和 deadline，后续交给 TodoAdd 校验，再写入数据库。
# 这里先创建提示词模板，不直接接收用户输入。
# from_template() 会自动识别模板中的 {now_time} 和 {message} 两个变量。
def add_prompt():
    return PromptTemplate.from_template(
        """
        请根据用户的要求创建一个todo。

        返回json，必须包含title和deadline两个字段。
        deadline必须返回完整的日期时间，格式必须是YYYY-MM-DD HH:MM:SS。
        deadline禁止返回：
        明天
        下午3点
        后天

        当前时间是：{now_time}
        用户要求：{message}"""
    )

"""
 调用示例：
    prompt = add_prompt()  # 创建提示词模板对象
    text = prompt.format(
        now_time="2026-09-26 21:00:00",
        message="明天下午3点提醒我买牛奶"
    )  # 填入两个变量，生成最终提示词字符串
 """


### 6、负责生成 AI 修改 Todo 提示词模板
# 作用：让 AI 从用户的自然语言中提取需要修改的 Todo 字段。
# 第一次调用 AI，只负责提取修改内容；不负责判断具体修改哪一条 Todo。
# 找到目标 Todo 由后续的匹配逻辑完成。
# from_template() 会自动识别 {now_time} 和 {message} 两个变量。
def update_prompt():
    return PromptTemplate.from_template(
        """
            请根据用户要求，提取需要修改的todo字段。
            只返回json格式，不要返回Markdown代码块。
            可以修改的字段：

            title:
            任务名称。用户要求修改任务名称时，必须返回修改后的完整标题。

            deadline:
            任务截止时间，格式必须为YYYY-MM-DD HH:MM:SS

            status:
            任务状态，只能是pending或者completed

            注意：
            1. 用户没有要求修改的字段，不要返回。
            2. 不要返回解释。
            3. deadline需要根据当前时间理解“今天、明天、下午”等时间表达。
            4. 用户要求将一个任务名称改成另一个名称时， 必须将新名称放入title字段，而不是返回旧名称。 
            5. 不要自行补充用户没有要求修改的字段。

            示例1： 用户要求：把买牛奶改成买咖啡 
            返回： {{"title": "买咖啡"}} 
            示例2： 用户要求：把买牛奶的状态改成已完成 
            返回： {{"status": "completed"}} 
            示例3： 用户要求：把买牛奶改成买咖啡，并标记为已完成 
            返回： {{"title": "买咖啡", "status": "completed"}}

            当前时间：
            {now_time}

            用户要求：
            {message}"""
    )

"""
 调用示例：
 prompt = update_prompt()  # 创建提示词模板对象
 text = prompt.format(
     now_time="2026-09-26 21:00:00",
     message="把买牛奶改成买咖啡"
 )  # 填入两个变量，生成最终提示词字符串
 """
#⭐上面的{{ }}是因为PromptTemplate使用{}作为变量标记。（转意的功能）
#⭐两个大括号表示：我想让AI看到一个普通的大括号，不是模板变量。（回归大括号本意）


### 7、负责生成 AI 匹配 Todo 提示词模板（可充当删除提示词模板）
# 作用：修改或删除 Todo 时，让 AI 根据用户描述，从数据库已有的 Todo 列表中找到目标 id。
# 这里只负责生成匹配提示词，不查询数据库，也不直接修改或删除数据。
# from_template() 会自动识别 {message} 和 {todos} 两个变量。
def match_todo_prompt():
    return PromptTemplate.from_template(
        """
            你是一个todo匹配助手。

            根据用户描述，
            从todo列表中找到用户想操作的那一个。

            只返回json格式。
            返回格式：{{\"id\":数字}}   

            不要返回解释。

            用户描述：{message}

            todo列表：{todos}"""
    )
"""
 调用示例：
 prompt = match_todo_prompt()  # 创建提示词模板对象
 text = prompt.format(
     message="删除买牛奶",
     todos="[{'id': 1, 'title': '买牛奶'}]"
 )  # 填入两个变量，生成最终提示词字符串
 注意：JSON 示例中的 {{ 和 }} 要保留。
 在 PromptTemplate 中，双大括号用于表示普通大括号，避免被识别为模板变量。
 """






